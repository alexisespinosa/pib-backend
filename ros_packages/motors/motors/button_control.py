import json
import math
import time
import logging
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.action.client import GoalStatus
from std_msgs.msg import String, Int32MultiArray
from datatypes.srv import ProxyRunProgramStart, ProxyRunProgramStop
from datatypes.msg import ProxyRunProgramStatus

from pib_motors.bricklet import ipcon
from tinkerforge.bricklet_rgb_led_button import BrickletRGBLEDButton
from tinkerforge.ip_connection import IPConnection

from pib_api_client import send_request, URL_PREFIX
from urllib.request import Request

BREATHING_HZ = 20
BREATHING_PERIOD_S = 2.0
BRIGHTNESS_MIN = 0.3
BRIGHTNESS_MAX = 1.0

TERMINAL_STATUSES = {
    GoalStatus.STATUS_SUCCEEDED,
    GoalStatus.STATUS_ABORTED,
    GoalStatus.STATUS_CANCELED,
}


def _resolve_programs(program_mappings: dict[str, str]) -> dict[str, str]:
    names_to_resolve = {
        v for v in program_mappings.values() if v and not _looks_like_uuid(v)
    }
    if not names_to_resolve:
        return program_mappings

    request = Request(URL_PREFIX + "/program", method="GET")
    success, data = send_request(request)
    if not success or not data:
        logging.warning("Could not fetch programs from API for name resolution")
        return program_mappings

    name_to_number = {p["name"]: p["programNumber"] for p in data.get("programs", [])}
    resolved = {}
    for pos, val in program_mappings.items():
        if val and not _looks_like_uuid(val) and val in name_to_number:
            resolved[pos] = name_to_number[val]
            logging.info(f"Resolved program '{val}' -> {name_to_number[val]}")
        else:
            resolved[pos] = val
    return resolved


def _looks_like_uuid(s: str) -> bool:
    return len(s) == 36 and s.count("-") == 4


class ButtonState:
    def __init__(self, bricklet: BrickletRGBLEDButton, position: str, default_color: tuple[int, int, int]):
        self.bricklet = bricklet
        self.position = position
        self.default_color = default_color
        self.program_number: str = ""
        self.proxy_goal_id: str = ""
        self.running = False


class ButtonControlNode(Node):

    def __init__(self):
        super().__init__("button_control")

        self.declare_parameter("button_f_color", [0, 255, 0])
        self.declare_parameter("button_g_color", [255, 255, 255])
        self.declare_parameter("button_h_color", [255, 0, 0])
        self.declare_parameter("brightness", 0.3)
        self.declare_parameter("button_f_program", "")
        self.declare_parameter("button_g_program", "")
        self.declare_parameter("button_h_program", "")

        self._buttons: dict[str, ButtonState] = {}
        self._lock = Lock()
        self._breathing_start = 0.0

        self._discover_bricklets()

        if not self._buttons:
            self.get_logger().warn("No RGB LED Button Bricklets found")
            return

        self._apply_default_colors()
        self._setup_program_mappings()
        self._register_button_callbacks()

        self.event_pub = self.create_publisher(String, "/buttons/event", 10)
        self.create_subscription(
            Int32MultiArray, "/buttons/set_color", self._on_set_color, 10
        )

        self.start_client = self.create_client(
            ProxyRunProgramStart, "proxy_run_program_start"
        )
        self.stop_client = self.create_client(
            ProxyRunProgramStop, "proxy_run_program_stop"
        )

        self.create_subscription(
            ProxyRunProgramStatus, "proxy_run_program_status",
            self._on_program_status, 10,
        )

        self._breathing_timer = None
        self._update_breathing_timer()

        self.get_logger().info(
            f"Button control node ready — {len(self._buttons)} bricklet(s)"
        )

    def _discover_bricklets(self):
        discovered = {}

        def enum_cb(uid, connected_uid, position, hw_ver, fw_ver, dev_id, enum_type):
            if dev_id == BrickletRGBLEDButton.DEVICE_IDENTIFIER:
                if enum_type == IPConnection.ENUMERATION_TYPE_AVAILABLE:
                    discovered[uid] = position

        ipcon.register_callback(
            IPConnection.CALLBACK_ENUMERATE, enum_cb
        )
        ipcon.enumerate()
        time.sleep(1.5)

        color_params = {
            "f": self.get_parameter("button_f_color").value,
            "g": self.get_parameter("button_g_color").value,
            "h": self.get_parameter("button_h_color").value,
        }

        for uid, position in discovered.items():
            color = color_params.get(position, [255, 255, 255])
            bricklet = BrickletRGBLEDButton(uid, ipcon)
            self._buttons[uid] = ButtonState(
                bricklet=bricklet,
                position=position,
                default_color=tuple(color),
            )
            self.get_logger().info(
                f"Found RGB LED Button: UID={uid} pos={position}"
            )

    def _set_default_color(self, state: ButtonState):
        brightness = self.get_parameter("brightness").value * BRIGHTNESS_MIN
        r, g, b = state.default_color
        state.bricklet.set_color(
            int(r * brightness), int(g * brightness), int(b * brightness)
        )

    def _apply_default_colors(self):
        for state in self._buttons.values():
            self._set_default_color(state)

    def _setup_program_mappings(self):
        raw = {
            "f": self.get_parameter("button_f_program").value,
            "g": self.get_parameter("button_g_program").value,
            "h": self.get_parameter("button_h_program").value,
        }
        resolved = _resolve_programs(raw)
        for state in self._buttons.values():
            prog = resolved.get(state.position, "")
            if prog:
                state.program_number = prog
                self.get_logger().info(
                    f"Button {state.position}: mapped to program {prog}"
                )

    def _register_button_callbacks(self):
        for uid, state in self._buttons.items():
            state.bricklet.register_callback(
                BrickletRGBLEDButton.CALLBACK_BUTTON_STATE_CHANGED,
                lambda button_state, _uid=uid: self._on_button_state_changed(
                    _uid, button_state
                ),
            )

    def _on_button_state_changed(self, uid: str, button_state: int):
        state = self._buttons.get(uid)
        if state is None:
            return

        pressed = button_state == BrickletRGBLEDButton.BUTTON_STATE_PRESSED
        event = {
            "uid": uid,
            "position": state.position,
            "state": "pressed" if pressed else "released",
        }
        msg = String()
        msg.data = json.dumps(event)
        self.event_pub.publish(msg)

        if not pressed:
            return

        if not state.program_number:
            return

        with self._lock:
            if state.running:
                self._stop_program(state)
            else:
                self._start_program(state)

    def _start_program(self, state: ButtonState):
        if not self.start_client.service_is_ready():
            self.get_logger().warn("proxy_run_program_start service not available")
            return

        req = ProxyRunProgramStart.Request()
        req.program_number = state.program_number
        future = self.start_client.call_async(req)
        future.add_done_callback(
            lambda f, s=state: self._on_start_response(s, f)
        )

    def _on_start_response(self, state: ButtonState, future):
        try:
            resp = future.result()
            with self._lock:
                state.proxy_goal_id = resp.proxy_goal_id
                state.running = True
                self._breathing_start = time.monotonic()
                self._update_breathing_timer()
            self.get_logger().info(
                f"Button {state.position}: started program "
                f"(goal={resp.proxy_goal_id})"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to start program: {e}")

    def _stop_program(self, state: ButtonState):
        if not state.proxy_goal_id:
            return
        if not self.stop_client.service_is_ready():
            self.get_logger().warn("proxy_run_program_stop service not available")
            return

        req = ProxyRunProgramStop.Request()
        req.proxy_goal_id = state.proxy_goal_id
        future = self.stop_client.call_async(req)
        future.add_done_callback(
            lambda f, s=state: self._on_stop_response(s, f)
        )

    def _on_stop_response(self, state: ButtonState, future):
        try:
            future.result()
            with self._lock:
                state.running = False
                state.proxy_goal_id = ""
                self._set_default_color(state)
                self._update_breathing_timer()
            self.get_logger().info(f"Button {state.position}: stopped program")
        except Exception as e:
            self.get_logger().error(f"Failed to stop program: {e}")

    def _on_program_status(self, msg: ProxyRunProgramStatus):
        if msg.status not in TERMINAL_STATUSES:
            return
        with self._lock:
            for state in self._buttons.values():
                if state.proxy_goal_id == msg.proxy_goal_id and state.running:
                    state.running = False
                    state.proxy_goal_id = ""
                    self._set_default_color(state)
                    self._update_breathing_timer()
                    self.get_logger().info(
                        f"Button {state.position}: program ended (status={msg.status})"
                    )
                    break

    def _update_breathing_timer(self):
        any_running = any(s.running for s in self._buttons.values())
        if any_running and self._breathing_timer is None:
            self._breathing_start = time.monotonic()
            self._breathing_timer = self.create_timer(
                1.0 / BREATHING_HZ, self._breathing_tick
            )
        elif not any_running and self._breathing_timer is not None:
            self._breathing_timer.cancel()
            self.destroy_timer(self._breathing_timer)
            self._breathing_timer = None

    def _breathing_tick(self):
        brightness = self.get_parameter("brightness").value
        t = time.monotonic() - self._breathing_start
        phase = math.sin(2.0 * math.pi * t / BREATHING_PERIOD_S)
        scale = brightness * (BRIGHTNESS_MIN + (BRIGHTNESS_MAX - BRIGHTNESS_MIN) * (phase + 1.0) / 2.0)

        for state in self._buttons.values():
            if state.running:
                r = int(state.default_color[0] * scale)
                g = int(state.default_color[1] * scale)
                b = int(state.default_color[2] * scale)
                state.bricklet.set_color(r, g, b)

    def _on_set_color(self, msg: Int32MultiArray):
        data = msg.data
        if len(data) != 4:
            self.get_logger().warn("set_color expects [button_index, r, g, b]")
            return
        idx, r, g, b = data[0], data[1], data[2], data[3]
        buttons_list = list(self._buttons.values())
        if idx < 0 or idx >= len(buttons_list):
            self.get_logger().warn(f"Invalid button index: {idx}")
            return
        state = buttons_list[idx]
        state.default_color = (r, g, b)
        if not state.running:
            self._set_default_color(state)


def main(args=None):
    rclpy.init(args=args)
    node = ButtonControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
