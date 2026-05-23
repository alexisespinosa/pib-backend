import os
import sys
from dataclasses import dataclass
from io import BytesIO
from queue import Queue, Empty
from threading import Thread

import pygame

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from datatypes.msg import DisplayImage, DisplayOverlay, ImageFormat, ImageId, ProxyRunProgramStatus

from display.face import FaceRenderer, SCREEN_W, SCREEN_H

os.environ.setdefault("DISPLAY", ":0.0")
os.environ.setdefault("SDL_VIDEODRIVER", "x11")

STATIC_IMAGE_DIR: str = os.getenv(
    "STATIC_IMAGE_DIR",
    "/home/pib/ros_working_dir/src/display/static_images",
)

FPS = 30


# ── Data types for queue messages ───────────────────────────────────

@dataclass
class CameraFrame:
    """A camera image (JPEG/PNG bytes) to display."""
    data: bytes
    format_value: int


@dataclass
class OverlayData:
    rects: list  # [(x, y, w, h, label), ...]


@dataclass
class EmotionCommand:
    name: str


# ── Display node (ROS) ──────────────────────────────────────────────

MAX_OVERLAY_ITEMS = 10


class DisplayNode(Node):

    _PROGRAM_ACCEPTED = 1
    _PROGRAM_EXECUTING = 2
    _PROGRAM_SUCCEEDED = 4
    _PROGRAM_CANCELED = 5
    _PROGRAM_ABORTED = 6

    def __init__(self, command_queue: Queue):
        super().__init__("display")
        self._queue = command_queue
        self._program_running = False

        self.create_subscription(
            DisplayImage, "display_image", self._on_display_image, 1
        )
        self.create_subscription(
            DisplayOverlay, "display_overlay", self._on_display_overlay, 1
        )
        self.create_subscription(
            String, "display_emotion", self._on_emotion, 1
        )
        self.create_subscription(
            ProxyRunProgramStatus, "proxy_run_program_status",
            self._on_program_status, 1
        )

        self._queue.put(EmotionCommand("sleeping"))
        self.get_logger().info("Display node ready (pygame face renderer)")

    def _on_display_image(self, msg: DisplayImage):
        image_id = msg.id.value
        if image_id == ImageId.NONE:
            self._queue.put(EmotionCommand("sleeping"))
            return
        if image_id == ImageId.PIB_EYES_ANIMATED:
            self._queue.put(EmotionCommand("sleeping"))
            return

        if image_id == ImageId.CUSTOM:
            data = bytes(msg.data)
        else:
            filepath = _IMAGE_ID_TO_PATH.get(image_id)
            if filepath and os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    data = f.read()
            else:
                self.get_logger().warn(f"Unknown image id: {image_id}")
                return

        self._queue.put(CameraFrame(data=data, format_value=msg.format.value))

    def _on_display_overlay(self, msg: DisplayOverlay):
        rects = []
        for i in range(len(msg.x)):
            label = msg.labels[i] if i < len(msg.labels) else ""
            rects.append((msg.x[i], msg.y[i], msg.width[i], msg.height[i], label))
        self._queue.put(OverlayData(rects=rects))

    def _on_emotion(self, msg: String):
        self._queue.put(EmotionCommand(msg.data.strip().lower()))

    def _on_program_status(self, msg: ProxyRunProgramStatus):
        status = msg.status
        if status in (self._PROGRAM_ACCEPTED, self._PROGRAM_EXECUTING):
            if not self._program_running:
                self._program_running = True
                self._queue.put(EmotionCommand("neutral"))
        elif status in (self._PROGRAM_SUCCEEDED, self._PROGRAM_ABORTED, self._PROGRAM_CANCELED):
            if self._program_running:
                self._program_running = False
                self._queue.put(EmotionCommand("sleeping"))


_IMAGE_ID_TO_PATH: dict[int, str] = {
    ImageId.PIB_EYES_ANIMATED: os.path.join(STATIC_IMAGE_DIR, "pib-eyes-animated.gif"),
}


# ── Pygame display loop ────────────────────────────────────────────

def _load_camera_frame(frame: CameraFrame) -> pygame.Surface | None:
    try:
        buf = BytesIO(frame.data)
        img = pygame.image.load(buf)
        return pygame.transform.scale(img, (SCREEN_W, SCREEN_H))
    except Exception:
        return None


def _draw_overlays(surface: pygame.Surface, overlay: OverlayData):
    font = pygame.font.SysFont("sans", 16)
    for x, y, w, h, label in overlay.rects:
        x1 = int((x - w / 2) * SCREEN_W)
        y1 = int((y - h / 2) * SCREEN_H)
        x2 = int((x + w / 2) * SCREEN_W)
        y2 = int((y + h / 2) * SCREEN_H)
        rect = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
        pygame.draw.rect(surface, (0, 255, 0), rect, 2)
        if label:
            text = font.render(label, True, (0, 255, 0))
            surface.blit(text, (x1, y1 - 18))


def run_display(command_queue: Queue):
    pygame.init()
    pygame.font.init()
    pygame.mouse.set_visible(False)

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.FULLSCREEN | pygame.NOFRAME)
    pygame.display.set_caption("pib")
    clock = pygame.time.Clock()

    face = FaceRenderer()

    # Display mode: "face" or "camera"
    mode = "face"
    camera_surface: pygame.Surface | None = None
    current_overlay: OverlayData | None = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Drain command queue
        while True:
            try:
                cmd = command_queue.get_nowait()
            except Empty:
                break

            if isinstance(cmd, EmotionCommand):
                mode = "face"
                face.set_emotion(cmd.name)
                current_overlay = None
            elif isinstance(cmd, CameraFrame):
                surf = _load_camera_frame(cmd)
                if surf:
                    camera_surface = surf
                    mode = "camera"
            elif isinstance(cmd, OverlayData):
                current_overlay = cmd

        # Render
        if mode == "face":
            face.draw(screen)
        elif mode == "camera" and camera_surface is not None:
            screen.blit(camera_surface, (0, 0))
            if current_overlay:
                _draw_overlays(screen, current_overlay)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


# ── ROS thread ──────────────────────────────────────────────────────

def run_ros_node(command_queue: Queue):
    rclpy.init()
    executor = SingleThreadedExecutor()
    node = DisplayNode(command_queue)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ── Main ────────────────────────────────────────────────────────────

def main(args=None):
    command_queue: Queue = Queue(maxsize=0)
    Thread(daemon=True, target=run_ros_node, args=(command_queue,)).start()
    run_display(command_queue)


if __name__ == "__main__":
    main()
