import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32
import usb.core

from ros_audio_io.tuning import Tuning

SUBSCRIBER_POLL_PERIOD_S = 1.0


class HearingNode(Node):
    def __init__(self):
        super().__init__("hearing_node")

        self.dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if self.dev is None:
            self.get_logger().error(
                "ReSpeaker Mic Array v2.0 not found! Make sure it's connected."
            )
            self.mic_tuning = None
        else:
            self.mic_tuning = Tuning(self.dev)

        self.doa_pub = self.create_publisher(Int32, "/hearing/doa", 10)
        self.vad_pub = self.create_publisher(Bool, "/hearing/voice_activity", 10)

        self._doa_publishing = False
        self._vad_publishing = False

        try:
            interval = float(os.getenv("HEARING_PUBLISH_INTERVAL", "0.2"))
        except ValueError:
            self.get_logger().warn(
                "Invalid HEARING_PUBLISH_INTERVAL, defaulting to 0.2s"
            )
            interval = 0.2

        self.publish_timer = self.create_timer(interval, self._publish_callback)
        self.subscriber_poll_timer = self.create_timer(
            SUBSCRIBER_POLL_PERIOD_S, self._check_lazy_subscribers
        )

        self.get_logger().info(
            f"Hearing node initialized (publish interval: {interval:.3f}s, "
            f"device found: {self.mic_tuning is not None})"
        )

    def _check_lazy_subscribers(self):
        doa_subs = self.doa_pub.get_subscription_count()
        if doa_subs > 0 and not self._doa_publishing:
            self._doa_publishing = True
            self.get_logger().info(
                f"DoA consumers appeared ({doa_subs}); publishing /hearing/doa"
            )
        elif doa_subs == 0 and self._doa_publishing:
            self._doa_publishing = False
            self.get_logger().info("No DoA consumers; pausing /hearing/doa")

        vad_subs = self.vad_pub.get_subscription_count()
        if vad_subs > 0 and not self._vad_publishing:
            self._vad_publishing = True
            self.get_logger().info(
                f"VAD consumers appeared ({vad_subs}); "
                "publishing /hearing/voice_activity"
            )
        elif vad_subs == 0 and self._vad_publishing:
            self._vad_publishing = False
            self.get_logger().info(
                "No VAD consumers; pausing /hearing/voice_activity"
            )

    def _publish_callback(self):
        if self.mic_tuning is None:
            return

        if self._doa_publishing:
            msg = Int32()
            msg.data = self.mic_tuning.direction
            self.doa_pub.publish(msg)

        if self._vad_publishing:
            msg = Bool()
            msg.data = bool(self.mic_tuning.is_voice())
            self.vad_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HearingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
