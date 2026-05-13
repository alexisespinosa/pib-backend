import json
import os

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, String
from vosk import Model, KaldiRecognizer

SUBSCRIBER_POLL_PERIOD_S = 1.0


class SttNode(Node):
    def __init__(self):
        super().__init__("stt_node")

        model_path = os.getenv("STT_MODEL_PATH", "/app/vosk-model")
        try:
            self.model = Model(model_path)
            self.get_logger().info(f"Vosk model loaded from {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load Vosk model from {model_path}: {e}")
            self.model = None

        self.speech_pub = self.create_publisher(String, "/hearing/speech", 10)
        self._active = False
        self._audio_sub = None
        self._recognizer = None

        self.subscriber_poll_timer = self.create_timer(
            SUBSCRIBER_POLL_PERIOD_S, self._check_lazy_subscribers
        )

        self.get_logger().info(
            f"STT node initialized (model loaded: {self.model is not None})"
        )

    def _check_lazy_subscribers(self):
        subs = self.speech_pub.get_subscription_count()
        if subs > 0 and not self._active:
            self._activate()
        elif subs == 0 and self._active:
            self._deactivate()

    def _activate(self):
        if self.model is None:
            return
        self._recognizer = KaldiRecognizer(self.model, 16000)
        self._audio_sub = self.create_subscription(
            Int16MultiArray, "audio_stream", self._on_audio, 10
        )
        self._active = True
        self.get_logger().info(
            "Speech subscribers appeared; STT active"
        )

    def _deactivate(self):
        if self._audio_sub is not None:
            self.destroy_subscription(self._audio_sub)
            self._audio_sub = None
        self._recognizer = None
        self._active = False
        self.get_logger().info("No speech subscribers; STT paused")

    def _on_audio(self, msg):
        if self._recognizer is None:
            return
        audio_bytes = np.array(msg.data, dtype=np.int16).tobytes()
        if self._recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(self._recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                speech_msg = String()
                speech_msg.data = text
                self.speech_pub.publish(speech_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SttNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
