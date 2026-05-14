import os
from queue import Queue
from threading import Thread

import pyaudio
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from piper.voice import PiperVoice


class TtsNode(Node):
    def __init__(self):
        super().__init__("tts_node")

        model_path = os.getenv("TTS_MODEL_PATH", "/app/piper-voice")
        model_file = f"{model_path}/model.onnx"
        try:
            self.voice = PiperVoice.load(model_file)
            self.get_logger().info(f"Piper voice loaded from {model_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to load Piper voice from {model_file}: {e}")
            self.voice = None

        self._queue: Queue[str] = Queue()
        Thread(target=self._playback_worker, daemon=True).start()

        self.create_subscription(String, "/speech/say", self._on_say, 10)
        self.get_logger().info("TTS node ready")

    def _on_say(self, msg):
        text = msg.data.strip()
        if text and self.voice is not None:
            self._queue.put(text)

    def _playback_worker(self):
        pya = pyaudio.PyAudio()
        while True:
            text = self._queue.get()
            try:
                rate = self.voice.config.sample_rate
                stream = pya.open(
                    format=pyaudio.paInt16, channels=1, rate=rate, output=True
                )
                for chunk in self.voice.synthesize_stream_raw(text):
                    stream.write(chunk)
                stream.stop_stream()
                stream.close()
            except Exception as e:
                self.get_logger().error(f"TTS playback failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TtsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
