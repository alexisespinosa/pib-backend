#!/usr/bin/python3
import base64

import cv2
import depthai as dai
import rclpy
from datatypes.srv import GetCameraImage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float64, Int32, Int32MultiArray, String

# All measurements published by this node carry this frame_id.
# Future sensors (left/right mono, depth) will get sibling frame_ids.
FRAME_ID = "oak_d_lite_rgb"

# Latched QoS: late subscribers immediately receive the most recent message.
# Used for CameraInfo since intrinsics are static after device-open.
LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)


class ErrorPublisher(Node):

    # def __new__(cls, error_message):
    #    print("creating new ErrorPublisher with Error message" + error_message)

    def __init__(self):
        super().__init__("error_publisher")
        self.publisher_ = self.create_publisher(String, "camera_topic", 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.current_image = ""

    def timer_callback(self):
        msg = String()
        msg.data = "Camera not available: "
        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)


class CameraNode(Node):

    def __init__(self):
        super().__init__("camera_node")
        # Legacy publisher — kept until Cerebra migrates to /vision/raw_frame_b64
        # (step 4 of docs/vision-architecture.md migration plan).
        self.legacy_camera_topic_pub = self.create_publisher(
            String, "camera_topic", 10
        )
        # Always-on /vision/* topics (sub-step 3.2 of the migration plan).
        self.raw_frame_pub = self.create_publisher(
            CompressedImage, "/vision/raw_frame", 10
        )
        self.raw_frame_b64_pub = self.create_publisher(
            String, "/vision/raw_frame_b64", 10
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo, "/vision/camera_info", LATCHED_QOS
        )

        self.timer_subscription = self.create_subscription(
            Float64, "timer_period_topic", self.timer_period_callback, 10
        )
        self.quality_factor_subscription = self.create_subscription(
            Int32, "quality_factor_topic", self.quality_factor_callback, 10
        )
        self.preview_size_subscription = self.create_subscription(
            Int32MultiArray, "size_topic", self.preview_size_callback, 10
        )

        # Initialize default preview size and quality factor
        self.preview_width = 1280
        self.preview_height = 720
        self.quality_factor = 80

        # Initialize pipeline when camera is available
        self.camera_available = self.init_pipeline()

        if self.camera_available:
            self.get_camera_image_service = self.create_service(
                GetCameraImage, "get_camera_image", self.get_camera_image_callback
            )
            self.publish_camera_info()
            self.get_logger().info("Camera service initialized.")
        else:
            self.get_logger().error("Camera not available.")

        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def get_camera_image_callback(self, request, response):
        self.get_logger().info(f"LEN IMAGE: {len(self.current_image)}")
        response.image_base64 = self.current_image
        return response

    def init_pipeline(self) -> bool:
        try:
            self.pipeline = dai.Pipeline()

            # Define a source - color camera
            self.camRgb = self.pipeline.createColorCamera()
            self.camRgb.setPreviewSize(self.preview_width, self.preview_height)
            self.camRgb.setInterleaved(False)

            # Create output
            xoutRgb = self.pipeline.createXLinkOut()
            xoutRgb.setStreamName("rgb")
            self.camRgb.preview.link(xoutRgb.input)

            # Try to connect to device
            self.device = dai.Device(self.pipeline)

            # Output queue will be used to get the rgb frames from the output defined above
            self.queue = self.device.getOutputQueue(
                name="rgb", maxSize=4, blocking=False
            )
            return True

        except Exception as e:
            self.get_logger().error(f"Camera not found: {e}")
            self.device = None
            self.queue = None
            return False

    def publish_camera_info(self):
        """Publish a latched CameraInfo built from the OAK-D's on-device calibration."""
        try:
            calib = self.device.readCalibration()
            K = calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, self.preview_width, self.preview_height
            )
            distortion = calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
        except Exception as e:
            self.get_logger().error(f"Failed to read calibration for CameraInfo: {e}")
            return

        msg = CameraInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.width = self.preview_width
        msg.height = self.preview_height
        # depthai returns 14 distortion coefficients; OpenCV's plumb_bob model
        # uses the first 5 (k1, k2, p1, p2, k3).
        msg.distortion_model = "plumb_bob"
        msg.d = [float(d) for d in distortion[:5]]
        msg.k = [float(v) for row in K for v in row]
        # No rectification for a single (non-stereo) camera.
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # Projection P = [K | 0] for an unrectified single camera.
        msg.p = [
            float(K[0][0]), float(K[0][1]), float(K[0][2]), 0.0,
            float(K[1][0]), float(K[1][1]), float(K[1][2]), 0.0,
            float(K[2][0]), float(K[2][1]), float(K[2][2]), 0.0,
        ]
        self.camera_info_pub.publish(msg)
        self.get_logger().info(
            f"Published CameraInfo: {msg.width}x{msg.height} "
            f"fx={K[0][0]:.1f} fy={K[1][1]:.1f} cx={K[0][2]:.1f} cy={K[1][2]:.1f}"
        )

    def timer_callback(self):
        if not self.queue:
            return
        image_rgb = self.queue.tryGet()  # non-blocking call
        if image_rgb is None:
            return
        # data is originally represented as a flat 1D array, it needs to be converted into HxWxC form
        frame = image_rgb.getCvFrame()

        # Encode once, publish to all three topics.
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality_factor]
        )
        if not ok:
            self.get_logger().warn("cv2.imencode returned failure; dropping frame")
            return
        jpg_bytes = buffer.tobytes()
        jpg_b64 = base64.b64encode(buffer).decode("utf-8")

        stamp = self.get_clock().now().to_msg()

        # Legacy /camera_topic (Cerebra still consumes this). Remove after step 4.
        legacy_msg = String()
        legacy_msg.data = jpg_b64
        self.current_image = legacy_msg.data
        self.legacy_camera_topic_pub.publish(legacy_msg)

        # /vision/raw_frame (binary CompressedImage, ROS-native)
        compressed_msg = CompressedImage()
        compressed_msg.header.stamp = stamp
        compressed_msg.header.frame_id = FRAME_ID
        compressed_msg.format = "jpeg"
        compressed_msg.data = jpg_bytes
        self.raw_frame_pub.publish(compressed_msg)

        # /vision/raw_frame_b64 (String, for Cerebra rosbridge after step 4)
        b64_msg = String()
        b64_msg.data = jpg_b64
        self.raw_frame_b64_pub.publish(b64_msg)

    def timer_period_callback(self, msg):
        self.timer_period = msg.data
        self.timer.cancel()  # cancel the old timer
        self.timer = self.create_timer(
            self.timer_period, self.timer_callback
        )  # create a new timer with updated period

    def quality_factor_callback(self, msg):
        self.quality_factor = msg.data

    def preview_size_callback(self, msg):
        self.preview_width, self.preview_height = msg.data

        # Reset pipeline with new preview size
        self.device.close()
        if self.init_pipeline():
            # CameraInfo dimensions (and possibly intrinsics) changed — re-publish.
            self.publish_camera_info()


def spin_camera(times):
    cnt = times
    if cnt == 0:
        print(
            "Couldn't restart camera due to displayed error/s, publishing error message"
        )
        rclpy.spin(error_publisher)
    else:
        try:
            camera_node = CameraNode()
            rclpy.spin(camera_node)
        except Exception as exc:
            error_publisher.timer_callback()
            print(exc)
        finally:
            if "camera_node" in locals():
                camera_node.destroy_node()
                print("camera_node destroyed")
            cnt = times - 1
            print("Retry starting camera..." + str(cnt))
            spin_camera(cnt)
    return


def main(args=None):
    rclpy.init()
    global error_publisher
    error_publisher = ErrorPublisher()
    print("Starting camera")
    spin_camera(3)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
