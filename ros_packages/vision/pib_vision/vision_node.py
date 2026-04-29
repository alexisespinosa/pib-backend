#!/usr/bin/python3
import base64
import time

import depthai as dai
import rclpy
from datatypes.srv import GetCameraImage
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float64, Int32, Int32MultiArray, String
from tf2_ros import StaticTransformBroadcaster

# All RGB-derived measurements published by this node carry this frame_id.
# Future sensors (left/right mono, depth) will get sibling frame_ids.
FRAME_ID = "oak_d_lite_rgb"
# Logical root of the OAK-D's internal TF subtree. NOT connected to any
# robot-body frame here — see docs/vision-architecture.md §4.3.
DEVICE_FRAME_ID = "oak_d_lite_link"

# Latched QoS: late subscribers immediately receive the most recent message.
# Used for CameraInfo since intrinsics are static after device-open.
LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

# How long to wait between retries when the OAK-D's USB connection is in
# the "device in use" state (typically right after a prior process closed
# the connection — USB takes a few seconds to fully release).
RETRY_DELAY_SECONDS = 5


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

        self.tf_broadcaster = StaticTransformBroadcaster(self)

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
            self.publish_static_transforms()
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

            # Color camera. Video output goes to the on-device MJPEG encoder
            # below; preview output is reserved for NN inputs (sub-step 3.4).
            self.camRgb = self.pipeline.createColorCamera()
            self.camRgb.setVideoSize(self.preview_width, self.preview_height)
            self.camRgb.setInterleaved(False)

            # On-device MJPEG encoder. The OAK has a hardware codec; using it
            # frees the Pi's CPU from cv2.imencode (~30% -> <5%).
            self.video_encoder = self.pipeline.createVideoEncoder()
            self.video_encoder.setDefaultProfilePreset(
                self.camRgb.getFps(),
                dai.VideoEncoderProperties.Profile.MJPEG,
            )
            self.video_encoder.setQuality(self.quality_factor)
            self.camRgb.video.link(self.video_encoder.input)

            # Stream the encoded JPEG bitstream to the host.
            xout_jpeg = self.pipeline.createXLinkOut()
            xout_jpeg.setStreamName("jpeg")
            self.video_encoder.bitstream.link(xout_jpeg.input)

            # Try to connect to device.
            self.device = dai.Device(self.pipeline)
            self.queue = self.device.getOutputQueue(
                name="jpeg", maxSize=4, blocking=False
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

    def publish_static_transforms(self):
        """Publish the OAK-D's internal TF subtree.

        v1: a single identity transform oak_d_lite_link -> oak_d_lite_rgb.
        Future stereo/depth capabilities add sibling transforms here using
        depthai's `getCameraExtrinsics()` for the actual physical offsets.

        Intentionally does NOT connect oak_d_lite_link to any robot-body
        frame (e.g. pib_base_link) — that is owned by a future
        robot-description publisher with the actual mounting measurement.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = DEVICE_FRAME_ID
        t.child_frame_id = FRAME_ID
        # Identity: translation defaults to (0,0,0); set quaternion to (0,0,0,1).
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info(
            f"Published static TF: {DEVICE_FRAME_ID} -> {FRAME_ID} (identity)"
        )

    def timer_callback(self):
        if not self.queue:
            return
        jpeg_pkt = self.queue.tryGet()  # already MJPEG-encoded by the OAK
        if jpeg_pkt is None:
            return

        jpg_bytes = bytes(jpeg_pkt.getData())
        jpg_b64 = base64.b64encode(jpg_bytes).decode("utf-8")

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
        # MJPEG quality is baked into the on-device VideoEncoder at pipeline
        # build, so applying a new value requires rebuilding. Causes a brief
        # streaming interruption (~1-2s) — acceptable for a tuning op.
        self.device.close()
        self.init_pipeline()

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
        return
    try:
        camera_node = CameraNode()
        if not camera_node.camera_available:
            # init_pipeline failed (commonly "device in use" right after a
            # restart). Trigger the retry path below by raising — without
            # this, rclpy.spin would block forever on a node with no
            # working pipeline.
            raise RuntimeError("OAK-D pipeline init failed; will retry")
        rclpy.spin(camera_node)
    except Exception as exc:
        error_publisher.timer_callback()
        print(exc)
    finally:
        if "camera_node" in locals():
            camera_node.destroy_node()
            print("camera_node destroyed")
        cnt = times - 1
        print(
            f"Retry starting camera in {RETRY_DELAY_SECONDS}s... ({cnt} attempt(s) left)"
        )
        time.sleep(RETRY_DELAY_SECONDS)
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
