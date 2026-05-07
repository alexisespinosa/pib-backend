#!/usr/bin/python3
import base64
import time

import blobconverter
import depthai as dai
import rclpy
from datatypes.srv import GetCameraImage
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float64, Int32, Int32MultiArray, String
from tf2_ros import StaticTransformBroadcaster
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

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

# --- Face detection capability (sub-step 3.4) -----------------------------
# Curated zoo model pre-baked at Docker build time per §4.5 of the design
# doc. blobconverter.from_zoo() at runtime is a cache hit (no network).
FACE_NN_NAME = "face-detection-retail-0004"
FACE_NN_SHAVES = 6
FACE_NN_INPUT_SIZE = 300            # the model expects 300x300 RGB
FACE_NN_CONFIDENCE_THRESHOLD = 0.5
# vision_msgs/ObjectHypothesis class label. Single class; no class id space
# to share with other capabilities.
FACE_CLASS_ID = "face"

# --- Stereo depth capability ------------------------------------------------
# Uses the OAK-D Lite's dedicated stereo depth engine (not SHAVE cores),
# so it runs alongside the face NN without contention.
DEPTH_MEDIAN_FILTER = dai.MedianFilter.KERNEL_7x7
DEPTH_FRAME_ID = "oak_d_lite_depth"
# Size of the ROI around the queried pixel (in normalized coordinates).
# A small region averages out noise; 0.02 = ~2% of frame = ~13x14 pixels.
DEPTH_ROI_HALF_SIZE = 0.01

# Polling interval for subscriber-count gating of lazy capabilities.
# Trade-off: longer interval = less idle CPU, longer activation latency.
# 1s is fine for human-driven subscribe events (Cerebra page open, user
# program start). For event-driven activation we'd switch to MatchedEvent.
SUBSCRIBER_POLL_PERIOD_S = 1.0


class ErrorPublisher(Node):

    def __init__(self):
        super().__init__("error_publisher")
        self.publisher_ = self.create_publisher(String, "/vision/raw_frame_b64", 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = "Camera not available"
        self.publisher_.publish(msg)


class CameraNode(Node):

    def __init__(self):
        super().__init__("camera_node")
        self.raw_frame_pub = self.create_publisher(
            CompressedImage, "/vision/raw_frame", 10
        )
        self.raw_frame_b64_pub = self.create_publisher(
            String, "/vision/raw_frame_b64", 10
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo, "/vision/camera_info", LATCHED_QOS
        )

        # Lazy capability: face detections (sub-step 3.4). Publishing is gated
        # by subscriber count via _check_lazy_subscribers() below. NN inference
        # itself runs on the OAK device unconditionally per the v1 design.
        self.face_detections_pub = self.create_publisher(
            Detection2DArray, "/vision/face_detections", 10
        )
        self._face_publishing = False

        self.depth_result_pub = self.create_publisher(
            Int32, "/vision/depth_result", 10
        )
        self.depth_query_sub = self.create_subscription(
            Int32MultiArray, "/vision/depth_query",
            self._on_depth_query, 10,
        )
        self._depth_enabled = False

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

        self.current_image = ""

        if self.camera_available:
            self.get_camera_image_service = self.create_service(
                GetCameraImage, "get_camera_image", self.get_camera_image_callback
            )
            self.publish_camera_info()
            self.publish_static_transforms()
            self.get_logger().info("Camera node initialized.")
        else:
            self.get_logger().error("Camera not available.")

        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Subscriber-count poll for lazy capability gating.
        self.subscriber_poll_timer = self.create_timer(
            SUBSCRIBER_POLL_PERIOD_S, self._check_lazy_subscribers
        )

    def get_camera_image_callback(self, request, response):
        response.image_base64 = self.current_image
        return response

    def init_pipeline(self) -> bool:
        try:
            self.pipeline = dai.Pipeline()

            # Color camera shared by two outputs:
            #   .video    -> on-device MJPEG encoder (always-on raw_frame topic)
            #   .preview  -> NN input(s) for face detection (and future capabilities)
            self.camRgb = self.pipeline.createColorCamera()
            self.camRgb.setVideoSize(self.preview_width, self.preview_height)
            self.camRgb.setPreviewSize(FACE_NN_INPUT_SIZE, FACE_NN_INPUT_SIZE)
            # Stretch the 16:9 sensor frame into the 1:1 NN input. False ->
            # consumers can recover pixel coords in the video frame by
            # multiplying normalized bbox coords by (video_width, video_height).
            self.camRgb.setPreviewKeepAspectRatio(False)
            self.camRgb.setInterleaved(False)

            # On-device MJPEG encoder. The OAK has a hardware codec; using it
            # frees the Pi's CPU from cv2.imencode (~30% -> ~20%).
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

            # Face detection NN. Always loaded per v1 design (§4.4); the host
            # only reads the output queue when there are subscribers — see
            # _check_lazy_subscribers() and timer_callback().
            self.face_nn = self.pipeline.createMobileNetDetectionNetwork()
            self.face_nn.setBlobPath(
                blobconverter.from_zoo(
                    name=FACE_NN_NAME, shaves=FACE_NN_SHAVES
                )
            )
            self.face_nn.setConfidenceThreshold(FACE_NN_CONFIDENCE_THRESHOLD)
            self.face_nn.input.setBlocking(False)
            self.camRgb.preview.link(self.face_nn.input)

            xout_face_nn = self.pipeline.createXLinkOut()
            xout_face_nn.setStreamName("face_nn")
            self.face_nn.out.link(xout_face_nn.input)

            # Stereo depth with on-device SpatialLocationCalculator.
            # Only the queried depth value crosses USB (not the full frame).
            mono_left = self.pipeline.createMonoCamera()
            mono_left.setResolution(
                dai.MonoCameraProperties.SensorResolution.THE_480_P
            )
            mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

            mono_right = self.pipeline.createMonoCamera()
            mono_right.setResolution(
                dai.MonoCameraProperties.SensorResolution.THE_480_P
            )
            mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

            stereo = self.pipeline.createStereoDepth()
            stereo.setDefaultProfilePreset(
                dai.node.StereoDepth.PresetMode.HIGH_DENSITY
            )
            stereo.initialConfig.setMedianFilter(DEPTH_MEDIAN_FILTER)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(False)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)

            spatial_calc = self.pipeline.createSpatialLocationCalculator()
            spatial_calc.setWaitForConfigInput(False)
            default_cfg = dai.SpatialLocationCalculatorConfigData()
            default_cfg.roi = dai.Rect(
                dai.Point2f(0.4, 0.4), dai.Point2f(0.6, 0.6)
            )
            default_cfg.depthThresholds.lowerThreshold = 200
            default_cfg.depthThresholds.upperThreshold = 10000
            spatial_calc.initialConfig.addROI(default_cfg)
            stereo.depth.link(spatial_calc.inputDepth)

            xin_spatial_cfg = self.pipeline.createXLinkIn()
            xin_spatial_cfg.setStreamName("spatial_cfg")
            xin_spatial_cfg.out.link(spatial_calc.inputConfig)

            xout_spatial = self.pipeline.createXLinkOut()
            xout_spatial.setStreamName("spatial_data")
            spatial_calc.out.link(xout_spatial.input)

            # Try to connect to device.
            self.device = dai.Device(self.pipeline)
            self.queue = self.device.getOutputQueue(
                name="jpeg", maxSize=4, blocking=False
            )
            self.face_nn_queue = self.device.getOutputQueue(
                name="face_nn", maxSize=4, blocking=False
            )
            self.spatial_cfg_queue = self.device.getInputQueue(
                name="spatial_cfg"
            )
            self.spatial_data_queue = self.device.getOutputQueue(
                name="spatial_data", maxSize=4, blocking=False
            )
            return True

        except Exception as e:
            self.get_logger().error(f"Camera not found: {e}")
            self.device = None
            self.queue = None
            self.face_nn_queue = None
            self.spatial_cfg_queue = None
            self.spatial_data_queue = None
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
        t_depth = TransformStamped()
        t_depth.header.stamp = t.header.stamp
        t_depth.header.frame_id = DEVICE_FRAME_ID
        t_depth.child_frame_id = DEPTH_FRAME_ID
        t_depth.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform([t, t_depth])
        self.get_logger().info(
            f"Published static TF: {DEVICE_FRAME_ID} -> {FRAME_ID}, "
            f"{DEVICE_FRAME_ID} -> {DEPTH_FRAME_ID} (identity)"
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

        # /vision/raw_frame (binary CompressedImage, ROS-native)
        compressed_msg = CompressedImage()
        compressed_msg.header.stamp = stamp
        compressed_msg.header.frame_id = FRAME_ID
        compressed_msg.format = "jpeg"
        compressed_msg.data = jpg_bytes
        self.raw_frame_pub.publish(compressed_msg)

        # /vision/raw_frame_b64 (String, for Cerebra via rosbridge)
        b64_msg = String()
        b64_msg.data = jpg_b64
        self.raw_frame_b64_pub.publish(b64_msg)
        self.current_image = jpg_b64

        # Lazy: face detections, gated by subscriber count. The NN itself
        # keeps running on-device whether or not we read its output.
        if self._face_publishing:
            self._publish_face_detections(stamp)

        if self._depth_enabled:
            self._poll_depth_results()

    def _publish_face_detections(self, stamp):
        """Drain the NN output queue and publish a Detection2DArray.

        Bbox coordinates from depthai are normalized [0..1] of the NN input
        frame (which is the camera preview, stretched to 1:1). We publish in
        pixel coordinates of the raw_frame (camRgb.video size) so consumers
        can correlate detections with the published image.
        """
        nn_pkt = self.face_nn_queue.tryGet()
        if nn_pkt is None:
            return

        msg = Detection2DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = FRAME_ID

        for det in nn_pkt.detections:
            d = Detection2D()
            d.header = msg.header
            d.bbox.center.position.x = (
                (det.xmin + det.xmax) / 2.0 * self.preview_width
            )
            d.bbox.center.position.y = (
                (det.ymin + det.ymax) / 2.0 * self.preview_height
            )
            d.bbox.size_x = (det.xmax - det.xmin) * self.preview_width
            d.bbox.size_y = (det.ymax - det.ymin) * self.preview_height

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = FACE_CLASS_ID
            hyp.hypothesis.score = float(det.confidence)
            d.results.append(hyp)

            msg.detections.append(d)

        self.face_detections_pub.publish(msg)

    def _on_depth_query(self, msg):
        if not self._depth_enabled or self.spatial_cfg_queue is None:
            return
        x, y = msg.data[0], msg.data[1]
        nx = x / self.preview_width
        ny = y / self.preview_height
        cfg = dai.SpatialLocationCalculatorConfigData()
        cfg.roi = dai.Rect(
            dai.Point2f(
                max(0.0, nx - DEPTH_ROI_HALF_SIZE),
                max(0.0, ny - DEPTH_ROI_HALF_SIZE),
            ),
            dai.Point2f(
                min(1.0, nx + DEPTH_ROI_HALF_SIZE),
                min(1.0, ny + DEPTH_ROI_HALF_SIZE),
            ),
        )
        cfg.depthThresholds.lowerThreshold = 200
        cfg.depthThresholds.upperThreshold = 10000
        spatial_cfg = dai.SpatialLocationCalculatorConfig()
        spatial_cfg.addROI(cfg)
        self.spatial_cfg_queue.send(spatial_cfg)

    def _poll_depth_results(self):
        result = self.spatial_data_queue.tryGet()
        if result is None:
            return
        spatial_data = result.getSpatialLocations()
        if not spatial_data:
            return
        loc = spatial_data[0]
        depth_mm = int(loc.spatialCoordinates.z)
        msg = Int32()
        msg.data = depth_mm
        self.depth_result_pub.publish(msg)

    def _check_lazy_subscribers(self):
        """Toggle publishing flags for lazy capabilities based on whether
        anyone is subscribed. Runs every SUBSCRIBER_POLL_PERIOD_S seconds.
        """
        face_subs = self.face_detections_pub.get_subscription_count()
        if face_subs > 0 and not self._face_publishing:
            self._face_publishing = True
            self.get_logger().info(
                f"Face detection consumers appeared ({face_subs}); "
                "publishing /vision/face_detections"
            )
        elif face_subs == 0 and self._face_publishing:
            self._face_publishing = False
            self.get_logger().info(
                "No face detection consumers; pausing /vision/face_detections"
            )

        depth_subs = self.depth_result_pub.get_subscription_count()
        if depth_subs > 0 and not self._depth_enabled:
            self._depth_enabled = True
            self.get_logger().info(
                f"Depth consumers appeared ({depth_subs}); "
                "depth queries enabled"
            )
        elif depth_subs == 0 and self._depth_enabled:
            self._depth_enabled = False
            self.get_logger().info(
                "No depth consumers; depth queries disabled"
            )

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
