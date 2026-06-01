#!/usr/bin/python3
import base64
import json
import time

import blobconverter
import depthai as dai
import numpy as np
import rclpy
from datatypes.srv import EnrollFace, GetCameraImage
from geometry_msgs.msg import TransformStamped
from pib_api_client import person_client
import threading

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float64, Int32, Int32MultiArray, String
from tf2_ros import StaticTransformBroadcaster
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

FRAME_ID = "oak_d_lite_rgb"
DEVICE_FRAME_ID = "oak_d_lite_link"

LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

RETRY_DELAY_SECONDS = 5

# --- Face detection ---
FACE_NN_NAME = "face-detection-retail-0004"
FACE_NN_SHAVES = 6
FACE_NN_INPUT_SIZE = 300
FACE_NN_CONFIDENCE_THRESHOLD = 0.5
FACE_CLASS_ID = "face"

# --- Face recognition ---
FACE_REC_NN_NAME = "face-recognition-arcface-112x112"
FACE_REC_NN_ZOO_TYPE = "depthai"
FACE_REC_NN_SHAVES = 4
FACE_REC_INPUT_SIZE = 112
FACE_REC_SIMILARITY_THRESHOLD = 0.65

# --- Stereo depth ---
DEPTH_MEDIAN_FILTER = dai.MedianFilter.KERNEL_7x7
DEPTH_FRAME_ID = "oak_d_lite_depth"
DEPTH_ROI_HALF_SIZE = 0.01

SUBSCRIBER_POLL_PERIOD_S = 1.0

ENROLL_CAPTURE_INTERVAL_S = 0.6


def _cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


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

        self.face_detections_pub = self.create_publisher(
            Detection2DArray, "/vision/face_detections", 10
        )
        self._face_publishing = False

        self.face_recognitions_pub = self.create_publisher(
            Detection2DArray, "/vision/face_recognitions", 10
        )
        self._recognition_publishing = False

        self.depth_result_pub = self.create_publisher(
            Int32, "/vision/depth_result", 10
        )
        self.depth_query_sub = self.create_subscription(
            Int32MultiArray, "/vision/depth_query",
            self._on_depth_query, 10,
        )
        self._depth_enabled = False

        self._enroll_cb_group = MutuallyExclusiveCallbackGroup()
        self.enroll_face_srv = self.create_service(
            EnrollFace, "/vision/enroll_face", self._enroll_face_callback,
            callback_group=self._enroll_cb_group,
        )
        self._enroll_name = ""
        self._enroll_target_count = 0
        self._enroll_captured = []
        self._enroll_last_capture_time = 0.0
        self._enroll_done = threading.Event()

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

        self.preview_width = 1280
        self.preview_height = 720
        self.quality_factor = 80

        self._known_faces = {}
        self._load_known_faces()
        self._active_tracklets = []
        self._tracked_identities = {}

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

        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.subscriber_poll_timer = self.create_timer(
            SUBSCRIBER_POLL_PERIOD_S, self._check_lazy_subscribers
        )

    def _load_known_faces(self):
        self._known_faces = {}
        success, data = person_client.get_all_persons()
        if not success or data is None:
            self.get_logger().warn("Could not load known faces from API")
            return
        for person in data.get("persons", []):
            person_id = person["personId"]
            name = person["name"]
            ok, emb_data = person_client.get_embeddings(person_id)
            if not ok or emb_data is None:
                continue
            embeddings = []
            for e in emb_data.get("embeddings", []):
                embeddings.append(np.array(json.loads(e["embedding"]), dtype=np.float32))
            if embeddings:
                self._known_faces[name] = embeddings
        self.get_logger().info(
            f"Loaded {len(self._known_faces)} known face(s): "
            f"{list(self._known_faces.keys())}"
        )

    def _identify_face(self, embedding):
        best_name = "unknown"
        best_score = 0.0
        for name, embeddings in self._known_faces.items():
            for known_emb in embeddings:
                score = _cosine_similarity(embedding, known_emb)
                if score > best_score:
                    best_score = score
                    best_name = name
        if best_score < FACE_REC_SIMILARITY_THRESHOLD:
            return "unknown", best_score
        return best_name, best_score

    def _enroll_face_callback(self, request, response):
        name = request.name.strip()
        count = max(1, min(request.count, 20))
        if not name:
            response.success = False
            response.captured = 0
            return response

        self.get_logger().info(f"Enrollment started for '{name}' ({count} embeddings)")
        self._enroll_done.clear()
        self._enroll_name = name
        self._enroll_target_count = count
        self._enroll_captured = []
        self._enroll_last_capture_time = 0.0

        timeout = count * ENROLL_CAPTURE_INTERVAL_S + 10.0
        if not self._enroll_done.wait(timeout=timeout):
            self.get_logger().warn("Enrollment timed out")

        success, person_data = person_client.get_all_persons()
        person_id = None
        if success and person_data:
            for p in person_data.get("persons", []):
                if p["name"] == name:
                    person_id = p["personId"]
                    break
        if person_id is None:
            ok, new_person = person_client.create_person(name)
            if ok and new_person:
                person_id = new_person["personId"]

        stored = 0
        if person_id:
            for emb in self._enroll_captured:
                ok, _ = person_client.add_embedding(person_id, emb.tolist())
                if ok:
                    stored += 1

        self._enroll_name = ""
        self._enroll_target_count = 0
        self._enroll_captured = []

        if stored > 0:
            self._load_known_faces()

        self.get_logger().info(f"Enrollment complete for '{name}': {stored} embeddings stored")
        response.success = stored > 0
        response.captured = stored
        return response

    def get_camera_image_callback(self, request, response):
        response.image_base64 = self.current_image
        return response

    def init_pipeline(self) -> bool:
        try:
            self.pipeline = dai.Pipeline()

            self.camRgb = self.pipeline.createColorCamera()
            self.camRgb.setVideoSize(self.preview_width, self.preview_height)
            self.camRgb.setPreviewSize(FACE_NN_INPUT_SIZE, FACE_NN_INPUT_SIZE)
            self.camRgb.setPreviewKeepAspectRatio(False)
            self.camRgb.setInterleaved(False)

            self.video_encoder = self.pipeline.createVideoEncoder()
            self.video_encoder.setDefaultProfilePreset(
                self.camRgb.getFps(),
                dai.VideoEncoderProperties.Profile.MJPEG,
            )
            self.video_encoder.setQuality(self.quality_factor)
            self.camRgb.video.link(self.video_encoder.input)

            xout_jpeg = self.pipeline.createXLinkOut()
            xout_jpeg.setStreamName("jpeg")
            self.video_encoder.bitstream.link(xout_jpeg.input)

            # --- Face detection NN ---
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

            # --- Object tracker (persistent face IDs across frames) ---
            # All inputs non-blocking so tracker cannot stall the pipeline.
            tracker = self.pipeline.createObjectTracker()
            tracker.setDetectionLabelsToTrack([1])
            tracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
            tracker.setTrackerIdAssignmentPolicy(
                dai.TrackerIdAssignmentPolicy.SMALLEST_ID
            )
            tracker.inputDetections.setBlocking(False)
            tracker.inputDetections.setQueueSize(1)
            tracker.inputDetectionFrame.setBlocking(False)
            tracker.inputDetectionFrame.setQueueSize(1)
            tracker.inputTrackerFrame.setBlocking(False)
            tracker.inputTrackerFrame.setQueueSize(1)
            self.face_nn.out.link(tracker.inputDetections)
            self.face_nn.passthrough.link(tracker.inputDetectionFrame)
            self.camRgb.preview.link(tracker.inputTrackerFrame)

            xout_tracker = self.pipeline.createXLinkOut()
            xout_tracker.setStreamName("tracker")
            tracker.out.link(xout_tracker.input)

            # --- Face recognition two-stage pipeline ---
            # Use passthrough to get the exact frame the detection NN processed
            script = self.pipeline.createScript()
            script.setProcessor(dai.ProcessorType.LEON_CSS)
            self.face_nn.out.link(script.inputs["face_det_in"])
            self.face_nn.passthrough.link(script.inputs["frame"])

            script.setScript("""
while True:
    face_dets = node.io['face_det_in'].get()
    img = node.io['frame'].get()
    for det in face_dets.detections:
        xmin = max(0.0, det.xmin)
        ymin = max(0.0, det.ymin)
        xmax = min(1.0, det.xmax)
        ymax = min(1.0, det.ymax)
        if xmax - xmin < 0.01 or ymax - ymin < 0.01:
            continue
        cfg = ImageManipConfig()
        cfg.setCropRect(xmin, ymin, xmax, ymax)
        cfg.setResize(112, 112)
        cfg.setKeepAspectRatio(False)
        node.io['manip_cfg'].send(cfg)
        node.io['manip_img'].send(img)
""")

            # Face recognition ImageManip: crops and resizes each face
            face_rec_manip = self.pipeline.createImageManip()
            face_rec_manip.initialConfig.setResize(
                FACE_REC_INPUT_SIZE, FACE_REC_INPUT_SIZE
            )
            face_rec_manip.setWaitForConfigInput(True)
            face_rec_manip.inputImage.setQueueSize(4)

            script.outputs["manip_cfg"].link(face_rec_manip.inputConfig)
            script.outputs["manip_img"].link(face_rec_manip.inputImage)

            # Face recognition NN
            face_rec_nn = self.pipeline.createNeuralNetwork()
            face_rec_nn.setBlobPath(
                blobconverter.from_zoo(
                    name=FACE_REC_NN_NAME,
                    zoo_type=FACE_REC_NN_ZOO_TYPE,
                    shaves=FACE_REC_NN_SHAVES,
                )
            )
            face_rec_manip.out.link(face_rec_nn.input)

            xout_face_rec = self.pipeline.createXLinkOut()
            xout_face_rec.setStreamName("face_rec")
            face_rec_nn.out.link(xout_face_rec.input)

            # --- Stereo depth ---
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

            # --- Connect to device ---
            self.device = dai.Device(self.pipeline)
            self.queue = self.device.getOutputQueue(
                name="jpeg", maxSize=4, blocking=False
            )
            self.tracker_queue = self.device.getOutputQueue(
                name="tracker", maxSize=4, blocking=False
            )
            self.face_nn_queue = self.device.getOutputQueue(
                name="face_nn", maxSize=4, blocking=False
            )
            self.face_rec_queue = self.device.getOutputQueue(
                name="face_rec", maxSize=4, blocking=False
            )
            self.spatial_cfg_queue = self.device.getInputQueue(
                name="spatial_cfg"
            )
            self.spatial_data_queue = self.device.getOutputQueue(
                name="spatial_data", maxSize=4, blocking=False
            )

            self.get_logger().info("Face recognition NN loaded")
            return True

        except Exception as e:
            self.get_logger().error(f"Camera not found: {e}")
            self.device = None
            self.queue = None
            self.tracker_queue = None
            self.face_nn_queue = None
            self.face_rec_queue = None
            self.spatial_cfg_queue = None
            self.spatial_data_queue = None
            return False

    def publish_camera_info(self):
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
        msg.distortion_model = "plumb_bob"
        msg.d = [float(d) for d in distortion[:5]]
        msg.k = [float(v) for row in K for v in row]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
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
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = DEVICE_FRAME_ID
        t.child_frame_id = FRAME_ID
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
        jpeg_pkt = self.queue.tryGet()
        if jpeg_pkt is None:
            return

        jpg_bytes = bytes(jpeg_pkt.getData())
        jpg_b64 = base64.b64encode(jpg_bytes).decode("utf-8")

        stamp = self.get_clock().now().to_msg()

        compressed_msg = CompressedImage()
        compressed_msg.header.stamp = stamp
        compressed_msg.header.frame_id = FRAME_ID
        compressed_msg.format = "jpeg"
        compressed_msg.data = jpg_bytes
        self.raw_frame_pub.publish(compressed_msg)

        b64_msg = String()
        b64_msg.data = jpg_b64
        self.raw_frame_b64_pub.publish(b64_msg)
        self.current_image = jpg_b64

        if self._face_publishing or self._recognition_publishing or self._enroll_name:
            self._update_tracked_faces()

        if self._face_publishing:
            self._publish_face_detections(stamp)

        if self._recognition_publishing or self._enroll_name:
            self._process_recognition_embeddings()

        if self._recognition_publishing:
            self._publish_face_recognitions(stamp)

        if self._depth_enabled:
            self._poll_depth_results()

    def _update_tracked_faces(self):
        tracker_pkt = self.tracker_queue.tryGet()
        if tracker_pkt is None:
            return
        self._active_tracklets = []
        for t in tracker_pkt.tracklets:
            if t.status == dai.Tracklet.TrackingStatus.REMOVED:
                self._tracked_identities.pop(t.id, None)
            elif t.status != dai.Tracklet.TrackingStatus.LOST:
                self._active_tracklets.append(t)

    def _publish_face_detections(self, stamp):
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

    def _process_recognition_embeddings(self):
        rec_pkt = self.face_rec_queue.tryGet()
        if rec_pkt is None:
            return

        embedding = np.array(rec_pkt.getFirstLayerFp16(), dtype=np.float32)

        if self._enroll_name and len(self._enroll_captured) < self._enroll_target_count:
            now = time.monotonic()
            if now - self._enroll_last_capture_time >= ENROLL_CAPTURE_INTERVAL_S:
                self._enroll_captured.append(embedding.copy())
                self._enroll_last_capture_time = now
                self.get_logger().info(
                    f"Enrollment capture {len(self._enroll_captured)}/"
                    f"{self._enroll_target_count} for '{self._enroll_name}'"
                )
                if len(self._enroll_captured) >= self._enroll_target_count:
                    self._enroll_done.set()

        name, score = self._identify_face(embedding)
        if name != "unknown" and self._active_tracklets:
            largest = max(
                self._active_tracklets,
                key=lambda t: (
                    (t.srcImgDetection.xmax - t.srcImgDetection.xmin)
                    * (t.srcImgDetection.ymax - t.srcImgDetection.ymin)
                ),
            )
            self._tracked_identities[largest.id] = (name, score)

    def _publish_face_recognitions(self, stamp):
        msg = Detection2DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = FRAME_ID

        for t in self._active_tracklets:
            det = t.srcImgDetection
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

            cached = self._tracked_identities.get(t.id, ("unknown", 0.0))
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cached[0]
            hyp.hypothesis.score = cached[1]
            d.results.append(hyp)
            msg.detections.append(d)

        self.face_recognitions_pub.publish(msg)

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

        rec_subs = self.face_recognitions_pub.get_subscription_count()
        if rec_subs > 0 and not self._recognition_publishing:
            self._recognition_publishing = True
            self.get_logger().info(
                f"Face recognition consumers appeared ({rec_subs}); "
                "publishing /vision/face_recognitions"
            )
        elif rec_subs == 0 and self._recognition_publishing:
            self._recognition_publishing = False
            self.get_logger().info(
                "No face recognition consumers; pausing /vision/face_recognitions"
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
        self.timer.cancel()
        self.timer = self.create_timer(
            self.timer_period, self.timer_callback
        )

    def quality_factor_callback(self, msg):
        self.quality_factor = msg.data
        self.device.close()
        self.init_pipeline()

    def preview_size_callback(self, msg):
        self.preview_width, self.preview_height = msg.data
        self.device.close()
        if self.init_pipeline():
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
            raise RuntimeError("OAK-D pipeline init failed; will retry")
        executor = MultiThreadedExecutor()
        executor.add_node(camera_node)
        executor.spin()
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
