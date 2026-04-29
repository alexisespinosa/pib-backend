# pib vision architecture

Status: design draft — no implementation yet
Last updated: 2026-04-27

## 1. Context

pib's current vision flow puts depthai pipeline code directly in user-authored Blockly programs. The "Face Detector" Blockly block emits a Python class that opens its own `dai.Device(self.pipeline)`, runs NN inference on the OAK-D Lite, calls `cv2.imshow(...)`, and reads keystrokes via `cv2.waitKey(...)`. This has three structural problems:

1. **Hardware ownership is in user code.** depthai allows exactly one host process to connect to a given OAK device. The `ros-camera` ROS node also owns the device. When both want it, one fails — and `ros-camera` typically wins because it boots first. Production users hit this same conflict; the workaround has been to manually stop `ros-camera` before running a face-detector program.

2. **GUI/display dependencies in user code.** `cv2.imshow` requires X11 forwarded into the user-program container. Even if it worked, user programs running inside Docker would be coupled to the host's display server, the cv2 runtime, and a window manager.

3. **The lesson collapses.** pib is an educational platform. If a Blockly block ships a fully-formed face-detector class with depthai pipeline definition, NN model loading, and a render loop, the kid isn't learning robotics — they're learning to drag a magic block. The kid should compose behavior from primitives that *aren't* secretly running their own hardware control loops.

This doc describes the target architecture: a ROS2-native vision daemon that owns the OAK-D exclusively, exposes per-capability topics that activate lazily based on consumers, and lets user programs subscribe to typed detection results without touching depthai or cv2.

## 2. Principles

1. **Hardware ownership is layered.** Exactly one process owns the OAK-D. Device access lives in a single hardware-owning ROS node. Behavior and user code never call `dai.Device(...)`.

2. **Capabilities activate on demand.** NN models cost device compute and host CPU. They run only when something is subscribed to their output topic. No subscriber → no published messages (and eventually, no on-device inference).

3. **The default install ships no behavior.** The system boots into a "blank slate" — hardware-owning nodes are running, no opinionated behavior is enabled, the kid is the source of all decisions. Optional behavior nodes exist as installable packages but are not on by default.

4. **User programs are the primary surface.** Blockly user programs are the path of first contact for the user. They consume vision topics directly. Behavior nodes are an advanced concept users meet later, if at all.

5. **Standard ROS messages where possible.** Use `vision_msgs`, `sensor_msgs`, `geometry_msgs` rather than pib-custom types unless there is a concrete need. Interop with `rqt`, `rviz`, and broader ROS tooling is a feature.

6. **Forward-compatible with depth.** The OAK-D Lite has stereo cameras for depth. The catalog reserves namespace and conventions for depth, point clouds, and 3D detections from day one, even if they aren't implemented yet.

## 3. Architecture overview

```
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 0  Hardware-owning nodes (always running, capabilities lazy)   │
│                                                                      │
│   ros-vision      owns OAK-D, publishes raw frames + detections      │
│   ros-motors      owns Tinkerforge servos + relay                    │
│   ros-display     owns the face screen (Tk + X11)                    │
│   ros-voice-*     owns audio devices                                 │
│   ros-audio-io    owns audio I/O                                     │
└──────────────────────────────────────────────────────────────────────┘
                              │ ROS topics / services
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 1  Behavior nodes (OPT-IN, not auto-started)                   │
│                                                                      │
│   Empty in the default install. Owners install behavior packages     │
│   when giving pib a specific role (greeter, dog-recognizer,          │
│   physical-therapy-assistant, etc.). Behavior nodes consume          │
│   Layer 0 outputs, publish actuator commands, and may expose new     │
│   topics that user programs can consume.                             │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 2  Blockly user programs (primary surface)                     │
│                                                                      │
│   On-demand, per-user logic. Subscribe to Layer 0 topics directly,   │
│   call services, write to actuator topics. Run by ros-programs as    │
│   Popen subprocesses. No depthai, no cv2, no hardware coupling.      │
└──────────────────────────────────────────────────────────────────────┘
```

Hardware-owning nodes are always running but their *capabilities* are lazy — see §4.2.

## 4. `ros-vision` specification

`ros-vision` is the renamed/extended replacement for `ros-camera`. It owns the OAK-D Lite exclusively and publishes both raw camera output and detection results from on-device NN models. Same Docker container shape as `ros-camera` today (privileged, USB device access).

### 4.1 Topic catalog

```
ALWAYS-ON (the device is open, frames stream)
  /vision/raw_frame             sensor_msgs/CompressedImage
  /vision/raw_frame_b64         std_msgs/String                  Cerebra compat (b64 JPEG)
  /vision/camera_info           sensor_msgs/CameraInfo           latched

LAZY — published only when subscribers present
  /vision/face_detections       vision_msgs/Detection2DArray

DEFERRED — reserved namespace, not implemented yet
  /vision/person_detections     vision_msgs/Detection2DArray
  /vision/object_detections     vision_msgs/Detection2DArray
  /vision/depth                 sensor_msgs/Image (encoding 16UC1)
  /vision/depth_camera_info     sensor_msgs/CameraInfo           latched
  /vision/point_cloud           sensor_msgs/PointCloud2
  /vision/face_detections_3d    vision_msgs/Detection3DArray
  /vision/person_detections_3d  vision_msgs/Detection3DArray
  /vision/hand_keypoints        TBD (vision_msgs has no first-class type yet)
  /vision/pose_keypoints        TBD
```

#### Why two raw-frame topics

`/vision/raw_frame` uses `sensor_msgs/CompressedImage` for ROS-native consumers (`rqt_image_view`, future ROS nodes). `/vision/raw_frame_b64` uses `std_msgs/String` (base64-encoded JPEG) because Cerebra subscribes via rosbridge, which JSON-encodes message contents — base64-in-a-string is straightforward over JSON, while binary in a `sensor_msgs` payload is awkward. Cerebra's existing camera page already consumes a String topic; we keep that interface to avoid forcing a frontend change in the same PR.

Long-term, Cerebra can migrate to `sensor_msgs/CompressedImage` (rosbridge supports `cbor-raw` compression for binary topics) and the b64 topic can be deprecated. Not in scope now.

### 4.2 Subscription-driven activation

For each lazy topic, `ros-vision` tracks subscription count and toggles publishing on transitions:

- 0 → N: enable publishing. In v1 (see §4.4) this just means "start writing messages"; the on-device NN is already running. In v2 it would also mean "rebuild the depthai pipeline to include this NN."
- N → 0: disable publishing. In v1, stop writing messages (NN keeps running on-device). In v2, rebuild pipeline without this NN.

Implementation uses ROS2's `MatchedEvent` callbacks on each `Publisher` (preferred — event-driven) or polled `get_subscription_count()` on a periodic timer (fallback if matched events don't behave on the publisher type we use).

**Always-on topics** (`/vision/raw_frame`, `/vision/raw_frame_b64`, `/vision/camera_info`) bypass this — they always publish, regardless of subscribers. Cerebra's camera page expects to be able to subscribe and immediately get frames.

### 4.3 Frame conventions

Each measurement's `Header` carries the `frame_id` of the optical sensor that produced it:

| frame_id | sensor | when used |
|---|---|---|
| `oak_d_lite_link` | device root | TF parent of all OAK-internal optical frames |
| `oak_d_lite_rgb` | RGB camera | raw_frame, all 2D detections (NN models run on RGB) |
| `oak_d_lite_left` | left mono | future stereo / depth source |
| `oak_d_lite_right` | right mono | future stereo / depth source |
| `oak_d_lite_depth` | depth image | future depth output, conventionally aligned to one of the above |

Static transforms between these frames come from depthai's calibration data on the device (`device.readCalibration()`). They are published once at startup via `tf2_ros.StaticTransformBroadcaster`. This makes downstream nodes able to convert "a face at pixel (300, 200) in `oak_d_lite_rgb`" into a 3D position in `oak_d_lite_depth` once depth is implemented.

For v1 (RGB-only, no depth), the only published transform is `oak_d_lite_link → oak_d_lite_rgb` as identity. Sibling transforms (`oak_d_lite_link → oak_d_lite_left/right/depth`) are added when those capabilities come online, sourced from `getCameraExtrinsics()`.

**Scope limit**: `ros-vision` deliberately does NOT publish a transform connecting `oak_d_lite_link` to any robot-body frame (e.g. `pib_base_link`). That edge of the TF tree is owned by a future robot-description publisher (URDF + `robot_state_publisher`, or a dedicated `ros-pib-state` node) which has the actual measurement of how the OAK-D is mounted on pib. Until that exists, consumers asking "where is this face in pib's body frame?" will receive a TF lookup error — which is the honest answer. Publishing a placeholder identity transform there would silently produce wrong-but-consistent positions; we'd rather have a loud error.

### 4.4 v1 vs v2 implementation strategy

**v1** (initial release):
- All NN models supported by the system are loaded into the depthai pipeline at startup. Currently: face detection only.
- The pipeline runs continuously on the device while the device is open.
- Subscription-driven activation gates *publishing*, not *inference*. No subscriber → host doesn't read the NN output queue and doesn't construct/publish messages. Saves host CPU and ROS bandwidth, but device compute is constant.
- Trade-off: cheaper to implement, no glitches when capabilities toggle, but uses the device's NN compute capacity unconditionally.
- Acceptable as long as total NN load fits the OAK-D Lite's 4 SHAVE cores. Face-detection alone uses one model — plenty of headroom.

**v2** (when needed):
- Pipeline rebuild on capability set change. Subscribing to a previously-quiet capability triggers `dai.Device.close()` + new pipeline construction + `dai.Device(new_pipeline)`. Brief outage (~1-3s) during reload, including raw frame streaming.
- Necessary when total simultaneous NN load exceeds device capacity (~3-4 models depending on size).
- Code is structured so the active capability set is a list — adding the rebuild trigger is a localized change, not a refactor.

The v1→v2 boundary is a runtime tuning, not an API change. Consumers don't notice.

## 5. What user programs see

A face detector Blockly program after the refactor:

```python
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

rclpy.init()
node = Node('user_program')
latest_faces = []

def on_faces(msg):
    global latest_faces
    latest_faces = msg.detections

node.create_subscription(Detection2DArray, '/vision/face_detections', on_faces, 10)

# user logic — read latest_faces, do whatever (motor commands, etc.)
```

No `import depthai`. No `import cv2`. No `dai.Device(...)`. The "Face Detector: start" Blockly block emits the subscriber wiring; the "Run face detector" block reads `latest_faces`. The user program is small, hardware-decoupled, and runs in a slim container.

The `ros-programs` container therefore does *not* need cv2, depthai, blobconverter, or X11 forwarding. Its dependency list shrinks back to what it has today plus `vision_msgs`.

## 6. Migration plan

Multi-PR effort. Order matters because each step depends on the previous:

1. **This design doc**, merged to `feat/vision-architecture` branch in `pib-backend`. Becomes the contract.
2. **`pib_vision_msgs` package** — only created if we discover during step 4 that `vision_msgs` doesn't cover something. Most likely we don't need this package for face detection alone.
3. **`ros-vision` service** — built up across multiple commits on the same branch:
   - **3.1** Rename `ros-camera` → `ros-vision` (Compose service, image, Python package, boot scripts, production setup script references). No behavior change.
   - **3.2** Implement always-on topics: `/vision/raw_frame` (`sensor_msgs/CompressedImage`), `/vision/raw_frame_b64` (`std_msgs/String`), `/vision/camera_info` (`sensor_msgs/CameraInfo` latched). Keep the legacy `/camera_topic` publisher live so Cerebra doesn't break before step 4.
   - **3.3** Publish static TF for `oak_d_lite_rgb` (and identity transform for the device root). Sourced from `device.readCalibration()` once at startup.
   - **3.4** Implement face-detection capability — depthai NN node added to the pipeline, `/vision/face_detections` (`vision_msgs/Detection2DArray`) published behind a subscription-count gate (matched events preferred, polled `get_subscription_count()` as fallback).
   - **3.5** **Restructure checkpoint.** After 3.4, evaluate `vision_node.py`. Trigger thresholds for splitting into modules (e.g. `vision_node.py` + `pipeline.py` + `publishers/{raw_frame,face_detection,camera_info,tf}.py`):
     - File exceeds ~400 LOC and reading it requires scrolling between unrelated concerns, **or**
     - The next lazy capability is queued and the subscription-gating logic would be duplicated.
     If neither trigger fires, defer the restructure to whichever future commit first hits one. The point is to let the structure be motivated by real duplication, not by speculation about what publishers might want in common.
4. **Cerebra adjustment** — update the camera page topic name from `camera_topic` to `/vision/raw_frame_b64`.
5. **Blockly generator refactor** — `face_detector_start_stop` and `face_detector_running` (in `pib-blockly`) emit ROS subscriber code instead of depthai pipeline. The `FaceDetector` class declaration in `function-declarations.ts` is removed.
6. **Programs container slim-down** — remove the dependencies that user programs no longer need (none added, but document that cv2/depthai/blobconverter must NOT be added back for vision purposes).
7. **Deprecation message** — old depthai-in-user-program path emits a clear error directing users to re-save their Blockly program (which then uses the new generators).

Each step is a separate PR. Steps 1-3 land on `feat/vision-architecture`; once that's merged to main, steps 4-7 can land independently.

## 7. Out of scope (for now)

Explicit list of things this doc does *not* commit to:

- **Any Layer 1 behavior nodes.** We've defined the layer; we've shipped nothing in it.
- **Person, object, hand, pose detection.** Topic names reserved; no implementation.
- **Depth and 3D detection.** Frames and topics reserved; no implementation.
- **Cerebra camera page migration to `sensor_msgs/CompressedImage`.** Eventually yes, not now.
- **Pipeline rebuild on capability change (v2).** Designed-for, not implemented.
- **Replacing rosbridge's String-based camera transport.** Tracked as eventual work, not blocking.
- **Migration from base64-string-camera to standard ROS image topics in Cerebra.** Same.

These are deliberate parking-lot items, not gaps in the design.
