# FoGaze: A Dual-Camera System for Real-Time Gaze-to-Object Focus Estimation Using Appearance-Based Regression and RGB-D Sensing

**Authors:** Chanachol Polk*, *et al.*
*KAITO–Kanagawa Exchange Program, School Project*
*Corresponding author: chanachol.polk@gmail.com*

---

***Abstract*** — Determining *which* real-world object a person is attending to is a foundational problem for assistive technology, human–robot interaction, and augmented-reality interfaces. We present **FoGaze**, a low-cost, fully software-based system that fuses an appearance-based gaze estimator with an RGB-D scene sensor to infer the object a user is focusing on in real time. A face-facing webcam drives a MediaPipe FaceLandmarker pipeline that extracts a 486-dimensional eye/head feature vector; a Ridge-regression model, calibrated per user against a 4×4 grid of on-screen targets, maps these features to a gaze point in the scene-camera coordinate frame. Concurrently, a PrimeSense structured-light sensor supplies a registered colour-plus-depth stream, on which a YOLOv8-nano detector localises candidate objects. The predicted gaze point is matched against detected bounding boxes to identify the focused object, and the sensor's depth channel lifts that object into 3-D for publication to ROS 2 / RViz. Gaze jitter is suppressed by a cascade of Kalman, exponential-moving-average, and One-Euro filters. The system runs interactively on commodity CPU hardware without a dedicated eye-tracker, and supports hands-free interaction through triple-blink gestures and spoken object feedback. We describe the architecture, calibration procedure, and engineering trade-offs, and discuss limitations and directions for quantitative evaluation.

***Index Terms*** — gaze estimation, eye tracking, object detection, RGB-D sensing, human–computer interaction, assistive technology, ridge regression, ROS 2.

---

## I. Introduction

Eye gaze is a strong proxy for human visual attention and intent. Knowing not merely *where on a screen* a person looks, but *which physical object* they attend to, unlocks applications in assistive communication for people with motor impairments, hands-free robot tasking, attention analytics, and context-aware augmented reality. Commercial solutions to this problem typically rely on dedicated infrared eye-trackers, head-mounted scene cameras, or both, which are costly and intrusive.

This paper presents **FoGaze**, a system that addresses *gaze-to-object focus estimation* using only commodity hardware: a single face-facing webcam and one consumer RGB-D sensor. Rather than tracking gaze on a 2-D display, FoGaze maps a user's gaze directly into the coordinate frame of a scene camera observing the real world, then intersects the gaze point with detected objects to determine focus.

The central contributions of this work are:

1. **A two-stream architecture** that decouples gaze estimation (face stream) from scene understanding (RGB-D stream) and unifies them in a single calibrated coordinate frame.
2. **A practical per-user calibration procedure** combining a 4×4 spatial target grid with head-pose sampling, trained as a regularised linear regressor over MediaPipe-derived features.
3. **Depth-aware object focus**, where the focused object is lifted to 3-D using registered depth and published to ROS 2 for robotic or visualisation use.
4. **Hands-free interaction primitives** — triple-blink selection and text-to-speech feedback — built on the same feature stream.

The remainder of the paper is organised as follows. Section II reviews related work. Section III details the system architecture. Section IV describes feature extraction and gaze modelling. Section V covers calibration. Section VI explains scene understanding and focus determination. Section VII describes filtering and interaction. Section VIII reports implementation details. Section IX discusses limitations and future work, and Section X concludes.

## II. Related Work

**Appearance-based gaze estimation.** Classical model-based gaze trackers fit geometric eye models to infrared corneal reflections. Appearance-based methods instead learn a direct mapping from eye images or facial landmarks to gaze, trading geometric precision for hardware simplicity. FoGaze follows the appearance-based paradigm, building on the open-source *EyeTrax* approach, which extracts dense facial-landmark features via MediaPipe and regresses them to screen coordinates.

**Facial landmark detection.** Google's MediaPipe Face Mesh / FaceLandmarker provides 478 3-D landmarks, including refined iris points, from a single RGB image at interactive rates on CPU. FoGaze uses these landmarks as the substrate for both gaze features and blink detection.

**Real-time object detection.** Single-stage detectors such as the YOLO family offer favourable accuracy/latency trade-offs for embedded and real-time use. FoGaze adopts YOLOv8-nano at reduced input resolution to keep scene analysis within the per-frame budget on CPU.

**RGB-D sensing.** Structured-light sensors such as the PrimeSense/Carmine class, accessed through OpenNI2, provide registered colour and metric depth. FoGaze repurposes such a sensor both as the scene camera and as the depth source for 3-D object localisation.

FoGaze's novelty lies less in any single component than in their integration: a calibrated bridge from a webcam-based gaze regressor into the metric frame of an RGB-D scene sensor, closed by object detection and published to a robotics middleware.

## III. System Architecture

FoGaze comprises two synchronised input streams that converge on a focus-determination stage (Fig. 1).

```
 ┌──────────────┐    486-D features   ┌──────────────────────┐
 │ Face camera  │ ──────────────────► │ MediaPipe FaceLandmkr │
 │  (USB webcam)│                     └──────────┬───────────┘
 └──────────────┘                                │
                                                 ▼
                                      ┌─────────────────────┐  gaze (x,y)
                                      │ Ridge regressor      │ ─────────────┐
                                      │ (StandardScaler+Ridge)│             │
                                      └─────────────────────┘              ▼
 ┌──────────────┐  colour+depth   ┌───────────────┐  bboxes   ┌────────────────────┐
 │ PrimeSense   │ ──────────────► │ YOLOv8-n      │ ────────► │ Focus determination │
 │ RGB-D sensor │                 │ detector      │           │ (point-in-bbox)     │
 └──────┬───────┘                 └───────────────┘           └─────────┬──────────┘
        │ depth                                                          │
        └──────────────────────► depth_at_bbox() ──── 3-D position ──────┘
                                                                          ▼
                                                            ROS 2 MarkerArray + TF
                                                              (RViz visualisation)
```

***Fig. 1.*** *FoGaze data flow. The face stream produces a 2-D gaze point in the scene frame; the RGB-D stream produces detected objects and per-object depth; the focus stage intersects them and lifts the result to 3-D.*

**Face stream.** A standard USB webcam pointed at the user feeds frames to a MediaPipe FaceLandmarker. From the landmarks, a 486-dimensional feature vector is extracted (Section IV) and passed to the calibrated gaze model, which outputs a gaze coordinate expressed in the scene camera's pixel frame.

**Scene stream.** A PrimeSense sensor, accessed through OpenNI2, provides a colour stream (used as the scene image) and a registered 640×480, 1 mm-precision depth stream at 30 fps. YOLOv8-nano runs on the colour frames to produce object bounding boxes with class labels.

**Convergence.** Because calibration is performed against targets rendered on the scene image, the gaze prediction and the detector outputs share a common coordinate frame, making focus determination a direct geometric query.

## IV. Feature Extraction and Gaze Modelling

### A. Landmark-based features

For each face frame, the system extracts a **486-dimensional** feature vector replicating the EyeTrax formulation. The vector concatenates normalised 3-D coordinates of densely sampled landmarks around the left eye, the right eye (including the five refined iris landmarks per eye), and a set of mutual reference landmarks spanning the face (nose bridge, chin, cheeks, forehead) used to factor out global head pose and scale. An eye-aspect-ratio (EAR) statistic is computed in parallel for blink detection.

Normalising eye-region landmarks against the mutual reference set makes the representation approximately invariant to translation and scale, so that the learned mapping responds primarily to relative eye and head configuration rather than absolute face position in the frame.

### B. Regression model

Gaze prediction is a multi-output regression from the 486-D feature vector to a 2-D scene coordinate `(x, y)`. The pipeline is a scikit-learn `StandardScaler` followed by a **Ridge** regressor (L2-regularised linear regression), chosen as the default for its stability under limited calibration data and negligible inference cost. The implementation also supports `LinearSVR`, `ElasticNet`, and `MLPRegressor` as drop-in alternatives via the `--model` flag, enabling future comparison studies.

Ridge regression is attractive here because the per-user calibration set is small (tens to low-hundreds of samples) relative to the feature dimensionality; L2 regularisation mitigates over-fitting in this high-dimension/low-sample regime while keeping the model interpretable and fast.

## V. Calibration

Calibration personalises the gaze model to the current user and the current camera geometry. FoGaze uses a **pulse-and-capture** protocol over a spatial grid augmented with head-pose targets.

1. **Face acquisition.** The user looks toward the face camera; a two-second countdown arc confirms a stable face lock and restarts if the face is lost.
2. **Grid capture.** A circular target is overlaid on the scene image at each node of a **4×4 grid** (16 points, with a 10 % margin). For each target the system runs a one-second *pulse* (drawing the user's fixation) followed by a one-second *capture* window during which feature samples are collected. The user fixates the real-world region under the circle and confirms.
3. **Head-pose sampling.** Additional samples are gathered at five head orientations (centre, left, right, up, down) to broaden the model's coverage of head movement.
4. **Training.** Collected (feature, target) pairs train the `StandardScaler` + Ridge pipeline; the fitted model is serialised to disk (`.pkl`).

Controls allow the user to capture (`ENTER`), undo the last point (`BACKSPACE`), and finish (`ESC`); re-calibration can be triggered at any time during operation with `c`, without restarting the application.

**Lens-distortion correction.** Because consumer webcams exhibit barrel distortion that biases the landmark-to-gaze mapping, FoGaze includes an optional chessboard-based intrinsic calibration (`CameraCalibrator`) and applies a sensible default undistortion for typical webcams, improving the geometric consistency between calibration targets and runtime predictions.

## VI. Scene Understanding and Focus Determination

### A. Object detection

The scene colour stream is processed by **YOLOv8-nano**. To meet the real-time budget on CPU, inference runs at a reduced input size (320 px) and is **temporally sub-sampled**: detection executes every *N* frames (default *N* = 3, configurable via `--detection-interval`), with the most recent detections cached and reused on the intervening frames. This frame-skipping exploits the relative slowness of scene change compared with gaze movement.

### B. Focus determination

Given the smoothed gaze point `g = (x, y)` and the set of detected bounding boxes `{b_i}`, the focused object is the detection whose box contains `g`. When multiple boxes overlap the gaze point, the most specific (smallest containing) box is preferred. The focused object is highlighted in the UI and announced through the interaction layer.

### C. Depth lifting and ROS 2 publication

For each detection, `depth_at_bbox()` samples the registered depth stream within the bounding box to estimate the object's distance. The focused object is thereby promoted from a 2-D image detection to a metric 3-D position. FoGaze publishes detections as a `visualization_msgs/MarkerArray` and broadcasts a `map → fogaze_base` transform over TF, allowing the scene and the user's focus to be visualised live in **RViz** and consumed by downstream robotic behaviours.

### D. Spatial relations

A lightweight zoning function partitions the scene frame into regions and computes coarse spatial relations between objects (e.g., left-of, above), providing human-readable context for the focused object beyond its class label.

## VII. Filtering and Hands-Free Interaction

### A. Gaze smoothing

Raw per-frame gaze predictions are noisy. FoGaze applies a configurable smoothing cascade:

- **Kalman + EMA** (default) — a Kalman filter for motion-consistent tracking, followed by an exponential moving average (`--ema-alpha`, default 0.8) for additional steadiness;
- **KDE** — a kernel-density confidence smoother (`--kde-confidence`); and
- **One-Euro filter** — an adaptive low-pass filter that trades latency against jitter as a function of gaze velocity, applied at the tracker level.

The smoother is selectable at runtime (`--filter`) so that responsiveness and stability can be tuned to the task.

### B. Blink and speech interaction

A **triple-blink detector** consumes the EAR signal over a sliding 1.5-second window to recognise an intentional triple blink as a hands-free "select" gesture. A **text-to-speech** layer (`pyttsx3`), rate-limited by a cooldown, announces the currently focused object aloud. Together these enable an eyes-and-voice interaction loop with no manual input — important for the assistive-technology use case.

## VIII. Implementation

FoGaze is implemented in Python 3.10+ and runs on Linux (developed on Ubuntu 22.04, GNOME + XWayland). Key dependencies are OpenCV, MediaPipe (pinned `<0.10.10` to avoid a breaking API migration), NumPy, scikit-learn, Ultralytics YOLO, OpenNI2 (PrimeSense access), and `pyttsx3`. The interactive UI is built with Dear ImGui over a GLFW/OpenGL context, providing a dark-theme overlay with an animated gaze cursor and trail, bounding-box rendering with focus highlighting, a face picture-in-picture view, an optional depth-colourmap view, and an FPS/HUD panel. On Wayland the application forces `QT_QPA_PLATFORM=xcb` for backend compatibility, and caches the trained model under the user's cache directory. ROS 2 publishing is optional and degrades gracefully when the middleware is absent.

Engineering choices that keep the system real-time on CPU include: reduced-resolution, frame-skipped detection; a linear gaze model with negligible inference cost; and reuse of a single MediaPipe landmark pass for gaze, blink, and head-pose features.

## IX. Discussion and Future Work

**Limitations.** The current work is an engineering system description rather than a controlled accuracy study; we have not yet reported quantitative gaze-angular error, focus-classification accuracy, or end-to-end latency. The linear gaze model is sensitive to head movement beyond the calibrated range and to lighting changes that perturb landmark detection. Depth-from-bbox sampling can be biased by occlusion or background pixels inside the box. Focus determination by point-in-box is ambiguous for overlapping or nested objects.

**Future work.** Planned directions include: (i) a formal evaluation protocol measuring gaze error in degrees and object-focus accuracy/F1 against ground-truth fixations; (ii) comparison of the Ridge baseline against the SVR/ElasticNet/MLP alternatives already supported; (iii) head-pose-conditioned or non-linear gaze models to extend the operating envelope; (iv) instance-segmentation masks in place of axis-aligned boxes for sharper focus boundaries; and (v) temporal fixation modelling to distinguish deliberate focus from saccadic transit.

## X. Conclusion

We presented FoGaze, a dual-camera system that estimates which real-world object a user is focusing on by fusing an appearance-based, per-user-calibrated gaze regressor with an RGB-D scene sensor and a YOLOv8 detector. By calibrating the gaze model directly into the scene-camera frame, the system reduces focus estimation to a geometric intersection, lifts the focused object to 3-D via registered depth, and exposes the result to ROS 2 for robotics and visualisation. With hands-free blink selection and spoken feedback, FoGaze demonstrates a low-cost, accessible path toward real-world gaze-driven interaction on commodity hardware. Quantitative evaluation against the directions outlined above is the primary next step.

## Acknowledgement

This work was carried out as part of the KAITO–Kanagawa exchange school project. The gaze-feature and calibration approach builds upon the open-source EyeTrax project.

## References

[1] A. Bulling, "Pervasive Attentive User Interfaces," *Computer*, vol. 49, no. 1, pp. 94–98, 2016.

[2] C.-K. Zhang, "EyeTrax: Webcam-Based Gaze Estimation," GitHub repository. [Online]. Available: https://github.com/ck-zhang/EyeTrax

[3] C. Lugaresi *et al.*, "MediaPipe: A Framework for Building Perception Pipelines," *arXiv:1906.08172*, 2019.

[4] Y. Kartynnik, A. Ablavatski, I. Grishchenko, and M. Grundmann, "Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs," in *Proc. CVPR Workshop on Computer Vision for AR/VR*, 2019.

[5] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[6] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in *Proc. IEEE CVPR*, 2016, pp. 779–788.

[7] K. Khoshelham and S. O. Elberink, "Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications," *Sensors*, vol. 12, no. 2, pp. 1437–1454, 2012.

[8] G. Casiez, N. Roussel, and D. Vogel, "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems," in *Proc. ACM CHI*, 2012, pp. 2527–2530.

[9] R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems," *J. Basic Eng.*, vol. 82, no. 1, pp. 35–45, 1960.

[10] A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems," *Technometrics*, vol. 12, no. 1, pp. 55–67, 1970.

[11] F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *J. Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[12] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, "Robot Operating System 2: Design, Architecture, and Uses in the Wild," *Science Robotics*, vol. 7, no. 66, 2022.

[13] K. Krafka *et al.*, "Eye Tracking for Everyone," in *Proc. IEEE CVPR*, 2016, pp. 2176–2184.

[14] Z. Zhang, "A Flexible New Technique for Camera Calibration," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 22, no. 11, pp. 1330–1334, 2000.
