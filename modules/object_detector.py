"""
Object Detector Module

Wraps Ultralytics YOLO for real-time object detection.  Returns structured
detection dictionaries with bounding-box, class label, and image crop.

Supports both the closed-vocabulary COCO models (e.g. ``yolov8n.pt``) and the
open-vocabulary YOLO-World models (``yolov8s-world.pt``).  For YOLO-World we
call ``set_classes`` once with an expanded vocabulary so the (heavy) text
encoder runs a single time and per-frame cost stays close to a plain YOLOv8s.
"""

from pathlib import Path

from ultralytics import YOLO
import numpy as np


# Expanded vocabulary used when a YOLO-World model is loaded.  COCO's 80
# classes plus common desk / household / manipulation targets so the gaze
# pipeline can name more of what the user looks at.
WORLD_CLASSES = [
    "person", "bottle", "cup", "mug", "glass", "can", "wine glass", "bowl",
    "plate", "fork", "knife", "spoon", "chopsticks", "kettle", "teapot",
    "jar", "box", "book", "notebook", "pen", "pencil", "marker", "eraser",
    "ruler", "scissors", "stapler", "laptop", "keyboard", "mouse", "monitor",
    "tv", "remote", "cell phone", "smartphone", "tablet", "charger",
    "headphones", "earbuds", "camera", "speaker", "microphone", "clock",
    "watch", "wallet", "keys", "backpack", "bag", "handbag", "umbrella",
    "chair", "stool", "couch", "table", "desk", "lamp", "plant", "vase",
    "flower pot", "picture frame", "mirror", "door handle", "light switch",
    "power outlet", "trash can", "tissue box", "toothbrush", "toothpaste",
    "comb", "razor", "soap", "shampoo bottle", "towel", "spray bottle",
    "medicine bottle", "pill bottle", "apple", "banana", "orange", "lemon",
    "tomato", "potato", "carrot", "bread", "sandwich", "snack", "candy",
    "cookie", "chocolate bar", "food container", "lunch box", "thermos",
    "screwdriver", "hammer", "pliers", "wrench", "tape", "glue stick",
    "battery", "usb drive", "cable", "adapter", "calculator", "card",
    "credit card", "coin", "medal", "controller", "joystick", "fan",
    "heater", "hair dryer", "iron", "blender", "microwave", "toaster",
    "coffee maker", "rice cooker", "pan", "pot", "lid", "cutting board",
    "napkin", "straw", "lighter", "matchbox", "candle", "gloves", "hat",
    "shoe", "sock", "sunglasses", "ball", "toy", "doll",
]


def available_models(models_dir: Path):
    """UI model menu: (label, weights_path/name, is_world)."""
    return [
        ("yolov8n (fast)", str(models_dir / "yolov8n.pt"), False),
        ("yolov8s (better)", str(models_dir / "yolov8s.pt"), False),
        ("yolov8s-world (open-vocab)", "yolov8s-worldv2.pt", True),
    ]


def _cuda_usable():
    """True only if a CUDA kernel actually *runs* on this GPU.

    ``torch.cuda.is_available()`` lies on a GPU whose compute capability the
    installed PyTorch build was not compiled for (e.g. a Pascal sm_61 card
    like the MX250 under a cu13 wheel that only ships sm_75+).  The import
    succeeds and ``is_available()`` is True, but the first kernel raises
    ``CUDA error: no kernel image is available for execution on the device``.
    So we launch a tiny real kernel and confirm it completes.
    """
    try:
        import warnings
        import torch
        if not torch.cuda.is_available():
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = torch.zeros(1, device="cuda:0")
            _ = (x + 1).cpu()  # forces a kernel launch + sync
        return True
    except Exception as e:
        cap = ""
        try:
            import torch
            cc = torch.cuda.get_device_capability(0)
            cap = f" (GPU sm_{cc[0]}{cc[1]} not in this torch build)"
        except Exception:
            pass
        print(f"[ObjectDetector] CUDA present but unusable{cap}: "
              f"{type(e).__name__}. Falling back to CPU.")
        return False


def _iou(a, b):
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DetectionStabilizer:
    """Temporal tracker that de-flickers and de-jitters YOLO detections.

    Each inference, detections are matched to existing tracks by class + IoU.
    A matched track's box is low-pass filtered (EMA) so it stops shaking; an
    unmatched track is kept alive for ``max_age`` more inferences so a single
    missed frame doesn't make the box blink out; a fresh detection must be seen
    ``min_hits`` times before it is shown so spurious one-frame boxes never
    appear.  Operates per inference cycle (not per displayed frame).
    """

    def __init__(self, iou_match=0.3, smooth=0.5, max_age=3, min_hits=2):
        self.iou_match = iou_match   # min IoU to call it the same object
        self.smooth = smooth         # EMA weight on the NEW box (0..1); lower=smoother
        self.max_age = max_age       # inferences a track survives unmatched
        self.min_hits = min_hits     # inferences before a track is shown
        self._tracks = []

    def reset(self):
        self._tracks = []

    def update(self, dets):
        """Feed this inference's raw detections, return stabilized ones."""
        for t in self._tracks:
            t["_matched"] = False
        n_existing = len(self._tracks)

        for d in dets:
            best, best_iou = -1, self.iou_match
            for i in range(n_existing):
                t = self._tracks[i]
                if t["_matched"] or t["class_name"] != d["class_name"]:
                    continue
                iou = _iou(d["bbox"], t["raw_bbox"])
                if iou >= best_iou:
                    best_iou, best = iou, i
            if best >= 0:
                t = self._tracks[best]
                a = self.smooth
                t["bbox"] = [a * nd + (1 - a) * od
                             for nd, od in zip(d["bbox"], t["bbox"])]
                t["raw_bbox"] = d["bbox"]
                t["confidence"] = a * d["confidence"] + (1 - a) * t["confidence"]
                t["misses"], t["_matched"] = 0, True
                t["hits"] += 1
            else:
                self._tracks.append({
                    "bbox": [float(v) for v in d["bbox"]],
                    "raw_bbox": d["bbox"],
                    "class_name": d["class_name"],
                    "confidence": d["confidence"],
                    "misses": 0, "hits": 1, "_matched": True,
                })

        # Age unmatched tracks; drop the ones that have been gone too long.
        survivors = []
        for t in self._tracks:
            if not t["_matched"]:
                t["misses"] += 1
            if t["misses"] <= self.max_age:
                survivors.append(t)
        self._tracks = survivors

        # Emit only confirmed tracks, with rounded integer boxes.
        out = []
        for t in self._tracks:
            if t["hits"] >= self.min_hits:
                out.append({
                    "bbox": [int(round(v)) for v in t["bbox"]],
                    "class_name": t["class_name"],
                    "confidence": float(t["confidence"]),
                })
        return out


def resolve_device(device):
    """Map ``"auto"``/``None`` to a *working* cuda device, else cpu.

    Validates that CUDA kernels actually run before returning a cuda device,
    so an incompatible GPU never crashes inference — it degrades to CPU.
    """
    want = "cuda:0" if device in (None, "auto") else device
    if str(want).startswith("cuda"):
        if _cuda_usable():
            return want
        if device not in (None, "auto"):
            print(f"[ObjectDetector] Requested '{device}' but GPU is unusable; "
                  f"using CPU instead.")
        return "cpu"
    return want


class ObjectDetector:
    """YOLO-based object detector with lightweight result formatting."""

    def __init__(self, model_path="models/yolov8n.pt", confidence=0.35,
                 device="cpu", imgsz=320, iou=0.45, max_det=100):
        """
        Args:
            model_path: Path to the YOLO weights file.
                        (Downloaded automatically if it does not exist.)
            confidence: Detection confidence threshold (0.0 – 1.0).  Lower =
                        more (but less certain) detections.
            device:     Torch device ("cpu", "cuda:0", "auto").  "auto" picks
                        cuda when a GPU is available, otherwise cpu.
            imgsz:      Inference image size (pixels).  Larger = sees more
                        small/distant objects but slower.  640 is the YOLO
                        default; 320 is faster but misses small things.
            iou:        NMS IoU threshold.  Higher keeps more overlapping
                        boxes (better in cluttered scenes).
            max_det:    Cap on detections returned per frame.
        """
        self.confidence = confidence
        self.device = resolve_device(device)
        self.imgsz = imgsz
        self.iou = iou
        self.max_det = max_det

        # Temporal stabilizer: stops boxes flickering / jittering frame to frame.
        self.stabilize = True
        self._stabilizer = DetectionStabilizer()

        self.model_path = str(model_path)
        self.is_world = False
        self.model = None
        self._load(self.model_path)

        # Frame-skip cache
        self._cached_detections = []
        self._frame_counter = 0
        self._detection_interval = 3

    # ── model management ─────────────────────────────────────────────────
    def _load(self, model_path):
        """(Re)load weights, move to device, and prime YOLO-World vocab."""
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model_path = str(model_path)
        self.is_world = "world" in Path(self.model_path).stem.lower()
        if self.is_world:
            try:
                self.model.set_classes(WORLD_CLASSES)
            except Exception as e:
                print(f"[ObjectDetector] set_classes failed: {e}")
        print(f"[ObjectDetector] Loaded {self.model_path} on {self.device}"
              f"{' (open-vocab)' if self.is_world else ''}")

    def swap_model(self, model_path):
        """Switch to a different model at runtime (blocks while it loads)."""
        if str(model_path) == self.model_path and self.model is not None:
            return
        self._load(model_path)
        self._cached_detections = []
        self._frame_counter = 0
        self._stabilizer.reset()

    @property
    def model_name(self):
        return Path(self.model_path).stem

    def set_detection_interval(self, n):
        """Run inference every *n* frames (1 = every frame)."""
        self._detection_interval = max(1, n)

    def detect(self, frame):
        """Run inference on *frame* (BGR numpy array).

        When *detection_interval* > 1, only every Nth frame triggers a
        full inference pass; intermediate frames reuse the last result.

        Returns a list of dicts with keys:
            bbox        – [x1, y1, x2, y2]  (int, clipped to frame bounds)
            class_name  – str (``"unknown"`` when the model is unsure)
            confidence  – float
            crop        – numpy array (the object ROI)
        """
        self._frame_counter += 1

        if self._frame_counter % self._detection_interval != 0:
            return self._cached_detections

        results = self.model(frame, conf=self.confidence, iou=self.iou,
                             max_det=self.max_det, verbose=False,
                             imgsz=self.imgsz)

        h, w = frame.shape[:2]

        # Build this inference's raw boxes (no crop yet — crop is taken from
        # the *stabilized* box below so it matches what's drawn).
        raw = []
        result = results[0] if results else None
        if result is not None and result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                raw.append({
                    "bbox": [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
                    "class_name": result.names.get(cls_id, "unknown"),
                    "confidence": float(box.conf[0]),
                })

        # De-flicker + de-jitter (track across inferences). Always run it — even
        # on an empty result — so tracks age out gracefully instead of blinking.
        detections = self._stabilizer.update(raw) if self.stabilize else raw

        # Attach the crop from each final box.
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            d["crop"] = (frame[y1:y2, x1:x2]
                         if y2 > y1 and x2 > x1 else np.array([]))

        self._cached_detections = detections
        return detections
