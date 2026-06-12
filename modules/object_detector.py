"""
Object Detector Module

Wraps Ultralytics YOLO for real-time object detection.  Returns structured
detection dictionaries with bounding-box, class label, and image crop.
"""

from ultralytics import YOLO
import numpy as np


class ObjectDetector:
    """YOLO-based object detector with lightweight result formatting."""

    def __init__(self, model_path="models/yolov8n.pt", confidence=0.5,
                 device="cpu", imgsz=320):
        """
        Args:
            model_path: Path to the YOLO weights file.
                        (Downloaded automatically if it does not exist.)
            confidence: Detection confidence threshold (0.0 – 1.0).
            device:     Torch device ("cpu", "cuda:0", etc.).  Defaults to
                        "cpu" for broad GPU-compatibility issues.
            imgsz:      Inference image size (pixels).  Smaller = faster.
                        320 is a good balance for real-time.
        """
        self.model = YOLO(model_path)
        self.model.to(device)
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz

        # Frame-skip cache
        self._cached_detections = []
        self._frame_counter = 0
        self._detection_interval = 3

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

        results = self.model(frame, conf=self.confidence, verbose=False,
                             imgsz=self.imgsz)
        if not results:
            return []

        result = results[0]
        detections = []

        if result.boxes is None:
            self._cached_detections = detections
            return detections

        h, w = frame.shape[:2]

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = result.names.get(cls_id, "unknown")

            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else np.array([])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_name": class_name,
                "confidence": conf,
                "crop": crop,
            })

        self._cached_detections = detections
        return detections
