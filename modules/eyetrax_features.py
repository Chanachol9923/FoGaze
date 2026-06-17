"""486D feature extraction + EAR blink — replicates EyeTrax's approach exactly."""

import time
import numpy as np
import cv2
import mediapipe as mp

# fmt: off
_LEFT_EYE = [
    107,  66, 105,  63,  70,
     55,  65,  52,  53,  46,
    468, 469, 470, 471, 472,
    133,  33,
    173, 157, 158, 159, 160, 161, 246,
    155, 154, 153, 145, 144, 163,   7,
    243, 190,  56,  28,  27,  29,  30, 247,
    130,  25, 110,  24,  23,  22,  26, 112,
    244, 189, 221, 222, 223, 224, 225, 113,
    226,  31, 228, 229, 230, 231, 232, 233,
    193, 245, 128, 121, 120, 119, 118, 117,
    111,  35, 124, 143, 156,
]
_RIGHT_EYE = [
    336, 296, 334, 293, 300,
    285, 295, 282, 283, 276,
    473, 476, 475, 474, 477,
    362, 263,
    398, 384, 385, 386, 387, 388, 466,
    382, 381, 380, 374, 373, 390, 249,
    463, 414, 286, 258, 257, 259, 260, 467,
    359, 255, 339, 254, 253, 252, 256, 341,
    464, 413, 441, 442, 443, 444, 445, 342,
    446, 261, 448, 449, 450, 451, 452, 453,
    417, 465, 357, 350, 349, 348, 347, 346,
    340, 265, 353, 372, 383,
]
_MUTUAL = [
      4,  10, 151,   9, 152, 234, 454,  58, 288,
]
# fmt: on

_SUBSET = _LEFT_EYE + _RIGHT_EYE + _MUTUAL
_FEAT_DIM = len(_SUBSET) * 3 + 3  # 486


class FeatureExtractor:
    """Extract 486D pose-normalised features + detect blinks."""

    def __init__(self, ear_history_len=50, blink_threshold_ratio=0.8, min_history=15):
        self._ear_history = []
        self._ear_hist_len = ear_history_len
        self._blink_ratio = blink_threshold_ratio
        self._min_hist = min_history
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self):
        self._face_mesh.close()

    def process(self, frame):
        """Detect face, extract features.  Returns (486D_array | None, blink_bool)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None, False

        lm = results.multi_face_landmarks[0]
        all_pts = np.array([(p.x, p.y, p.z) for p in lm.landmark], dtype=np.float32)

        # ── Pose normalisation ──
        left_corner = all_pts[33]
        right_corner = all_pts[263]
        top_of_head = all_pts[10]
        eye_center = (left_corner + right_corner) * 0.5

        shifted = all_pts - eye_center

        x_axis = right_corner - left_corner
        xn = np.linalg.norm(x_axis)
        if xn < 1e-9:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis /= xn

        y_approx = top_of_head - eye_center
        ya = np.linalg.norm(y_approx)
        if ya > 1e-9:
            y_approx /= ya

        y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
        yn = np.linalg.norm(y_axis)
        if yn > 1e-9:
            y_axis /= yn
        else:
            y_axis = np.array([0.0, 1.0, 0.0])

        z_axis = np.cross(x_axis, y_axis)
        zn = np.linalg.norm(z_axis)
        if zn > 1e-9:
            z_axis /= zn

        R = np.column_stack((x_axis, y_axis, z_axis))
        rotated = (R.T @ shifted.T).T

        # Scale by inter-ocular distance
        left_r = R.T @ (left_corner - eye_center)
        right_r = R.T @ (right_corner - eye_center)
        ied = np.linalg.norm(right_r - left_r)
        if ied > 1e-7:
            rotated /= ied

        # Subset + flatten + append angles
        subset = rotated[_SUBSET].flatten()

        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll = np.arctan2(R[2, 1], R[2, 2])

        feats = np.concatenate([subset, [yaw, pitch, roll]]).astype(np.float64)

        # ── Blink (EAR) ──
        blink = self._detect_blink(lm)
        return feats, blink

    def _detect_blink(self, landmarks):
        def _ear(l_outer, l_inner, l_top, l_bottom):
            outer = np.array([landmarks.landmark[l_outer].x,
                              landmarks.landmark[l_outer].y])
            inner = np.array([landmarks.landmark[l_inner].x,
                              landmarks.landmark[l_inner].y])
            top = np.array([landmarks.landmark[l_top].x,
                            landmarks.landmark[l_top].y])
            bottom = np.array([landmarks.landmark[l_bottom].x,
                               landmarks.landmark[l_bottom].y])
            ew = np.linalg.norm(outer - inner)
            eh = np.linalg.norm(top - bottom)
            return eh / (ew + 1e-9)

        left_ear = _ear(33, 133, 159, 145)
        right_ear = _ear(263, 362, 386, 374)
        ear = (left_ear + right_ear) * 0.5

        self._ear_history.append(ear)
        if len(self._ear_history) > self._ear_hist_len:
            self._ear_history.pop(0)

        if len(self._ear_history) >= self._min_hist:
            thr = float(np.mean(self._ear_history)) * self._blink_ratio
        else:
            thr = 0.2

        return ear < thr
