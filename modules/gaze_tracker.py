import cv2
import numpy as np
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh_conn = mp.solutions.face_mesh_connections


class OneEuroFilter:
    __slots__ = ('min_cutoff', 'beta', 'dcutoff',
                 '_x_prev', '_dx_prev', '_t_prev')

    def __init__(self, min_cutoff=1.0, beta=0.007, dcutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.reset()

    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt) if dt > 0 else 1.0

    def __call__(self, x, t=None):
        if t is None:
            t = time.perf_counter()
        if self._x_prev is None:
            self._x_prev = float(x)
            self._dx_prev = 0.0
            self._t_prev = t
            return float(x)
        dt = max(t - self._t_prev, 1e-6)
        dx = (float(x) - self._x_prev) / dt
        ad = self._alpha(self.dcutoff, dt)
        dx_smooth = ad * dx + (1.0 - ad) * self._dx_prev
        fc = self.min_cutoff + self.beta * abs(dx_smooth)
        ap = self._alpha(fc, dt)
        x_smooth = ap * float(x) + (1.0 - ap) * self._x_prev
        self._x_prev = x_smooth
        self._dx_prev = dx_smooth
        self._t_prev = t
        return x_smooth


class GazeTracker:
    """Gaze estimation with 3-level processing:
       Level 1: Full frame FaceMesh -> head-pose, rough landmarks
       Level 2: Zoomed face crop -> refined iris via intensity centroid
       Level 3: Zoomed iris crop -> finest pupil centre detection
    """

    _LEFT_EYE = {
        "iris": [468, 469, 470, 471, 472],
        "outer": 33, "inner": 133,
        "top": 159, "bottom": 145,
    }
    _RIGHT_EYE = {
        "iris": [473, 474, 475, 476, 477],
        "outer": 263, "inner": 362,
        "top": 386, "bottom": 374,
    }

    _EAR_LEFT = [33, 160, 158, 133, 153, 144]
    _EAR_RIGHT = [362, 387, 385, 263, 380, 373]

    _PNP_IDXS = [1, 152, 33, 263, 61, 291]
    _MODEL_3D = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ], dtype=np.float64)

    _IRIS_INDICES = {468, 469, 470, 471, 472, 473, 474, 475, 476, 477}

    # ── Face contour indices for simplified mesh ──
    _JAW = [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389,
            356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
            148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162]
    _L_BROW = [46, 53, 52, 65, 55]
    _R_BROW = [285, 295, 282, 283, 276]
    _NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
    _LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                  409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
    _L_EYE_CONT = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                   173, 157, 158, 159, 160, 161, 246]
    _R_EYE_CONT = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                   466, 388, 387, 386, 385, 384, 398]

    def __init__(self,
                 gaze_radius=0.30,
                 head_pose_strength=0.0,
                 hp_yaw_sign=1.0,
                 hp_pitch_sign=-1.0,
                 oef_min_cutoff=1.0,
                 oef_beta=0.007,
                 oef_dcutoff=1.0,
                 zoom_refine=False,
                 pip_size=160,
                 iris_zoom_size=100):
        self.gaze_radius = gaze_radius
        self.hp_strength = head_pose_strength
        self.hp_yaw_sign = hp_yaw_sign
        self.hp_pitch_sign = hp_pitch_sign
        self.zoom_refine = zoom_refine
        self.pip_size = pip_size
        self.iris_zoom_size = iris_zoom_size

        self._lm_filters = {}
        self._lm_cutoff = max(oef_min_cutoff * 2, 2.0)
        self._lm_beta = oef_beta * 0.3

        self._prev_nose_pos = None
        self._head_speed = 0.0
        self._head_speed_filtered = 0.0

        self._roll_filter = OneEuroFilter(2.0, 0.005, 1.0)

        self._hp_yaw = 0.0
        self._hp_pitch = 0.0
        self._hp_roll = 0.0

        self._oef_x = OneEuroFilter(oef_min_cutoff, oef_beta, oef_dcutoff)
        self._oef_y = OneEuroFilter(oef_min_cutoff, oef_beta, oef_dcutoff)
        self._oef_base_min_cutoff = oef_min_cutoff
        self._oef_base_beta = oef_beta
        self._oef_base_dcutoff = oef_dcutoff

        self._lost_time = None
        self._raw_lm_ = None
        self._t_ = 0.0

        self._refined_iris = {}
        self._pip_face = None
        self._pip_iris = None
        self._last_crop_bbox = None

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ── Filter helpers ───────────────────────────────────────────────────

    def _reset_landmark_filters(self):
        for f in self._lm_filters.values():
            f[0].reset()
            f[1].reset()
        self._prev_nose_pos = None
        self._head_speed = 0.0
        self._head_speed_filtered = 0.0
        self._roll_filter.reset()

    def _reset_output_filters(self):
        self._oef_x.reset()
        self._oef_y.reset()
        self._hp_yaw = 0.0
        self._hp_pitch = 0.0
        self._hp_roll = 0.0

    def _lm(self, idx, t):
        if idx not in self._lm_filters:
            self._lm_filters[idx] = (
                OneEuroFilter(self._lm_cutoff, self._lm_beta, 0.5),
                OneEuroFilter(self._lm_cutoff, self._lm_beta, 0.5),
            )
        fx, fy = self._lm_filters[idx]
        if idx in self._refined_iris:
            rx, ry = self._refined_iris[idx]
            return fx(rx, t), fy(ry, t)
        p = self._raw_lm_.landmark[idx]
        return fx(p.x, t), fy(p.y, t)

    # ── Face crop ────────────────────────────────────────────────────────

    def get_face_crop(self, frame, landmarks, margin=1.4, out_size=300):
        if landmarks is None:
            return None, None
        h, w = frame.shape[:2]
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        hw = (x_max - x_min) * margin * 0.5
        hh = (y_max - y_min) * margin * 0.5

        x1 = int(max(0, (cx - hw) * w))
        x2 = int(min(w, (cx + hw) * w))
        y1 = int(max(0, (cy - hh) * h))
        y2 = int(min(h, (cy + hh) * h))

        if x2 <= x1 or y2 <= y1:
            return None, None

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (out_size, out_size))
        return crop, (x1, y1, x2, y2)

    # ── Level 2: Intensity refinement on face crop ──────────────────────

    @staticmethod
    def _refine_iris_intensity(gray, cx, cy, win=20):
        h, w = gray.shape[:2]
        x1 = int(max(0, cx - win))
        x2 = int(min(w, cx + win))
        y1 = int(max(0, cy - win))
        y2 = int(min(h, cy + win))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return float(cx), float(cy)

        roi = gray[y1:y2, x1:x2]

        # Bilateral filter — edge-preserving
        filtered = cv2.bilateralFilter(roi, 7, 50, 50)

        # CLAHE for local contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        enhanced = clahe.apply(filtered)

        _, thresh = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        best_center = None
        best_score = -1e9

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 8:
                continue

            peri = float(cv2.arcLength(c, True))
            circ = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0

            M = cv2.moments(c)
            if M["m00"] < 1e-6:
                continue
            bx = float(M["m10"] / M["m00"])
            by = float(M["m01"] / M["m00"])

            dist = np.hypot(bx - (cx - x1), by - (cy - y1))
            dist_penalty = dist / (win * 0.5)
            score = area * circ * (1.0 - 0.5 * dist_penalty)

            if score > best_score:
                best_score = score
                best_center = (bx + x1, by + y1)

        if best_center is not None and best_score > 5:
            return float(best_center[0]), float(best_center[1])

        moments = cv2.moments(thresh)
        if moments["m00"] > 10:
            return float(moments["m10"] / moments["m00"] + x1), \
                   float(moments["m01"] / moments["m00"] + y1)

        return float(cx), float(cy)

    # ── Level 3: Iris zoom + finest pupil refinement ───────────────────

    def get_iris_zoom(self, frame, center_px, zoom=5, out_size=100):
        """Return (zoomed_img with pupil overlay, refined_norm (x,y)) or (None, None)."""
        cx, cy = center_px
        h, w = frame.shape[:2]
        crop_size = out_size // zoom
        half = crop_size // 2

        x1 = int(max(0, cx - half))
        x2 = int(min(w, cx + half))
        y1 = int(max(0, cy - half))
        y2 = int(min(h, cy + half))

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, None

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        local_px = float(cx - x1)
        local_py = float(cy - y1)

        # Bilateral filter
        filtered = cv2.bilateralFilter(enhanced, 5, 30, 30)

        _, thresh = cv2.threshold(filtered, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        pupil_contour = None
        best_score = -1e9
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 6:
                continue
            peri = float(cv2.arcLength(c, True))
            circ = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0
            M = cv2.moments(c)
            if M["m00"] < 1e-6:
                continue
            bx = float(M["m10"] / M["m00"])
            by = float(M["m01"] / M["m00"])
            dist = np.hypot(bx - local_px, by - local_py)
            score = area * circ * (1.0 - 0.4 * (dist / (crop_size * 0.5)))
            if score > best_score:
                best_score = score
                local_px, local_py = bx, by
                pupil_contour = c

        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        if pupil_contour is not None and best_score > 3:
            cv2.drawContours(enhanced_color, [pupil_contour], -1,
                             (0, 255, 0), 1)
        cv2.circle(enhanced_color, (int(local_px), int(local_py)), 2,
                   (0, 0, 255), -1)
        cv2.circle(enhanced_color, (int(local_px), int(local_py)), 4,
                   (0, 0, 255), 1)

        zoomed = cv2.resize(enhanced_color, (out_size, out_size),
                            interpolation=cv2.INTER_NEAREST)

        refined_frame_x = float(x1 + local_px)
        refined_frame_y = float(y1 + local_py)
        refined_norm = (refined_frame_x / w, refined_frame_y / h)

        return zoomed, refined_norm

    # ── Simplified mesh drawing on face crop ────────────────────────────

    def _draw_simple_mesh(self, crop, landmarks, bbox, fw, fh):
        x1, y1, x2, y2 = bbox
        cw, ch = x2 - x1, y2 - y1
        ch2, cw2 = crop.shape[:2]

        def _p(idx):
            p = landmarks.landmark[idx]
            fx = p.x * fw
            fy = p.y * fh
            cx = int((fx - x1) / cw * cw2)
            cy = int((fy - y1) / ch * ch2)
            return cx, cy

        cv2.polylines(crop,
                      [np.array([_p(i) for i in self._JAW], np.int32)],
                      isClosed=False, color=(100, 100, 100), thickness=1,
                      lineType=cv2.LINE_AA)
        cv2.polylines(crop,
                      [np.array([_p(i) for i in self._L_BROW], np.int32)],
                      isClosed=False, color=(160, 160, 160), thickness=1,
                      lineType=cv2.LINE_AA)
        cv2.polylines(crop,
                      [np.array([_p(i) for i in self._R_BROW], np.int32)],
                      isClosed=False, color=(160, 160, 160), thickness=1,
                      lineType=cv2.LINE_AA)
        cv2.polylines(crop,
                      [np.array([_p(i) for i in self._NOSE], np.int32)],
                      isClosed=False, color=(160, 160, 160), thickness=1,
                      lineType=cv2.LINE_AA)
        cv2.polylines(crop,
                      [np.array([_p(i) for i in self._LIP_OUTER], np.int32)],
                      isClosed=True, color=(160, 160, 160), thickness=1,
                      lineType=cv2.LINE_AA)
        cv2.polylines(crop,
                      [np.array([_p(i) for i in self._L_EYE_CONT], np.int32)],
                      isClosed=True, color=(0, 200, 0), thickness=1,
                      lineType=cv2.LINE_AA)
        cv2.polylines(crop,
                      [np.array([_p(i) for i in self._R_EYE_CONT], np.int32)],
                      isClosed=True, color=(0, 200, 0), thickness=1,
                      lineType=cv2.LINE_AA)
        cv2.circle(crop, _p(468), 2, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(crop, _p(473), 2, (255, 0, 255), -1, cv2.LINE_AA)

    # ── Head motion tracking ─────────────────────────────────────────────

    def _track_head_motion(self, t):
        nx, ny = self._lm(1, t)
        if self._prev_nose_pos is not None:
            pnx, pny, pt = self._prev_nose_pos
            dt = max(t - pt, 1e-6)
            dist = np.hypot(nx - pnx, ny - pny)
            speed = dist / dt
            self._head_speed_filtered = (0.9 * self._head_speed_filtered
                                         + 0.1 * speed)
            self._head_speed = speed
        else:
            self._head_speed = 0.0
            self._head_speed_filtered = 0.0
        self._prev_nose_pos = (nx, ny, t)

    # ── Head pose (solvePnP) ──────────────────────────────────────────

    def _head_pose(self, t, fw, fh):
        img_pts = np.array([
            [self._lm(i, t)[0] * fw, self._lm(i, t)[1] * fh]
            for i in self._PNP_IDXS
        ], dtype=np.float64)

        focal = fw
        cam = np.array([
            [focal, 0, fw / 2],
            [0, focal, fh / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist = np.zeros((4, 1))

        success, rvec, _ = cv2.solvePnP(
            self._MODEL_3D, img_pts, cam, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        yaw = pitch = 0.0
        if success:
            R, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            if sy > 1e-6:
                yaw = float(np.degrees(np.arctan2(-R[2, 0], sy)))
                pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))

        hp_a = 0.7
        self._hp_yaw = hp_a * self._hp_yaw + (1 - hp_a) * yaw
        self._hp_pitch = hp_a * self._hp_pitch + (1 - hp_a) * pitch
        return self._hp_yaw, self._hp_pitch

    # ── Iris ratio (uses _refined_iris via _lm) ──────────────────────

    def _eye_ratio(self, t, eye_def):
        xs, ys = [], []
        for idx in eye_def["iris"]:
            ix, iy = self._lm(idx, t)
            xs.append(ix)
            ys.append(iy)
        iris = np.array([float(np.mean(xs)), float(np.mean(ys))])

        ox, oy = self._lm(eye_def["outer"], t)
        ix, iy = self._lm(eye_def["inner"], t)
        tx, ty = self._lm(eye_def["top"], t)
        bx, by = self._lm(eye_def["bottom"], t)

        outer = np.array([ox, oy])
        inner = np.array([ix, iy])
        top = np.array([tx, ty])
        bottom = np.array([bx, by])

        ew = float(np.linalg.norm(inner - outer))
        eh = float(np.linalg.norm(bottom - top))

        rx = (iris[0] - outer[0]) / ew if ew > 0 else 0.5
        ry = (iris[1] - top[1]) / eh if eh > 0 else 0.5
        return np.clip([rx, ry], 0.0, 1.0)

    # ── Roll compensation ────────────────────────────────────────────────

    def _get_roll_angle(self, t):
        lx, ly = self._lm(33, t)
        rx, ry = self._lm(263, t)
        angle = np.arctan2(ly - ry, lx - rx)
        return self._roll_filter(angle, t)

    # ── Gaze estimation ─────────────────────────────────────────────────

    def _estimate(self, t, fw, fh):
        rr = self._eye_ratio(t, self._RIGHT_EYE)
        ratio = rr

        ox = ratio[0] - 0.5
        oy = ratio[1] - 0.5

        roll = self._get_roll_angle(t)
        self._hp_roll = roll
        cos_a = np.cos(-roll)
        sin_a = np.sin(-roll)
        ox_rot = ox * cos_a - oy * sin_a
        oy_rot = ox * sin_a + oy * cos_a
        ox, oy = ox_rot, oy_rot

        yaw, pitch = self._head_pose(t, fw, fh)
        if self.hp_strength > 0:
            ox += self.hp_yaw_sign * yaw * self.hp_strength / fw
            oy += self.hp_pitch_sign * pitch * self.hp_strength / fh

        speed_weight = 1.0 - min(self._head_speed_filtered * 5.0, 0.5)
        adaptive_radius = self.gaze_radius * max(speed_weight, 0.5)

        gx = fw * 0.5 - ox * (fw * adaptive_radius)
        gy = fh * 0.5 + oy * (fh * adaptive_radius)
        return float(gx), float(gy)

    # ── Public API ───────────────────────────────────────────────────────

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        now = time.perf_counter()

        # ── Level 1: Full frame FaceMesh ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._raw_lm_ = None
            self._refined_iris.clear()
            self._pip_face = None
            self._pip_iris = None
            if self._lost_time is None:
                self._lost_time = now
            elif now - self._lost_time > 1.0:
                self._reset_landmark_filters()
                self._reset_output_filters()
            if self._oef_x._x_prev is not None:
                gx = self._oef_x(self._oef_x._x_prev, now)
                gy = self._oef_y(self._oef_y._x_prev, now)
            else:
                gx, gy = w / 2.0, h / 2.0
            return None, (float(np.clip(gx, 0, w - 1)),
                          float(np.clip(gy, 0, h - 1)))

        self._lost_time = None
        self._raw_lm_ = results.multi_face_landmarks[0]
        self._t_ = now

        # ── Level 2: Zoomed face crop + intensity refinement ──
        crop, bbox = self.get_face_crop(frame, self._raw_lm_)
        self._refined_iris.clear()

        if crop is not None:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            x1, y1, x2, y2 = bbox
            cw, ch = x2 - x1, y2 - y1
            crop_h, crop_w = crop.shape[:2]

            for idx in self._IRIS_INDICES:
                p = self._raw_lm_.landmark[idx]
                fpx, fpy = int(p.x * w), int(p.y * h)
                cpx = int((fpx - x1) / cw * crop_w)
                cpy = int((fpy - y1) / ch * crop_h)
                cpx = int(np.clip(cpx, 0, crop_w - 1))
                cpy = int(np.clip(cpy, 0, crop_h - 1))

                crx, cry = self._refine_iris_intensity(crop_gray, cpx, cpy)

                full_x = x1 + (crx / crop_w) * cw
                full_y = y1 + (cry / crop_h) * ch
                self._refined_iris[idx] = (full_x / w, full_y / h)

            # Draw simplified mesh on crop for PIP display
            self._draw_simple_mesh(crop, self._raw_lm_, bbox, w, h)
            self._pip_face = crop
        else:
            self._pip_face = None

        # ── Level 3: Iris zoom (right eye only, finest pupil refinement) ──
        iz = self.iris_zoom_size

        right_key = 473
        if right_key in self._refined_iris:
            rxn, ryn = self._refined_iris[right_key]
        else:
            rxn = self._raw_lm_.landmark[right_key].x
            ryn = self._raw_lm_.landmark[right_key].y
        rpx, rpy = int(rxn * w), int(ryn * h)

        iris_img, iris_ref = self.get_iris_zoom(frame, (rpx, rpy))
        if iris_img is not None:
            # Add label
            cv2.putText(iris_img, "R", (4, iz - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1,
                        cv2.LINE_AA)
            self._pip_iris = iris_img
            if iris_ref is not None:
                for idx in self._RIGHT_EYE["iris"]:
                    self._refined_iris[idx] = iris_ref
        else:
            self._pip_iris = None

        # ── Head motion tracking ──
        self._track_head_motion(now)

        # ── Estimate gaze ──
        gx, gy = self._estimate(now, w, h)

        # ── Adaptive output filter ──
        speed_factor = 1.0 + self._head_speed_filtered * 50.0
        adaptive_cutoff = self._oef_base_min_cutoff * min(speed_factor, 10.0)
        self._oef_x.min_cutoff = adaptive_cutoff
        self._oef_y.min_cutoff = adaptive_cutoff

        gx = self._oef_x(gx, now)
        gy = self._oef_y(gy, now)

        gx = float(np.clip(gx, 0, w - 1))
        gy = float(np.clip(gy, 0, h - 1))
        return self._raw_lm_, (gx, gy)

    # ── Feature extraction for multi-dimensional calibration ────────

    def get_features(self):
        if self._raw_lm_ is None:
            return {
                "iris_right": (0.5, 0.5),
                "yaw": float(self._hp_yaw),
                "pitch": float(self._hp_pitch),
                "roll": float(self._hp_roll),
            }
        t = self._t_
        rr = self._eye_ratio(t, self._RIGHT_EYE)
        return {
            "iris_right": (float(rr[0]), float(rr[1])),
            "yaw": float(self._hp_yaw),
            "pitch": float(self._hp_pitch),
            "roll": float(self._hp_roll),
        }

    def get_feature_vector(self):
        f = self.get_features()
        return np.array([
            0.5, 0.5,  # left eye placeholder (single-eye mode)
            f["iris_right"][0], f["iris_right"][1],
            f["yaw"], f["pitch"], f["roll"],
        ], dtype=np.float64)

    # ── Blink (EAR) ──────────────────────────────────────────────────────

    def get_ear(self):
        if self._raw_lm_ is None:
            return 0.0

        def _ear(indices):
            pts = [np.array(self._lm(i, self._t_)) for i in indices]
            v1 = np.linalg.norm(pts[1] - pts[5])
            v2 = np.linalg.norm(pts[2] - pts[4])
            h = np.linalg.norm(pts[0] - pts[3])
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

        return (_ear(self._EAR_LEFT) + _ear(self._EAR_RIGHT)) * 0.5

    # ── Drawing ──────────────────────────────────────────────────────────

    def draw_eye(self, frame, landmarks, fw=None, fh=None):
        if landmarks is None:
            return
        h_f, w_f = frame.shape[:2]
        fw = fw or w_f
        fh = fh or h_f

        def _pt(idx):
            p = landmarks.landmark[idx]
            return (int(p.x * fw), int(p.y * fh))

        cv2.polylines(frame,
                      [np.array([_pt(i) for i in self._L_EYE_CONT],
                                np.int32)],
                      isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.polylines(frame,
                      [np.array([_pt(i) for i in self._R_EYE_CONT],
                                np.int32)],
                      isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.circle(frame, _pt(468), 2, (0, 255, 255), -1)
        cv2.circle(frame, _pt(473), 2, (255, 0, 255), -1)

    def draw_pip(self, frame):
        """Draw face-mesh PIP (bottom-right) and iris-zoom PIP (bottom-left)."""
        h, w = frame.shape[:2]
        gap = 10

        # ── Iris zoom PIP (bottom-left, right eye only) ──
        if self._pip_iris is not None:
            ih, iw = self._pip_iris.shape[:2]
            ix1, iy1 = gap, h - ih - gap
            frame[iy1:iy1 + ih, ix1:ix1 + iw] = self._pip_iris
            cv2.rectangle(frame, (ix1 - 1, iy1 - 1),
                          (ix1 + iw, iy1 + ih), (255, 0, 255), 2)
            cv2.putText(frame, "PUPIL", (ix1 + 2, iy1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1,
                        cv2.LINE_AA)

        # ── Face crop mesh PIP (bottom-right) ──
        if self._pip_face is not None:
            s = self.pip_size
            x1 = w - s - gap
            y1 = h - s - gap
            pip = cv2.resize(self._pip_face, (s, s))
            frame[y1:y1 + s, x1:x1 + s] = pip
            cv2.rectangle(frame, (x1 - 1, y1 - 1),
                          (x1 + s, y1 + s), (0, 255, 0), 2)
            cv2.putText(frame, "FACE", (x1 + 2, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                        cv2.LINE_AA)

    def release(self):
        self.face_mesh.close()
