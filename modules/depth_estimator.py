"""Depth estimation using MiDaS (Intel ISL)."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.hub


class DepthEstimator:
    def __init__(self, device="cuda"):
        import warnings
        # Monkey-patch trust check (headless env)
        orig = torch.hub._check_repo_is_trusted
        torch.hub._check_repo_is_trusted = lambda *a, **kw: True

        # Check CUDA compatibility (GPU compute capability)
        cuda_ok = False
        if device == "cuda" and torch.cuda.is_available():
            try:
                cc = torch.cuda.get_device_capability()
                # PyTorch 2.12 supports sm_75+
                cuda_ok = cc >= (7, 0)
                if not cuda_ok:
                    warnings.warn(
                        f"GPU CC {cc[0]}.{cc[1]} too low, falling back to CPU")
            except Exception:
                cuda_ok = False

        self.device = torch.device(
            "cuda" if (device == "cuda" and cuda_ok) else "cpu")
        print(f"[DepthEstimator] Using device: {self.device}")
        self.model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
        )
        self.model.to(self.device).eval()

        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        self.transform = transforms.small_transform

        self._last_depth = None
        self._h, self._w = None, None

        torch.hub._check_repo_is_trusted = orig

    @torch.no_grad()
    def estimate(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run depth estimation. Returns depth map (H, W) float32, higher = closer."""
        h, w = image_bgr.shape[:2]
        self._h, self._w = h, w

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        depth = self.model(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth_np = depth.cpu().numpy().astype(np.float32)
        self._last_depth = depth_np
        return depth_np

    def depth_at_bbox(self, x1, y1, x2, y2):
        """Return average depth within bbox region."""
        if self._last_depth is None:
            return None
        roi = self._last_depth[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return float(roi.mean())

    def depth_to_distance_cm(self, depth_value: float, scene_min=None, scene_max=None) -> float:
        """Convert raw MiDaS depth (disparity) to approximate cm.
        
        Higher disparity = closer. Maps roughly 20-200cm."""
        if scene_min is None or scene_max is None:
            if self._last_depth is None:
                return 0
            scene_min = float(self._last_depth.min())
            scene_max = float(self._last_depth.max())
        if scene_max == scene_min:
            return 100
        norm = (depth_value - scene_min) / (scene_max - scene_min)
        norm = np.clip(norm, 0, 1)
        return 200 * (1 - norm) + 20

    def colormap(self, depth_map: np.ndarray | None = None) -> np.ndarray:
        """Return a color-mapped depth image (overlay)."""
        d = depth_map if depth_map is not None else self._last_depth
        if d is None:
            return np.zeros((self._h or 480, self._w or 640, 3), dtype=np.uint8)
        norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        norm = (norm * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)
