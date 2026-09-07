from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from wholebody.codecs.base import BaseCodec
from wholebody.core.registry import CODECS


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Apply Gaussian blur to heatmaps."""
    B, K, H, W = heatmaps.shape
    blurred = np.zeros_like(heatmaps)
    for b in range(B):
        for k in range(K):
            blurred[b, k] = cv2.GaussianBlur(heatmaps[b, k], (kernel, kernel), 0)
    return blurred


@CODECS.register("UDPHeatmapCodec")
class UDPHeatmapCodec(BaseCodec):
    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 192),
        heatmap_size: Tuple[int, int] = (64, 48),
        sigma: float = 2.0,
    ) -> None:
        self.input_size = tuple(input_size)
        self.heatmap_size = tuple(heatmap_size)
        self.sigma = sigma
        self.kernel = 11  # typically 11 for sigma=2

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Only decode is implemented for testing.")

    def decode(
        self,
        encoded: Union[torch.Tensor, np.ndarray],
        metainfo: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(encoded, torch.Tensor):
            heatmaps = encoded.detach().cpu().numpy()
        else:
            heatmaps = encoded.copy()

        B, K, H, W = heatmaps.shape

        # Gaussian blur + Taylor expansion (UDP decoding)
        heatmaps = np.maximum(gaussian_blur(heatmaps, self.kernel), 1e-10)
        heatmaps = np.log(heatmaps)

        heatmaps_reshaped = heatmaps.reshape((B, K, -1))
        idx = np.argmax(heatmaps_reshaped, axis=2)
        maxvals = np.amax(heatmaps_reshaped, axis=2)

        preds = np.zeros((B, K, 2), dtype=np.float32)
        preds[:, :, 0] = idx % W
        preds[:, :, 1] = idx // W

        for b in range(B):
            for k in range(K):
                px = int(preds[b, k, 0])
                py = int(preds[b, k, 1])
                if 1 < px < W - 1 and 1 < py < H - 1:
                    dx = 0.5 * (heatmaps[b, k, py, px + 1] - heatmaps[b, k, py, px - 1])
                    dy = 0.5 * (heatmaps[b, k, py + 1, px] - heatmaps[b, k, py - 1, px])
                    dxx = 0.25 * (heatmaps[b, k, py, px + 2] - 2 * heatmaps[b, k, py, px] + heatmaps[b, k, py, px - 2])
                    dxy = 0.25 * (heatmaps[b, k, py + 1, px + 1] - heatmaps[b, k, py - 1, px + 1] - heatmaps[b, k, py + 1, px - 1] + heatmaps[b, k, py - 1, px - 1])
                    dyy = 0.25 * (heatmaps[b, k, py + 2, px] - 2 * heatmaps[b, k, py, px] + heatmaps[b, k, py - 2, px])
                    derivative = np.matrix([[dx], [dy]])
                    hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
                    if dxx * dyy - dxy ** 2 != 0:
                        hessianinv = hessian.I
                        offset = -hessianinv * derivative
                        offset = np.squeeze(np.array(offset.T), axis=0)
                        preds[b, k, 0] += offset[0]
                        preds[b, k, 1] += offset[1]

        # Map back using UDP scaling
        if metainfo is not None and len(metainfo) == B:
            for b in range(B):
                meta = metainfo[b]
                c = np.array(meta["center"], dtype=np.float32)
                s = np.array(meta["scale"], dtype=np.float32) * 200.0
                scale_x = s[0] / (W - 1.0)
                scale_y = s[1] / (H - 1.0)
                
                preds[b, :, 0] = preds[b, :, 0] * scale_x + c[0] - s[0] * 0.5
                preds[b, :, 1] = preds[b, :, 1] * scale_y + c[1] - s[1] * 0.5

        return preds, maxvals
