from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from wholebody.codecs.base import BaseCodec
from wholebody.core.registry import CODECS


@CODECS.register("MSRAHeatmapCodec")
class MSRAHeatmapCodec(BaseCodec):
    """MSRA Gaussian Heatmap encoder and decoder with sub-pixel refinement.
    
    Args:
        input_size: Tuple (height, width) of input image fed into model.
        heatmap_size: Tuple (height, width) of output heatmap from head.
        sigma: Standard deviation of 2D Gaussian kernel.
        unbiased: If True, uses unbiased sub-pixel estimation (DarkPose style).
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 192),
        heatmap_size: Tuple[int, int] = (64, 48),
        sigma: float = 2.0,
        unbiased: bool = False,
    ) -> None:
        self.input_size = tuple(input_size)
        self.heatmap_size = tuple(heatmap_size)
        self.sigma = sigma
        self.unbiased = unbiased

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate 2D Gaussian heatmaps from transformed keypoint coordinates.
        
        Args:
            keypoints: (K, 2) in input_size crop coordinate space [x, y].
            keypoints_visible: (K,) visibility / weight flags (0: invisible, 1: occluded, 2: visible).
        
        Returns:
            Dict containing:
              - 'heatmaps': (K, H_out, W_out) float32 ndarray
              - 'keypoint_weights': (K,) float32 ndarray
        """
        num_keypoints = keypoints.shape[0]
        h_out, w_out = self.heatmap_size
        h_in, w_in = self.input_size

        heatmaps = np.zeros((num_keypoints, h_out, w_out), dtype=np.float32)
        if keypoints_visible is None:
            keypoint_weights = np.ones((num_keypoints,), dtype=np.float32)
        else:
            keypoint_weights = (keypoints_visible > 0).astype(np.float32)

        # Scale factors
        scale_x = w_out / float(w_in)
        scale_y = h_out / float(h_in)

        # 3-sigma radius
        radius = int(3 * self.sigma)
        size = 2 * radius + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

        for k in range(num_keypoints):
            if keypoint_weights[k] <= 0:
                continue

            # Target coordinate on heatmap grid
            feat_x = keypoints[k, 0] * scale_x
            feat_y = keypoints[k, 1] * scale_y

            # Check if keypoint is within heatmap bounds
            mu_x = int(feat_x + 0.5)
            mu_y = int(feat_y + 0.5)

            ul = [int(mu_x - radius), int(mu_y - radius)]
            br = [int(mu_x + radius + 1), int(mu_y + radius + 1)]

            if ul[0] >= w_out or ul[1] >= h_out or br[0] < 0 or br[1] < 0:
                keypoint_weights[k] = 0.0
                continue

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], w_out) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], h_out) - ul[1]

            # Image range
            img_x = max(0, ul[0]), min(br[0], w_out)
            img_y = max(0, ul[1]), min(br[1], h_out)

            heatmaps[k, img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                heatmaps[k, img_y[0]:img_y[1], img_x[0]:img_x[1]],
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            )

        return {
            "heatmaps": heatmaps,
            "keypoint_weights": keypoint_weights,
        }

    def decode(
        self,
        encoded: Union[torch.Tensor, np.ndarray],
        metainfo: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode heatmaps to original image keypoint coordinates and confidence scores.
        
        Args:
            encoded: Heatmap tensor of shape (B, K, H_out, W_out)
            metainfo: List of B metainfo dicts containing inverse transform or center/scale.
            
        Returns:
            Tuple of (pred_keypoints: ndarray (B, K, 2), pred_scores: ndarray (B, K))
        """
        if isinstance(encoded, torch.Tensor):
            heatmaps = encoded.detach().cpu().numpy()
        else:
            heatmaps = encoded

        batch_size, num_keypoints, h_out, w_out = heatmaps.shape
        h_in, w_in = self.input_size

        # Reshape to (B, K, H*W) to find argmax
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_keypoints, -1))
        idx = np.argmax(heatmaps_reshaped, axis=2)
        maxvals = np.amax(heatmaps_reshaped, axis=2)

        preds = np.zeros((batch_size, num_keypoints, 2), dtype=np.float32)
        preds[:, :, 0] = idx % w_out  # x in heatmap coords
        preds[:, :, 1] = idx // w_out  # y in heatmap coords

        # Sub-pixel post-processing (quarter offset refinement)
        for b in range(batch_size):
            for k in range(num_keypoints):
                px = int(preds[b, k, 0])
                py = int(preds[b, k, 1])
                if 1 < px < w_out - 1 and 1 < py < h_out - 1:
                    diff_x = heatmaps[b, k, py, px + 1] - heatmaps[b, k, py, px - 1]
                    diff_y = heatmaps[b, k, py + 1, px] - heatmaps[b, k, py - 1, px]
                    preds[b, k, 0] += np.sign(diff_x) * 0.25
                    preds[b, k, 1] += np.sign(diff_y) * 0.25

        # Scale heatmap coords back to input_size crop
        preds[:, :, 0] *= float(w_in) / float(w_out)
        preds[:, :, 1] *= float(h_in) / float(h_out)

        # Transform from crop coordinates back to original image space if metainfo is provided
        if metainfo is not None and len(metainfo) == batch_size:
            for b in range(batch_size):
                meta = metainfo[b]
                if "warp_mat_inv" in meta:
                    warp_inv = meta["warp_mat_inv"]
                    # preds[b]: (K, 2)
                    pts_homo = np.concatenate([preds[b], np.ones((num_keypoints, 1), dtype=np.float32)], axis=1)
                    preds[b] = np.dot(pts_homo, warp_inv.T)[:, :2]
                elif "center" in meta and "scale" in meta:
                    # Alternative center/scale transform
                    c = np.array(meta["center"], dtype=np.float32)
                    s = np.array(meta["scale"], dtype=np.float32) * 200.0
                    scale_factor = s / np.array([w_in, h_in], dtype=np.float32)
                    preds[b] = (preds[b] - np.array([w_in / 2.0, h_in / 2.0])) * scale_factor + c

        return preds, maxvals
