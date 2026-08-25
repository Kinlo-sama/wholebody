from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from wholebody.codecs.base import BaseCodec
from wholebody.core.registry import CODECS


@CODECS.register("RegressionCodec")
class RegressionCodec(BaseCodec):
    """Direct normalized coordinate regression codec [0, 1]."""

    def __init__(self, input_size: Tuple[int, int] = (256, 192)) -> None:
        self.input_size = tuple(input_size)  # (H, W)

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        h_in, w_in = self.input_size
        norm_keypoints = keypoints.copy()
        norm_keypoints[:, 0] /= float(w_in)
        norm_keypoints[:, 1] /= float(h_in)

        if keypoints_visible is None:
            keypoint_weights = np.ones((keypoints.shape[0],), dtype=np.float32)
        else:
            keypoint_weights = (keypoints_visible > 0).astype(np.float32)

        return {
            "target_coords": norm_keypoints.astype(np.float32),
            "keypoint_weights": keypoint_weights.astype(np.float32),
        }

    def decode(
        self,
        encoded: Union[torch.Tensor, np.ndarray],
        metainfo: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(encoded, torch.Tensor):
            coords = encoded.detach().cpu().numpy()
        else:
            coords = encoded

        batch_size, num_keypoints, _ = coords.shape
        h_in, w_in = self.input_size

        preds = coords.copy()
        preds[:, :, 0] *= float(w_in)
        preds[:, :, 1] *= float(h_in)
        scores = np.ones((batch_size, num_keypoints), dtype=np.float32)

        if metainfo is not None and len(metainfo) == batch_size:
            for b in range(batch_size):
                meta = metainfo[b]
                if "warp_mat_inv" in meta:
                    warp_inv = meta["warp_mat_inv"]
                    pts_homo = np.concatenate([preds[b], np.ones((num_keypoints, 1), dtype=np.float32)], axis=1)
                    preds[b] = np.dot(pts_homo, warp_inv.T)[:, :2]

        return preds, scores
