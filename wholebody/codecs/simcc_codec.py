import numpy as np
import torch
from typing import Tuple, Optional
from wholebody.core.registry import CODECS

@CODECS.register("SimCCLabel")
class SimCCCodec:
    def __init__(
        self,
        input_size: Tuple[int, int],
        sigma: float = 6.0,
        simcc_split_ratio: float = 2.0,
        normalize: bool = False,
    ):
        """
        input_size: [Height, Width] of the image (e.g., [256, 192])
        sigma: Standard deviation for the Gaussian label smoothing
        simcc_split_ratio: Ratio to expand the coordinate bins
        normalize: Whether to normalize the gaussian distributions
        """
        self.input_size = input_size # [H, W]
        self.sigma = sigma
        self.simcc_split_ratio = simcc_split_ratio
        self.normalize = normalize

    def encode(self, keypoints: np.ndarray, keypoints_visible: np.ndarray) -> dict:
        """
        keypoints: [N, K, 2] in image space (x, y)
        keypoints_visible: [N, K]
        Returns dict with:
            keypoint_x_labels: [N, K, W * ratio]
            keypoint_y_labels: [N, K, H * ratio]
            keypoint_weights: [N, K]
        """
        N, K, _ = keypoints.shape
        H, W = self.input_size
        
        W_simcc = int(np.round(W * self.simcc_split_ratio))
        H_simcc = int(np.round(H * self.simcc_split_ratio))

        target_x = np.zeros((N, K, W_simcc), dtype=np.float32)
        target_y = np.zeros((N, K, H_simcc), dtype=np.float32)
        keypoint_weights = keypoints_visible.copy()

        # Map coords to SimCC space
        keypoints_split = np.round(keypoints * self.simcc_split_ratio).astype(np.int64)

        x_grid = np.arange(0, W_simcc, 1, dtype=np.float32)
        y_grid = np.arange(0, H_simcc, 1, dtype=np.float32)

        radius = self.sigma * 3

        for n in range(N):
            for k in range(K):
                if keypoints_visible[n, k] < 0.5:
                    continue

                mu_x, mu_y = keypoints_split[n, k]

                if mu_x < 0 or mu_x >= W_simcc or mu_y < 0 or mu_y >= H_simcc:
                    keypoint_weights[n, k] = 0
                    continue

                # Create 1D Gaussians
                target_x[n, k] = np.exp(-((x_grid - mu_x)**2) / (2 * self.sigma**2))
                target_y[n, k] = np.exp(-((y_grid - mu_y)**2) / (2 * self.sigma**2))

        if self.normalize:
            norm_value = self.sigma * np.sqrt(np.pi * 2)
            target_x /= norm_value
            target_y /= norm_value

        return dict(
            keypoint_x_labels=target_x,
            keypoint_y_labels=target_y,
            keypoint_weights=keypoint_weights
        )

    def decode(self, encoded: Tuple[torch.Tensor, torch.Tensor], metainfo: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        encoded: Tuple of (simcc_x, simcc_y)
            simcc_x: [B, K, W * ratio]
            simcc_y: [B, K, H * ratio]
        Returns:
            keypoints: [B, K, 2] in original image space (x, y)
            scores: [B, K]
        """
        simcc_x, simcc_y = encoded
        
        B, K, Wx = simcc_x.shape
        _, _, Hy = simcc_y.shape

        # Find argmax
        x_locs = simcc_x.argmax(dim=2) # [B, K]
        y_locs = simcc_y.argmax(dim=2) # [B, K]

        # Get scores at argmax
        x_scores = simcc_x.amax(dim=2)
        y_scores = simcc_y.amax(dim=2)
        
        # Total confidence score is often the product or average of x and y confidence
        scores = (x_scores + y_scores) / 2.0

        # Create output tensor
        keypoints = torch.stack([x_locs.float(), y_locs.float()], dim=-1)

        # Map back to network input space
        keypoints /= self.simcc_split_ratio
        
        # Move to CPU numpy
        keypoints = keypoints.cpu().numpy()
        scores = scores.cpu().numpy()
        
        # Apply metainfo transformations to map back to original image space
        if metainfo is not None:
            for i, meta in enumerate(metainfo):
                if "input_center" in meta and "input_scale" in meta:
                    center = meta["input_center"]
                    scale = meta["input_scale"]
                    # Usually input_size = [W, H] for scaling, wait our input_size is [H, W]
                    H, W = self.input_size
                    # Scale is usually [W, H] in pixels / 200
                    w = scale[0] * 200.0
                    h = scale[1] * 200.0
                    
                    # Convert from [0, W] to [-0.5, 0.5]
                    keypoints[i, :, 0] = (keypoints[i, :, 0] / W - 0.5) * w + center[0]
                    keypoints[i, :, 1] = (keypoints[i, :, 1] / H - 0.5) * h + center[1]

        return keypoints, scores
