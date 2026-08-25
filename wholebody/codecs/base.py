from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class BaseCodec(ABC):
    """Abstract Base Class for Keypoint Encoders/Decoders.
    
    Codecs bridge datasets/transforms and model outputs:
      - `encode()` transforms ground truth coordinates into learning targets (Heatmaps, SimCC, normalized vectors).
      - `decode()` transforms raw model outputs into metric coordinates in the original image space.
    """

    @abstractmethod
    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Encode keypoint coordinates into model training targets."""
        pass

    @abstractmethod
    def decode(
        self,
        encoded: Union[torch.Tensor, np.ndarray],
        metainfo: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode model outputs into (keypoints, scores) in original coordinate space.
        
        Returns:
            Tuple of:
              - keypoints: (B, N, K, 2) or (B, K, 2)
              - scores: (B, N, K) or (B, K)
        """
        pass
