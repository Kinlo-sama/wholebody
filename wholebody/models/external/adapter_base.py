from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from wholebody.models.base import BasePoseEstimator
from wholebody.structures.data_sample import PoseDataSample


class BaseExternalModelAdapter(BasePoseEstimator):
    """Base Adapter class to seamlessly wrap 3rd-party models (RTMPose, DWPose, ViTPose, MediaPipe).
    
    Allows external model weights and architectures to conform to the wholebody framework API.
    """

    def __init__(self, external_module: nn.Module) -> None:
        super().__init__()
        self.external_module = external_module

    @abstractmethod
    def convert_state_dict(self, external_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert external weight keys to internal adapter naming."""
        pass
