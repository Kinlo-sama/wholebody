from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from wholebody.core.registry import KEYPOINT_SPECS
from wholebody.datasets.pipelines import Compose
from wholebody.structures.keypoint_spec import KeypointSpec
from wholebody.structures.data_sample import PoseDataSample


class BasePoseDataset(Dataset, ABC):
    """Abstract Base Class for Human Pose and Whole-Body Datasets.
    
    Provides standardized pipeline execution and custom batch collation.
    """

    def __init__(
        self,
        pipeline: List[Union[Dict[str, Any], Any]],
        keypoint_spec: Union[KeypointSpec, str, Dict[str, Any]],
        metainfo: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
    ) -> None:
        super().__init__()
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)

        # Resolve keypoint specification
        if isinstance(keypoint_spec, str):
            self.keypoint_spec: KeypointSpec = KEYPOINT_SPECS.get(keypoint_spec)
        elif isinstance(keypoint_spec, dict):
            self.keypoint_spec = KEYPOINT_SPECS.build(keypoint_spec)
        elif isinstance(keypoint_spec, KeypointSpec):
            self.keypoint_spec = keypoint_spec
        else:
            raise TypeError(f"Invalid keypoint_spec: {keypoint_spec}")

        self.metainfo = metainfo or {}
        self.data_list: List[Dict[str, Any]] = self.load_annotations()

    @abstractmethod
    def load_annotations(self) -> List[Dict[str, Any]]:
        """Load and normalize raw annotations into a list of sample dicts."""
        pass

    def __len__(self) -> int:
        return len(self.data_list)

    def prepare_data(self, idx: int) -> Dict[str, Any]:
        """Prepare sample dict and inject keypoint spec metadata."""
        raw_data = self.data_list[idx].copy()
        raw_data["flip_indices"] = self.keypoint_spec.flip_indices
        raw_data["keypoint_weights"] = self.keypoint_spec.weights
        raw_data["sigmas"] = self.keypoint_spec.sigmas
        return raw_data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data_info = self.prepare_data(idx)
        results = self.pipeline(data_info)
        if results is None:
            # Fallback to next sample if transform failed
            return self.__getitem__((idx + 1) % len(self))
        return results

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch into stacked image tensor and list of PoseDataSample."""
        inputs_list = [item["inputs"] for item in batch]
        data_samples = [item["data_samples"] for item in batch]
        stacked_inputs = torch.stack(inputs_list, dim=0)
        return {
            "inputs": stacked_inputs,
            "data_samples": data_samples,
        }
