from wholebody.datasets.base import BasePoseDataset
from wholebody.datasets.synthetic import SyntheticWholeBodyDataset
from wholebody.datasets.coco_wholebody import COCOWholeBodyDataset

__all__ = [
    "BasePoseDataset",
    "SyntheticWholeBodyDataset",
    "COCOWholeBodyDataset"
]
from .combined_dataset import CombinedDataset
__all__.append("CombinedDataset")
from .dummy_dataset import DummyPoseDataset
__all__.append("DummyPoseDataset")
