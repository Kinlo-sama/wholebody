"""WholeBody: A Modular Research Framework for Whole-Body Human Pose Estimation."""

__version__ = "0.1.0"

from wholebody.core import (
    DeviceManager,
    Registry,
    Config,
    MODELS,
    BACKBONES,
    NECKS,
    HEADS,
    LOSSES,
    CODECS,
    DATASETS,
    TRANSFORMS,
    METRICS,
    HOOKS,
    KEYPOINT_SPECS,
)
from wholebody.structures import (
    KeypointSpec,
    JointSpec,
    SkeletonEdge,
    PoseDataSample,
    InstanceData,
    BoundingBox,
)
from wholebody.inference import init_model, PosePredictor
from wholebody.visualization import SkeletonVisualizer
from wholebody.engine import Trainer, Evaluator, CheckpointManager

__all__ = [
    "__version__",
    "DeviceManager",
    "Registry",
    "Config",
    "MODELS",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "LOSSES",
    "CODECS",
    "DATASETS",
    "TRANSFORMS",
    "METRICS",
    "HOOKS",
    "KEYPOINT_SPECS",
    "KeypointSpec",
    "JointSpec",
    "SkeletonEdge",
    "PoseDataSample",
    "InstanceData",
    "BoundingBox",
    "init_model",
    "PosePredictor",
    "SkeletonVisualizer",
    "Trainer",
    "Evaluator",
    "CheckpointManager",
]
