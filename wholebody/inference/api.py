from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn

from wholebody.core.config import Config
from wholebody.core.device import DeviceManager
from wholebody.core.registry import MODELS
from wholebody.datasets.transforms.affine import TopDownAffine
from wholebody.datasets.transforms.formatting import Normalize, PackPoseInputs, ToTensor
from wholebody.datasets.pipelines import Compose
from wholebody.engine.checkpointer import CheckpointManager
from wholebody.models.base import BasePoseEstimator
from wholebody.structures.data_sample import InstanceData, PoseDataSample
from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.inference.api")


def init_model(
    config: Union[str, Path, Config, Dict[str, Any]],
    checkpoint: Optional[Union[str, Path]] = None,
    device: str = "auto",
) -> BasePoseEstimator:
    """Initialize a pose estimator model from configuration and optional checkpoint weights."""
    if isinstance(config, (str, Path)):
        cfg = Config.from_file(config)
    elif isinstance(config, dict) and not isinstance(config, Config):
        cfg = Config.from_dict(config)
    else:
        cfg = config

    device_manager = DeviceManager(device=device)
    model: BasePoseEstimator = MODELS.build(cfg.model)

    if checkpoint is not None:
        CheckpointManager.load_checkpoint(
            filepath=checkpoint,
            model=model,
            strict=False,
            device=device_manager.get_device(),
        )

    model = device_manager.to_device(model)
    model.eval()
    model.cfg = cfg
    model.device_manager = device_manager
    return model


class PosePredictor:
    """High-level, user-friendly inference engine for whole-body pose estimation."""

    def __init__(
        self,
        model: BasePoseEstimator,
        input_size: Tuple[int, int] = (256, 192),
    ) -> None:
        self.model = model
        self.device_manager = getattr(model, "device_manager", DeviceManager(device="auto"))
        self.input_size = tuple(input_size)

        # Preprocessing pipeline for inference
        self.pipeline = Compose([
            TopDownAffine(input_size=self.input_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            PackPoseInputs(),
        ])

    def predict(
        self,
        image_or_path: Union[str, Path, np.ndarray],
        bboxes: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> PoseDataSample:
        """Run pose prediction on a single image."""
        if isinstance(image_or_path, (str, Path)):
            img = cv2.imread(str(image_or_path))
            if img is None:
                raise FileNotFoundError(f"Failed to read image at: {image_or_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_or_path

        h, w = img.shape[:2]

        if bboxes is None:
            # Default to full image bbox
            box = np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)
        elif isinstance(bboxes, list):
            box = np.array(bboxes, dtype=np.float32)
        else:
            box = bboxes.astype(np.float32)

        if box.ndim == 1:
            box = box.reshape(1, 4)

        # Process each person bbox
        box_w = box[0, 2] - box[0, 0]
        box_h = box[0, 3] - box[0, 1]
        center = np.array([box[0, 0] + box_w / 2.0, box[0, 1] + box_h / 2.0], dtype=np.float32)
        scale = np.array([box_w / 200.0, box_h / 200.0], dtype=np.float32)

        sample_dict = {
            "img": img,
            "img_shape": (h, w, 3),
            "bboxes": box,
            "center": center,
            "scale": scale,
            "rotation": 0.0,
        }

        transformed = self.pipeline(sample_dict)
        input_tensor = self.device_manager.to_device(transformed["inputs"].unsqueeze(0))
        data_sample: PoseDataSample = transformed["data_samples"]

        with torch.no_grad():
            with self.device_manager.autocast():
                pred_samples = self.model(
                    inputs=input_tensor,
                    data_samples=[data_sample],
                    mode="predict",
                )

        return pred_samples[0]
