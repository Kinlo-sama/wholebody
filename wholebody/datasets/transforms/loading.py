from typing import Any, Dict
import cv2
import numpy as np

from wholebody.core.registry import TRANSFORMS
from wholebody.datasets.transforms.base import BaseTransform


@TRANSFORMS.register("LoadImageFromFile")
class LoadImageFromFile(BaseTransform):
    """Load an image from file."""

    def __init__(self, to_rgb: bool = True) -> None:
        self.to_rgb = to_rgb

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if "img_path" not in results:
            raise KeyError("results dict must contain 'img_path'")

        img_path = results["img_path"]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image {img_path}")

        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results["img"] = img
        results["ori_shape"] = img.shape
        return results
