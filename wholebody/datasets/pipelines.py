from typing import Any, Dict, List, Optional, Union

from wholebody.core.registry import TRANSFORMS
from wholebody.datasets.transforms.base import BaseTransform


class Compose:
    """Sequential composition of data pipeline transforms."""

    def __init__(self, transforms: List[Union[Dict[str, Any], BaseTransform]]) -> None:
        self.transforms: List[BaseTransform] = []
        for t in transforms:
            if isinstance(t, dict):
                self.transforms.append(TRANSFORMS.build(t))
            elif isinstance(t, BaseTransform) or callable(t):
                self.transforms.append(t)
            else:
                raise TypeError(f"Invalid transform item: {t}")

    def __call__(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self) -> str:
        transform_names = [t.__class__.__name__ for t in self.transforms]
        return f"Compose({transform_names})"
