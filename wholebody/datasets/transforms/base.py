from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTransform(ABC):
    """Abstract Base Class for all data transformation and augmentation modules."""

    def __call__(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.transform(results)

    @abstractmethod
    def transform(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform input results dictionary in-place or return a new one."""
        pass
