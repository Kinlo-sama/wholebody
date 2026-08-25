from typing import Any, Dict, Optional


class BaseHook:
    """Abstract Base Class for Trainer event hooks."""

    def before_train(self, trainer: Any) -> None:
        pass

    def after_train(self, trainer: Any) -> None:
        pass

    def before_train_epoch(self, trainer: Any) -> None:
        pass

    def after_train_epoch(self, trainer: Any) -> None:
        pass

    def before_train_iter(self, trainer: Any, batch_idx: int, data: Dict[str, Any]) -> None:
        pass

    def after_train_iter(self, trainer: Any, batch_idx: int, data: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        pass

    def before_val_epoch(self, trainer: Any) -> None:
        pass

    def after_val_epoch(self, trainer: Any, metrics: Dict[str, float]) -> None:
        pass
