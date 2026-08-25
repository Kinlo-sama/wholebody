import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.core.config")


class Config(dict):
    """Hierarchical configuration class with dot-attribute access and inheritance.
    
    Supports:
      - Recursive inheritance via `_base_` list or string
      - Attribute access (`cfg.model.head.num_keypoints`)
      - CLI key-value overrides (`model.head.num_keypoints=133`)
      - Clean YAML serialization and deserialization
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'. Available: {list(self.keys())}")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super().__setitem__(key, value)

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file, resolving `_base_` inheritance."""
        filepath = Path(filename).resolve()
        if not filepath.is_file():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}

        base_cfg_dict: Dict[str, Any] = {}
        if "_base_" in cfg_dict:
            bases = cfg_dict.pop("_base_")
            if isinstance(bases, (str, Path)):
                bases = [bases]

            for base_file in bases:
                base_path = (filepath.parent / base_file).resolve()
                base_cfg = cls.from_file(base_path)
                base_cfg_dict = cls._deep_merge(base_cfg_dict, base_cfg.to_dict())

        # Merge base config with child config (child overrides base)
        merged_dict = cls._deep_merge(base_cfg_dict, cfg_dict)
        config = cls(merged_dict)
        config._filename = str(filepath)
        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Construct a Config from a python dictionary."""
        return cls(data)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override dictionary into base dictionary."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def merge_from_cli_args(self, overrides: List[str]) -> None:
        """Apply dot-notated CLI overrides like ['model.backbone.out_channels=512', 'training.epochs=100']."""
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Invalid CLI override '{item}'. Expected format 'key.subkey=value'")
            key_path, val_str = item.split("=", 1)
            keys = key_path.strip().split(".")
            parsed_val = self._parse_val(val_str.strip())

            curr = self
            for k in keys[:-1]:
                if k not in curr or not isinstance(curr[k], dict):
                    curr[k] = Config()
                curr = curr[k]
            curr[keys[-1]] = parsed_val
            logger.info(f"Applied CLI override: {key_path} = {parsed_val}")

    @staticmethod
    def _parse_val(val_str: str) -> Any:
        """Parse string to int, float, bool, None, or literal string."""
        if val_str.lower() == "true":
            return True
        if val_str.lower() == "false":
            return False
        if val_str.lower() in ("none", "null"):
            return None
        try:
            return int(val_str)
        except ValueError:
            pass
        try:
            return float(val_str)
        except ValueError:
            pass
        # Handle list representations e.g. [256,192]
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                import ast
                return ast.literal_eval(val_str)
            except Exception:
                pass
        return val_str

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config back into a standard Python dictionary."""
        result: Dict[str, Any] = {}
        for k, v in self.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [item.to_dict() if isinstance(item, Config) else item for item in v]
            else:
                result[k] = v
        return result

    def dump(self, filename: Union[str, Path]) -> None:
        """Save configuration as a YAML file."""
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self.to_dict()))

    def __repr__(self) -> str:
        return f"Config({yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)})"
