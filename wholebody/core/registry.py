import difflib
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.core.registry")


class Registry:
    """Modular Registry for dynamic component instantiation and decoupling.
    
    Supports:
      - Decorator-based registration: @MODELS.register()
      - Explicit registration: MODELS.register_module(cls, name="Custom")
      - Nested construction: builds submodules if child dicts contain 'type'
      - Typo diagnosis with difflib suggestions
      - Hierarchical registries (child -> parent fallback)
    """

    def __init__(self, name: str, parent: Optional["Registry"] = None) -> None:
        self._name = name
        self._module_dict: Dict[str, Type[Any]] = {}
        self._parent = parent

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type[Any]]:
        return self._module_dict

    def register(
        self,
        name: Optional[str] = None,
        force: bool = False,
        module: Optional[Type[Any]] = None,
    ) -> Union[Type[Any], Callable[[Type[Any]], Type[Any]]]:
        """Register a module. Can be used as a decorator or direct function call."""
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        def _register(cls: Type[Any]) -> Type[Any]:
            self._register_module(module=cls, module_name=name, force=force)
            return cls

        return _register

    def register_module(
        self,
        module: Type[Any],
        name: Optional[str] = None,
        force: bool = False,
    ) -> Type[Any]:
        """Explicitly register a class under a given name."""
        self._register_module(module=module, module_name=name, force=force)
        return module

    def _register_module(
        self,
        module: Any,
        module_name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        name = module_name or getattr(module, "__name__", str(module))
        if name in self._module_dict and not force:
            existing = self._module_dict[name]
            if (
                existing is module
                or getattr(existing, "__name__", None) == getattr(module, "__name__", None)
                or (hasattr(existing, "name") and getattr(existing, "name") == getattr(module, "name", None))
            ):
                self._module_dict[name] = module
                return
            raise KeyError(
                f"'{name}' is already registered in registry '{self._name}' at {self._module_dict[name]}."
            )
        self._module_dict[name] = module

    def get(self, key: str) -> Type[Any]:
        """Retrieve a registered class by name."""
        if key in self._module_dict:
            return self._module_dict[key]

        if self._parent is not None:
            return self._parent.get(key)

        # Generate helpful typo suggestion
        all_keys = list(self._module_dict.keys())
        matches = difflib.get_close_matches(key, all_keys, n=3, cutoff=0.5)
        msg = f"Cannot find '{key}' in registry '{self._name}'."
        if matches:
            msg += f" Did you mean: {', '.join(matches)}?"
        else:
            msg += f" Available components: {sorted(all_keys)}"
        raise KeyError(msg)

    def contains(self, key: str) -> bool:
        """Check whether a key exists in this registry or parent."""
        if key in self._module_dict:
            return True
        if self._parent is not None:
            return self._parent.contains(key)
        return False

    def build(self, cfg: Union[Dict[str, Any], Any], default_args: Optional[Dict[str, Any]] = None) -> Any:
        """Build an instance from a configuration dict or Config object."""
        if cfg is None:
            return None

        # If already an instantiated object, return as-is
        if not isinstance(cfg, dict) and not hasattr(cfg, "to_dict"):
            return cfg

        args = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

        if default_args is not None:
            for k, v in default_args.items():
                args.setdefault(k, v)

        if "type" not in args:
            raise KeyError(f"Configuration for building from registry '{self._name}' must contain key 'type'. Given: {args}")

        type_str = args.pop("type")
        if isinstance(type_str, str):
            obj_cls = self.get(type_str)
        elif inspect.isclass(type_str) or inspect.isfunction(type_str):
            obj_cls = type_str
        else:
            raise TypeError(f"'type' must be a string or class, got {type(type_str)}")

        try:
            return obj_cls(**args)
        except TypeError as e:
            raise TypeError(f"Failed to instantiate {obj_cls.__name__} from registry '{self._name}' with args {args}: {e}") from e

    def __repr__(self) -> str:
        return f"Registry(name='{self._name}', items={list(self._module_dict.keys())})"


# Global Registries for wholebody framework
MODELS = Registry("models")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")
LOSSES = Registry("losses")
CODECS = Registry("codecs")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
METRICS = Registry("metrics")
HOOKS = Registry("hooks")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
KEYPOINT_SPECS = Registry("keypoint_specs")
