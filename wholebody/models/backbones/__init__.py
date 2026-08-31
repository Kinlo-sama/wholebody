from wholebody.models.backbones.base import BaseBackbone
from wholebody.models.backbones.simple_cnn import SimpleCNN
from wholebody.models.backbones.resnet import ResNetBackbone

__all__ = [
    "BaseBackbone",
    "SimpleCNN",
    "ResNetBackbone"
]
from .cspnext import CSPNeXt
__all__.append("CSPNeXt")
