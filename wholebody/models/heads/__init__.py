from wholebody.models.heads.base import BaseHead
from wholebody.models.heads.heatmap_head import HeatmapHead

__all__ = [
    "BaseHead",
    "HeatmapHead",
]
from .rtmw_head import RTMWHead
__all__.append("RTMWHead")
