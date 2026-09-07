from wholebody.codecs.base import BaseCodec
from wholebody.codecs.heatmap import MSRAHeatmapCodec
from wholebody.codecs.regression import RegressionCodec

__all__ = [
    "BaseCodec",
    "MSRAHeatmapCodec",
    "RegressionCodec",
]
from .simcc_codec import SimCCCodec
__all__.append("SimCCCodec")
from .udp_heatmap import UDPHeatmapCodec
