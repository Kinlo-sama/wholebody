from typing import Any, Dict, Union

from wholebody.codecs.base import BaseCodec
from wholebody.core.registry import CODECS, TRANSFORMS
from wholebody.datasets.transforms.base import BaseTransform


@TRANSFORMS.register("GenerateTarget")
class GenerateTarget(BaseTransform):
    """Generate model targets (e.g. heatmaps, SimCC, normalized vectors) using a registered Codec."""

    def __init__(self, codec: Union[Dict[str, Any], BaseCodec]) -> None:
        if isinstance(codec, dict):
            self.codec: BaseCodec = CODECS.build(codec)
        else:
            self.codec = codec

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if "keypoints" not in results:
            return results

        keypoints = results["keypoints"]
        keypoints_visible = results.get("keypoints_visible", None)

        encoded_targets = self.codec.encode(
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
        )
        results.update(encoded_targets)
        return results
