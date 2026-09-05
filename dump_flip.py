import sys
from wholebody.structures.keypoint_spec import create_coco_wholebody_133_spec
spec = create_coco_wholebody_133_spec()
print(spec.flip_indices)
