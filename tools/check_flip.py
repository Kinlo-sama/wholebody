from wholebody.structures.keypoint_spec import KEYPOINT_SPECS
spec = KEYPOINT_SPECS.get("coco_wholebody_133")
print("My flip indices:", spec.flip_indices)
