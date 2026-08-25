import unittest
from wholebody.structures.keypoint_spec import (
    KeypointSpec,
    create_coco_17_spec,
    create_coco_wholebody_133_spec,
)


class TestKeypointSpec(unittest.TestCase):

    def test_coco_17_spec(self):
        spec = create_coco_17_spec()
        self.assertEqual(spec.num_keypoints, 17)
        self.assertIn("body", spec.groups)
        self.assertEqual(len(spec.flip_indices), 17)
        # Flip symmetry check: left_eye <-> right_eye
        self.assertEqual(spec.flip_indices[1], 2)
        self.assertEqual(spec.flip_indices[2], 1)

    def test_coco_wholebody_133_spec(self):
        spec = create_coco_wholebody_133_spec()
        self.assertEqual(spec.num_keypoints, 133)
        self.assertIn("body", spec.groups)
        self.assertIn("feet", spec.groups)
        self.assertIn("face", spec.groups)
        self.assertIn("left_hand", spec.groups)
        self.assertIn("right_hand", spec.groups)
        self.assertEqual(len(spec.groups["face"]), 68)
        self.assertEqual(len(spec.groups["left_hand"]), 21)
        self.assertEqual(len(spec.groups["right_hand"]), 21)


if __name__ == "__main__":
    unittest.main()
