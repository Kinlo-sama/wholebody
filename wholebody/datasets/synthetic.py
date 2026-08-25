import math
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from wholebody.core.registry import DATASETS
from wholebody.datasets.base import BasePoseDataset
from wholebody.structures.keypoint_spec import KeypointSpec


@DATASETS.register("SyntheticWholeBodyDataset")
class SyntheticWholeBodyDataset(BasePoseDataset):
    """Synthetic Whole-Body Dataset for fast testing, validation, and demo experiments.
    
    Generates synthetic RGB images containing procedural stick figures whose joint
    positions strictly match the specified KeypointSpec (COCO-17, WholeBody-133, etc.).
    """

    def __init__(
        self,
        num_samples: int = 50,
        img_size: Tuple[int, int] = (512, 512),
        pipeline: Optional[List[Dict[str, Any]]] = None,
        keypoint_spec: Union[KeypointSpec, str, Dict[str, Any]] = "coco_wholebody_133",
        test_mode: bool = False,
    ) -> None:
        self.num_samples = num_samples
        self.img_size = tuple(img_size)
        super().__init__(
            pipeline=pipeline or [],
            keypoint_spec=keypoint_spec,
            test_mode=test_mode,
        )

    def load_annotations(self) -> List[Dict[str, Any]]:
        data_list: List[Dict[str, Any]] = []
        h, w = self.img_size
        num_kpts = self.keypoint_spec.num_keypoints

        for idx in range(self.num_samples):
            # Render a procedural stick figure
            img = np.ones((h, w, 3), dtype=np.uint8) * 40

            # Center position and human scale
            cx = w * 0.5 + np.sin(idx) * 20.0
            cy = h * 0.5 + np.cos(idx) * 20.0
            scale_factor = 1.0 + 0.1 * np.sin(idx * 2)

            kpts = np.zeros((num_kpts, 2), dtype=np.float32)
            vis = np.ones((num_kpts,), dtype=np.float32) * 2.0  # 2: visible

            # Procedural body positions
            # Nose
            if 0 < num_kpts: kpts[0] = [cx, cy - 140 * scale_factor]
            if 1 < num_kpts: kpts[1] = [cx - 10 * scale_factor, cy - 145 * scale_factor]
            if 2 < num_kpts: kpts[2] = [cx + 10 * scale_factor, cy - 145 * scale_factor]
            if 3 < num_kpts: kpts[3] = [cx - 20 * scale_factor, cy - 140 * scale_factor]
            if 4 < num_kpts: kpts[4] = [cx + 20 * scale_factor, cy - 140 * scale_factor]
            # Shoulders
            if 5 < num_kpts: kpts[5] = [cx - 40 * scale_factor, cy - 100 * scale_factor]
            if 6 < num_kpts: kpts[6] = [cx + 40 * scale_factor, cy - 100 * scale_factor]
            # Elbows
            if 7 < num_kpts: kpts[7] = [cx - 60 * scale_factor, cy - 40 * scale_factor]
            if 8 < num_kpts: kpts[8] = [cx + 60 * scale_factor, cy - 40 * scale_factor]
            # Wrists
            if 9 < num_kpts: kpts[9] = [cx - 70 * scale_factor, cy + 20 * scale_factor]
            if 10 < num_kpts: kpts[10] = [cx + 70 * scale_factor, cy + 20 * scale_factor]
            # Hips
            if 11 < num_kpts: kpts[11] = [cx - 30 * scale_factor, cy + 20 * scale_factor]
            if 12 < num_kpts: kpts[12] = [cx + 30 * scale_factor, cy + 20 * scale_factor]
            # Knees
            if 13 < num_kpts: kpts[13] = [cx - 35 * scale_factor, cy + 90 * scale_factor]
            if 14 < num_kpts: kpts[14] = [cx + 35 * scale_factor, cy + 90 * scale_factor]
            # Ankles
            if 15 < num_kpts: kpts[15] = [cx - 40 * scale_factor, cy + 160 * scale_factor]
            if 16 < num_kpts: kpts[16] = [cx + 40 * scale_factor, cy + 160 * scale_factor]

            # Whole-body extensions if present
            if num_kpts >= 133:
                # Feet (17-22)
                for f_idx in range(3):
                    kpts[17 + f_idx] = [kpts[15, 0] + (f_idx - 1) * 8, kpts[15, 1] + 15]
                    kpts[20 + f_idx] = [kpts[16, 0] + (f_idx - 1) * 8, kpts[16, 1] + 15]

                # Face (23-90)
                face_cx, face_cy = kpts[0, 0], kpts[0, 1]
                for f_i in range(68):
                    angle = (f_i / 68.0) * 2 * math.pi
                    r = 25.0 * scale_factor
                    kpts[23 + f_i] = [face_cx + r * math.cos(angle), face_cy + r * math.sin(angle)]

                # Hands (91-111 left hand around left wrist kpts[9], 112-132 right hand around right wrist kpts[10])
                lw_x, lw_y = kpts[9, 0], kpts[9, 1]
                for h_i in range(21):
                    kpts[91 + h_i] = [lw_x + (h_i % 5 - 2) * 5, lw_y + (h_i // 5) * 6]

                rw_x, rw_y = kpts[10, 0], kpts[10, 1]
                for h_i in range(21):
                    kpts[112 + h_i] = [rw_x + (h_i % 5 - 2) * 5, rw_y + (h_i // 5) * 6]

            # Draw visual skeleton onto synthetic image
            for edge in self.keypoint_spec.edges:
                j1, j2 = edge.joint1_id, edge.joint2_id
                if j1 < num_kpts and j2 < num_kpts:
                    pt1 = (int(kpts[j1, 0]), int(kpts[j1, 1]))
                    pt2 = (int(kpts[j2, 0]), int(kpts[j2, 1]))
                    cv2.line(img, pt1, pt2, (200, 200, 200), 2)

            for j in range(num_kpts):
                pt = (int(kpts[j, 0]), int(kpts[j, 1]))
                cv2.circle(img, pt, 3, (0, 255, 0), -1)

            # Compute bounding box
            min_x = max(0.0, float(np.min(kpts[:, 0]) - 30))
            min_y = max(0.0, float(np.min(kpts[:, 1]) - 30))
            max_x = min(float(w), float(np.max(kpts[:, 0]) + 30))
            max_y = min(float(h), float(np.max(kpts[:, 1]) + 30))
            bbox = np.array([min_x, min_y, max_x, max_y], dtype=np.float32)

            box_w = max_x - min_x
            box_h = max_y - min_y
            center = np.array([min_x + box_w / 2.0, min_y + box_h / 2.0], dtype=np.float32)
            scale = np.array([box_w / 200.0, box_h / 200.0], dtype=np.float32)

            sample_dict = {
                "img_id": idx,
                "img": img,
                "img_shape": (h, w, 3),
                "raw_keypoints": kpts.copy(),
                "keypoints": kpts.copy(),
                "keypoints_visible": vis.copy(),
                "bboxes": bbox.reshape(1, 4),
                "center": center,
                "scale": scale,
                "rotation": 0.0,
            }
            data_list.append(sample_dict)

        return data_list
