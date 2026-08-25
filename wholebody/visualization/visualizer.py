from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from wholebody.core.registry import KEYPOINT_SPECS
from wholebody.structures.keypoint_spec import KeypointSpec
from wholebody.structures.data_sample import PoseDataSample


class SkeletonVisualizer:
    """Universal whole-body skeleton visualizer driven dynamically by KeypointSpec."""

    def __init__(
        self,
        keypoint_spec: Union[str, KeypointSpec, Dict[str, Any]] = "coco_wholebody_133",
        line_thickness: int = 2,
        circle_radius: int = 3,
        kpt_thr: float = 0.3,
    ) -> None:
        if isinstance(keypoint_spec, str):
            self.spec: KeypointSpec = KEYPOINT_SPECS.get(keypoint_spec)
        elif isinstance(keypoint_spec, dict):
            self.spec = KEYPOINT_SPECS.build(keypoint_spec)
        else:
            self.spec = keypoint_spec

        self.line_thickness = line_thickness
        self.circle_radius = circle_radius
        self.kpt_thr = kpt_thr

    def draw_skeleton(
        self,
        img: np.ndarray,
        keypoints: Union[np.ndarray, torch.Tensor],
        scores: Optional[Union[np.ndarray, torch.Tensor]] = None,
        bboxes: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """Render joints, limbs, and bounding boxes onto image canvas."""
        canvas = img.copy()

        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()

        if keypoints.ndim == 2:
            keypoints = keypoints[np.newaxis, ...]
        if scores is not None and scores.ndim == 1:
            scores = scores[np.newaxis, ...]

        num_instances = keypoints.shape[0]
        num_kpts = keypoints.shape[1]

        # Draw bounding boxes
        if bboxes is not None:
            if bboxes.ndim == 1:
                bboxes = bboxes[np.newaxis, ...]
            for b in range(len(bboxes)):
                box = bboxes[b].astype(int)
                cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Draw skeletons for each person instance
        for inst_idx in range(num_instances):
            inst_kpts = keypoints[inst_idx]
            inst_scores = scores[inst_idx] if scores is not None else np.ones((num_kpts,), dtype=np.float32)

            # Draw limbs (edges)
            for edge in self.spec.edges:
                j1, j2 = edge.joint1_id, edge.joint2_id
                if j1 < num_kpts and j2 < num_kpts:
                    if inst_scores[j1] >= self.kpt_thr and inst_scores[j2] >= self.kpt_thr:
                        pt1 = (int(inst_kpts[j1, 0]), int(inst_kpts[j1, 1]))
                        pt2 = (int(inst_kpts[j2, 0]), int(inst_kpts[j2, 1]))
                        # Color: RGB to BGR for OpenCV
                        color_bgr = (edge.color[2], edge.color[1], edge.color[0])
                        cv2.line(canvas, pt1, pt2, color_bgr, edge.thickness or self.line_thickness)

            # Draw keypoint circles
            for j_id in range(min(num_kpts, len(self.spec.joints))):
                if inst_scores[j_id] >= self.kpt_thr:
                    pt = (int(inst_kpts[j_id, 0]), int(inst_kpts[j_id, 1]))
                    joint_spec = self.spec.joints[j_id]
                    color_bgr = (joint_spec.color[2], joint_spec.color[1], joint_spec.color[0])
                    cv2.circle(canvas, pt, self.circle_radius, color_bgr, -1)

        return canvas

    def draw_sample(self, img: np.ndarray, sample: PoseDataSample) -> np.ndarray:
        """Helper to draw predictions from a PoseDataSample."""
        if not sample.pred_instances or sample.pred_instances.keypoints is None:
            return img
        return self.draw_skeleton(
            img=img,
            keypoints=sample.pred_instances.keypoints,
            scores=sample.pred_instances.keypoint_scores,
            bboxes=sample.pred_instances.bboxes,
        )
