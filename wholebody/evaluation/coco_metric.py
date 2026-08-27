import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from wholebody.core.registry import METRICS
from wholebody.evaluation.metrics import BaseMetric
from wholebody.structures.data_sample import PoseDataSample
from wholebody.utils.logger import get_logger

@METRICS.register("CocoWholeBodyMetric")
class CocoWholeBodyMetric(BaseMetric):
    """Official COCO evaluation metric using pycocotools."""

    def __init__(self, ann_file: str) -> None:
        super().__init__(name="CocoWholeBody")
        self.ann_file = ann_file
        self.logger = get_logger("wholebody.evaluation.coco")
        
        # Load official COCO Ground Truth once
        self.logger.info(f"Loading COCO Ground Truth from {ann_file}...")
        self.coco_gt = COCO(ann_file)
        self.reset()

    def reset(self) -> None:
        self.results: List[Dict] = []

    def process(self, data_samples: List[PoseDataSample]) -> None:
        """Accumulate predictions for the current batch."""
        for sample in data_samples:
            if not sample.pred_instances:
                continue

            # COCO requires: image_id, category_id (1 for person), keypoints, score
            image_id = sample.metainfo.get("img_id", 0) # Your Dataset must pass img_id!
            pred_kpts = sample.pred_instances.keypoints.detach().cpu().numpy() # (K, 2)
            
            if pred_kpts.ndim == 3:
                pred_kpts = pred_kpts[0]
            
            # Format to COCO: [x, y, visibility, x, y, visibility...]
            # We predict x,y. We set visibility to 1 (predicted).
            num_kpts = pred_kpts.shape[0]
            coco_kpts = np.zeros(num_kpts * 3, dtype=float)
            coco_kpts[0::3] = pred_kpts[:, 0]
            coco_kpts[1::3] = pred_kpts[:, 1]
            coco_kpts[2::3] = 1.0  # Predicted points get visibility 1
            
            # Score: we can take the mean of keypoint confidence scores
            scores = sample.pred_instances.keypoint_scores.detach().cpu().numpy()
            avg_score = float(np.mean(scores))

            self.results.append({
                "image_id": int(image_id),
                "category_id": 1,  # 1 is person in COCO
                "keypoints": coco_kpts.tolist(),
                "score": avg_score
            })

    def compute_metrics(self) -> Dict[str, float]:
        """Save results to JSON and run official COCOeval."""
        if not self.results:
            return {"AP": 0.0}

        self.logger.info(f"Evaluating {len(self.results)} predictions...")

        # Create a temporary JSON file for pycocotools to read
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.results, f)
            temp_path = f.name

        try:
            # Load predictions into COCO format
            coco_dt = self.coco_gt.loadRes(temp_path)
            
            # Run evaluation
            coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='keypoints')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            stats = coco_eval.stats
            # stats[0] is AP at OKS=0.50:0.95 (The main metric)
            # stats[1] is AP at OKS=0.50 (AP50)
            # stats[2] is AP at OKS=0.75 (AP75)
            
            return {
                "Coco/AP": float(stats[0]),
                "Coco/AP50": float(stats[1]),
                "Coco/AP75": float(stats[2]),
                "Coco/AR": float(stats[5]),
            }
        finally:
            Path(temp_path).unlink()  # Clean up temp file