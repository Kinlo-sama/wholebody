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
        
        # HACK: ONLY BODY
        for ann_id, ann in self.coco_gt.anns.items():
            if 'keypoints' in ann:
                # Merge ALL 133 points (Body=17, Foot=6, Face=68, L-Hand=21, R-Hand=21)
                body = ann.get('keypoints', [0]*51)
                foot = ann.get('foot_kpts', [0]*18)
                face = ann.get('face_kpts', [0]*204)
                left_hand = ann.get('lefthand_kpts', [0]*63)
                right_hand = ann.get('righthand_kpts', [0]*63)
                merged = body + foot + face + left_hand + right_hand
                ann['keypoints'] = merged
                ann['num_keypoints'] = sum(1 for v in merged[2::3] if v > 0)
                w = ann['bbox'][2] / 200.0 * 1.25
                h = ann['bbox'][3] / 200.0 * 1.25
                aspect_ratio = 288.0 / 384.0
                if w > h * aspect_ratio:
                    h = w / aspect_ratio
                else:
                    w = h * aspect_ratio
                ann['area'] = (w * 200.0) * (h * 200.0)
        
        self.results: List[Dict] = []

    def reset(self) -> None:
        self.results = []

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
                
            # Evaluate all 133 points
            
            # Format to COCO: [x, y, visibility, x, y, visibility...]
            # We predict x,y. We set visibility to 1 (predicted).
            num_kpts = pred_kpts.shape[0]
            coco_kpts = np.zeros(num_kpts * 3, dtype=float)
            coco_kpts[0::3] = pred_kpts[:, 0]
            coco_kpts[1::3] = pred_kpts[:, 1]
            all_scores = sample.pred_instances.keypoint_scores.detach().cpu().numpy()
            coco_kpts[2::3] = all_scores
            
            # Compute a single score. Here we use the mean of keypoint scores.
            
            # Weighted score: Only use scores > threshold to avoid penalizing good predictions
            # with low-confidence invisible points.
            body_scores = all_scores[:17]
            valid_scores = body_scores[body_scores > 0.2]
            if len(valid_scores) > 0:
                mean_kpt_score = float(np.mean(valid_scores))
            else:
                mean_kpt_score = float(np.mean(body_scores))
                
            bbox_score = sample.metainfo.get("bbox_score", 1.0)
            final_score = mean_kpt_score * bbox_score
            
            self.results.append({
                "image_id": image_id,
                "category_id": 1,
                "keypoints": coco_kpts.tolist(),
                "score": final_score,
                "area": float((sample.metainfo.get("scale")[0] * 200.0) * (sample.metainfo.get("scale")[1] * 200.0))
            })

    def compute_metrics(self) -> Dict[str, float]:
        """Compute AP and AR using COCOeval."""
        if not self.results:
            self.logger.warning("No predictions found for evaluation.")
            return {}

        self.logger.info(f"Evaluating {len(self.results)} predictions...")
        
        # Apply NMS BEFORE evaluation to remove redundant boxes on the same person!
        from collections import defaultdict
        from wholebody.evaluation.nms import oks_nms
        
        img_to_preds = defaultdict(list)
        for res in self.results:
            img_to_preds[res["image_id"]].append(res)
            
        filtered_results = []
        from wholebody.structures.keypoint_spec import KEYPOINT_SPECS
        spec = KEYPOINT_SPECS.get("coco_wholebody_133")
        sigmas = np.array(spec.sigmas, dtype=np.float32)

        for img_id, preds in img_to_preds.items():
            filtered_preds = oks_nms(preds, thr=0.7, sigmas=sigmas)
            filtered_results.extend(filtered_preds)
            
        self.logger.info(f"After NMS: {len(filtered_results)} predictions remain.")

        # Create a temporary JSON file for pycocotools to read
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(filtered_results, f)
            temp_path = f.name

        try:
            from wholebody.core.registry import KEYPOINT_SPECS
            
            # Load predictions into COCO format
            coco_dt = self.coco_gt.loadRes(temp_path)
            
            # Run evaluation
            coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='keypoints')
            
            # INJECT 17 SIGMAS
            spec = KEYPOINT_SPECS.get("coco_wholebody_133")
            coco_eval.params.kpt_oks_sigmas = np.array(spec.sigmas, dtype=np.float32)
            
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
