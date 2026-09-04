import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from wholebody.core.registry import METRICS
from wholebody.evaluation.metrics import BaseMetric
from wholebody.structures.data_sample import PoseDataSample
from wholebody.utils.logger import get_logger
from wholebody.evaluation.nms import oks_nms
from wholebody.structures.keypoint_spec import KEYPOINT_SPECS


@METRICS.register("CocoWholeBodyMetric")
class CocoWholeBodyMetric(BaseMetric):
    """Official COCO-WholeBody evaluation metric using xtcocotools."""

    def __init__(self, ann_file: str) -> None:
        super().__init__(name="CocoWholeBody")
        self.ann_file = ann_file
        self.logger = get_logger("wholebody.evaluation.coco")
        
        # Load official COCO Ground Truth once using xtcocotools
        self.logger.info(f"Loading COCO-WholeBody Ground Truth from {ann_file}...")
        self.coco_gt = COCO(ann_file)
        self.results: List[Dict] = []

    def reset(self) -> None:
        self.results = []

    def process(self, data_samples: List[PoseDataSample]) -> None:
        """Accumulate predictions for the current batch."""
        for sample in data_samples:
            if not sample.pred_instances:
                continue

            image_id = sample.metainfo.get("img_id", 0)
            pred_kpts = sample.pred_instances.keypoints.detach().cpu().numpy()
            
            if pred_kpts.ndim == 3:
                pred_kpts = pred_kpts[0]
                
            num_kpts = pred_kpts.shape[0]
            coco_kpts = np.zeros(num_kpts * 3, dtype=float)
            coco_kpts[0::3] = pred_kpts[:, 0]
            coco_kpts[1::3] = pred_kpts[:, 1]
            
            all_scores = sample.pred_instances.keypoint_scores.detach().cpu().numpy()
            coco_kpts[2::3] = all_scores
            
            # Ranking score: use body points (17) for score ranking like MMPose
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
        """Compute AP and AR using xtcocotools COCOeval."""
        if not self.results:
            self.logger.warning("No predictions found for evaluation.")
            return {}

        self.logger.info(f"Evaluating {len(self.results)} predictions...")
        
        # Apply NMS to remove redundant boxes on the same person
        from collections import defaultdict
        img_to_preds = defaultdict(list)
        for res in self.results:
            img_to_preds[res["image_id"]].append(res)
            
        filtered_results = []
        spec = KEYPOINT_SPECS.get("coco_wholebody_133")
        sigmas = np.array(spec.sigmas, dtype=np.float32)

        for img_id, preds in img_to_preds.items():
            # Standard thr=0.9 for MMPose
            filtered_preds = oks_nms(preds, thr=0.9, sigmas=sigmas)
            filtered_results.extend(filtered_preds)
            
        self.logger.info(f"After NMS: {len(filtered_results)} predictions remain.")

        # Format specifically for xtcocotools which expects 5 separate arrays
        xtcoco_results = []
        for res in filtered_results:
            kpts = res["keypoints"]
            formatted_res = res.copy()
            formatted_res["keypoints"] = kpts[:51]
            formatted_res["foot_kpts"] = kpts[51:69]
            formatted_res["face_kpts"] = kpts[69:273]
            formatted_res["lefthand_kpts"] = kpts[273:336]
            formatted_res["righthand_kpts"] = kpts[336:399]
            xtcoco_results.append(formatted_res)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(xtcoco_results, f)
            temp_path = f.name

        try:
            # Load predictions into xtcocotools
            coco_dt = self.coco_gt.loadRes(temp_path)
            
            # Run evaluation natively for keypoints_wholebody
            # xtcocotools COCOeval takes sigmas and use_area in the constructor!
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints_wholebody', sigmas, use_area=True)
            coco_eval.params.useSegm = None
            
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            stats = coco_eval.stats
            
            return {
                "WholeBody/AP": float(stats[0]),
                "WholeBody/AP50": float(stats[1]),
                "WholeBody/AP75": float(stats[2]),
                "WholeBody/AR": float(stats[5]),
            }
        finally:
            Path(temp_path).unlink()

