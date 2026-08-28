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
        
        # HACK: COCOeval expects all keypoints in the 'keypoints' list.
        # COCO-WholeBody splits them. We must merge them in memory so COCOeval doesn't crash!
        for ann_id, ann in self.coco_gt.anns.items():
            if 'keypoints' in ann:
                # 17 body
                merged = list(ann['keypoints'])
                
                # 6 feet
                if 'foot_kpts' in ann and ann['foot_kpts']:
                    merged.extend(ann['foot_kpts'])
                else:
                    merged.extend([0] * (6 * 3))
                    
                # 68 face
                if 'face_kpts' in ann and ann['face_kpts']:
                    merged.extend(ann['face_kpts'])
                else:
                    merged.extend([0] * (68 * 3))
                    
                # 21 left hand
                if 'lefthand_kpts' in ann and ann['lefthand_kpts']:
                    merged.extend(ann['lefthand_kpts'])
                else:
                    merged.extend([0] * (21 * 3))
                    
                # 21 right hand
                if 'righthand_kpts' in ann and ann['righthand_kpts']:
                    merged.extend(ann['righthand_kpts'])
                else:
                    merged.extend([0] * (21 * 3))
                    
                ann['keypoints'] = merged
                ann['num_keypoints'] = sum(1 for i in range(2, len(merged), 3) if merged[i] > 0)
                
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
            
            # MMPose 'bbox_keypoint' scoring: Only average keypoints with score > 0.2!
            scores = sample.pred_instances.keypoint_scores.detach().cpu().numpy()
            
            valid_scores = scores[scores > 0.2]
            if len(valid_scores) > 0:
                mean_kpt_score = float(np.mean(valid_scores))
            else:
                mean_kpt_score = 0.0
                
            # CRITICAL: If using a detector, we MUST multiply by the bounding box confidence!
            bbox_score = sample.metainfo.get("bbox_score", 1.0)
            final_score = mean_kpt_score * bbox_score

            self.results.append({
                "image_id": int(image_id),
                "category_id": 1,  # 1 is person in COCO
                "keypoints": coco_kpts.tolist(),
                "score": final_score
            })

    def compute_metrics(self) -> Dict[str, float]:
        """Save results to JSON and run official COCOeval."""
        if not self.results:
            return {"AP": 0.0}

        self.logger.info(f"Evaluating {len(self.results)} predictions...")

        # STEP 4: NMS Filtering
        # MMPose performs OKS NMS to remove overlapping detections.
        # We group by image_id and apply a bounding box NMS based on keypoint extents.
        from collections import defaultdict
        from wholebody.evaluation.nms import oks_nms
        
        img_to_preds = defaultdict(list)
        for res in self.results:
            img_to_preds[res["image_id"]].append(res)
            
        filtered_results = []
        for img_id, preds in img_to_preds.items():
            filtered_preds = oks_nms(preds)
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
            
            # INJECT 133 SIGMAS: COCOeval defaults to 17, which causes the broadcast error.
            spec = KEYPOINT_SPECS.get("coco_wholebody_133")
            coco_eval.params.kpt_oks_sigmas = spec.sigmas
            
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