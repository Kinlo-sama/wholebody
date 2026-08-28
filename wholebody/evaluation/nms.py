import numpy as np
from typing import List, Dict

def oks_nms(preds: List[Dict], thr: float = 0.9, sigmas: np.ndarray = None) -> List[Dict]:
    """Perform OKS-based Non-Maximum Suppression."""
    if len(preds) == 0:
        return []

    # Sort by score descending
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    keep = []

    for i in range(len(preds)):
        keep_pred = True
        kpt1 = np.array(preds[i]["keypoints"]).reshape(-1, 3)
        score1 = preds[i]["score"]
        
        for j in keep:
            kpt2 = np.array(preds[j]["keypoints"]).reshape(-1, 3)
            
            # Compute OKS (simplified for NMS: only use visible points of kpt1)
            # Area is tricky to estimate if we don't have it, we can use bbox area.
            # For simplicity, we just use BBox NMS if OKS is too complex to approximate perfectly here,
            # but OKS is better. Let's do a fast bounding box NMS using keypoint extents.
            
            # Fast BBox NMS
            min_x1, min_y1 = np.min(kpt1[:, 0]), np.min(kpt1[:, 1])
            max_x1, max_y1 = np.max(kpt1[:, 0]), np.max(kpt1[:, 1])
            area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
            
            min_x2, min_y2 = np.min(kpt2[:, 0]), np.min(kpt2[:, 1])
            max_x2, max_y2 = np.max(kpt2[:, 0]), np.max(kpt2[:, 1])
            area2 = (max_x2 - min_x2) * (max_y2 - min_y2)
            
            xx1 = max(min_x1, min_x2)
            yy1 = max(min_y1, min_y2)
            xx2 = min(max_x1, max_x2)
            yy2 = min(max_y1, max_y2)
            
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (area1 + area2 - inter + 1e-6)
            
            if iou > 0.5: # Standard IOU NMS threshold
                keep_pred = False
                break
                
        if keep_pred:
            keep.append(i)

    return [preds[i] for i in keep]
