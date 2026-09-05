import numpy as np
from typing import List, Dict

def compute_oks(kpt1: np.ndarray, kpt2: np.ndarray, sigmas: np.ndarray, area1: float, area2: float = None) -> float:
    """Compute OKS between two keypoint arrays.
    kpt1, kpt2: (K, 3) where [:, 2] is score/visibility.
    """
    # d2 = (dx^2 + dy^2)
    d2 = (kpt1[:, 0] - kpt2[:, 0])**2 + (kpt1[:, 1] - kpt2[:, 1])**2
    vars = (sigmas * 2)**2
    
    # Use average area just like MMPose oks_iou: ((a_g + a_d) / 2 + eps)
    if area2 is not None:
        area = (area1 + area2) / 2.0
    else:
        area = area1
        
    # OKS formula (no visibility filtering by default in MMPose oks_nms)
    e = d2 / (2 * area * vars + 1e-9)
    oks = np.sum(np.exp(-e)) / len(e)
    return float(oks)

def oks_nms(preds: List[Dict], thr: float = 0.9, sigmas: np.ndarray = None) -> List[Dict]:
    """Perform OKS-based Non-Maximum Suppression."""
    if len(preds) == 0:
        return []

    # Sort by score descending
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    keep = []

    if sigmas is None:
        # Fallback to a generic sigma if none provided
        sigmas = np.ones(len(preds[0]["keypoints"])//3) * 0.05

    for i in range(len(preds)):
        keep_pred = True
        kpt1 = np.array(preds[i]["keypoints"]).reshape(-1, 3)
        
        # Estimate area from kpt1 (rough approximation for OKS)
        # Better: if 'bbox' is present, use bbox area * 0.53 (MMPose standard)
        if "area" in preds[i]:
            area1 = preds[i]["area"]
        elif "bbox" in preds[i]:
            # COCO JSON format is [x, y, w, h] or sometimes we don't have it in preds.
            # Wait, in our coco_metric.py, do we save bbox? Let me check.
            # We didn't save bbox in `results.append(...)`! We only saved image_id, category_id, keypoints, score.
            # So "bbox" is not in preds!
            # If not in preds, we must use keypoint extents.
            min_x1, min_y1 = np.min(kpt1[:, 0]), np.min(kpt1[:, 1])
            max_x1, max_y1 = np.max(kpt1[:, 0]), np.max(kpt1[:, 1])
            area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
            area1 = max(area1, 1.0)
        else:
            min_x1, min_y1 = np.min(kpt1[:, 0]), np.min(kpt1[:, 1])
            max_x1, max_y1 = np.max(kpt1[:, 0]), np.max(kpt1[:, 1])
            area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
            area1 = max(area1, 1.0)
            min_x1, min_y1 = np.min(kpt1[:, 0]), np.min(kpt1[:, 1])
            max_x1, max_y1 = np.max(kpt1[:, 0]), np.max(kpt1[:, 1])
            area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
            area1 = max(area1, 1.0)
        
        for j in keep:
            kpt2 = np.array(preds[j]["keypoints"]).reshape(-1, 3)
            
            if "area" in preds[j]:
                area2 = preds[j]["area"]
            else:
                min_x2, min_y2 = np.min(kpt2[:, 0]), np.min(kpt2[:, 1])
                max_x2, max_y2 = np.max(kpt2[:, 0]), np.max(kpt2[:, 1])
                area2 = max((max_x2 - min_x2) * (max_y2 - min_y2), 1.0)
                
            oks = compute_oks(kpt1, kpt2, sigmas, area1, area2)
            
            if oks > thr: # Default MMPose OKS NMS threshold is 0.9
                keep_pred = False
                break
                
        if keep_pred:
            keep.append(i)

    return [preds[i] for i in keep]
