import numpy as np
from typing import List, Dict

def compute_oks(kpt1, kpt2, sigmas, area):
    """Compute OKS between two keypoint arrays.
    kpt1, kpt2: (K, 3) where [:, 2] is score/visibility.
    """
    # Only use keypoints that both predictions think are visible
    # Or just use keypoints from kpt1 that have score > 0.05
    k1_x = kpt1[:, 0]
    k1_y = kpt1[:, 1]
    k1_s = kpt1[:, 2]
    
    k2_x = kpt2[:, 0]
    k2_y = kpt2[:, 1]
    k2_s = kpt2[:, 2]
    
    # Distance squared
    dx = k1_x - k2_x
    dy = k1_y - k2_y
    d2 = dx**2 + dy**2
    
    # Sigmas and area
    vars = (sigmas * 2)**2
    
    # Filter points where BOTH have reasonable confidence
    vis = (k1_s > 0.0) & (k2_s > 0.0)
    if np.sum(vis) == 0:
        return 0.0
        
    # OKS formula
    e = d2 / (2 * area * vars + 1e-9)
    oks = np.sum(np.exp(-e[vis])) / np.sum(vis)
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
        if "bbox" in preds[i]:
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
            
            oks = compute_oks(kpt1, kpt2, sigmas, area1)
            
            if oks > thr: # Default MMPose OKS NMS threshold is 0.9
                keep_pred = False
                break
                
        if keep_pred:
            keep.append(i)

    return [preds[i] for i in keep]
