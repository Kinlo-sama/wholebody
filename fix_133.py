with open('wholebody/evaluation/coco_metric.py', 'r') as f:
    content = f.read()

# Replace the 17-point hack with the 133-point merge
old_hack = """                # 17 body ONLY
                merged = list(ann['keypoints'])[:17*3]"""
new_hack = """                # Merge ALL 133 points (Body=17, Foot=6, Face=68, L-Hand=21, R-Hand=21)
                body = ann.get('keypoints', [0]*51)
                foot = ann.get('foot_kpts', [0]*18)
                face = ann.get('face_kpts', [0]*204)
                left_hand = ann.get('lefthand_kpts', [0]*63)
                right_hand = ann.get('righthand_kpts', [0]*63)
                merged = body + foot + face + left_hand + right_hand"""
content = content.replace(old_hack, new_hack)

# Remove the pred slicing
content = content.replace("pred_kpts = pred_kpts[:17]", "# Evaluate all 133 points")

# Use all scores instead of slicing
content = content.replace("coco_kpts[2::3] = all_scores[:17]", "coco_kpts[2::3] = all_scores")

# Fix NMS sigmas to use all 133
content = content.replace("sigmas = np.array(spec.sigmas[:17], dtype=np.float32)", "sigmas = np.array(spec.sigmas, dtype=np.float32)")

# Fix COCOeval sigmas to use all 133
content = content.replace("coco_eval.params.kpt_oks_sigmas = spec.sigmas[:17]", "coco_eval.params.kpt_oks_sigmas = np.array(spec.sigmas, dtype=np.float32)")

with open('wholebody/evaluation/coco_metric.py', 'w') as f:
    f.write(content)

