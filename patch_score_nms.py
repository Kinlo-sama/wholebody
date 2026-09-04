with open('wholebody/evaluation/coco_metric.py', 'r') as f:
    content = f.read()

# Revert mean_kpt_score to use ONLY body points (first 17) for score ranking
old_score = """            valid_scores = all_scores[all_scores > 0.2]
            if len(valid_scores) > 0:
                mean_kpt_score = float(np.mean(valid_scores))
            else:
                mean_kpt_score = float(np.mean(all_scores))"""
new_score = """            body_scores = all_scores[:17]
            valid_scores = body_scores[body_scores > 0.2]
            if len(valid_scores) > 0:
                mean_kpt_score = float(np.mean(valid_scores))
            else:
                mean_kpt_score = float(np.mean(body_scores))"""
content = content.replace(old_score, new_score)

# For oks_nms, let's use a lower threshold to force filtering
# The default MMPose thr is 0.9. If we lower it to 0.7, it will aggressively filter duplicates.
content = content.replace("oks_nms(preds, thr=0.9, sigmas=sigmas)", "oks_nms(preds, thr=0.7, sigmas=sigmas)")

with open('wholebody/evaluation/coco_metric.py', 'w') as f:
    f.write(content)
