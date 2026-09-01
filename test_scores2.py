with open('wholebody/evaluation/coco_metric.py', 'r') as f:
    content = f.read()

# Change `[:17]` to ALL scores for mean calculation
content = content.replace("scores = sample.pred_instances.keypoint_scores.detach().cpu().numpy()[:17]\n            coco_kpts[2::3] = scores",
                          "all_scores = sample.pred_instances.keypoint_scores.detach().cpu().numpy()\n            coco_kpts[2::3] = all_scores[:17]")

# Change valid_scores to use all_scores
content = content.replace("valid_scores = scores[scores > 0.2]", "valid_scores = all_scores[all_scores > 0.2]")
content = content.replace("mean_kpt_score = float(np.mean(scores))", "mean_kpt_score = float(np.mean(all_scores))")

with open('wholebody/evaluation/coco_metric.py', 'w') as f:
    f.write(content)
