with open('wholebody/evaluation/coco_metric.py', 'r') as f:
    content = f.read()

content = content.replace("coco_kpts[2::3] = 1.0  # Predicted points get visibility 1", 
                          "scores = sample.pred_instances.keypoint_scores.detach().cpu().numpy()[:17]\n            coco_kpts[2::3] = scores")

# Also need to remove the duplicate `scores = ...` line below it
content = content.replace("            # Compute a single score. Here we use the mean of keypoint scores.\n            scores = sample.pred_instances.keypoint_scores.detach().cpu().numpy()[:17]",
                          "            # Compute a single score. Here we use the mean of keypoint scores.")

with open('wholebody/evaluation/coco_metric.py', 'w') as f:
    f.write(content)
