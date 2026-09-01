with open('wholebody/evaluation/coco_metric.py', 'r') as f:
    content = f.read()

# I need to insert "                ann['area'] = ann['bbox'][2] * ann['bbox'][3]\n"
if "ann['area'] = ann['bbox'][2] * ann['bbox'][3]" not in content:
    content = content.replace("ann['num_keypoints'] = sum(1 for v in merged[2::3] if v > 0)\n", 
                              "ann['num_keypoints'] = sum(1 for v in merged[2::3] if v > 0)\n                ann['area'] = ann['bbox'][2] * ann['bbox'][3]\n")
    with open('wholebody/evaluation/coco_metric.py', 'w') as f:
        f.write(content)
