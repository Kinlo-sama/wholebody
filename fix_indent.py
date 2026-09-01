with open('wholebody/evaluation/coco_metric.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("                                w = ann['bbox'][2]"):
        new_lines.append("                w = ann['bbox'][2] / 200.0 * 1.25\n")
    else:
        new_lines.append(line)

with open('wholebody/evaluation/coco_metric.py', 'w') as f:
    f.writelines(new_lines)
