with open('wholebody/evaluation/coco_metric.py', 'r') as f:
    content = f.read()

fix_area = """                w = ann['bbox'][2] / 200.0 * 1.25
                h = ann['bbox'][3] / 200.0 * 1.25
                aspect_ratio = 288.0 / 384.0
                if w > h * aspect_ratio:
                    h = w / aspect_ratio
                else:
                    w = h * aspect_ratio
                ann['area'] = (w * 200.0) * (h * 200.0)"""

content = content.replace("ann['area'] = ann['bbox'][2] * ann['bbox'][3]", fix_area)

with open('wholebody/evaluation/coco_metric.py', 'w') as f:
    f.write(content)
