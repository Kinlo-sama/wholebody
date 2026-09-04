with open('wholebody/evaluation/coco_metric.py', 'r') as f:
    content = f.read()

# Pass area to preds
content = content.replace('"score": final_score', '"score": final_score,\n                "area": float((sample.metainfo.get("scale")[0] * 200.0) * (sample.metainfo.get("scale")[1] * 200.0))')

# Re-enable NMS
content = content.replace("filtered_preds = preds", "filtered_preds = oks_nms(preds, thr=0.9, sigmas=sigmas)")

with open('wholebody/evaluation/coco_metric.py', 'w') as f:
    f.write(content)

with open('wholebody/evaluation/nms.py', 'r') as f:
    nms_content = f.read()

nms_content = nms_content.replace('if "bbox" in preds[i]:', 'if "area" in preds[i]:\n            area1 = preds[i]["area"]\n        elif "bbox" in preds[i]:')

with open('wholebody/evaluation/nms.py', 'w') as f:
    f.write(nms_content)

