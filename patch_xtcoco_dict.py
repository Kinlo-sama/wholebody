with open('wholebody/evaluation/coco_metric.py', 'r') as f:
    content = f.read()

# Replace the JSON dumping block to format the dictionary for xtcocotools
old_dump = """        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(filtered_results, f)
            temp_path = f.name"""

new_dump = """        # Format specifically for xtcocotools which expects 5 separate arrays
        xtcoco_results = []
        for res in filtered_results:
            kpts = res["keypoints"]
            formatted_res = res.copy()
            formatted_res["keypoints"] = kpts[:51]
            formatted_res["foot_kpts"] = kpts[51:69]
            formatted_res["face_kpts"] = kpts[69:273]
            formatted_res["lefthand_kpts"] = kpts[273:336]
            formatted_res["righthand_kpts"] = kpts[336:399]
            xtcoco_results.append(formatted_res)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(xtcoco_results, f)
            temp_path = f.name"""

content = content.replace(old_dump, new_dump)

with open('wholebody/evaluation/coco_metric.py', 'w') as f:
    f.write(content)

