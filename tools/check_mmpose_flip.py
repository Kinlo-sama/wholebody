import re
with open('mmpose/configs/_base_/datasets/coco_wholebody.py') as f:
    content = f.read()
    match = re.search(r'flip_indices\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if match:
        # Strip newlines and spaces to print neatly
        nums = [n.strip() for n in match.group(1).replace('\n', '').split(',') if n.strip()]
        print([int(n) for n in nums])
