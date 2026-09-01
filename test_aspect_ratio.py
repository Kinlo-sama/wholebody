with open('wholebody/datasets/coco_wholebody.py', 'r') as f:
    content = f.read()

# I need to insert aspect ratio fix after `scale = scale * 1.25`
fix_code = """
                # Fix aspect ratio
                aspect_ratio = 288.0 / 384.0  # W / H
                w, h = scale[0], scale[1]
                if w > h * aspect_ratio:
                    scale[1] = w / aspect_ratio
                else:
                    scale[0] = h * aspect_ratio
"""

if "# Fix aspect ratio" not in content:
    content = content.replace("scale = scale * 1.25", "scale = scale * 1.25\n" + fix_code)
    with open('wholebody/datasets/coco_wholebody.py', 'w') as f:
        f.write(content)
