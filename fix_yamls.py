import os
import glob

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix dataset name
    content = content.replace('type: CocoWholeBodyDataset', 'type: COCOWholeBodyDataset')
    
    # Fix final layer kernel size
    content = content.replace('final_layer_kernel_size: 1', 'final_layer_kernel_size: 7')
    
    # Fix pipeline steps
    content = content.replace('- type: LoadImage\n', '- type: LoadImageFromFile\n')
    
    # Remove GetBBoxCenterScale and padding
    lines = content.split('\n')
    new_lines = []
    skip = False
    for line in lines:
        if '- type: GetBBoxCenterScale' in line:
            skip = True
            continue
        if skip and 'padding: 1.25' in line:
            skip = False
            continue
        # Also need to add keypoint_spec: coco_wholebody_133 under img_prefix if not exists
        if 'img_prefix:' in line and 'keypoint_spec' not in content:
            new_lines.append(line)
            new_lines.append('      keypoint_spec: coco_wholebody_133')
            continue
            
        new_lines.append(line)
        
    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines))

for f in glob.glob('configs/experiments/rtmw-*.yaml'):
    if '256x192' in f or '384x288' in f:
        fix_file(f)
        print(f"Fixed {f}")
