import re

with open('wholebody/models/heads/rtmcc_head.py', 'r') as f:
    content = f.read()

replacement = """        # The codec returns a dict
        encoded = self.codec.encode(keypoints.cpu().numpy(), weights.cpu().numpy())
        gt_x = torch.from_numpy(encoded['keypoint_x_labels']).to(device)
        gt_y = torch.from_numpy(encoded['keypoint_y_labels']).to(device)
        gt_weight = torch.from_numpy(encoded['keypoint_weights']).to(device)"""

pattern = r"        gt_x, gt_y, gt_weight = self.codec.encode\(keypoints.cpu\(\).numpy\(\), weights.cpu\(\).numpy\(\)\)\n        gt_x = torch.from_numpy\(gt_x\).to\(device\)\n        gt_y = torch.from_numpy\(gt_y\).to\(device\)\n        gt_weight = torch.from_numpy\(gt_weight\).to\(device\)"

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('wholebody/models/heads/rtmcc_head.py', 'w') as f:
    f.write(new_content)
