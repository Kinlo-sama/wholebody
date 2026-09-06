with open('wholebody/codecs/simcc_codec.py', 'r') as f:
    content = f.read()

replacement = """        if isinstance(self.sigma, (list, tuple)):
            sigma_x, sigma_y = self.sigma
        else:
            sigma_x = sigma_y = self.sigma

        radius_x = sigma_x * 3
        radius_y = sigma_y * 3

        for n in range(N):
            for k in range(K):
                if keypoints_visible[n, k] < 0.5:
                    continue

                mu_x, mu_y = keypoints_split[n, k]
                
                if mu_x < 0 or mu_x >= W_simcc or mu_y < 0 or mu_y >= H_simcc:
                    keypoint_weights[n, k] = 0
                    continue

                # Create 1D Gaussians
                target_x[n, k] = np.exp(-((x_grid - mu_x)**2) / (2 * sigma_x**2))
                target_y[n, k] = np.exp(-((y_grid - mu_y)**2) / (2 * sigma_y**2))

        if self.normalize:
            target_x /= (sigma_x * np.sqrt(np.pi * 2))
            target_y /= (sigma_y * np.sqrt(np.pi * 2))"""

import re
# Regex to match the block starting from 'radius = self.sigma * 3' to 'target_y /= norm_value'
pattern = r"        radius = self.sigma \* 3.*?target_y /= norm_value"
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('wholebody/codecs/simcc_codec.py', 'w') as f:
    f.write(new_content)
