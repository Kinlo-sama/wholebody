import torch
import torch.nn as nn
c1 = nn.Conv2d(256, 256, 3, padding=1)
c2 = nn.Conv2d(256, 256, 3, padding=1, groups=256)
print("Standard conv weight shape:", c1.weight.shape)
print("Depthwise conv weight shape:", c2.weight.shape)

try:
    c1.load_state_dict(c2.state_dict(), strict=False)
    print("Loaded depthwise into standard!")
except Exception as e:
    print("Failed to load depthwise into standard:", e)
    
try:
    c2.load_state_dict(c1.state_dict(), strict=False)
    print("Loaded standard into depthwise!")
except Exception as e:
    print("Failed to load standard into depthwise:", e)
