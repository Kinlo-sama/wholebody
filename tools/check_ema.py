import torch
import sys

for path in ["weights/rtmw-m.pth", "weights/rtmw-x_384x288.pth"]:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Keys in {path}:", list(ckpt.keys()))
