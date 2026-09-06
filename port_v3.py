import torch

ckpt = torch.load("weights/dw-mm_ucoco.pth", map_location='cpu')
state = ckpt.get('state_dict', ckpt)

# It seems state_dict is already perfectly unwrapped!
torch.save(state, "weights/dw-mm_ucoco_v3.pth")
print(f"Extraidos {len(state)} pesos desde state_dict.")
