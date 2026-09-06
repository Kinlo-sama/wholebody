import torch
from wholebody.core.config import Config
from wholebody.core.registry import MODELS

cfg = Config.from_file("configs/experiments/dwpose-m_256x192.yaml")
model = MODELS.build(cfg.model)
ckpt = torch.load("weights/dw-mm_ucoco_ported.pth", map_location='cpu')
model.load_state_dict(ckpt)
model.eval()

# Fake image batch
x = torch.randn(1, 3, 256, 192)

with torch.no_grad():
    feats = model.extract_feat(x)
    pred_x, pred_y = model.head(feats)
    
print("pred_x max:", pred_x.max().item(), "min:", pred_x.min().item())
print("pred_y max:", pred_y.max().item(), "min:", pred_y.min().item())

