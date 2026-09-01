import torch
from wholebody.core.config import Config
from wholebody.inference.api import init_model

cfg = Config.fromfile('configs/experiments/rtmw-l_384x288.yaml')
model = init_model(config=cfg, checkpoint='weights/rtmw-l_384x288_ported.pth', device='cpu')

x = torch.randn(1, 3, 384, 288)
with torch.no_grad():
    feats = model.extract_feat(x)
    pred_x, pred_y = model.head.forward(feats)
    
    print(f"Pred X Min: {pred_x.min().item():.3f}, Max: {pred_x.max().item():.3f}, Mean: {pred_x.mean().item():.3f}")
    print(f"Pred Y Min: {pred_y.min().item():.3f}, Max: {pred_y.max().item():.3f}, Mean: {pred_y.mean().item():.3f}")
