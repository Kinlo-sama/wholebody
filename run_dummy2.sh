cat << 'PY_EOF' > test_inference_dummy2.py
import torch
from wholebody.core.config import Config
from wholebody.core.registry import MODELS

cfg = Config.from_file("configs/experiments/dwpose-m_256x192.yaml")
model = MODELS.build(cfg.model)
ckpt = torch.load("weights/dw-mm_ucoco_ported.pth", map_location='cpu')
model.load_state_dict(ckpt, strict=False)
model.eval()

# Fake image batch
torch.manual_seed(42)
x = torch.randn(1, 3, 256, 192)

with torch.no_grad():
    feats = model.extract_feat(x)
    pred_x, pred_y = model.head(feats)
    
print("pred_x argmax (first 10 kpts):", pred_x.argmax(dim=2)[0, :10].tolist())
print("pred_y argmax (first 10 kpts):", pred_y.argmax(dim=2)[0, :10].tolist())

PY_EOF
python test_inference_dummy2.py
