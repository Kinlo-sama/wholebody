import torch
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')
from wholebody.core.config import Config
from wholebody.core.registry import MODELS
from wholebody.engine.checkpointer import load_partial_state_dict

import wholebody.models

cfg = Config.from_file('configs/experiments/resnet50_wholebody_384x288.yaml')
model = MODELS.build(cfg.model)
ckpt = torch.load('weights/resnet50_wholebody_384x288_ported.pth', map_location='cpu', weights_only=False)
if 'state_dict' in ckpt:
    state = ckpt['state_dict']
elif 'ema_state_dict' in ckpt:
    state = ckpt['ema_state_dict']
else:
    state = ckpt
load_partial_state_dict(model, state, strict=False)
print("SUCCESS!")
