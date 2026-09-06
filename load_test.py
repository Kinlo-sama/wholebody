import sys
sys.path.append('.')
import torch
from wholebody.core.config import Config
from wholebody.core.registry import MODELS
import wholebody.models
cfg = Config.from_file('configs/experiments/dwpose-m_256x192.yaml')
model = MODELS.build(cfg.model)
print("Model built successfully!")
