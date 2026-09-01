import sys
sys.path.insert(0, '.')
from wholebody.core.config import Config
from wholebody.core.registry import MODELS
config = Config.from_file('configs/experiments/rtmw-l_384x288.yaml')
model = MODELS.build(config.model)
print(f"Total keys in model.state_dict(): {len(model.state_dict())}")
