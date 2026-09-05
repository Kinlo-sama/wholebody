import argparse
import torch

try:
    from mmengine.config import Config as MMConfig
    from mmpose.registry import MODELS as MM_MODELS
    from mmpose.utils import register_all_modules
except ImportError:
    pass

def main():
    register_all_modules(init_default_scope=True)
    cfg = MMConfig.fromfile('mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb704-270e_cocktail14-256x192.py')
    model = MM_MODELS.build(cfg.model)
    print("MMPose Stem BN eps:", model.backbone.stem[0].bn.eps)
    print("MMPose Stem BN momentum:", model.backbone.stem[0].bn.momentum)

if __name__ == '__main__':
    main()
