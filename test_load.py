import torch
from wholebody.models.backbones.cspnext import CSPNeXt
from wholebody.models.necks.cspnext_pafpn import CSPNeXtPAFPN
from wholebody.models.heads.rtmw_head import RTMWHead

# 1. Instantiate our models for RTMW-L 256x192
backbone = CSPNeXt(
    arch='P5',
    expand_ratio=0.5,
    deepen_factor=1.0,
    widen_factor=1.0,
    channel_attention=True
)

neck = CSPNeXtPAFPN(
    in_channels=[256, 512, 1024],
    out_channels=None,
    out_indices=(1, 2),
    num_csp_blocks=2,
    expand_ratio=0.5
)

head = RTMWHead(
    in_channels=1024,
    out_channels=133,
    input_size=(192, 256), # [H, W] in our config for 256x192? Wait. [256, 192]? The config says input_size=(288, 384) [H, W] for the big one, and (192, 256) for small?
    in_featuremap_size=(6, 8) 
)

ckpt = torch.load('weights/rtmw-l_ported.pth', map_location='cpu', weights_only=False)
state_dict = ckpt.get('state_dict', ckpt)

# Filter keys for backbone
bb_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
missing, unexpected = backbone.load_state_dict(bb_state_dict, strict=False)
print("Backbone Missing:", len(missing))
print("Backbone Unexpected:", len(unexpected))

# Filter keys for neck
nk_state_dict = {k.replace('neck.', ''): v for k, v in state_dict.items() if k.startswith('neck.')}
missing, unexpected = neck.load_state_dict(nk_state_dict, strict=False)
print("Neck Missing:", len(missing))
print("Neck Unexpected:", len(unexpected))

# Filter keys for head
hd_state_dict = {k.replace('head.', ''): v for k, v in state_dict.items() if k.startswith('head.')}
missing, unexpected = head.load_state_dict(hd_state_dict, strict=False)
print("Head Missing:", len(missing))
print("Head Unexpected:", len(unexpected))
