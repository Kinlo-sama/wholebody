import sys
import torch

from wholebody.core.config import Config
from wholebody.core.registry import MODELS
from wholebody.structures.data_sample import PoseDataSample, InstanceData

student_cfg = dict(
    type='TopDownPoseEstimator',
    pretrained=None,
    test_cfg=dict(),
    backbone=dict(
        type='CSPNeXt', arch='P5', expand_ratio=0.5, deepen_factor=0.33, widen_factor=0.5, out_indices=[4], channel_attention=True
    ),
    neck=None,
    head=dict(
        type='RTMCCHead', in_channels=512, out_channels=133, input_size=[256, 192],
        in_featuremap_size=[8, 6], simcc_split_ratio=2.0, final_layer_kernel_size=7,
        gau_cfg=dict(hidden_dims=256, s=128, expansion_factor=2, dropout_rate=0.0, drop_path=0.0, act_fn='SiLU', use_rel_bias=False, pos_enc=False),
        codec=dict(type='SimCCLabel', input_size=[256, 192], sigma=[4.9, 5.66], simcc_split_ratio=2.0, normalize=False, use_dark=False)
    )
)

teacher_cfg = dict(
    type='TopDownPoseEstimator',
    pretrained=None,
    test_cfg=dict(),
    backbone=dict(
        type='CSPNeXt', arch='P5', expand_ratio=0.5, deepen_factor=1.0, widen_factor=1.0, out_indices=[4], channel_attention=True
    ),
    neck=None,
    head=dict(
        type='RTMCCHead', in_channels=1024, out_channels=133, input_size=[256, 192],
        in_featuremap_size=[8, 6], simcc_split_ratio=2.0, final_layer_kernel_size=7,
        gau_cfg=dict(hidden_dims=256, s=128, expansion_factor=2, dropout_rate=0.0, drop_path=0.0, act_fn='SiLU', use_rel_bias=False, pos_enc=False),
        codec=dict(type='SimCCLabel', input_size=[256, 192], sigma=[4.9, 5.66], simcc_split_ratio=2.0, normalize=False, use_dark=False)
    )
)

distiller_cfg = dict(
    type='PoseEstimatorDistiller',
    student=student_cfg,
    teacher=teacher_cfg,
    distill_cfg=[
        dict(methods=[dict(type='KDLoss', weight=1.0)]),
        dict(methods=[dict(type='FeaLoss', student_channels=512, teacher_channels=1024, alpha_fea=0.00007)])
    ]
)

distiller = MODELS.build(distiller_cfg)

x = torch.randn(2, 3, 256, 192)
data_samples = []
for _ in range(2):
    sample = PoseDataSample()
    sample.gt_instances = InstanceData()
    sample.gt_instances.keypoints = torch.rand(133, 2) * 100
    sample.gt_instances.keypoint_weights = torch.ones(133)
    data_samples.append(sample)

losses = distiller.forward_train(x, data_samples)
print("✅ Losses Computed Successfully:")
for k, v in losses.items():
    print(f"  - {k}: {v.item():.4f}")
    
total_loss = sum(losses.values())
total_loss.backward()
print("✅ Backward pass successful (Gradients computed!)")
