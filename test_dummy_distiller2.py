import sys
sys.path.append('.')
import torch

from wholebody.models.backbones.cspnext import CSPNeXt
from wholebody.models.heads.rtmcc_head import RTMCCHead
from wholebody.models.base import TopDownPoseEstimator
from wholebody.models.distillers.pose_estimator_distiller import PoseEstimatorDistiller
from wholebody.structures.data_sample import PoseDataSample, InstanceData

student_backbone = CSPNeXt(arch='P5', expand_ratio=0.5, deepen_factor=0.33, widen_factor=0.5, out_indices=[4], channel_attention=True)
student_head = RTMCCHead(in_channels=512, out_channels=133, input_size=[256, 192], in_featuremap_size=[8, 6], simcc_split_ratio=2.0, final_layer_kernel_size=7, codec=dict(type='SimCCLabel', input_size=[256, 192], sigma=[4.9, 5.66], simcc_split_ratio=2.0, normalize=False, use_dark=False))
student = TopDownPoseEstimator(backbone=student_backbone, head=student_head)

teacher_backbone = CSPNeXt(arch='P5', expand_ratio=0.5, deepen_factor=1.0, widen_factor=1.0, out_indices=[4], channel_attention=True)
teacher_head = RTMCCHead(in_channels=1024, out_channels=133, input_size=[256, 192], in_featuremap_size=[8, 6], simcc_split_ratio=2.0, final_layer_kernel_size=7, codec=dict(type='SimCCLabel', input_size=[256, 192], sigma=[4.9, 5.66], simcc_split_ratio=2.0, normalize=False, use_dark=False))
teacher = TopDownPoseEstimator(backbone=teacher_backbone, head=teacher_head)

# Manually create distiller
class DummyDistiller(torch.nn.Module):
    def __init__(self, s, t):
        super().__init__()
        self.student = s
        self.teacher = t
        from wholebody.models.losses.kd_loss import KDLoss
        from wholebody.models.losses.fea_loss import FeaLoss
        self.kd = KDLoss()
        self.fea = FeaLoss(512, 1024)
        
    def loss(self, x, data_samples):
        s_feats = self.student.extract_feat(x)
        s_heatmaps = self.student.head.forward(s_feats)
        
        with torch.no_grad():
            t_feats = self.teacher.extract_feat(x)
            t_heatmaps = self.teacher.head.forward(t_feats)
            
        losses = {}
        losses.update(self.student.head.loss(s_feats, data_samples))
        
        losses['loss_fea'] = self.fea(s_feats[-1] if isinstance(s_feats, tuple) else s_feats, t_feats[-1] if isinstance(t_feats, tuple) else t_feats)
        losses['loss_kd'] = self.kd(s_heatmaps, t_heatmaps, 1.0, data_samples[0].gt_instances.keypoint_weights.unsqueeze(0).repeat(x.size(0), 1))
        
        return losses

distiller = DummyDistiller(student, teacher)
x = torch.randn(2, 3, 256, 192)

data_samples = []
for _ in range(2):
    sample = PoseDataSample()
    sample.gt_instances = InstanceData()
    sample.gt_instances.keypoints = torch.rand(133, 2) * 100
    sample.gt_instances.keypoint_weights = torch.ones(133)
    data_samples.append(sample)

try:
    losses = distiller.loss(x, data_samples)
    print("Losses Computed Successfully!")
    for k, v in losses.items():
        print(f"  {k}: {v.item()}")
        
    total_loss = sum(losses.values())
    total_loss.backward()
    print("Backward pass successful!")
except Exception as e:
    import traceback
    traceback.print_exc()

