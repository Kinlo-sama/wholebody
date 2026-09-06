import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union

from wholebody.core.registry import MODELS, LOSSES
from wholebody.structures.data_sample import PoseDataSample
from wholebody.models.base import BasePoseEstimator

@MODELS.register("PoseEstimatorDistiller")
class PoseEstimatorDistiller(BasePoseEstimator):
    def __init__(self,
                 student: Dict[str, Any],
                 teacher: Dict[str, Any],
                 distill_cfg: List[Dict[str, Any]],
                 teacher_pretrained: Optional[str] = None):
        super().__init__()
        
        self.student: BasePoseEstimator = MODELS.build(student)
        self.teacher: BasePoseEstimator = MODELS.build(teacher)
        
        if teacher_pretrained is not None:
            from wholebody.engine.checkpointer import load_partial_state_dict
            ckpt = torch.load(teacher_pretrained, map_location='cpu', weights_only=False)
            state = ckpt.get('ema_state_dict', ckpt.get('state_dict', ckpt))
            load_partial_state_dict(self.teacher, state, strict=False)
            
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.teacher.eval()
        
        self.distill_losses = nn.ModuleDict()
        self.distill_methods = distill_cfg
        
        for loss_cfg in self.distill_methods:
            for method_cfg in loss_cfg['methods']:
                loss_name = method_cfg.pop('type')
                self.distill_losses[loss_name] = LOSSES.build({'type': loss_name, **method_cfg})
                
    def extract_feat(self, inputs: torch.Tensor):
        return self.student.extract_feat(inputs)
        
    def forward_train(self, inputs: torch.Tensor, data_samples: List[PoseDataSample]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            self.teacher.eval()
            t_feats = self.teacher.extract_feat(inputs)
            t_heatmaps = self.teacher.head.forward(t_feats)
            
        s_feats = self.student.extract_feat(inputs)
        s_heatmaps = self.student.head.forward(s_feats)
        
        losses = {}
        
        # Determine decay factor for DWPose distillation
        decay_factor = 1.0
        try:
            from mmengine.logging import MessageHub
            hub = MessageHub.get_current_instance()
            epoch = hub.get_info('epoch')
            max_epochs = hub.get_info('max_epochs')
            decay_factor = max(0.0, 1.0 - (epoch / max_epochs))
        except Exception:
            pass # Fallback to no decay if not using mmengine runner
        
        # Base student loss
        s_base_loss = self.student.head.loss(s_feats, data_samples)
        losses.update(s_base_loss)
        
        # Distillation losses
        if 'FeaLoss' in self.distill_losses:
            if isinstance(s_feats, (list, tuple)):
                s_feat = s_feats[-1]
                t_feat = t_feats[-1]
            else:
                s_feat = s_feats
                t_feat = t_feats
            losses['loss_fea'] = self.distill_losses['FeaLoss'](s_feat, t_feat) * decay_factor
            
        if 'KDLoss' in self.distill_losses:
            weight = data_samples[0].gt_instances.keypoint_weights if hasattr(data_samples[0], 'gt_instances') else torch.ones_like(s_heatmaps[0])
            beta = 1.0 # default beta
            losses['loss_kd'] = self.distill_losses['KDLoss'](s_heatmaps, t_heatmaps, beta, weight) * decay_factor
            
        return losses

    def forward_predict(self, inputs: torch.Tensor, data_samples: List[PoseDataSample]) -> List[PoseDataSample]:
        return self.student.forward_predict(inputs, data_samples)
        
    def forward_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.student.forward_tensor(inputs)
