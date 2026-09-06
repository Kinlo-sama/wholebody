with open('wholebody/models/distillers/pose_estimator_distiller.py', 'r') as f:
    content = f.read()

replacement = """        losses = {}
        
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
            
        return losses"""

import re
pattern = r"        losses = \{\}.*?return losses"
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('wholebody/models/distillers/pose_estimator_distiller.py', 'w') as f:
    f.write(new_content)
