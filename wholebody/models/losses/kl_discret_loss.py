import torch
import torch.nn as nn
import torch.nn.functional as F

from wholebody.core.registry import LOSSES

@LOSSES.register("KLDiscretLoss")
class KLDiscretLoss(nn.Module):
    def __init__(self, beta=1.0, label_softmax=False, use_target_weight=True):
        super(KLDiscretLoss, self).__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.use_target_weight = use_target_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.beta, dim=1)
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        num_joints = pred_simcc[0].size(1)
        loss = 0

        for pred, gt in zip(pred_simcc, gt_simcc):
            weight = target_weight
            pred = pred.reshape(-1, pred.size(-1))
            gt = gt.reshape(-1, gt.size(-1))
            weight = weight.reshape(-1)

            _loss = self.criterion(pred, gt)
            if self.use_target_weight:
                _loss = _loss * weight
            loss += _loss.sum()

        return loss / num_joints
