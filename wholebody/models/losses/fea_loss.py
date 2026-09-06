import torch
import torch.nn as nn
from wholebody.core.registry import LOSSES

@LOSSES.register("FeaLoss")
class FeaLoss(nn.Module):
    """PyTorch version of feature-based distillation
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        alpha_fea (float, optional): Weight of dis_loss. Defaults to 0.00007
    """
    def __init__(self,
                 student_channels: int,
                 teacher_channels: int,
                 alpha_fea: float = 0.00007,
                 ):
        super(FeaLoss, self).__init__()
        self.alpha_fea = alpha_fea

        if teacher_channels != student_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

    def forward(self, preds_S, preds_T):
        if self.align is not None:
            outs = self.align(preds_S)
        else:
            outs = preds_S

        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
        dis_loss = loss_mse(outs, preds_T) / N * self.alpha_fea

        return dis_loss
