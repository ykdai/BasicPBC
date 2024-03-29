import torch
from torch import abs_, nn

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class L_direct(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(L_direct, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, loss):
        Abs_loss = torch.mean(loss)
        return self.loss_weight * Abs_loss
