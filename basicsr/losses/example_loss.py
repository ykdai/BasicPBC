from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ExampleLoss(nn.Module):
    """Example Loss.

    Args:
        loss_weight (float): Loss weight for Example loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(ExampleLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * F.l1_loss(pred, target, reduction='mean')
