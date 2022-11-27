# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class FocalLoss(BaseWeightedLoss):
    """Focus Loss.

    Args:
        alpha (float): The alpha for calculating the modulating factor.
        gamma (float): The gamma for calculating the modulating factor.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self, alpha=2.0, gamma=4.0, reduction='mean', loss_weight=1.0):
        super(FocalLoss, self).__init__(loss_weight=loss_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def _forward(self, pred, target):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.

        Returns:
            torch.Tensor: The calculated loss
        """
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
