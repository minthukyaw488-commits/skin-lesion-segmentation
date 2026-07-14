import torch
import torch.nn as nn


def dice_score(preds, targets, smooth=1e-6):
    preds   = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def iou_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds   = (preds > threshold).float().view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union        = preds.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        return 1 - dice_score(preds, targets, self.smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce         = nn.BCELoss()
        self.dice        = DiceLoss()

    def forward(self, preds, targets):
        return self.bce_weight * self.bce(preds, targets) + \
               self.dice_weight * self.dice(preds, targets)