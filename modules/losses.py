import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dice_loss(pred, target, smooth=1e-6):
    num_classes = pred.shape[1]
    pred_soft = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
    cardinality = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice_score = (2. * intersection + smooth) / (cardinality + smooth)
    
    loss = 1 - dice_score.mean()
    return loss

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        d_loss = dice_loss(pred, target)
        return self.weight_ce * ce_loss + self.weight_dice * d_loss
