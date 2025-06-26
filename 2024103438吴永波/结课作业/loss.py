import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        logits = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        mask = (targets != self.ignore_index).unsqueeze(1)
        logits = logits * mask
        targets_one_hot = targets_one_hot * mask
        dims = (0, 2, 3)
        intersection = torch.sum(logits * targets_one_hot, dims)
        union = torch.sum(logits + targets_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice

if __name__ == '__main__':
    loss_fn = CombinedLoss()
    logits = torch.randn(2, 20, 512, 1024)
    targets = torch.randint(0, 20, (2, 512, 1024))
    loss = loss_fn(logits, targets)
    print('Loss:', loss.item()) 