import torch


def dice_score(outputs, labels, smooth=1e-6):
    """Calculates the Dice coefficient."""
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()

    outputs = outputs.view(-1)
    labels = labels.view(-1)

    intersection = (outputs * labels).sum()
    dice = (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)
    return dice.item()


def iou_score(outputs, labels):
    """Calculates the Intersection over Union (IoU) score."""
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()

    outputs = outputs.view(-1)
    labels = labels.view(-1)

    intersection = (outputs * labels).sum()
    total = (outputs + labels).sum()
    union = total - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])