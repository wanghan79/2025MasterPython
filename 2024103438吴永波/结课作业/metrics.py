import torch
import numpy as np

class SegmentationMetric:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, labels):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        mask = labels != self.ignore_index
        preds = preds[mask]
        labels = labels[mask]
        self.confusion_matrix += self._fast_hist(labels, preds)

    def _fast_hist(self, label, pred):
        k = (label >= 0) & (label < self.num_classes)
        return np.bincount(
            self.num_classes * label[k].astype(int) + pred[k],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def get_scores(self):
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / (hist.sum() + 1e-10)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return {
            'Pixel Acc': acc,
            'Mean Acc': acc_cls,
            'Mean IoU': mean_iu,
            'FreqW Acc': fwavacc,
            'IoU': iu
        }

    def get_dice(self):
        hist = self.confusion_matrix
        dice = 2 * np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) + 1e-10)
        mean_dice = np.nanmean(dice)
        return {
            'Dice': dice,
            'Mean Dice': mean_dice
        }

if __name__ == '__main__':
    metric = SegmentationMetric(num_classes=20)
    preds = torch.randint(0, 20, (2, 512, 1024))
    labels = torch.randint(0, 20, (2, 512, 1024))
    metric.update(preds, labels)
    print(metric.get_scores())
    print(metric.get_dice()) 