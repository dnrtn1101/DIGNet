import numpy as np


def iou(pred, target, num_classes=10):
    ious = []
    iou_sum = 0
    for cls in range(num_classes):
        pred_ind = (pred == cls)
        target_ind = (target == cls)

        intersection = (pred_ind[target_ind]).sum()
        union = pred_ind.sum() + target_ind.sum()

        if union == 0:
            ious.append(float('nan'))

        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iou_sum += float(intersection) / float(max(union, 1))

    return iou_sum/num_classes * 100, ious