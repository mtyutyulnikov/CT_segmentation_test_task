def dice_coef(preds, targets, smooth=0.001):
    """Takes in two tensors with predicted classes. Returns Dice coefficient"""
    assert preds.shape == targets.shape
    intersection = ((preds == targets) & (targets == 1)).sum()
    coef = (2 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return coef
