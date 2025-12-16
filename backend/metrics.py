import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim(img1, img2):
    val = structural_similarity_index_measure(img1, img2, data_range=1.0)
    # If it’s a tuple (value, map), take only the value
    if isinstance(val, (tuple, list)):
        val = val[0]
    # Convert to a float scalar
    return val.item() if torch.is_tensor(val) else float(val)

def mse(img1, img2):
    with torch.no_grad():
        criterion = nn.MSELoss()
        return criterion(img1, img2)



def iou(y_true, y_pred, threshold=0.5, smooth=1e-15):
    """
    Intersection over Union (IoU) for binary segmentation.
    y_true: tensor of shape (B, 1, H, W)
    y_pred: tensor of shape (B, 1, H, W) with raw logits or probabilities
    """
    # Apply threshold to predictions
    y_pred = (y_pred > threshold).float()

    # Flatten tensors
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    # Calculate intersection and union
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    # IoU
    return (intersection + smooth) / (union + smooth)

def dice_coef(y_true, y_pred, threshold=0.5, smooth = 1e-15):
    """
    Dice coefficient for binary segmentation.
    y_true: tensor of shape (B, 1, H, W)
    y_pred: tensor of shape (B, 1, H, W) with raw logits or probabilities
    """
    # Apply threshold
    y_pred = (y_pred > threshold).float()
    
    # Flatten
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    # Intersection and Dice formula
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-15):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        # If using sigmoid in last layer, ensure predictions are between 0 and 1
        y_true = y_true.contiguous().view(-1)
        y_pred = y_pred.contiguous().view(-1)

        intersection = (y_true * y_pred).sum()
        dice = (2. * intersection + self.smooth) / (y_true.sum() + y_pred.sum() + self.smooth)
        return 1 - dice
