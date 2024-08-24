import torch
import torch.nn as nn

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # Apply sigmoid to get probabilities
        outputs = torch.sigmoid(outputs)
        
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Intersection and Union
        intersection = (outputs * targets).sum()
        union = outputs.sum() + targets.sum() - intersection
        
        # Calculate IoU (Jaccard Index)
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Jaccard loss is 1 - IoU
        return 1 - iou


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # Binary Cross-Entropy with logits
        bce_loss = self.bce_loss(outputs, targets)
        
        # Apply sigmoid to get probabilities
        outputs = torch.sigmoid(outputs)
        
        # Focal loss calculation
        pt = torch.where(targets == 1, outputs, 1 - outputs)  # Probabilities for the target class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, jaccard_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.jaccard_loss = JaccardLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.jaccard_weight = jaccard_weight
        self.focal_weight = focal_weight

    def forward(self, outputs, targets):
        # Compute both losses
        jaccard = self.jaccard_loss(outputs, targets)
        focal = self.focal_loss(outputs, targets)
        
        # Combine losses
        combined_loss = self.jaccard_weight * jaccard + self.focal_weight * focal
        
        return combined_loss
