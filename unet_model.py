import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101

class Unet(nn.Module):
    def __init__(self, pretained=True, feature_extraction=True, device = None) -> None:
        super().__init__()
        self.model = deeplabv3_resnet101(pretrained=pretained, num_classes=1)
        if feature_extraction:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        
        # for i in range(4):
        #     self.model.classifier[i] = torch.nn.Identity()
        #     self.model.aux_classifier[i] = torch.nn.Identity()
        # self.model.classifier[4] = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        # self.model.aux_classifier[4] = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.model = self.model.to(device)
    
    def forward(self, x):
        x = self.model(x)['out']
        return x
    