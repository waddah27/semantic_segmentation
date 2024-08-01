import torch
import torch.nn as nn
from unet import UNet
from classifier import Classifier

class UnetWithClassifier(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.unet = UNet(in_ch, num_classes) # TODO: USE PRETRAINED MODEL
        self.classifier = Classifier()

    def forward(self, x):
        seg_out = self.unet(x)
        cls_out = self.classifier(seg_out)
        # return torch.argmax(out).item()
        return seg_out, cls_out
if __name__ == '__main__':
    model = UnetWithClassifier(3, 2)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(out)