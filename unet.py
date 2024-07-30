import torch
import torch.nn as nn
from unet_utils import DoupleConv, DownSampling, UpSampling

class UNet(nn.Module):
    def __init__(self, in_ch, num_classes):
        """
        The model is based on UNet. it expects input of shape (batch_size, in_ch, height, width)
        and output of shape (batch_size, num_classes, height, width)
        """
        super().__init__()
        self.down1 = DownSampling(in_ch, 64)
        self.down2 = DownSampling(64, 128)
        self.down3 = DownSampling(128, 256)
        self.down4 = DownSampling(256, 512)
        self.bottleneck = DoupleConv(512, 1024)
        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        dconv1, down1 = self.down1(x)
        dconv2, down2 = self.down2(down1)
        dconv3, down3 = self.down3(down2)
        dconv4, down4 = self.down4(down3)
        bottleneck = self.bottleneck(down4)
        up1 = self.up1(bottleneck, dconv4)
        up2 = self.up2(up1, dconv3)
        up3 = self.up3(up2, dconv2)
        up4 = self.up4(up3, dconv1)
        out = self.out(up4)
        return out

if __name__ == '__main__':
    N, C, H, W = 1, 3, 256, 256
    n_classes = 10
    x = torch.randn(N, C, H, W)
    model = UNet(in_ch=C, num_classes=n_classes)
    out = model(x)
    print(out.shape)