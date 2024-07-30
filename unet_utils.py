import torch
import torch.nn as nn
class DoupleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoupleConv, self).__init__()
        self.do_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.do_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownSampling, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoupleConv(in_ch, out_ch)

    def forward(self, x):
        dconv =  self.double_conv(x)
        down = self.pool(dconv)
        return dconv, down

class UpSampling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampling, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.double_conv = DoupleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        up = self.up(x1)
        cat = torch.cat([up, x2], dim=1)
        return self.double_conv(cat)


