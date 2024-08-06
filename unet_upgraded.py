import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.ReLU()):
        super().__init__()
        expantion_rate = 4
        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(7, 7), stride=1, padding=3, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, expantion_rate * out_ch, kernel_size=(1, 1), stride=1, padding=1),
            activation,
            nn.Conv2d(expantion_rate * out_ch, out_ch, kernel_size=(1, 1), stride=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=(7, 7), stride=1, padding=3, groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, expantion_rate * out_ch, kernel_size=(1, 1), stride=1, padding=1),
            activation,
            nn.Conv2d(expantion_rate * out_ch, out_ch, kernel_size=(1, 1), stride=1),
        )

    def forward(self, x):
        return self.encoder_block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.ReLU()):
        super().__init__()
        expantion_rate = 4
        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(7, 7), stride=1, padding=3, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, expantion_rate * out_ch, kernel_size=(1, 1), stride=1, padding=1),
            activation,
            nn.Conv2d(expantion_rate * out_ch, out_ch, kernel_size=(1, 1), stride=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=(7, 7), stride=1, padding=3, groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, expantion_rate * out_ch, kernel_size=(1, 1), stride=1, padding=1),
            activation,
            nn.Conv2d(expantion_rate * out_ch, out_ch, kernel_size=(1, 1), stride=1),
        )

    def forward(self, x):
        return self.decoder_block(x)



class UNet2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Config
        in_channels  = 4   # Input images have 4 channels
        out_channels = 3   # Mask has 3 channels
        n_filters    = 32  # Scaled down from 64 in original paper
        activation   = nn.ReLU()
        self.downsampling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        

