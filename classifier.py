import torch
import torch.nn as nn
from torch.nn import functional as F
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(64 * 64, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 64 * 64)  # Flatten the output from U-Net
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
