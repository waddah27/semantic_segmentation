import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetWithClassifier(nn.Module):
    def __init__(self, encoder_name="vgg16"):
        super(UNetWithClassifier, self).__init__()
        # Load a pre-trained U-Net model
        self.unet = smp.DeepLabV3(
            encoder_name=encoder_name,        # Choose encoder architecture
            encoder_weights="imagenet",     # Load pre-trained weights for the encoder
            classes=1,                      # Output channels (1 for binary segmentation)
            activation=None                # No activation here, we'll use it later
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 128, 512),  # Assuming the output size from the U-Net is 128x128
            nn.Dropout(0.2, inplace=True),
            nn.ReLU(),
            nn.Linear(512, 1)  # Binary classification
        )

    def forward(self, x):
        seg_output = self.unet(x)
        seg_output = torch.sigmoid(seg_output)  # Sigmoid activation for binary mask

        # Flatten the segmentation output
        flat_output = seg_output.view(seg_output.size(0), -1)
        cls_output = self.classifier(flat_output)
        cls_output = torch.sigmoid(cls_output)

        # Reshape the classification output
        cls_output = cls_output.view(cls_output.size(0), 1, 1, 1)

        return cls_output, seg_output


if __name__ == "__main__":
    model = UNetWithClassifier()
    print(model)