import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.models.segmentation as segmentation


from unet2 import UNet2

import torch.nn.functional as F

class ClassifierHead(nn.Module):
    def __init__(self, input_channels, num_classes=1):
        super(ClassifierHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc1 = nn.Linear(input_channels, 1024)  # Hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)  # Output layer

    def forward(self, x):
        x = self.global_pool(x)  # Apply global average pooling
        x = torch.flatten(x, 1)  # Flatten the output of pooling
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SegmentationWithClassifier(nn.Module):
    def __init__(self, num_classes=1, train_segmentation=False):
        super(SegmentationWithClassifier, self).__init__()
        self.unet = segmentation.deeplabv3_resnet101(pretrained=True)
        self.unet.classifier = nn.Identity()  # Remove the segmentation head
        self.classifier_head = ClassifierHead(input_channels=2048, num_classes=num_classes)  # Adjust input_channels
        # Freeze the U-Net parameters
        if not train_segmentation:
            self.freeze_unet()

    def freeze_unet(self):
        for param in self.unet.parameters():
            param.requires_grad = False
    def forward(self, x):
        features = self.unet(x)['out']
        segmentation_output = features
        classification_output = self.classifier_head(features)
        return classification_output, segmentation_output



# from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
class UNetWithClassifier(nn.Module):
    def __init__(self, use_unet2=False, encoder_name="vgg16", activation=nn.LeakyReLU()):
        super(UNetWithClassifier, self).__init__()
        self.use_unet2 = use_unet2
        # Load a pre-trained U-Net model
        if self.use_unet2:
            self.unet = UNet2(in_ch=3, out_ch=1, activation=activation)

        else:
            self.unet = smp.Unet(
                encoder_name=encoder_name,        # Choose encoder architecture
                encoder_weights="imagenet",     # Load pre-trained weights for the encoder
                classes=1,                      # Output channels (1 for binary segmentation)
                activation=None,            # No activation here, we'll use it later
            )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 128, 512),  # Assuming the output size from the U-Net is 128x128
            nn.Dropout(0.2, inplace=True),
            nn.ReLU(),
            nn.Linear(512, 1)  # Binary classification
        )

    def forward(self, x):
        if self.use_unet2:
            seg_output = self.unet(x)
        else:
            seg_output = self.unet(x)['out']
        seg_output = torch.sigmoid(seg_output)  # Sigmoid activation for binary mask

        # Flatten the segmentation output
        flat_output = seg_output.view(seg_output.size(0), -1)
        cls_output = self.classifier(flat_output)
        cls_output = torch.sigmoid(cls_output)

        # Reshape the classification output
        cls_output = cls_output.view(cls_output.size(0), 1, 1, 1)

        return cls_output, seg_output


if __name__ == "__main__":
    from torchsummary import summary


    model = UNetWithClassifier()

    # summary(model, input_size=(3, 128, 128))  # Adjust input size as needed
    print(model.unet)