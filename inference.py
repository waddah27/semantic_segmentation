import torch
import cv2
import segmentation_models_pytorch as smp
from trainer import ModelWrapper
from torchvision.models.segmentation import deeplabv3_resnet101
from unet_model import Unet
from augmentation import transform
# Example usage
# Load model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Unet(device=device)
model = smp.Unet(
    encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2
    encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
).to(device)
model.load_state_dict(torch.load('weights/model_fold_1.pth'))
model.eval()

model_wrapper = ModelWrapper(device=device, transform=transform)
image_path = 'dataset/images/000000002907.jpg'
mask_path = 'dataset/masks/000000002907.png'
prediction, logits  = model_wrapper.infer(model, image_path, thresh=0.4)
model_wrapper.visualize_results(image_path, prediction, mask_path)