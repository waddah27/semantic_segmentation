import torch
import cv2
from trainer import ModelWrapper
from torchvision.models.segmentation import deeplabv3_resnet101
from unet_model import Unet
from augmentation import transform
# Example usage
# Load model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(device=device)
model.load_state_dict(torch.load('weights/model_fold_1.pth'))
model.eval()

model_wrapper = ModelWrapper(device=device, transform=transform)
image_path = 'dataset/images/000000002907.jpg'
mask_path = 'dataset/masks/000000002907.png'
prediction = model_wrapper.infer(model, image_path, thresh=0.5)
model_wrapper.visualize_results(image_path, prediction, mask_path)