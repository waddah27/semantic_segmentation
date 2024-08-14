import torch
from trainer import ModelWrapper
from torchvision.models.segmentation import deeplabv3_resnet101
from unet_model import Unet
from augmentation import transform
# Example usage
# Load model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(device=device)
model.load_state_dict(torch.load('weights/model_weights_new.pth'))
model.eval()

model_wrapper = ModelWrapper(device=device, transform=transform)
image_path = 'dataset/images/000000561795.jpg'
mask_path = 'dataset/masks/000000561795.png'
prediction = model_wrapper.infer(model, image_path)
model_wrapper.visualize_results(image_path, prediction, mask_path)