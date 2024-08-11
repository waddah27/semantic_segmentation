# import argparse
# import os
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from unet import UNet
# from PIL import Image
# import matplotlib.pyplot as plt
# class ModelWrapper:
#     def __init__(self, weights_path, device="cpu"):
#         self.transforms = transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.ToTensor()
#             ])
#         self.device = device
#         self.model = UNet(3, 1)
#         self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))

#     def predict(self, img_path):
#         img = Image.open(img_path).convert('RGB')
#         img = self.transforms(img).float().to(self.device)
#         img = img.unsqueeze(0)
#         pred_mask = self.model(img)
#         img = img.squeeze(0).cpu().detach()#.numpy()
#         img = img.permute(1, 2, 0)
#         pred_mask = pred_mask.squeeze(0).cpu().detach()#.numpy()
#         pred_mask = pred_mask.permute(1, 2, 0)
#         pred_mask[pred_mask<0] = 0
#         pred_mask[pred_mask>0] = 1
#         return img, pred_mask

#     def pred_show_single_img(self, img_path):
#         img, pred_mask = self.predict(img_path)
#         fig = plt.figure()
#         fig.add_subplot(1, 2, 1)
#         plt.imshow(img)
#         fig.add_subplot(1, 2, 2)
#         plt.imshow(pred_mask)
#         plt.show()

#     def pred_show_multiple_img(self, data_dir):
#         for img_path in os.listdir(data_dir):
#             self.pred_show_single_img(os.path.join(data_dir, img_path))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_save_path", default="model_weights", type=str)
#     parser.add_argument("--data_path", default="\dataset", type=str)
#     args = parser.parse_args()
#     img_dir = args.data_path
#     model_save_path = os.path.join(args.model_save_path, "unet.pth")
#     model = ModelWrapper(model_save_path, device="cpu")
#     model.pred_show_single_img(img_dir)

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet  # Replace with your U-Net model import

# Load model
model = UNet(in_channels=3, out_channels=2)
model.load_state_dict(torch.load('path/to/your/trained_model.pth'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Prepare transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)

def infer(image_path):
    image = load_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        output = torch.argmax(output, dim=1).cpu().numpy().squeeze()
    return output

def visualize_results(image_path, prediction):
    original_image = Image.open(image_path).convert('RGB')
    original_image = original_image.resize((256, 256))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Prediction')
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage
image_path = 'path/to/your/test_image.jpg'
prediction = infer(image_path)
visualize_results(image_path, prediction)
