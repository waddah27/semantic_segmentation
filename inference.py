import argparse
import os
import torch
import torch.nn as nn
from torchvision import transforms
from unet import UNet
from PIL import Image
import matplotlib.pyplot as plt
class ModelWrapper:
    def __init__(self, weights_path, device="cpu"):
        self.transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
            ])
        self.device = device
        self.model = UNet(3, 1)
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))

    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img).float().to(self.device)
        img = img.unsqueeze(0)
        pred_mask = self.model(img)
        img = img.squeeze(0).cpu().detach()#.numpy()
        img = img.permute(1, 2, 0)
        pred_mask = pred_mask.squeeze(0).cpu().detach()#.numpy()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask<0] = 0
        pred_mask[pred_mask>0] = 1
        return img, pred_mask

    def pred_show_single_img(self, img_path):
        img, pred_mask = self.predict(img_path)
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(img)
        fig.add_subplot(1, 2, 2)
        plt.imshow(pred_mask)
        plt.show()

    def pred_show_multiple_img(self, data_dir):
        for img_path in os.listdir(data_dir):
            self.pred_show_single_img(os.path.join(data_dir, img_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", default="model_weights", type=str)
    parser.add_argument("--data_path", default="\dataset", type=str)
    args = parser.parse_args()
    img_dir = args.data_path
    model_save_path = os.path.join(args.model_save_path, "unet.pth")
    model = ModelWrapper(model_save_path, device="cpu")
    model.pred_show_multiple_img(img_dir)
