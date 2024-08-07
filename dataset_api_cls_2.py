import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ObjectDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.image_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')
        self.transform_image = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        self.mask_files = [f for f in os.listdir(self.mask_dir) if f.endswith('.png')]

        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        # Reshape: (H, W, C) -> (C, H, W)
        # image = image.transpose((2, 0, 1))
        # mask = mask.transpose((2, 0, 1))

        # Convert mask to binary
        mask_arr = np.array(mask)
        label = torch.sigmoid(torch.tensor(255)) if np.any(mask_arr) else torch.sigmoid(torch.tensor(0))  # Sigmoid activation for binary mask

        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        mask = torch.sigmoid(mask) # Sigmoid activation for binary mask

        return image, label, mask


