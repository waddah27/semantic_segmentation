import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from augmentation import get_training_augmentation, get_validation_augmentation
from torchvision import transforms


class ObjectDataset(Dataset):
    def __init__(self, root, augmentation=None):
        self.root = root
        self.image_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
        self.augmentation = augmentation

        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 0, 1, 0)  # Convert to binary

        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # Normalize and rearrange dimensions
        mask = torch.tensor(mask).unsqueeze(0).float() / 255.0  # Add channel dimension and normalize

        return image, mask.squeeze(0), mask.squeeze(0)  # Return image, label, and mask


class CustomSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        # Define a transform to convert masks to binary
        self.binary_mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # Convert mask to tensor
            transforms.Lambda(lambda x: (x > 0).float())  # Convert to binary (0 or 1)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
        mask = self.binary_mask_transform(mask)
        return image, mask


class AugmentedSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB').resize((256, 256)))
        mask = np.array(Image.open(self.mask_paths[idx]).resize((256, 256)))
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

    def __len__(self):
        return len(self.image_paths)