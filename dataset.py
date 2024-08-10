# import os
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from augmentation import get_training_augmentation, get_validation_augmentation, get_preprocessing

# training_augmentation = get_training_augmentation()
# testing_augmentation = get_validation_augmentation()
# class ObjectDataset(Dataset):
#     def __init__(self, root, augmentation=training_augmentation):
#         self.root = root
#         self.image_dir = os.path.join(root, 'images')
#         self.mask_dir = os.path.join(root, 'masks')

#         # self.transform_image = transforms.Compose([
#         #     transforms.Resize((128, 128)),
#         #     transforms.ToTensor(),
#         #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         # ])
#         # self.transform_mask = transforms.Compose([
#         #     transforms.Resize((128, 128)),
#         #     transforms.ToTensor(),
#         #     # transforms.Normalize(mean=[0.485], std=[0.229])
#         # ])
#         self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
#         self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
#         self.augmentation = augmentation

#         assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_dir, self.image_files[idx])
#         mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

#         # image = Image.open(img_path).convert('RGB')
#         # mask = Image.open(mask_path).convert('L')
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#         # Convert mask to binary
#         # mask_arr = np.array(mask)
#         label = torch.sigmoid(torch.tensor(255)) if np.any(mask) else torch.sigmoid(torch.tensor(0))  # Sigmoid activation for binary mask
#         # apply augmentations
#         if self.augmentation is not None:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
#         # image = self.transform_image(image)
#         # mask = self.transform_mask(mask)
#         # mask = torch.sigmoid(mask) # Sigmoid activation for binary mask

#         # Convert to tensor
#         image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to CHW format
#         mask = torch.tensor(mask, dtype=torch.float32).squeeze(0)  # Remove channel dimension


#         return image, label, mask  # image, label, mask


import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from augmentation import get_training_augmentation, get_validation_augmentation

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
