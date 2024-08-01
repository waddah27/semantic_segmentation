import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ObjectDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.image_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        # Convert mask to binary
        mask_arr = np.array(mask)
        label = 1 if np.any(mask_arr) else 0

        image = self.transform(image)
        mask = self.transform(mask)

        return image, label, mask


