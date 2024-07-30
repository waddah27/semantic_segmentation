from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms
class ObjectDataset(Dataset):
    def __init__(self, root):
        self.root = root

        self.img_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()
        ])

        self.img_names = sorted(os.listdir(self.img_dir))
        self.mask_names = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        return self.transform(img), self.transform(mask)

