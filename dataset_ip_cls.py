import os
import numpy as np
from PIL import Image
from dataset_api import ObjectDataset
from torchvision import transforms

class ClassificationDataset(ObjectDataset):
    def __init__(self, root):
        super().__init__(root)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
        label = 1 if np.any(mask_np) else 0
        return self.transform(img), self.transform(mask), label
