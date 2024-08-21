import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models.segmentation as segmentation
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.models.segmentation import FCN_ResNet50_Weights
from PIL import Image

class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]
        self.mask_paths = [os.path.join(mask_dir, msk) for msk in os.listdir(mask_dir) if msk.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        if self.transform:
            image, mask = self.transform(image), self.transform(mask)
        return image, mask

def get_transform():
    return Compose([
        Resize((256, 256)),
        ToTensor()
    ])




def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)['out']
                    outputs = torch.sigmoid(outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(preds, labels.squeeze(1))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


def main():
    # Prepare data
    image_dir = 'dataset/images'
    mask_dir = 'dataset/masks'
    dataset = CustomSegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform=get_transform())
    dataloaders = {
        'train': DataLoader(dataset, batch_size=4, shuffle=True),
        'val': DataLoader(dataset, batch_size=4, shuffle=False)
    }

    # Initialize model, criterion, optimizer
    model = segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1))  # Assuming 2 classes: background + person
    criterion = nn.CrossEntropyLoss()  # For multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)


if __name__ == '__main__':
    main()