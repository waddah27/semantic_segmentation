import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
root = '/content/dataset'
torch.cuda.empty_cache()
batch_size = 8  # Reduced batch size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10
# Define your custom dataset
class CustomSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match model input
    transforms.ToTensor()
])

# Paths to your images and masks
image_dir = os.path.join(root, 'images')  # '/content/dataset/images'
mask_dir = os.path.join(root, 'masks')  #'/content/dataset/masks'

image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]
mask_paths = [os.path.join(mask_dir, msk) for msk in os.listdir(mask_dir) if msk.endswith('.png')]

# Shuffle and split dataset
dataset = CustomSegmentationDataset(image_paths, mask_paths, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# Load a pretrained model
model = deeplabv3_resnet101(pretrained=True)
for param in model.parameters():
  param.requires_grad = False
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)  # Adjust for number of classes (e.g., 2 for binary segmentation)
model = model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Loss and metrics per epochs
history = {
    'Train_loss':[],
    'Val_loss':[],
    'acc':[],
    'recall':[]
}
# Training loop
for epoch in range(epochs):  # Number of epochs
    print(f'Epoch {epoch+1}')
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device).long()  # Convert masks to long for CrossEntropyLoss
        optimizer.zero_grad()
        torch.cuda.empty_cache()  # Clear CUDA cache
        outputs = model(images)['out']
        # Ensure masks are in the shape [N, H, W]
        if masks.dim() == 4:
            masks = masks.squeeze(1)  # Remove channel dimension if it's 1
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    history['Train_loss'].append(train_loss)
    print(f'Train Loss: {train_loss}')

    # Evaluation
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    val_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images, masks = images.to(device), masks.to(device).long()  # Convert masks to long for CrossEntropyLoss
            outputs = model(images)['out']
            # Ensure masks are in the shape [N, H, W]
            if masks.dim() == 4:
                masks = masks.squeeze(1)  # Remove channel dimension if it's 1
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(masks.cpu().numpy())

    # Compute metrics
    val_loss = val_loss/len(test_loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    precision = precision_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
    recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
    history['Val_loss'].append(val_loss)
    history['acc'].append(accuracy)
    history['recall'].append(recall)
    print(f'Val Loss: {val_loss}')
    print(f'Test Accuracy: {100 * accuracy:.2f}%')
    print(f'Test Precision: {precision:.2f}')
    print(f'Test Recall: {recall:.2f}')

# save results history as a pd dataframe
results_csv = pd.DataFrame(history)
print(results_csv)
results_csv.plot()
results_csv.to_csv('results.csv')
print('Training finished.')

# Save the model
torch.save(model.state_dict(), 'segmentation_model.pth')
