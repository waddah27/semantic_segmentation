import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from augmentation import transform
from trainer import ModelWrapper
from unet_model import Unet
from dataset import CustomSegmentationDataset



def main():
    torch.cuda.empty_cache()
    batch_size = 8  # Reduced batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 10
    learning_rate = 1e-4
    data_path = 'dataset'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", default="weights", type=str)
    parser.add_argument("--data_path", default=data_path, type=str)
    parser.add_argument("--epochs", default=epochs, type=int)
    parser.add_argument("--batch_size", default=batch_size, type=int)
    parser.add_argument("--learning_rate", default=learning_rate, type=float)
    parser.add_argument("--device", default=device, action='store_true')
    args = parser.parse_args()

    # Paths to images and masks
    image_dir = os.path.join(args.data_path, 'images')  # '/content/dataset/images'
    mask_dir = os.path.join(args.data_path, 'masks')  #'/content/dataset/masks'

    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]
    mask_paths = [os.path.join(mask_dir, msk) for msk in os.listdir(mask_dir) if msk.endswith('.png')]

    # Shuffle and split dataset
    dataset = CustomSegmentationDataset(image_paths, mask_paths, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    # Load Unet pretrained model
    model = Unet(device=args.device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    model_wrapper = ModelWrapper(
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        device=args.device
    )

    model_wrapper.train(model=model, model_save_path=args.model_save_path)

if __name__=='__main__':
    main()