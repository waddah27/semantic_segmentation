# this script is used to train the semantic segmentation model
# using the dataset provided in the repository
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms
import numpy as np
from dataset_api import ObjectDataset
from unet import UNet
from torch import optim
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    weights_name = "unet.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", default="model_weights", type=str)
    parser.add_argument("--data_path", default="D:\Job\Other\pytorch\pytorch_pipelines\semantic_segmentation\dataset", type=str)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    args = parser.parse_args()
    # hyperparameters
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dataset
    data_path = args.data_path
    model_save_path = os.path.join(args.model_save_path, weights_name)
    dataset = ObjectDataset(data_path)
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(dataset, [0.8, 0.2], generator=generator)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    model = UNet(in_ch=3, num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    critertion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs)):
        model.train()
        train_running_loss = 0
        for batch_idx, (img, mask) in enumerate(tqdm(train_loader)):
            img = img.float().to(device)
            mask = mask.float().to(device)
            y_pred = model(img)

            optimizer.zero_grad()
            loss = critertion(y_pred, mask)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        train_loss = train_running_loss / batch_idx+1

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for batch_idx, (img, mask) in enumerate(val_loader):
                img = img.float().to(device)
                mask = mask.float().to(device)
                y_pred = model(img)
                loss = critertion(y_pred, mask)
                val_running_loss += loss.item()

        val_loss = val_running_loss / batch_idx+1
        print('--' * 30)
        print(f"Epoch: {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        print('--' * 30)
        torch.save(model.state_dict(), model_save_path)

