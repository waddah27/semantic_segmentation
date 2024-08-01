import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from unet_with_classifier import UnetWithClassifier
from dataset_ip_cls import ClassificationDataset
from torch.utils.data import DataLoader, random_split


if __name__ == "__main__":
    epochs = 2
    batch_size = 4
    learning_rate = 0.001
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", default="model_weights", type=str)
    parser.add_argument("--data_path", default="D:\Job\Other\pytorch\pytorch_pipelines\semantic_segmentation\dataset", type=str)
    parser.add_argument("--epochs", default=epochs, type=int)
    parser.add_argument("--batch_size", default=batch_size, type=int)
    parser.add_argument("--learning_rate", default=learning_rate, type=float)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(ClassificationDataset(args.data_path), [0.8, 0.2], generator=generator)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=4)
    model = UnetWithClassifier(3, 1)
    seg_criterion = nn.BCEWithLogitsLoss()
    cls_criterion = nn.CrossEntropyLoss()
    optmizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    for epoch in tqdm(range(epochs)):
        model.train()
        training_loss = 0
        for i, (img, mask, cls) in enumerate(tqdm(train_loader)):
            img, mask, cls = img.to(device), mask.to(device), cls.to(device)
            mask_pred, cls_pred = model(img)

            optmizer.zero_grad()
            seg_loss = seg_criterion(mask_pred, mask)
            cls_loss = cls_criterion(cls_pred, cls)
            loss = seg_loss + cls_loss
            loss.backward()
            optmizer.step()
            training_loss += loss.item()
        training_loss = training_loss / i+1
        print(f"Epoch: {epoch+1}, train_loss: {training_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (img, mask, cls) in enumerate(tqdm(val_loader)):
                img, mask, cls = img.to(device), mask.to(device), cls.to(device)
                y_pred = model(img)
                seg_loss = seg_criterion(y_pred, mask)
                cls_loss = cls_criterion(y_pred, cls)
                loss = seg_loss + cls_loss
                val_loss += loss.item()
        val_loss = val_loss / i+1
        print(f"Epoch: {epoch+1}, val_loss: {val_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(args.model_save_path, "unet_with_classifier.pt"))


