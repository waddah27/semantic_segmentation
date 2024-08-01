import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from unet_with_classifier_pretrained import UNetWithClassifier
from dataset_api_cls_2 import ObjectDataset
from torch.utils.data import DataLoader, random_split


if __name__ == "__main__":
    epochs = 2
    batch_size = 4
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", default="model_weights", type=str)
    parser.add_argument("--data_path", default="D:\Job\Other\pytorch\pytorch_pipelines\semantic_segmentation\dataset", type=str)
    parser.add_argument("--epochs", default=epochs, type=int)
    parser.add_argument("--batch_size", default=batch_size, type=int)
    parser.add_argument("--learning_rate", default=learning_rate, type=float)
    args = parser.parse_args()

    # Initialize models
    model = UNetWithClassifier().to(device)
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    generator = torch.Generator().manual_seed(42)
    train_data , test_data = random_split(ObjectDataset(args.data_path), [0.8, 0.2], generator=generator)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    # Loss and optimizer
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels, masks in train_loader:
            optimizer.zero_grad()

            cls_outputs, seg_outputs = model(images)
            cls_outputs = cls_outputs.squeeze()

            # Compute losses
            cls_loss = criterion_cls(cls_outputs, labels.float())
            seg_loss = nn.BCEWithLogitsLoss()(seg_outputs, masks)  # Adjust as needed for mask loss

            loss = cls_loss + seg_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
