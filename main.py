import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import segmentation_models_pytorch as smp
from trainer import ModelWrapper
from unet_with_classifier import UNetWithClassifier
from dataset import ObjectDataset
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
    parser.add_argument("--encoder_name", default="vgg16", type=str)
    args = parser.parse_args()

    # Initialize models
    encoder_name = args.encoder_name
    model = UNetWithClassifier(encoder_name=encoder_name).to(device)
    # for name, param in model.named_parameters():
    #     if 'classifier' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    generator = torch.Generator().manual_seed(42)
    train_data , test_data = random_split(ObjectDataset(args.data_path), [0.8, 0.2])#, generator=generator)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    # Loss and optimizer
    criterion_cls = smp.losses.SoftBCEWithLogitsLoss() #nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    num_epochs = args.epochs
    model_trainer = ModelWrapper(optimizer, scheduler, criterion_cls, train_loader, test_loader, num_epochs, device)
    model_trainer.train(model, model_save_path=args.model_save_path)

