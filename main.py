import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import argparse

from torch.utils.data import DataLoader, random_split
from augmentation import transform2
from losses import CombinedLoss
from trainer import ModelWrapper
from unet_model import Unet
from dataset import CustomSegmentationDataset, AugmentedSegmentationDataset



def main():
    
    torch.cuda.empty_cache()
    batch_size =  6 # Reduced batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 30
    k_folds = 5
    learning_rate = 2e-3
    feature_extraction = True # Set to False if you want to train all parameters of the model 
    data_path = 'dataset'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", default="weights", type=str)
    parser.add_argument("--data_path", default=data_path, type=str)
    parser.add_argument("--epochs", default=epochs, type=int)
    parser.add_argument("--batch_size", default=batch_size, type=int)
    parser.add_argument("--learning_rate", default=learning_rate, type=float)
    parser.add_argument("--device", default=device, action='store_true')
    parser.add_argument("--k_folds", default=k_folds, type=int)
    parser.add_argument("--feature_extraction", default=feature_extraction, action='store_true')
    args = parser.parse_args()

    # Paths to images and masks
    image_dir = os.path.join(args.data_path, 'images')  # '/content/dataset/images'
    mask_dir = os.path.join(args.data_path, 'masks')  #'/content/dataset/masks'

    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]
    mask_paths = [os.path.join(mask_dir, msk) for msk in os.listdir(mask_dir) if msk.endswith('.png')]
    
    # Shuffle and split dataset
    dataset = CustomSegmentationDataset(image_paths, mask_paths, transform=transform2)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    # Load Unet pretrained model
    # model = Unet(device=args.device, feature_extraction=args.feature_extraction, pretained=True)
    model = smp.Unet(
        encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
        activation=None
    )
    # Define optimizer and loss function
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    loss_fn = CombinedLoss() #torch.nn.BCEWithLogitsLoss()
    model_wrapper = ModelWrapper(
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        device=args.device
    )

    model_wrapper.train(model=model, model_save_path=args.model_save_path, k_folds=args.k_folds)

if __name__=='__main__':
    main()