import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (f1_score, precision_recall_curve, 
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold

from PIL import Image
class ModelWrapper:
    def __init__(self, optimizer=None, scheduler=None, loss_fn=None, train_loader=None, test_loader=None, epochs=10, device=None, transform=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.transform = transform 
        self.scheduler = scheduler
        # Loss and metrics per epochs
        self.history = {
            'Train_loss':[],
            'Val_loss':[],
            'acc':[],
            'recall':[],
            'prec':[],
            'f1':[],
            'auc_roc':[],
            'auc_pr':[] 
        }
    def train(self, model, model_save_path=None, k_folds=5):
        torch.cuda.empty_cache()
        # Prepare data for cross-validation
        all_images = []
        all_masks = []
        for images, masks in self.train_loader:
            all_images.append(images)
            all_masks.append(masks)
        all_images = torch.cat(all_images, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
            print(f'\nFold {fold + 1}/{k_folds}')
            
            # Create data loaders for this fold
            train_images, val_images = all_images[train_idx], all_images[val_idx]
            train_masks, val_masks = all_masks[train_idx], all_masks[val_idx]

            train_dataset = torch.utils.data.TensorDataset(train_images, train_masks)
            val_dataset = torch.utils.data.TensorDataset(val_images, val_masks)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_loader.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.test_loader.batch_size, shuffle=False)
            
            # Initialize model, optimizer, and loss function
            model = model.to(self.device)
            optimizer = self.optimizer
            loss_fn = self.loss_fn
            
            # Training loop
            for epoch in range(self.epochs):
                print(f'Epoch {epoch+1}')
                model.train()
                running_loss = 0.0
                for images, masks in tqdm(train_loader):
                    images, masks = images.to(self.device).float(), masks.to(self.device).float()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    outputs = model(images)#['out']
                    loss = loss_fn(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                train_loss = running_loss / len(train_loader)
                self.history['Train_loss'].append(train_loss)
                print(f'Train Loss: {train_loss}')
            
            # Evaluation
            model.eval()
            all_preds_probs = []
            all_labels = []
            val_loss = 0

            with torch.no_grad():
                for images, masks in tqdm(val_loader):
                    images, masks = images.to(self.device).float(), masks.to(self.device).float()
                    outputs = model(images)#['out']
                    loss = loss_fn(outputs, masks)
                    val_loss += loss.item()
                    outputs = torch.sigmoid(outputs)
                    all_preds_probs.append(outputs.cpu().numpy())
                    all_labels.append(masks.cpu().numpy())
                
                all_preds_probs = np.concatenate(all_preds_probs)
                all_labels = np.concatenate(all_labels)
                flat_labels = all_labels.flatten().astype(int)
                flat_preds_probs = all_preds_probs.flatten()
                binary_preds = (flat_preds_probs > 0.5).astype(int)
                
                accuracy = np.mean(binary_preds == flat_labels)
                precision = precision_score(flat_labels, binary_preds, average='weighted')
                recall = recall_score(flat_labels, binary_preds, average='weighted')
                f1 = f1_score(flat_labels, binary_preds, average='weighted')
                
                if len(set(flat_labels)) == 2:
                    roc_auc = roc_auc_score(flat_labels, flat_preds_probs)
                    precision_vals, recall_vals, _ = precision_recall_curve(flat_labels, flat_preds_probs)
                    pr_auc = np.trapz(recall_vals, precision_vals)
                else:
                    roc_auc = pr_auc = np.nan
                
                fold_results.append({
                    'Val_loss': val_loss / len(val_loader),
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'ROC AUC': roc_auc,
                    'PR AUC': pr_auc
                })
                print(f'Val Loss: {val_loss / len(val_loader)}')
                print(f'Test Accuracy: {100 * accuracy:.2f}%')
                print(f'Test Precision: {precision:.2f}')
                print(f'Test Recall: {recall:.2f}')
                print(f'Test F1 Score: {f1:.2f}')
                if not np.isnan(roc_auc):
                    print(f'Test ROC AUC: {roc_auc:.2f}')
                if not np.isnan(pr_auc):
                    print(f'Test PR AUC: {pr_auc:.2f}')
            # Step the scheduler with the validation loss
            if self.scheduler:
                self.scheduler.step(val_loss / len(val_loader))
            # Save model weights for this fold
            if model_save_path:
                save_path = os.path.join(model_save_path, f'model_fold_{fold + 1}.pth')
                torch.save(model.state_dict(), save_path)
                print(f'Model saved to {save_path}')
        # Aggregate results
        fold_results_df = pd.DataFrame(fold_results)
        mean_results = fold_results_df.mean()
        print(f'\nCross-Validation Results:\n{fold_results_df}')
        print(f'\nMean Results:\n{mean_results}')
        fold_results_df.to_csv('cross-validation.csv')

        # Save results history as a pd dataframe
        # Ensure all history lists have the same length
        max_len = max(len(v) for v in self.history.values())
        for key in self.history.keys():
            while len(self.history[key]) < max_len:
                self.history[key].append(np.nan)
        

        # Save results history as a pd dataframe
        results_csv = pd.DataFrame(self.history)
        print(results_csv)
        results_csv.plot()
        results_csv.to_csv('results.csv')
        print('Training finished.')
    
    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0)

   
    def infer(self, model, image_path, thresh=0.3):
        # Load and preprocess image
        torch.cuda.empty_cache()
        image = self.load_image(image_path).to(self.device)
        
        with torch.no_grad():
            # Get model output
            logits = model(image)#['out']
            
            ## Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(logits)

            # Apply a threshold to obtain binary masks
            threshold = thresh
            preds = (probabilities > threshold).cpu().numpy().squeeze()
            return preds

    def visualize_results(self, image_path, prediction, mask_path):
        original_image = Image.open(image_path).convert('RGB')
        original_mask = Image.open(mask_path)
        original_image = original_image.resize((256, 256))
        original_mask = original_mask.resize((256,256))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(original_image)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Prediction')
        plt.imshow(prediction, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title('original mask')
        plt.imshow(original_mask, cmap='gray')
        plt.axis('off')

        plt.show()

