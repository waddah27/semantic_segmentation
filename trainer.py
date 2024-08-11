import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (f1_score, precision_recall_curve, 
                             precision_score, recall_score, roc_auc_score)

class ModelWrapper:
    def __init__(self, optimizer, scheduler, loss_fn, train_loader, test_loader, epochs, device):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
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
    def train(self, model, model_save_path=None):
        # Training loop
        for epoch in range(self.epochs):  # Number of epochs
            print(f'Epoch {epoch+1}')
            model.train()
            running_loss = 0.0
            for images, masks in tqdm(self.train_loader):
                images, masks = images.to(self.device), masks.to(self.device).long()  # Convert masks to long for CrossEntropyLoss
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()  # Clear CUDA cache
                outputs = model(images)['out']
                # Ensure masks are in the shape [N, H, W]
                if masks.dim() == 4:
                    masks = masks.squeeze(1)  # Remove channel dimension if it's 1
                loss = self.loss_fn(outputs, masks)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(self.train_loader)
            self.history['Train_loss'].append(train_loss)
            print(f'Train Loss: {train_loss}')


            # Evaluation
            model.eval()  # Set model to evaluation mode
            all_preds_probs = []
            all_labels = []
            val_loss = 0
            
            with torch.no_grad():
                for images, masks in tqdm(self.test_loader):
                    images, masks = images.to(self.device), masks.to(self.device).long()  # Convert masks to long for CrossEntropyLoss
                    outputs = model(images)['out']
                    if masks.dim() == 4:
                        masks = masks.squeeze(1)  # Remove channel dimension if it's 1
            
                    loss = self.loss_fn(outputs, masks)
                    val_loss += loss.item()
            
                    # Collect probabilities and labels
                    outputs = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                    all_preds_probs.append(outputs.cpu().numpy())  # Collect probabilities
                    all_labels.append(masks.cpu().numpy())  # Collect true labels
            
            # Convert lists to numpy arrays
            all_preds_probs = np.concatenate(all_preds_probs)
            all_labels = np.concatenate(all_labels)
            
            # Flatten arrays for metrics calculations
            flat_labels = all_labels.flatten()
            flat_preds_probs = all_preds_probs[:, 1].flatten()  # Assuming binary classification, get probability for class 1
            
            # Compute metrics
            accuracy = np.mean(np.argmax(all_preds_probs, axis=1).flatten() == flat_labels)
            precision = precision_score(flat_labels, np.argmax(all_preds_probs, axis=1).flatten(), average='weighted')
            recall = recall_score(flat_labels, np.argmax(all_preds_probs, axis=1).flatten(), average='weighted')
            f1 = f1_score(flat_labels, np.argmax(all_preds_probs, axis=1).flatten(), average='weighted')
            
            # Compute ROC AUC and PR AUC
            if len(set(flat_labels)) == 2:  # Check if binary classification
                roc_auc = roc_auc_score(flat_labels, flat_preds_probs)
                precision_vals, recall_vals, _ = precision_recall_curve(flat_labels, flat_preds_probs)
                pr_auc = np.trapz(recall_vals, precision_vals)  # Area under Precision-Recall curve
            else:
                roc_auc = pr_auc = np.nan
            
            self.history['Val_loss'].append(val_loss / len(self.test_loader))
            self.history['acc'].append(accuracy)
            self.history['recall'].append(recall)
            self.history['prec'].append(precision)
            self.history['f1'].append(f1)
            self.history['auc_roc'].append(roc_auc)
            self.history['auc_pr'].append(pr_auc)
            f1_max = np.max(self.history['f1'])
            if f1 == f1_max:
                save_path = os.path.join(model_save_path, f'model_F1_{np.round(f1, 4)}.pth')
                torch.save(model.state_dict(), save_path)
            print(f'Val Loss: {val_loss / len(self.test_loader)}')
            print(f'Test Accuracy: {100 * accuracy:.2f}%')
            print(f'Test Precision: {precision:.2f}')
            print(f'Test Recall: {recall:.2f}')
            print(f'Test F1 Score: {f1:.2f}')
            if not np.isnan(roc_auc):
                print(f'Test ROC AUC: {roc_auc:.2f}')
            if not np.isnan(pr_auc):
                print(f'Test PR AUC: {pr_auc:.2f}')
            
            
        # save results history as a pd dataframe
        results_csv = pd.DataFrame(self.history)
        print(results_csv)
        results_csv.plot()
        results_csv.to_csv('results.csv')
        print('Training finished.')