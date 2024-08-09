# import torch
# from tqdm import tqdm
# class ModelWrapper:
#     def __init__(self, optimizer, scheduler, criterion, train_loader, val_loader, epochs, device):
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.loss_fn = criterion
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.epochs = epochs
#         self.device = device


#     def train(self, model, model_save_path=None):
#         for epoch in range(self.epochs):
#             model.train()
#             running_loss_total = 0.0
#             running_loss_cls = 0.0
#             running_loss_seg = 0.0

#             for images, labels, masks in tqdm(self.train_loader):
#                 images, labels, masks = images.to(self.device), labels.to(self.device), masks.to(self.device)

#                 cls_outputs, seg_outputs = model(images)
#                 cls_outputs = cls_outputs.squeeze()

#                 # Compute losses
#                 cls_loss = self.loss_fn(cls_outputs, labels.float())
#                 # seg_loss = self.loss_fn(seg_outputs, masks)
#                 loss = cls_loss #seg_loss + cls_loss
#                 running_loss_cls += cls_loss.item() * images.size(0)
#                 # running_loss_seg += seg_loss.item() * images.size(0)
#                 # running_loss_total += loss.item() * images.size(0)
#                 # Backward pass and optimization
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()


#             self.scheduler.step()
#             # epoch_total_loss = running_loss_total / len(self.train_loader.dataset)
#             epoch_loss_cls = running_loss_cls / len(self.train_loader.dataset)
#             # epoch_loss_seg = running_loss_seg / len(self.train_loader.dataset)
#             print(f"Epoch {epoch+1}/{self.epochs}, Train cls Loss: {epoch_loss_cls:.4f}")
#             # print(f"Epoch {epoch+1}/{self.epochs}, Train seg Loss: {epoch_loss_seg:.4f}")
#             # print(f"Epoch {epoch+1}/{self.epochs}, Train total Loss: {epoch_total_loss:.4f}")


#             model.eval()
#             val_seg_loss = 0
#             val_cls_loss = 0
#             val_total_loss = 0
#             with torch.no_grad():
#                 for i, (images, labels, masks) in enumerate(self.val_loader):
#                     images, labels, masks = images.to(self.device), labels.to(self.device), masks.to(self.device)
#                     cls_outputs, seg_outputs = model(images)
#                     cls_outputs = cls_outputs.squeeze()
#                     # seg_loss = self.loss_fn(seg_outputs, masks)
#                     cls_loss = self.loss_fn(cls_outputs, labels.float())
#                     loss = cls_loss #seg_loss + cls_loss
#                     # val_seg_loss += seg_loss.item() * images.size(0)
#                     val_cls_loss += cls_loss.item() * images.size(0)
#                     # val_total_loss += loss.item() * images.size(0)
#             # print(f"Epoch: {epoch+1}, Val seg Loss: {val_seg_loss/len(self.val_loader.dataset)}")
#             print(f"Epoch: {epoch+1}, Val cls Loss: {val_cls_loss/len(self.val_loader.dataset)}")
#             # print(f"Epoch: {epoch+1}, Val total Loss: {val_total_loss/len(self.val_loader.dataset)}")


import torch
from tqdm import tqdm

class ModelWrapper:
    def __init__(self, optimizer, scheduler, criterion, train_loader, val_loader, epochs, device):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device

    def train(self, model, model_save_path=None):
        for epoch in range(self.epochs):
            model.train()
            running_loss_cls = 0.0

            for images, labels, masks in tqdm(self.train_loader):
                images, labels, masks = images.to(self.device), labels.to(self.device), masks.to(self.device)

                cls_outputs, _ = model(images)

                # Compute losses
                cls_loss = self.loss_fn(cls_outputs.squeeze(1), labels.float())
                running_loss_cls += cls_loss.item() * images.size(0)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                cls_loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            epoch_loss_cls = running_loss_cls / len(self.train_loader.dataset)
            print(f"Epoch {epoch+1}/{self.epochs}, Train cls Loss: {epoch_loss_cls:.4f}")

            model.eval()
            val_cls_loss = 0
            with torch.no_grad():
                for images, labels, masks in self.val_loader:
                    images, labels, masks = images.to(self.device), labels.to(self.device), masks.to(self.device)
                    cls_outputs, _ = model(images)
                    cls_loss = self.loss_fn(cls_outputs.squeeze(1), labels.float())
                    val_cls_loss += cls_loss.item() * images.size(0)

            print(f"Epoch {epoch+1}/{self.epochs}, Val cls Loss: {val_cls_loss / len(self.val_loader.dataset):.4f}")

            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
