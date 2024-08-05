import torch

class ModelWrapper:
    def __init__(self, optimizer, criterion, train_loader, val_loader, epochs, device):
        self.optimizer = optimizer
        self.loss_fn = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
    def train(self, model, model_save_path=None):
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.loss_fn(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f"Epoch: {epoch+1}, train_loss: {running_loss/i+1}")

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(self.val_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            print(f"Epoch: {epoch+1}, val_loss: {val_loss/i+1}")