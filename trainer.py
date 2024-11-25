import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from metrics_utils import Metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18, ResNet18_Weights
from out_csv import export





class TrainingManager:
    def __init__(self, model, train_loader, val_loader,test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.L1Loss()#
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        self.train_losses = []
        self.val_losses = []


    def train(self, epochs):
        self.model.to(self.device)
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training

            self.model.train()
            total_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation
            avg_val_loss = None
            if self.val_loader is not None:
                self.model.eval()
                total_val_loss = 0
                y_true = []
                y_pred = []

                with torch.no_grad():
                    for images, labels in self.val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        y_true.extend(labels.cpu().numpy().flatten())
                        y_pred.extend(outputs.cpu().numpy().flatten())
                        val_loss = self.criterion(outputs, labels)
                        total_val_loss += val_loss.item()
                avg_val_loss = total_val_loss / len(self.val_loader)
                self.val_losses.append(avg_val_loss)

                # Accuracy számítás
                mae = Metrics.mae(np.array(y_true), np.array(y_pred))
                rmse = Metrics.rmse(np.array(y_true), np.array(y_pred))
                accuracy = Metrics.accuracy_within_tolerance(np.array(y_true), np.array(y_pred), tolerance=0.4)
                self.scheduler.step(avg_val_loss)
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping")
                        break
                # Epoch kiírás
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, "
                      f"Validation Loss: {avg_val_loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
                      f"Accuracy: {accuracy:.2f}%")
                if rmse < 2 or accuracy > 50:
                    export(self.model, self.test_loader, self.device, epoch, 16, rmse)
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
        export(self.model, self.test_loader, self.device, epochs, 16, best_loss)

    def validate(self):
        if self.val_loader is None:
            print("No validation data provided. Skipping validation.")
            return
        self.model.eval()
        total_loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                # Mentjük a valós és predikált értékeket
                y_true.extend(labels.cpu().numpy().flatten())
                y_pred.extend(outputs.cpu().numpy().flatten())

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_loader)

        # Pontossági metrikák kiszámítása
        mae = Metrics.mae(np.array(y_true), np.array(y_pred))
        rmse = Metrics.rmse(np.array(y_true), np.array(y_pred))
        accuracy = Metrics.accuracy_within_tolerance(np.array(y_true), np.array(y_pred), tolerance=0.4)

        print(f"Validation Loss: {avg_val_loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")

        return avg_val_loss


    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions.extend(outputs.cpu().numpy())
        return predictions
