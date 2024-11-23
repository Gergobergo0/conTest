import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from metrics_utils import Metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18, ResNet18_Weights



class TrainingManager:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.HuberLoss(delta=1.0)
        # self.criterion = nn.SmoothL1Loss()  # Kevésbé érzékeny kiugró értékekre
        # self.criterion = nn.MSELoss()  # Négyzetes hibák csökkentése
        # self.criterion = nn.L1Loss()  # Abszolút hiba csökkentése
        # self.criterion = nn.CrossEntropyLoss()  # Ha osztályozást szeretnél

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)  # Súlyok szétválasztott frissítése
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  # Stochastic Gradient Descent
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001, alpha=0.99, weight_decay=0.0001)  # RMSprop
        # self.optimizer = optim.Adagrad(self.model.parameters(), lr=0.01)  # Adagrad
        # self.optimizer = optim.Adam([
        #     {'params': self.model.model.fc.parameters(), 'lr': 0.001},  # Utolsó rétegek
        #     {'params': self.model.model.layer4.parameters(), 'lr': 0.0001}  # Mélyebb rétegek
        # ])

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-6)
        # self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.5)  # Lépésenkénti tanulási ráta csökkentés
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)  # Ha a validáció nem javul
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)  # Exponenciális csökkentés

        self.train_losses = []
        self.val_losses = []

    def train(self, epochs):
        self.model.to(self.device)
        best_loss = float('inf')
        patience = 5
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
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")


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
