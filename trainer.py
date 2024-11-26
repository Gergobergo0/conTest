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
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.4, patience=5, verbose=True)

        #self.scheduler = ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,  patience=3, verbose=True,min_lr=1e-6 )


        self.train_losses = []
        self.val_losses = []

    def hyper_parameters(self):
        return "self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=0.01)  self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.4, patience=3, verbose=True)"

    def train(self, epochs):
        self.model.to(self.device)
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            y_true_train = []
            y_pred_train = []

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)

                # Loss számítás és lépés
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # Training adat pontosság számítása
                y_true_train.extend(labels.cpu().numpy().flatten())
                y_pred_train.extend(outputs.detach().cpu().numpy().flatten())

            # Átlagos training loss kiszámítása
            avg_train_loss = total_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # Training accuracy kiszámítása
            train_mae = Metrics.mae(np.array(y_true_train), np.array(y_pred_train))
            train_rmse = Metrics.rmse(np.array(y_true_train), np.array(y_pred_train))
            train_accuracy = Metrics.accuracy_within_tolerance(np.array(y_true_train), np.array(y_pred_train),
                                                               tolerance=0.4)

            # Aktuális Learning Rate lekérése
            current_lr = self.optimizer.param_groups[0]['lr']

            # Validation
            avg_val_loss = None
            if self.val_loader is not None:
                self.model.eval()
                total_val_loss = 0
                y_true_val = []
                y_pred_val = []

                with torch.no_grad():
                    for images, labels in self.val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        y_true_val.extend(labels.cpu().numpy().flatten())
                        y_pred_val.extend(outputs.cpu().numpy().flatten())
                        val_loss = self.criterion(outputs, labels)
                        total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(self.val_loader)
                self.val_losses.append(avg_val_loss)

                # Validation metrikák kiszámítása
                val_mae = Metrics.mae(np.array(y_true_val), np.array(y_pred_val))
                val_rmse = Metrics.rmse(np.array(y_true_val), np.array(y_pred_val))
                val_accuracy = Metrics.accuracy_within_tolerance(np.array(y_true_val), np.array(y_pred_val),
                                                                 tolerance=0.4)

                # LR csökkentése
                self.scheduler.step(avg_val_loss)

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping")
                        break

                # Epoch eredmények kiírása
                print(
                    f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
                    f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, "
                    f"MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, LR: {current_lr:.6f}")

                if val_rmse < 1.09 or val_accuracy > 50:
                    export(self.model, self.test_loader, self.device, epoch, 16, val_rmse)
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, LR: {current_lr:.6f}")

        # Végső modell exportálása
        export(self.model, self.test_loader, self.device, epochs, 16, best_loss)

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
