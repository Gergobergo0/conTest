import os
import matplotlib.pyplot as plt
#
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import HoloDataset
from model import FocusNet, FocusNetPretrained
from trainer import TrainingManager

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(base_dir, "data_labels_train.csv")
    train_image_dir = os.path.join(base_dir, "train_data")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Az adatok 0-1 közé kerülnek
    ])

    # Train dataset betöltése
    train_dataset = HoloDataset(train_csv, train_image_dir, transform=transform)

    # Train-validation split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    # Modell inicializálása
    #model = FocusNet()
    model = FocusNetPretrained()

    device = torch.device("cuda")

    print(f"deivce: {device}")

    # Edzési folyamat
    manager = TrainingManager(model, train_loader, val_loader, device)
    manager.train(epochs=30)

    plt.plot(manager.train_losses, label="Training Loss")
    plt.plot(manager.val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()
