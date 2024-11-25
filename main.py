import os
import matplotlib.pyplot as plt
#
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import HoloDataset,HoloDataset_test
from model import FocusNet, FocusNetPretrained
from trainer import TrainingManager


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(base_dir, "data_labels_train.csv")
    train_image_dir = os.path.join(base_dir, "train_data")
    test_image_dir = os.path.join(base_dir, "test_data")

    transform = transforms.Compose([ #RGB
        transforms.Resize((128, 128)),  # Méretezés
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalizálás
    ])

    # Train dataset betöltése
    train_dataset = HoloDataset(train_csv, train_image_dir, transform=transform)
    test_dataset = HoloDataset_test(test_image_dir, transform=transform)

    # Train-validation split
    # Train-validation split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Modell inicializálása
    model = FocusNetPretrained()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manager = TrainingManager(model, train_loader, val_loader,test_loader, device)

    # Tanítás
    manager.train(epochs=50)
    manager.validate()
    plt.plot(manager.train_losses, label="Training Loss")
    plt.plot(manager.val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()
