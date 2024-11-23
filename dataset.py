import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class HoloDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_id = row['filename_id']
        amplitude_path = os.path.join(self.root_dir, f"{file_id}_amp.png")
        phase_path = os.path.join(self.root_dir, f"{file_id}_phase.png")

        amplitude = Image.open(amplitude_path).convert("L")  # Grayscale
        phase = Image.open(phase_path).convert("L")  # Grayscale

        # Kombinálás 3 csatornás RGB kép formájában
        combined = Image.merge("RGB", (amplitude, phase, amplitude))  # 3 csatornás

        if self.transform:
            combined = self.transform(combined)

        # Target érték
        target = torch.tensor(abs(round(row['defocus_label'])), dtype=torch.float32)

        return combined, target
