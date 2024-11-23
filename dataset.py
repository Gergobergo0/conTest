import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class HoloDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename_id = row['filename_id']
        defocus_label = row['defocus_label']
        abs_distance = torch.tensor([abs(defocus_label)], dtype=torch.float32)

        # Load images from the corresponding directory
        amp_img = Image.open(os.path.join(self.image_dir, f"{filename_id}_amp.png")).convert("L")
        phase_img = Image.open(os.path.join(self.image_dir, f"{filename_id}_phase.png")).convert("L")
        mask_img = Image.open(os.path.join(self.image_dir, f"{filename_id}_mask.png")).convert("L")

        if self.transform:
            amp_img = self.transform(amp_img)
            phase_img = self.transform(phase_img)
            mask_img = self.transform(mask_img)

        # Combine channels
        input_img = torch.cat([amp_img, phase_img, mask_img], dim=0)
        return input_img, abs_distance
