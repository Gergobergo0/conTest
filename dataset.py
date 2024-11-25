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
        mask_path = os.path.join(self.root_dir, f"{file_id}_mask.png")#rgbs

        try:
            amplitude = Image.open(amplitude_path).convert("L")
            phase = Image.open(phase_path).convert("L")
            mask = Image.open(mask_path).convert("L")#rgbs
        except FileNotFoundError:
            print(f"Missing file: {amplitude_path} or {phase_path}")
            raise

        amplitude_tensor = transforms.ToTensor()(amplitude)
        phase_tensor = transforms.ToTensor()(phase)
        mask_tensor = transforms.ToTensor()(mask) if os.path.exists(mask_path) else amplitude_tensor  # Ha nincs mask, akkor duplikálhatod az amplitúdót

        combined = torch.cat([amplitude_tensor, phase_tensor, mask_tensor], dim=0)

        # Kombinálás 3 csatornás RGB kép formájában
        #combined = Image.merge("RGB", (amplitude, phase, amplitude))  # 3 csatornás



        if self.transform:
            combined = self.transform(combined)

        # Target érték
        target = torch.tensor(abs(round(row['defocus_label'])), dtype=torch.float32).unsqueeze(-1)


        return combined, target








class HoloDataset_test(Dataset):                    # Erre azért van szükség, mert neki nincs csv-je
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_ids = self._get_image_ids()
        self.transform = transform

    def _get_image_ids(self):
        file_ids = set()
        for filename in os.listdir(self.root_dir):
            if filename.endswith("_amp.png") or filename.endswith("_phase.png"):
                base_id = filename.rsplit("_", 1)[0]  # Extracts the file ID before "_amp" or "_phase"
                file_ids.add(base_id)
        return sorted(file_ids)  # Sort for consistency

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        file_id = self.image_ids[idx]

        # Construct image paths
        amplitude_path = os.path.join(self.root_dir, f"{file_id}_amp.png")
        phase_path = os.path.join(self.root_dir, f"{file_id}_phase.png")
        mask_path = os.path.join(self.root_dir, f"{file_id}_mask.png") #rgb-s

        # Load images
        try:
            amplitude = Image.open(amplitude_path).convert("L")
            phase = Image.open(phase_path).convert("L")
            mask = Image.open(mask_path).convert("L")#rgb-s
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing file: {e.filename}")

        # Combine amplitude and phase images (e.g., as 2-channel tensor)
        #combined = Image.merge("RGB", (amplitude, phase, amplitude))  # 3-channel RGB format


        combined = torch.stack([
            transforms.ToTensor()(amplitude),
            transforms.ToTensor()(phase),
            transforms.ToTensor()(mask)
        ])
        # Apply transformations if provided
        if self.transform:
            combined = self.transform(combined)

        return combined, file_id  # Return the combined image and the file ID
