import csv
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

class ImagePairCSVDataset(Dataset):
    def __init__(self, csv_path, root_dir=None, transform=None):
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.transform = transform

        self.samples = []
        with open(self.csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgA = row["imgA"]
                imgB = row["imgB"]
                label_s = int(row["label_s"])   # same=1 / diff=0
                label_c = int(row["label_c"])
                self.samples.append((imgA, imgB, label_s, label_c))

    def __len__(self):
        return len(self.samples)

    def _resolve_path(self, p):
        p = Path(p)
        if self.root_dir is not None and not p.is_absolute():
            p = self.root_dir / p
        return p

    def __getitem__(self, idx):
        imgA_path, imgB_path, label_s, label_c = self.samples[idx]
        imgA = Image.open(self._resolve_path(imgA_path)).convert("RGB")  # RGB 이미지 파일 전달
        imgB = Image.open(self._resolve_path(imgB_path)).convert("RGB")

        if self.transform is not None:
            imgA = self.transform(imgA)   # (C,H,W)
            imgB = self.transform(imgB)

        label_s = torch.tensor(label_s, dtype=torch.float32)
        label_c = torch.tensor(label_c, dtype=torch.float32)
        return imgA, imgB, label_s, label_c