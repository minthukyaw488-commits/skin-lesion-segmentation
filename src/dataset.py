import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.img_size  = img_size
        self.augment   = augment

        self.image_ids = sorted([p.stem for p in self.image_dir.glob("*.jpg")])

        if len(self.image_ids) == 0:
            raise FileNotFoundError(f"No .jpg images found in {image_dir}")

        print(f"Found {len(self.image_ids)} images in {image_dir}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        image = Image.open(self.image_dir / f"{img_id}.jpg").convert("RGB")
        mask  = Image.open(self.mask_dir  / f"{img_id}_segmentation.png").convert("L")

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask  = mask.resize((self.img_size, self.img_size),  Image.NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = T.ToTensor()(image)
        image = T.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])(image)

        mask  = torch.from_numpy(np.array(mask))
        mask  = (mask > 127).float().unsqueeze(0)

        return image, mask, img_id

    def _augment(self, image, mask):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        angle = random.uniform(-30, 30)
        image = TF.rotate(image, angle)
        mask  = TF.rotate(mask,  angle)

        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            image = TF.adjust_contrast(image,   random.uniform(0.7, 1.3))

        return image, mask


def get_dataloaders(data_root="../data", img_size=256, batch_size=8, val_split=0.15):
    image_dir = os.path.join(data_root, "images")
    mask_dir  = os.path.join(data_root, "masks")

    full    = ISICDataset(image_dir, mask_dir, img_size=img_size)
    n_val   = int(len(full) * val_split)
    n_train = len(full) - n_val

    train_ds = ISICDataset(image_dir, mask_dir, img_size=img_size, augment=True)
    val_ds   = ISICDataset(image_dir, mask_dir, img_size=img_size, augment=False)
    train_ds.image_ids = full.image_ids[:n_train]
    val_ds.image_ids   = full.image_ids[n_train:]

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader