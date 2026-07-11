# ================================================================
# Skin Lesion Segmentation - U-Net + ResNet34 (Pretrained)
# Dataset: ISIC 2018 Task 1
# Expected Val Dice: 0.85+
# ================================================================

# ── Install ───────────────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "segmentation-models-pytorch", "-q"])

# ── Imports ───────────────────────────────────────────────────────
import os
import json
import time
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────
CFG = {
    "image_dir":  "/kaggle/input/datasets/novem18/isic-2018-task1/ISIC2018_Task1-2_Training_Input",
    "mask_dir":   "/kaggle/input/datasets/novem18/isic-2018-task1/ISIC2018_Task1_Training_GroundTruth",
    "output_dir": "/kaggle/working",
    "encoder":    "resnet34",
    "img_size":   256,
    "batch_size": 16,
    "epochs":     30,
    "lr":         1e-4,
    "val_split":  0.15,
    "threshold":  0.5,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
# ─────────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────
class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_ids, img_size=256, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.image_ids = image_ids
        self.img_size  = img_size
        self.augment   = augment

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        image = Image.open(self.image_dir / f"{img_id}.jpg").convert("RGB")
        mask  = Image.open(self.mask_dir  / f"{img_id}_segmentation.png").convert("L")

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask  = mask.resize((self.img_size, self.img_size),  Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() > 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
                image = TF.adjust_contrast(image,   random.uniform(0.7, 1.3))
            if random.random() > 0.5:
                image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))

        image = T.ToTensor()(image)
        image = T.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])(image)
        mask  = (torch.from_numpy(np.array(mask)) > 127).float().unsqueeze(0)

        return image, mask, img_id


def get_dataloaders():
    all_ids = sorted([p.stem for p in Path(CFG["image_dir"]).glob("*.jpg")])
    print(f"Total images: {len(all_ids)}")

    n_val   = int(len(all_ids) * CFG["val_split"])
    n_train = len(all_ids) - n_val

    train_ds = ISICDataset(CFG["image_dir"], CFG["mask_dir"],
                           all_ids[:n_train], CFG["img_size"], augment=True)
    val_ds   = ISICDataset(CFG["image_dir"], CFG["mask_dir"],
                           all_ids[n_train:], CFG["img_size"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader
# ─────────────────────────────────────────────────────────────────


# ── Model ─────────────────────────────────────────────────────────
model = smp.Unet(
    encoder_name    = CFG["encoder"],
    encoder_weights = "imagenet",   # pretrained on ImageNet
    in_channels     = 3,
    classes         = 1,
    activation      = None,         # raw logits output
).to(DEVICE)

print(f"Model: U-Net + {CFG['encoder']} (pretrained)")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# ─────────────────────────────────────────────────────────────────


# ── Loss & Metrics ────────────────────────────────────────────────
bce_loss  = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss(mode="binary")

def loss_fn(logits, targets):
    return 0.5 * bce_loss(logits, targets) + 0.5 * dice_loss(logits, targets)


def batch_metrics(logits, targets, thr=0.5):
    preds = (logits.sigmoid() > thr).long()
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, targets.long(), mode="binary"
    )
    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    iou  = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    return dice.item(), iou.item()
# ─────────────────────────────────────────────────────────────────


# ── Optimizer & Scheduler ─────────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-6)
# ─────────────────────────────────────────────────────────────────


# ── Training Loop ─────────────────────────────────────────────────
def train_one_epoch(loader):
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0

    loop = tqdm(loader, desc="Train", leave=False)
    for images, masks, _ in loop:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        d, i = batch_metrics(logits.detach(), masks)
        total_loss += loss.item()
        total_dice += d
        total_iou  += i
        loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n


@torch.no_grad()
def validate(loader):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0

    for images, masks, _ in tqdm(loader, desc="Val  ", leave=False):
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        logits = model(images)
        loss   = loss_fn(logits, masks)

        d, i = batch_metrics(logits, masks)
        total_loss += loss.item()
        total_dice += d
        total_iou  += i

    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n
# ─────────────────────────────────────────────────────────────────


# ── Visualization ─────────────────────────────────────────────────
def save_overlay_grid(loader, save_path, n=8):
    model.eval()
    images_list, masks_list, preds_list, ids_list = [], [], [], []

    with torch.no_grad():
        for images, masks, img_ids in loader:
            logits = model(images.to(DEVICE))
            preds  = (logits.sigmoid() > CFG["threshold"]).float().cpu()
            images_list.extend(images)
            masks_list.extend(masks)
            preds_list.extend(preds)
            ids_list.extend(img_ids)
            if len(images_list) >= n:
                break

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    fig, axes = plt.subplots(n, 4, figsize=(16, n*4))
    for i in range(n):
        img  = (images_list[i] * std + mean).clamp(0,1).permute(1,2,0).numpy()
        gt   = masks_list[i].squeeze().numpy()
        pred = preds_list[i].squeeze().numpy()

        overlay = img.copy()
        tp = (pred==1) & (gt==1)
        fn = (pred==0) & (gt==1)
        fp = (pred==1) & (gt==0)
        overlay[tp] = overlay[tp]*0.5 + np.array([0,1,0])*0.5
        overlay[fn] = overlay[fn]*0.5 + np.array([1,0,0])*0.5
        overlay[fp] = overlay[fp]*0.5 + np.array([0,0,1])*0.5

        for j, data in enumerate([img, gt, pred, overlay]):
            ax = axes[i][j]
            ax.imshow(data, cmap="gray" if j in [1,2] else None, vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(["Original","Ground Truth",
                               "Prediction","Overlay"][j],
                             fontsize=13, fontweight="bold")

    legend = [
        mpatches.Patch(color="green", label="True positive"),
        mpatches.Patch(color="red",   label="False negative"),
        mpatches.Patch(color="blue",  label="False positive"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Overlay grid saved: {save_path}")
# ─────────────────────────────────────────────────────────────────


# ── Main ──────────────────────────────────────────────────────────
def main():
    os.makedirs(CFG["output_dir"], exist_ok=True)

    train_loader, val_loader = get_dataloaders()

    best_dice = 0.0
    history   = []
    ckpt_path = os.path.join(CFG["output_dir"], "best_model.pth")

    print(f"\nTraining U-Net + {CFG['encoder']} for {CFG['epochs']} epochs...\n")

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(train_loader)
        val_loss,   val_dice,   val_iou   = validate(val_loader)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{CFG['epochs']} | "
              f"Train loss: {train_loss:.4f}  Dice: {train_dice:.4f}  IoU: {train_iou:.4f} | "
              f"Val loss: {val_loss:.4f}  Dice: {val_dice:.4f}  IoU: {val_iou:.4f} | "
              f"{elapsed:.1f}s")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_dice":     val_dice,
                "val_iou":      val_iou,
                "encoder":      CFG["encoder"],
            }, ckpt_path)
            print(f"  ✓ Best model saved (Dice: {best_dice:.4f})")

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss, "train_dice": train_dice, "train_iou": train_iou,
            "val_loss":   val_loss,   "val_dice":   val_dice,   "val_iou":   val_iou,
        })

    # Save metrics
    with open(os.path.join(CFG["output_dir"], "metrics.json"), "w") as f:
        json.dump({"best_dice": best_dice, "history": history}, f, indent=2)

    # Plot training curves
    epochs_list = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes,
        ["loss", "dice", "iou"],
        ["Loss", "Dice Score", "IoU Score"]):
        ax.plot(epochs_list, [h[f"train_{key}"] for h in history], label="Train")
        ax.plot(epochs_list, [h[f"val_{key}"]   for h in history], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"U-Net + ResNet34 | Best Val Dice: {best_dice:.4f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["output_dir"], "training_curves.png"), dpi=120)
    plt.close()
    print("Training curves saved.")

    # Save overlay grid
    save_overlay_grid(
        val_loader,
        os.path.join(CFG["output_dir"], "overlay_grid.png")
    )

    print(f"\n{'='*50}")
    print(f"  DONE! Best Val Dice: {best_dice:.4f}")
    print(f"  Encoder: ResNet34 (pretrained ImageNet)")
    print(f"{'='*50}")


main()
