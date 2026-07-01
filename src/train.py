import os
import json
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model   import get_model
from dataset import get_dataloaders
from losses  import BCEDiceLoss, dice_score, iou_score


CONFIG = {
    "data_root":   "../data",
    "img_size":    256,
    "batch_size":  4,
    "num_epochs":  10,
    "lr":          1e-4,
    "val_split":   0.15,
    "checkpoint":  "../results/best_model.pth",
    "metrics_log": "../results/metrics.json",
}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0

    loop = tqdm(loader, desc="Train", leave=False)
    for images, masks, _ in loop:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(preds.detach(), masks).item()
        total_iou  += iou_score(preds.detach(), masks).item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0

    for images, masks, _ in tqdm(loader, desc="Val  ", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        preds = model(images)
        loss  = criterion(preds, masks)

        total_loss += loss.item()
        total_dice += dice_score(preds, masks).item()
        total_iou  += iou_score(preds, masks).item()

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    os.makedirs("../results", exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        data_root  = CONFIG["data_root"],
        img_size   = CONFIG["img_size"],
        batch_size = CONFIG["batch_size"],
        val_split  = CONFIG["val_split"],
    )

    model     = get_model(device)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_dice = 0.0
    history   = []

    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...\n")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        t0 = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device)

        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{CONFIG['num_epochs']} | "
              f"Train loss: {train_loss:.4f}  Dice: {train_dice:.4f}  IoU: {train_iou:.4f} | "
              f"Val loss: {val_loss:.4f}  Dice: {val_dice:.4f}  IoU: {val_iou:.4f} | "
              f"{elapsed:.1f}s")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_dice":    val_dice,
                "val_iou":     val_iou,
            }, CONFIG["checkpoint"])
            print(f"  ✓ Best model saved (Dice: {best_dice:.4f})")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_dice": train_dice, "train_iou": train_iou,
            "val_loss":   val_loss,   "val_dice":   val_dice,   "val_iou":   val_iou,
        })

    with open(CONFIG["metrics_log"], "w") as f:
        json.dump({"best_dice": best_dice, "history": history}, f, indent=2)

    print(f"\nDone! Best Val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()