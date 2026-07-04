import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

from model import get_model
from dataset import get_dataloaders
from losses import dice_score, iou_score


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(image_tensor):
    image = image_tensor.cpu() * STD + MEAN
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def make_overlay(image, pred_mask, gt_mask):
    overlay = image.copy()
    tp = (pred_mask == 1) & (gt_mask == 1)
    fn = (pred_mask == 0) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)

    overlay[tp] = [0, 1, 0]
    overlay[fn] = [1, 0, 0]
    overlay[fp] = [0, 0, 1]
    return overlay


@torch.no_grad()
def evaluate(model, loader, device, num_samples=6):
    model.eval()
    total_dice, total_iou = 0, 0
    samples = []

    for images, masks, img_ids in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        total_dice += dice_score(preds, masks).item()
        total_iou += iou_score(preds, masks).item()

        for i in range(images.size(0)):
            if len(samples) >= num_samples:
                continue
            samples.append((
                images[i],
                masks[i, 0].cpu().numpy(),
                (preds[i, 0].cpu().numpy() > 0.5).astype(np.float32),
                img_ids[i],
            ))

    n = len(loader)
    return total_dice / n, total_iou / n, samples


def save_overlay_grid(samples, out_path):
    rows = len(samples)
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes[None, :]

    col_titles = ["Original", "Ground Truth", "Prediction", "Overlay"]
    for row, (image, gt_mask, pred_mask, img_id) in enumerate(samples):
        rgb = denormalize(image)
        overlay = make_overlay(rgb, pred_mask, gt_mask)

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(gt_mask, cmap="gray")
        axes[row, 2].imshow(pred_mask, cmap="gray")
        axes[row, 3].imshow(overlay)
        axes[row, 0].set_ylabel(img_id, fontsize=8)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col])

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved overlay grid to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../results/best_model.pth")
    parser.add_argument("--data_root", default="../data")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--out", default="../results/overlay_grid.png")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    _, val_loader = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )

    model = get_model(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(Val Dice: {checkpoint['val_dice']:.4f})")

    val_dice, val_iou, samples = evaluate(model, val_loader, device, args.num_samples)
    print(f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

    save_overlay_grid(samples, args.out)


if __name__ == "__main__":
    main()
