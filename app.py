import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import get_model

CHECKPOINT = Path(__file__).parent / "results" / "best_model.pth"
IMG_SIZE = 256

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = get_model(device)
checkpoint_info = "no checkpoint found — using untrained weights"
if CHECKPOINT.exists():
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    checkpoint_info = (f"checkpoint from epoch {ckpt['epoch']} "
                       f"(Val Dice: {ckpt['val_dice']:.4f})")
model.eval()

preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def segment(image, threshold):
    if image is None:
        return None, None

    original_size = image.size
    resized = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    x = preprocess(resized).unsqueeze(0).to(device)
    prob = model(x)[0, 0].cpu().numpy()

    mask = (prob > threshold).astype(np.uint8)
    mask_img = Image.fromarray(mask * 255).resize(original_size, Image.NEAREST)

    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    mask_full = np.array(mask_img) > 127
    overlay = rgb.copy()
    overlay[mask_full] = 0.5 * overlay[mask_full] + 0.5 * np.array([0, 1, 0])
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    return mask_img, overlay_img


demo = gr.Interface(
    fn=segment,
    inputs=[
        gr.Image(type="pil", label="Dermoscopic Image"),
        gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="Predicted Mask"),
        gr.Image(type="pil", label="Overlay"),
    ],
    title="Skin Lesion Segmentation (U-Net)",
    description=(
        "Upload a dermoscopic image to segment the lesion. "
        "Trained on ISIC 2018 Task 1 — Val Dice 0.7804. "
        f"Running on {device} with {checkpoint_info}."
    ),
)


if __name__ == "__main__":
    demo.launch()
