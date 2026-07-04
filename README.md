# Skin Lesion Segmentation with U-Net

> Binary segmentation of dermoscopic skin lesion images using U-Net trained on the ISIC 2018 dataset.

---

## Results

| Metric | Value |
|--------|-------|
| **Best Val Dice (DSC)** | **0.7804** |
| **Best Val IoU (Jaccard)** | **0.7257** |
| Epochs trained | 10 |
| Training platform | Kaggle (T4 GPU) |

### Training Curves

![Training Curves](assets/training_curves.png)

Loss drops consistently across all 10 epochs. Both Dice and IoU scores improve steadily with no sign of overfitting — the train/val gap stays tight throughout, which shows the augmentation pipeline is working well.

### Prediction Overlay

![Overlay Grid](assets/overlay_grid.png)

Each row shows: **Original** | **Ground Truth** | **Prediction** | **Overlay**

Overlay legend:
- 🟢 **Green** = True positive (correctly segmented lesion)
- 🔴 **Red** = False negative (missed lesion area)
- 🔵 **Blue** = False positive (over-predicted boundary)

Most predictions closely follow the ground truth boundary. The model handles varied lesion shapes, sizes, and skin tones well. Some boundary imprecision visible on irregular lesions (rows 4 and 6) — expected at 10 epochs and improvable with longer training.

---

## Architecture: U-Net

```
Input (3 × 256 × 256)
      │
 [Encoder]
   DoubleConv 64  → MaxPool
   DoubleConv 128 → MaxPool
   DoubleConv 256 → MaxPool
   DoubleConv 512 → MaxPool
      │
 [Bottleneck]
   DoubleConv 1024
      │
 [Decoder] ← skip connections from encoder
   Upsample + Concat → DoubleConv 512
   Upsample + Concat → DoubleConv 256
   Upsample + Concat → DoubleConv 128
   Upsample + Concat → DoubleConv 64
      │
 1×1 Conv → Sigmoid
      │
Output (1 × 256 × 256) — binary mask probability map
```

**Skip connections** pass fine-grained spatial detail from the encoder directly to the decoder, preserving sharp lesion boundaries that would otherwise be lost during downsampling.

---

## Dataset: ISIC 2018 Task 1

- **Source:** [ISIC Archive 2018](https://challenge.isic-archive.com/data/#2018)
- **Images:** 2,594 dermoscopic RGB images (.jpg)
- **Masks:** Corresponding binary segmentation masks (.png)
- **Split:** 85% train / 15% val

---

## Loss Function

Combined **BCE + Dice Loss**:

```
Loss = 0.5 × BCE + 0.5 × Dice
```

- **BCE** handles pixel-level accuracy
- **Dice** handles overlap quality — critical for imbalanced masks where lesion pixels are far fewer than background pixels

---

## Setup

```bash
git clone https://github.com/minthukyaw488-commits/skin-lesion-segmentation
cd skin-lesion-segmentation
pip install -r requirements.txt
```

### Data structure

```
data/
├── images/
│   ├── ISIC_0000000.jpg
│   └── ...
└── masks/
    ├── ISIC_0000000_segmentation.png
    └── ...
```

Download from [ISIC 2018 Task 1](https://challenge.isic-archive.com/data/#2018).

---

## Training

```bash
cd src
python train.py
```

Best checkpoint saved automatically to `results/best_model.pth`.

For GPU training (recommended), use the included Kaggle notebook — trains 30 epochs in ~2.5 hours on a free T4 GPU.

## Evaluation

```bash
cd src
python evaluate.py --checkpoint ../results/best_model.pth
```

Generates `results/overlay_grid.png` with side-by-side visual comparisons.

## Demo

```bash
python app.py
```

Launches a Gradio web UI: upload a dermoscopic image to get the predicted lesion mask and a green overlay at the original resolution, with an adjustable probability threshold. The app loads `results/best_model.pth` if present — train first (or download a trained checkpoint) to get real predictions.

---

## Project Structure

```
skin-lesion-segmentation/
├── src/
│   ├── model.py       ← U-Net architecture
│   ├── dataset.py     ← ISIC dataset loader + augmentation
│   ├── losses.py      ← Dice loss, BCE+Dice, IoU metric
│   ├── train.py       ← training loop with checkpointing
│   └── evaluate.py    ← evaluation + overlay visualization
├── app.py             ← Gradio demo
├── assets/
│   ├── training_curves.png
│   └── overlay_grid.png
├── requirements.txt
└── README.md
```

---

## Key Concepts Learned

- **U-Net encoder–decoder** with skip connections for boundary preservation
- **Dice loss** vs BCE — why segmentation needs overlap-aware losses
- **Mask-consistent augmentation** — flipping/rotating image AND mask together
- **IoU and Dice** as evaluation metrics for segmentation tasks
- **Overlay visualization** to diagnose TP / FN / FP model behavior

---

## Author

**NOVEM (MIN THU KYAW)**
Medical AI · Konyang University, Daejeon, South Korea
