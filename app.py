import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import streamlit as st
import segmentation_models_pytorch as smp

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Lesion Segmentation",
    page_icon="🔬",
    layout="wide",
)
# ─────────────────────────────────────────────────────────────────


# ── Load Model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    ).to(device)
    ckpt = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device, ckpt["val_dice"], ckpt["epoch"]

model, DEVICE, val_dice, epoch = load_model()
# ─────────────────────────────────────────────────────────────────


# ── Preprocessing ─────────────────────────────────────────────────
IMG_SIZE   = 256
preprocess = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])
# ─────────────────────────────────────────────────────────────────


# ── Inference ─────────────────────────────────────────────────────
@torch.no_grad()
def segment(image: Image.Image, threshold: float):
    original_size = image.size
    image = image.convert("RGB")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    prob   = logits.sigmoid()[0, 0].cpu().numpy()

    mask     = (prob > threshold).astype(np.uint8)
    mask_img = Image.fromarray(mask * 255).resize(original_size, Image.NEAREST)

    img_np  = np.array(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    overlay = img_np.copy()
    overlay[mask == 1] = 0.5 * overlay[mask == 1] + 0.5 * np.array([0, 1, 0])
    overlay_img = Image.fromarray(
        (overlay * 255).astype(np.uint8)
    ).resize(original_size)

    lesion_pct = mask.mean() * 100
    return mask_img, overlay_img, lesion_pct
# ─────────────────────────────────────────────────────────────────


# ── UI ────────────────────────────────────────────────────────────
st.title("🔬 Skin Lesion Segmentation")
st.markdown(
    f"**U-Net + ResNet34** pretrained encoder · "
    f"Trained on ISIC 2018 · "
    f"Val Dice: **{val_dice:.4f}** (epoch {epoch})"
)
st.markdown(
    "> ⚠️ This is a research demonstration — not a clinical diagnostic tool."
)
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader(
        "Upload a dermoscopy image",
        type=["jpg", "jpeg", "png"],
    )
    threshold = st.slider(
        "Threshold",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Lower = more sensitive, detects more area. Higher = more conservative."
    )
    run_btn = st.button("Run Segmentation", type="primary", use_container_width=True)

with col2:
    if uploaded and run_btn:
        image = Image.open(uploaded).convert("RGB")

        with st.spinner("Segmenting lesion..."):
            mask_img, overlay_img, lesion_pct = segment(image, threshold)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(image,       caption="Original",       use_column_width=True)
        with c2:
            st.image(mask_img,    caption="Predicted Mask", use_column_width=True)
        with c3:
            st.image(overlay_img, caption="Overlay",        use_column_width=True)

        st.divider()

        # Report
        status = "⚠️ Large lesion detected" if lesion_pct > 20 else "✅ Small/localized lesion"
        r1, r2, r3 = st.columns(3)
        r1.metric("Lesion Area", f"{lesion_pct:.1f}%")
        r2.metric("Threshold",   f"{threshold}")
        r3.metric("Status",      status)

        st.info(
            "**Green overlay** = predicted lesion region. "
            "Adjust the threshold slider to control sensitivity."
        )

    elif not uploaded:
        st.info("Upload a dermoscopy image on the left to get started.")

st.divider()
st.markdown(
    """
    #### How it works
    - **Model:** U-Net with ResNet34 encoder pretrained on ImageNet
    - **Dataset:** ISIC 2018 Task 1 — 2,594 dermoscopic images
    - **Loss:** BCE + Dice loss combined
    - **Performance:** Val Dice 0.8902 | Val IoU 0.8044

    **Author:** NOVEM (Min Thu Kyaw) · Medical AI · Konyang University ·
    [GitHub](https://github.com/minthukyaw488-commits/skin-lesion-segmentation)
    """
)
