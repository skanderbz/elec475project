# src/check_model.py

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

from config import (
    COCO_VAL_IMAGES,
    VAL_CACHE_FILE,
)

# --------------------------------------------------------
# Model Definition (same as training)
# --------------------------------------------------------

class ImageEncoderCLIP(torch.nn.Module):
    def __init__(self, embed_dim=512, proj_hidden_dim=1024):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone_out_dim = backbone.fc.in_features

        modules = list(backbone.children())[:-1]
        self.backbone = torch.nn.Sequential(*modules)

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(self.backbone_out_dim, proj_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(proj_hidden_dim, embed_dim),
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = feats.squeeze(-1).squeeze(-1)
        emb = self.proj(feats)
        return F.normalize(emb, dim=-1)


# --------------------------------------------------------
# CLIP Image Transform
# --------------------------------------------------------

CLIP_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
])


# --------------------------------------------------------
# Main Function
# --------------------------------------------------------

def check_model(model_path, top_k=5, img_path_override=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Load model -----
    model = ImageEncoderCLIP()
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"\nLoaded model checkpoint: {model_path}")

    # ----- Load caption cache -----
    cache = torch.load(VAL_CACHE_FILE, map_location="cpu")
    image_files = cache["image_files"]
    captions = cache["captions"]

    # move text embeddings to the same device
    text_embs = F.normalize(cache["text_embs"], dim=-1).to(device)

    # ----- Choose image (random OR user-specified) -----

    if img_path_override is None:
        # random val image
        idx = random.randint(0, len(image_files) - 1)
        img_file = image_files[idx]
        img_path = COCO_VAL_IMAGES / img_file
        print(f"\nRandom validation image: {img_file}")
    else:
        # use provided path
        img_path = Path(img_path_override)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        print(f"\nUsing custom image: {img_path}")

    # Load + transform
    img = Image.open(img_path).convert("RGB")
    img_tensor = CLIP_IMAGE_TRANSFORM(img).unsqueeze(0).to(device)

    # ----- Encode image -----
    with torch.no_grad():
        img_emb = model(img_tensor)
        img_emb = F.normalize(img_emb, dim=-1)

    # ----- Similarity with all captions -----
    sims = (img_emb @ text_embs.T).squeeze(0)
    topk = torch.topk(sims, k=top_k)

    # ----- Print predictions -----
    print("\nTop captions predicted by the model:\n")
    for score, cap_idx in zip(topk.values, topk.indices):
        print(f"  • {captions[cap_idx]}     (score={score.item():.3f})")

    # ----- Show image with captions -----
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.axis("off")

    # Draw captions on the image
    overlay_text = ""
    for score, cap_idx in zip(topk.values, topk.indices):
        overlay_text += f"{captions[cap_idx]}  (score={score.item():.3f})\n"

    # Add text box
    plt.text(
        5, 5,
        overlay_text,
        fontsize=10,
        color="white",
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.6, pad=5),
        wrap=True,
    )

    plt.title("Predicted Captions")
    plt.show()


# --------------------------------------------------------
# CLI
# --------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check model predictions")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pth model file")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--img", type=str, default=None,
                        help="Optional: path to a specific image")
    args = parser.parse_args()

    check_model(args.model, args.topk, args.img)
