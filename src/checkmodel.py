# src/check_model_cpu.py
#
# Simple qualitative check:
# - loads your trained image encoder (.pth)
# - loads COCO val text cache
# - picks a random val image
# - prints top-k captions
# - shows the image
#
# CPU ONLY – no CUDA at all.

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

from config import (
    COCO_VAL_IMAGES,
    VAL_CACHE_FILE,
)

print("=== RUNNING check_model_cpu.py (CPU ONLY) ===")

# --------------------------------------------------------
# Model Definition (same as training)
# --------------------------------------------------------

class ImageEncoderCLIP(nn.Module):
    def __init__(self, embed_dim=512, proj_hidden_dim=1024):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone_out_dim = backbone.fc.in_features

        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        self.proj = nn.Sequential(
            nn.Linear(self.backbone_out_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, embed_dim),
        )

    def forward(self, x):
        feats = self.backbone(x)              # [B, 2048, 1, 1]
        feats = feats.squeeze(-1).squeeze(-1) # [B, 2048]
        emb = self.proj(feats)                # [B, 512]
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
# Main Function (pure CPU)
# --------------------------------------------------------

def check_model_cpu(model_path, top_k=5):
    print("Using device: CPU only")

    # ----- Load model on CPU -----
    model = ImageEncoderCLIP()
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    # model stays on CPU (no .to())
    model.eval()

    print(f"\nLoaded model checkpoint: {model_path}")

    # ----- Load caption cache on CPU -----
    cache = torch.load(VAL_CACHE_FILE, map_location="cpu")
    image_files = cache["image_files"]
    captions = cache["captions"]
    text_embs = cache["text_embs"]          # [N, 512] on CPU
    text_embs = F.normalize(text_embs, dim=-1)

    print("text_embs device:", text_embs.device)

    # ----- Choose a random image -----
    idx = random.randint(0, len(image_files) - 1)
    img_file = image_files[idx]
    img_path = COCO_VAL_IMAGES / img_file

    print(f"\nRandom validation image: {img_file}")

    # Load + transform
    img = Image.open(img_path).convert("RGB")
    img_tensor = CLIP_IMAGE_TRANSFORM(img).unsqueeze(0)  # stays on CPU

    print("img_tensor device:", img_tensor.device)

    # ----- Encode image -----
    with torch.no_grad():
        img_emb = model(img_tensor)          # [1, 512] on CPU
        img_emb = F.normalize(img_emb, dim=-1)

    print("img_emb device:", img_emb.device)

    # ----- Similarity with all captions (CPU matmul) -----
    sims = (img_emb @ text_embs.T).squeeze(0)   # [N]

    topk = torch.topk(sims, k=top_k)

    # ----- Print predictions -----
    print("\nTop captions predicted by the model:\n")
    for score, cap_idx in zip(topk.values, topk.indices):
        print(f"  • {captions[cap_idx]}     (score={score.item():.3f})")

    # ----- Show image -----
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title("Random validation image")
    plt.axis("off")
    plt.show()


# --------------------------------------------------------
# CLI
# --------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check model predictions (CPU)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pth model file")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    check_model_cpu(args.model, args.topk)
