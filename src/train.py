# src/train.py

import argparse
from pathlib import Path
import time

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    PROJECT_ROOT,
    COCO_TRAIN_IMAGES,
    COCO_VAL_IMAGES,
    TRAIN_CACHE_FILE,
    VAL_CACHE_FILE,
)

MODELNAME = "5kS1kV10E128B"
SAMPLES = 5000
VALSAMPLES = 1000

# ----------------- CLIP Image Transform ----------------- #

CLIP_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


# ----------------- Dataset Class ----------------- #

class CocoClipDataset(Dataset):
    """
    Loads COCO images + precomputed CLIP text embeddings.
    Optionally takes a (random) subset for faster training.

    Returns:
        (image_tensor[3,224,224], text_embedding[512])
    """

    def __init__(
        self,
        cache_file: Path,
        image_root: Path,
        max_samples: int | None = None,
        random_subset: bool = True,
    ):
        cache_file = Path(cache_file)
        image_root = Path(image_root)

        print(f"Loading cache: {cache_file}")
        data = torch.load(cache_file, map_location="cpu")

        image_files = data["image_files"]   # list[str]
        captions    = data["captions"]      # list[str]
        text_embs   = data["text_embs"]     # Tensor[N, 512]

        N = len(image_files)
        if max_samples is not None and max_samples < N:
            if random_subset:
                # random subset of indices
                idx = torch.randperm(N)[:max_samples].tolist()
            else:
                # first max_samples (deterministic)
                idx = list(range(max_samples))

            image_files = [image_files[i] for i in idx]
            captions    = [captions[i] for i in idx]
            text_embs   = text_embs[idx]

        print(f"Dataset size: {len(image_files)} samples")

        self.image_files = image_files
        self.captions    = captions
        self.text_embs   = text_embs     # [M, 512] where M = subset size
        self.image_root  = image_root
        self.transform   = CLIP_IMAGE_TRANSFORM

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_root / self.image_files[idx]
        txt_emb = self.text_embs[idx]  # [512]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)  # [3, 224, 224]

        return img_tensor, txt_emb


# ----------------- Helper to build train/val datasets ----------------- #

def make_train_val_datasets(
    max_train_samples: int | None = SAMPLES,
    max_val_samples: int | None = VALSAMPLES,
):
    """
    Returns:
        train_dataset, val_dataset

    By default, uses a subset for faster training:
      - 10k train
      - 2k val
    Set max_*_samples=None to use the full split.
    """
    train_ds = CocoClipDataset(
        cache_file=TRAIN_CACHE_FILE,
        image_root=COCO_TRAIN_IMAGES,
        max_samples=max_train_samples,
        random_subset=True,     # random train subset
    )

    val_ds = CocoClipDataset(
        cache_file=VAL_CACHE_FILE,
        image_root=COCO_VAL_IMAGES,
        max_samples=max_val_samples,
        random_subset=False,    # deterministic val subset
    )

    return train_ds, val_ds


# ----------------- Model ----------------- #

class ImageEncoderCLIP(nn.Module):
    """
    ResNet50 backbone (ImageNet-pretrained) +
    2-layer projection head mapping to 512-dim CLIP space.
    """
    def __init__(self, embed_dim: int = 512, proj_hidden_dim: int = 1024):
        super().__init__()

        # ResNet50 backbone
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone_out_dim = backbone.fc.in_features  # 2048

        # Remove the original classification head (fc)
        modules = list(backbone.children())[:-1]  # keep everything up to avgpool
        self.backbone = nn.Sequential(*modules)

        # Projection head: Linear -> GELU -> Linear  (to 512-d CLIP space)
        self.proj = nn.Sequential(
            nn.Linear(self.backbone_out_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        feats = self.backbone(x)              # [B, 2048, 1, 1]
        feats = feats.squeeze(-1).squeeze(-1) # [B, 2048]
        emb = self.proj(feats)                # [B, 512]
        # L2-normalize to lie on unit sphere (CLIP style)
        emb = F.normalize(emb, dim=-1)
        return emb


# ------------- CLIP InfoNCE loss ------------- #

def clip_loss(image_embeds: torch.Tensor,
              text_embeds: torch.Tensor,
              temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE loss used in CLIP.

    image_embeds: [B, 512]
    text_embeds:  [B, 512]
    """
    # Ensure unit norm (should already be, but for safety)
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = image_embeds @ text_embeds.t()  # [B, B]
    logits = logits / temperature

    labels = torch.arange(image_embeds.size(0), device=image_embeds.device)

    loss_i = F.cross_entropy(logits, labels)       # image -> text
    loss_t = F.cross_entropy(logits.t(), labels)   # text -> image

    loss = (loss_i + loss_t) / 2.0
    return loss


# ------------- Training / Validation loops ------------- #

def train_one_epoch(model, dataloader, optimizer, device, temperature):
    model.train()
    running_loss = 0.0
    num_samples = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for images, text_embs in pbar:
        images = images.to(device, non_blocking=True)
        text_embs = text_embs.to(device, non_blocking=True)

        optimizer.zero_grad()
        img_embs = model(images)
        loss = clip_loss(img_embs, text_embs, temperature=temperature)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

        pbar.set_postfix(loss=running_loss / num_samples)

    return running_loss / num_samples


def validate(model, dataloader, device, temperature):
    model.eval()
    running_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Val", leave=False)
        for images, text_embs in pbar:
            images = images.to(device, non_blocking=True)
            text_embs = text_embs.to(device, non_blocking=True)

            img_embs = model(images)
            loss = clip_loss(img_embs, text_embs, temperature=temperature)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            pbar.set_postfix(loss=running_loss / num_samples)

    return running_loss / num_samples


# ----------------- Main script ----------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="ELEC 475 Lab 4 - Train ResNet50 image encoder for CLIP"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Datasets & loaders
    train_ds, val_ds = make_train_val_datasets()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    model = ImageEncoderCLIP(embed_dim=512, proj_hidden_dim=1024).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # for loss curves
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, args.temperature
        )
        val_loss = validate(
            model, val_loader, device, args.temperature
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}: "
            f"train_loss = {train_loss:.4f}, "
            f"val_loss = {val_loss:.4f}"
        )

        # Save best checkpoint (by val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / (MODELNAME + ".pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ↳ Saved new best checkpoint to {ckpt_path}")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.1f} minutes.")
    print(f"Best val loss: {best_val_loss:.4f}")

    # --------- Plot loss curves --------- #
    epochs = range(1, args.epochs + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CLIP Image Encoder Training")
    plt.legend()
    plt.grid(True)

    curve_path = plots_dir / (MODELNAME + '.jpg')
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()

    print(f"Saved loss curve to: {curve_path}")


if __name__ == "__main__":
    main()
