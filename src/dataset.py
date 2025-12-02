# src/dataset.py

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    COCO_TRAIN_IMAGES,
    COCO_VAL_IMAGES,
    TRAIN_CACHE_FILE,
    VAL_CACHE_FILE,
)

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
    Returns:
        (image_tensor[3,224,224], text_embedding[512])
    """

    def __init__(self, cache_file: Path, image_root: Path):
        cache_file = Path(cache_file)
        image_root = Path(image_root)

        print(f"Loading cache: {cache_file}")
        data = torch.load(cache_file, map_location="cpu")

        self.image_files = data["image_files"]   # list[str]
        self.captions    = data["captions"]      # list[str]
        self.text_embs   = data["text_embs"]     # Tensor[N, 512]

        self.image_root = image_root
        self.transform = CLIP_IMAGE_TRANSFORM

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_root / self.image_files[idx]
        txt_emb = self.text_embs[idx]  # [512]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)  # [3, 224, 224]

        return img_tensor, txt_emb


# ----------------- Helper to build train/val datasets ----------------- #

def make_train_val_datasets():
    """
    Returns:
        train_dataset, val_dataset
    """
    train_ds = CocoClipDataset(
        cache_file=TRAIN_CACHE_FILE,
        image_root=COCO_TRAIN_IMAGES,
    )

    val_ds = CocoClipDataset(
        cache_file=VAL_CACHE_FILE,
        image_root=COCO_VAL_IMAGES,
    )

    return train_ds, val_ds
