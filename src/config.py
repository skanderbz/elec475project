# src/config.py
from pathlib import Path

# Project root = folder that contains `src/` and `data/`
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- COCO paths ----
COCO_ROOT = PROJECT_ROOT / "data" / "coco2014"

COCO_TRAIN_IMAGES = COCO_ROOT / "images" / "train2014"
COCO_VAL_IMAGES   = COCO_ROOT / "images" / "val2014"

COCO_TRAIN_CAPTIONS = COCO_ROOT / "annotations" / "captions_train2014.json"
COCO_VAL_CAPTIONS   = COCO_ROOT / "annotations" / "captions_val2014.json"

# ---- Cache paths ----
CACHE_DIR = COCO_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CACHE_FILE = CACHE_DIR / "train_text_cache.pt"
VAL_CACHE_FILE   = CACHE_DIR / "val_text_cache.pt"

# ---- CLIP model ----
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ---- Optional: use subsets for speed (set to None for full dataset) ----
MAX_TRAIN_SAMPLES = None   # or None
MAX_VAL_SAMPLES   = None   # or None
