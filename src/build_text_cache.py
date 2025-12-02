# src/build_text_cache.py

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel

from config import (
    COCO_TRAIN_CAPTIONS,
    COCO_VAL_CAPTIONS,
    COCO_TRAIN_IMAGES,
    COCO_VAL_IMAGES,
    TRAIN_CACHE_FILE,
    VAL_CACHE_FILE,
    CLIP_MODEL_NAME,
    MAX_TRAIN_SAMPLES,
    MAX_VAL_SAMPLES,
)

def build_cache(captions_file: Path, images_root: Path, cache_file: Path, max_samples=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Building cache for {captions_file.name} on {device} ===")

    # ----- Load JSON -----
    with captions_file.open("r") as f:
        data = json.load(f)

    # map image_id -> file_name
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    annotations = data["annotations"]

    print("Total annotations in file:", len(annotations))
    if max_samples is not None:
        annotations = annotations[:max_samples]
        print("Using subset:", len(annotations))

    # ----- Init CLIP text encoder -----
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(
        CLIP_MODEL_NAME,
        use_safetensors=True,
    ).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    all_image_files = []
    all_captions = []
    all_embeddings = []

    batch_size = 256

    for i in tqdm(range(0, len(annotations), batch_size), desc="Encoding captions"):
        batch = annotations[i:i + batch_size]

        file_names = []
        caps = []

        for ann in batch:
            img_id = ann["image_id"]
            caption = ann["caption"]
            file_name = id_to_file.get(img_id)
            if file_name is None:
                continue

            img_path = images_root / file_name
            if not img_path.exists():
                continue

            file_names.append(file_name)
            caps.append(caption)

        if not caps:
            continue

        tokens = tokenizer(
            caps,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            # CLIP text features in the joint space, dim=512
            text_feats = clip_model.get_text_features(**tokens)
            text_feats = torch.nn.functional.normalize(text_feats, dim=-1)  # normalize rows

        all_image_files.extend(file_names)
        all_captions.extend(caps)
        all_embeddings.append(text_feats.cpu())

    if not all_embeddings:
        raise RuntimeError("No embeddings generated – check paths / captions file.")

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, 512]

    print("Final cache size:", all_embeddings.shape[0])

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_files": all_image_files,
            "captions": all_captions,
            "text_embs": all_embeddings,
        },
        cache_file,
    )

    print(f"Saved cache to: {cache_file}")


def main():
    build_cache(
        captions_file=COCO_TRAIN_CAPTIONS,
        images_root=COCO_TRAIN_IMAGES,
        cache_file=TRAIN_CACHE_FILE,
        max_samples=MAX_TRAIN_SAMPLES,
    )

    build_cache(
        captions_file=COCO_VAL_CAPTIONS,
        images_root=COCO_VAL_IMAGES,
        cache_file=VAL_CACHE_FILE,
        max_samples=MAX_VAL_SAMPLES,
    )


if __name__ == "__main__":
    main()
