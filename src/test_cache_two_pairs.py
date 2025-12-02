import torch
from pathlib import Path
import random
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CACHE = PROJECT_ROOT / "data" / "coco2014" / "cache" / "train_text_cache.pt"

print("Loading cache:", TRAIN_CACHE)
cache = torch.load(TRAIN_CACHE, map_location="cpu")

image_files = cache["image_files"]   # list[str]
captions    = cache["captions"]      # list[str]
embs        = cache["text_embs"]     # Tensor[N, 512] already normalized

print(f"Total entries in cache: {len(image_files)}")
print("Embeddings shape:", embs.shape)

# Build mapping: image_file -> list of indices
by_image = defaultdict(list)
for idx, fname in enumerate(image_files):
    by_image[fname].append(idx)

# Find an image with at least 2 captions
candidates = [f for f, idxs in by_image.items() if len(idxs) >= 2]
if not candidates:
    print("No image has multiple captions in this subset, try rebuilding cache with more samples.")
    exit()

img_file = random.choice(candidates)
idxs = by_image[img_file]
i1, i2 = random.sample(idxs, 2)

print("\n=== Same-image caption pair ===")
print("Image file:", img_file)
print("Caption 1:", captions[i1])
print("Caption 2:", captions[i2])

e1 = embs[i1].unsqueeze(0)  # [1, 512]
e2 = embs[i2].unsqueeze(0)  # [1, 512]
same_sim = (e1 @ e2.T).item()
print("Cosine similarity (same image):", round(same_sim, 4))

# Now pick two random captions from different images
all_indices = list(range(len(image_files)))

# ensure different-image pair
while True:
    j1, j2 = random.sample(all_indices, 2)
    if image_files[j1] != image_files[j2]:
        break

print("\n=== Different-image caption pair ===")
print("Image 1 file:", image_files[j1])
print("Caption 1:", captions[j1])
print("Image 2 file:", image_files[j2])
print("Caption 2:", captions[j2])

f1 = embs[j1].unsqueeze(0)
f2 = embs[j2].unsqueeze(0)
diff_sim = (f1 @ f2.T).item()
print("Cosine similarity (different images):", round(diff_sim, 4))

print("\n=== Summary ===")
print("Same-image similarity:     ", round(same_sim, 4))
print("Different-image similarity:", round(diff_sim, 4))
print("\n(For a good cache: same-image > different-image, usually by a decent margin.)")
