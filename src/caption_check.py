import json
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Path setup
# --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOT_FILE = PROJECT_ROOT / "data" / "coco2014" / "annotations" / "captions_val2014.json"
IMAGES_ROOT = PROJECT_ROOT / "data" / "coco2014" / "images" / "val2014"

# --------------------------------------------------------
# Load JSON
# --------------------------------------------------------
print("Loading:", ANNOT_FILE)

with open(ANNOT_FILE, "r") as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}  # map image_id → image info
annotations = data["annotations"]                    # list of caption entries

print("Loaded images:", len(images))
print("Loaded annotations:", len(annotations))

# --------------------------------------------------------
# Select a random image
# --------------------------------------------------------
random_image_id = random.choice(list(images.keys()))
img_info = images[random_image_id]

file_name = img_info["file_name"]
img_path = IMAGES_ROOT / file_name

print("\nRandom image:")
print("ID:", random_image_id)
print("Filename:", file_name)

if not img_path.exists():
    print("❌ ERROR — image file not found:", img_path)
    exit()

# --------------------------------------------------------
# Collect all captions for this image
# --------------------------------------------------------
captions = [ann["caption"] for ann in annotations if ann["image_id"] == random_image_id]

print("\nCaptions for this image:")
for c in captions:
    print(" -", c)

# --------------------------------------------------------
# Show image with first caption as title
# --------------------------------------------------------
img = Image.open(img_path)

plt.imshow(img)
plt.title(captions[0] if captions else "(No captions found)")
plt.axis("off")
plt.show()
