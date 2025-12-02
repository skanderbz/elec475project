from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

image = Image.open("cat.jpg").convert("RGB")
texts = ["a dog", "a cat", "a bowl of food"]

inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # [1, num_texts]
    probs = logits_per_image.softmax(dim=-1)

print("Probs:", probs)
