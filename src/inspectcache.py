import torch

CACHE_PATH = "../data/coco2014/cache/train_text_cache.pt"

cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)

print("CACHE TYPE:", type(cache))
print("CACHE KEYS:", cache.keys())
print()

for k, v in cache.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: Tensor shape = {v.shape}")
    else:
        print(f"{k}: type = {type(v)} length = {len(v)}")
