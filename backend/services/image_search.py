from typing import List, Dict, Any
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
from .utils import cosine_topk

class ImageSearch:
    def __init__(self, products: List[Dict[str, Any]]):
        self.products = products
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._emb = None

    def _embed_image(self, img: Image.Image):
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feats = self.clip_model.get_image_features(**inputs)
        v = feats.detach().cpu().numpy()
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)
        return v

    def _ensure_catalog_emb(self):
        if self._emb is not None:
            return
        vecs = []
        for p in self.products:
            try:
                resp = requests.get(p["image_url"], timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                v = self._embed_image(img)[0]
            except Exception:
                v = np.zeros((512,), dtype="float32")
            vecs.append(v)
        self._emb = np.vstack(vecs)

    def search_by_image_url(self, image_url: str, top_k: int = 8):
        self._ensure_catalog_emb()
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        q = self._embed_image(img)[0]
        idxs, _ = cosine_topk(self._emb, q, top_k)
        return [self.products[i] for i in idxs]

try:
    import torch
except Exception:
    pass
