
from typing import List, Dict, Any, Tuple
import io, requests
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from .utils import cosine_sim

class ImageSearcher:
    def __init__(self, products: List[Dict[str, Any]], image_root: str):
        self.products = products
        self.root = image_root.rstrip('/')
        self.model = SentenceTransformer("clip-ViT-B-32")
        self._emb = None
        self._paths = [self._to_path(p['image']) for p in products]
        self._build_index()

    def _to_path(self, rel: str) -> str:
        # rel like "/images/bag1.jpg" -> file path under image_root
        rel = rel.lstrip('/')
        # expected images prefix
        if rel.startswith('images/'):
            rel = rel[len('images/'):]
        return f"{self.root}/{rel}"

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def _embed_images(self, imgs: List[Image.Image]) -> np.ndarray:
        return self.model.encode(imgs, batch_size=8, convert_to_numpy=True, normalize_embeddings=True)

    def _build_index(self):
        imgs = []
        for p in self._paths:
            try:
                imgs.append(self._load_image(p))
            except Exception:
                imgs.append(Image.new('RGB',(224,224),(200,200,200)))
        self._emb = self._embed_images(imgs)

    def search_by_url(self, url: str, k: int = 8) -> List[Dict[str, Any]]:
        # accept remote or local file URL
        try:
            if url.startswith('http://') or url.startswith('https://'):
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert('RGB')
            else:
                img = Image.open(url).convert('RGB')
        except Exception:
            # fallback: return first k
            return self.products[:k]
        qv = self._embed_images([img])
        sims = cosine_sim(qv, self._emb)[0]
        order = np.argsort(-sims)[:k]
        return [self.products[i] for i in order]
