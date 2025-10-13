# Downloads & caches models so first query is fast
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Models cached.")
