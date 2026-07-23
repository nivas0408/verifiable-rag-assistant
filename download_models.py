import os
from pathlib import Path

# Force caching path to match settings
os.environ["HF_HOME"] = str(Path("./data/hf_cache").resolve())

print("Pre-downloading all-MiniLM-L6-v2 model weights to local cache...")
try:
    from sentence_transformers import SentenceTransformer
    # Initialize and download model to cache
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Successfully downloaded and cached embedding model to: {os.environ['HF_HOME']}")
except Exception as e:
    print(f"Error pre-downloading model weights: {e}")
    # Do not crash the build, print error
