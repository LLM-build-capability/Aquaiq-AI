"""
One-shot ingest for the Exercise C radar corpus.

Reads markdown files from exercise-c/rag-agent/data/, chunks and embeds them
using the same helpers as Exercise A (src/aquaiq_ai/), and writes to a
separate ChromaDB collection so the root chroma_db/ is never touched.

Run once before starting the RAG agent:
    make ingest
or manually:
    LLM_PROFILE=local python exercise-c/rag-agent/ingest_radar.py
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAG_DIR = os.path.dirname(os.path.abspath(__file__))
# aquaiq_ai/ sits alongside this script in rag-agent/; add rag-agent/ so it
# is importable as a top-level package.
sys.path.insert(0, RAG_DIR)
sys.path.insert(0, REPO_ROOT)

# Set env vars before importing Exercise A modules so they pick up the right
# paths. CHROMA_PERSIST_DIR as an absolute path causes os.path.join in
# ingest.py to ignore BASE_DIR (Python behaviour for absolute second arg).
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
os.environ.setdefault("LLM_PROFILE", "local")
os.environ.setdefault("CHROMA_PERSIST_DIR", CHROMA_DIR)
os.environ.setdefault("LOCAL_COLLECTION_NAME", "radar_local")

from dotenv import load_dotenv
load_dotenv(os.path.join(REPO_ROOT, ".env"))

import re
import chromadb
from chromadb.config import Settings

from aquaiq_ai.config import get_embedder, get_collection_name
from aquaiq_ai.embedding_helper import OllamaEmbedder

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Chunker — replicates the semantic_chunking logic from Exercise A's ingest.py
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_chunk_size: int = 500) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chunk_size and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 30]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
collection_name = get_collection_name()
db_path = os.environ["CHROMA_PERSIST_DIR"]

client = chromadb.Client(Settings(persist_directory=db_path, is_persistent=True))

# Skip if collection already exists and has data
try:
    existing = client.get_collection(collection_name)
    if existing.count() > 0:
        print(f"Collection '{collection_name}' already has {existing.count()} chunks — skipping.")
        print(f"Delete {db_path} to force re-ingest.")
        sys.exit(0)
    client.delete_collection(collection_name)
except Exception:
    pass

collection = client.create_collection(collection_name)
print(f"Created collection '{collection_name}' in {db_path}")

embedder = get_embedder()

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".md"))
if not files:
    print(f"No .md files found in {DATA_DIR}")
    sys.exit(1)

all_ids, all_embeddings, all_docs, all_metas = [], [], [], []

for filename in files:
    text = open(os.path.join(DATA_DIR, filename), encoding="utf-8").read()
    chunks = chunk_text(text)
    print(f"  {filename}: {len(chunks)} chunks")

    embeddings = embedder.embed_batch(chunks)
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        if emb is None:
            print(f"    chunk {i} dropped (embedding failed)")
            continue
        all_ids.append(f"{filename}-{i}")
        all_embeddings.append(emb)
        all_docs.append(chunk)
        all_metas.append({"source": filename, "chunk_index": i})

# Add in batches of 100
BATCH = 100
for start in range(0, len(all_ids), BATCH):
    end = start + BATCH
    collection.add(
        ids=all_ids[start:end],
        embeddings=all_embeddings[start:end],
        documents=all_docs[start:end],
        metadatas=all_metas[start:end],
    )
    print(f"  stored chunks {start}–{min(end, len(all_ids)) - 1}")

print(f"\nDone. {len(all_ids)} chunks in '{collection_name}'.")
