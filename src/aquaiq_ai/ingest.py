import os
import uuid
import re
import time
from itertools import count

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

from src.aqua_project.embeddings import AzureEmbedder

load_dotenv()

# Three dirname calls because this file is inside src/aqua_project/
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_PATH = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))

# Kept overlap at 2 sentences so nothing falls through the cracks
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
OVERLAP_SENTENCES = int(os.getenv("RAG_CHUNK_OVERLAP_SENTENCES", "2"))

print("Loading embedder...")
embedder = AzureEmbedder()

print("Setting up ChromaDB...")
chroma = chromadb.Client(Settings(persist_directory=DB_PATH, is_persistent=True))

# Checks if database is already present or not
existing = None
try:
    existing = chroma.get_collection("water_rag")
except:
    pass

if existing is not None:
    count = existing.count()
    print(f"\nDATABASE ALREADY EXISTS with {count} chunks!")
    print("   To re-ingest, delete the 'chroma_db' folder manually and run again.")
    print("\nExiting without making changes.")
    import sys

    sys.exit(0)
print("No existing database found. Creating new collection...")
collection = chroma.create_collection(name="water_rag")
print("Created new collection")


def load_pdfs():
    docs = []

    if not os.path.exists(DATA_PATH):
        print(f"Data folder not found: {DATA_PATH}")
        return []

    files = [f for f in os.listdir(DATA_PATH) if not f.startswith(".")]
    print(f"Found {len(files)} file(s)")

    for file in files:
        file_path = os.path.join(DATA_PATH, file)
        print(f"Reading: {file}")

        try:
            if file.endswith(".pdf"):
                reader = PdfReader(file_path)
                full_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

                if full_text.strip():
                    docs.append({
                        "name": file,
                        "text": full_text,
                        "pages": len(reader.pages)
                    })
                    print(f"Loaded {len(reader.pages)} pages, {len(full_text)} chars")
                else:
                    print(f"Warning: No text extracted - might be scanned doc")
            else:
                print(f"Skipping non-PDF: {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return docs


def semantic_chunking(text):
    # Split on periods, question marks, exclamation marks
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        # If adding this sentence would exceed chunk size
        if current_len + sent_len > CHUNK_SIZE and current:
            chunks.append(' '.join(current))

            # Keep last N sentences for overlap
            overlap = min(OVERLAP_SENTENCES, len(current))
            current = current[-overlap:] if overlap > 0 else []
            current_len = sum(len(s) for s in current)

        current.append(sent)
        current_len += sent_len

    # Dont forget the last chunk
    if current:
        chunks.append(' '.join(current))

    return chunks


def process_batch(batch_items):
    # This will runs in parallel across multiple threads.
    texts = [item["text"] for item in batch_items]

    try:
        embeddings = embedder.embed_batch(texts)
    except Exception as e:
        print(f" Batch embedding failed: {e}")
        return [], [], [], []

    ids = [str(uuid.uuid4()) for _ in batch_items]

    metadatas = []
    for item in batch_items:
        metadatas.append({
            "source": item["source"],
            "chunk_index": item["idx"],
            "total_chunks": item["total"],
            "chunk_size": len(item["text"])
        })

    return texts, embeddings, ids, metadatas


def run_ingestion():
    # main ingestion function
    print("\n" + "=" * 50)
    print("STARTING INGESTION")
    print("=" * 50)

    docs = load_pdfs()

    if not docs:
        print("No PDFs found. Add some to the 'data' folder and try again.")
        return

    total_chunks = 0
    batch_size = 20
    max_workers = 5

    for doc in docs:
        print(f"\nProcessing: {doc['name']}")

        chunks = semantic_chunking(doc['text'])
        total_chunks += len(chunks)

        print(f"  Created {len(chunks)} chunks")
        if chunks:
            avg_size = sum(len(c) for c in chunks) // len(chunks)
            print(f"  Avg chunk size: {avg_size} chars")

        # Prepare batch items
        batch_items = []
        for idx, chunk_text in enumerate(chunks):
            batch_items.append({
                "text": chunk_text,
                "source": doc['name'],
                "idx": idx,
                "total": len(chunks)
            })

        # Spliting into batches
        batches = [batch_items[i:i + batch_size] for i in range(0, len(batch_items), batch_size)]

        # Processing in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, b) for b in batches]

            for i, future in enumerate(as_completed(futures), 1):
                texts, embeddings, ids, metadatas = future.result()

                if texts and embeddings:
                    collection.add(
                        documents=texts,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                    print(f"Batch {i}/{len(batches)} done")
                else:
                    print(f"Batch {i}/{len(batches)} failed")

    print("\n" + "=" * 50)
    print(f"DONE! Total chunks stored: {total_chunks}")
    print(f"Database location: {DB_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    # Create data folder if it doesn't exist
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created folder: {DATA_PATH}")
        print("Please add your PDF files there and run this script again.")
    else:
        run_ingestion()