import os
import uuid
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_PATH = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "db")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

chroma = chromadb.Client(Settings(persist_directory=DB_PATH))
collection = chroma.get_or_create_collection(name="water_rag")


def load_docs():
    docs = []
    files = os.listdir(DATA_PATH)

    print(f"Found {len(files)} files")

    for file in files:
        if file.startswith("."):
            continue

        path = os.path.join(DATA_PATH, file)
        print(f"Reading: {file}")

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append((file, f.read()))

        elif file.endswith(".pdf"):
            reader = PdfReader(path)
            text = ""
            for p in reader.pages:
                text += p.extract_text() or ""
            docs.append((file, text))

    return docs


def chunk(text, size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        ch = text[start:end]

        if len(ch.strip()) > 50:
            chunks.append(ch)

        start += size - overlap

    return chunks


def embed_batch(texts):
    response = client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING"),
        input=texts
    )
    return [item.embedding for item in response.data]


def process_batch(batch):
    embeddings = embed_batch(batch)
    ids = [str(uuid.uuid4()) for _ in batch]
    return batch, embeddings, ids


def run():
    if collection.count() > 0:
        print("Data already exists. Skipping ingestion.")
        return

    docs = load_docs()

    batch_size = 20
    max_workers = 5
    total_chunks = 0

    for filename, text in docs:
        print(f"\nProcessing file: {filename}")

        chunks = chunk(text)
        total_chunks += len(chunks)

        print(f"Chunks: {len(chunks)}")

        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, b) for b in batches]

            for i, future in enumerate(as_completed(futures), 1):
                docs_batch, embeds_batch, ids_batch = future.result()

                collection.add(
                    documents=docs_batch,
                    embeddings=embeds_batch,
                    metadatas=[{"source": filename}] * len(docs_batch),
                    ids=ids_batch
                )

                print(f"Completed batch {i}/{len(batches)}")

    print("\nIngestion complete")
    print(f"Total chunks stored: {total_chunks}")


if __name__ == "__main__":
    run()