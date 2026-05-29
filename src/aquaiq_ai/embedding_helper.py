import os
import time
from functools import lru_cache
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()

# I tried to make this a simple wrapper but then added retry logic because Azure is hitting rate limits

class AzureEmbedder:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model = os.getenv("AZURE_OPENAI_EMBEDDING")
        # 3 retries seems to work most of the time
        self.max_retries = 3

    def embed(self, text):
        # Just one text at a time, simpler this way
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[text]
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Embedding failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)  # Wait a bit before retry
                else:
                    raise
        return None

    def embed_batch(self, texts):
        # Multiple texts at once - faster for ingestion
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"Batch embedding failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    raise
        return []


class OllamaEmbedder:
    """Local embedder via Ollama's OpenAI-compatible /v1/embeddings endpoint.

    Mirrors the AzureEmbedder API (embed, embed_batch) so callers don't change.
    Default model `nomic-embed-text` returns 768-dim vectors — keep this
    collection separate from any 1536-dim Azure-built collection.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
        self.model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.max_retries = 3

    def embed(self, text):
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[text],
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Ollama embedding failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    raise
        return None

    def embed_batch(self, texts):
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"Ollama batch embedding failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
        # Per-item fallback: at least one chunk in the batch likely exceeds the
        # embedder's context window. Try each chunk alone — return None for the
        # ones that still fail so the caller can drop them and keep the rest.
        results = []
        for t in texts:
            try:
                response = self.client.embeddings.create(model=self.model, input=[t])
                results.append(response.data[0].embedding)
            except Exception as e:
                print(f"  Per-item embed failed (chunk dropped): {str(e)[:120]}")
                results.append(None)
        return results