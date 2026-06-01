import os
import time
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()


class AzureEmbedder:
    """Cloud profile: Azure OpenAI text-embedding-3-small (1536 dims)."""

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model = os.getenv("AZURE_OPENAI_EMBEDDING")
        self.max_retries = 3

    def embed(self, text):
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
                    time.sleep(2)
                else:
                    raise
        return None

    def embed_batch(self, texts):
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


class LocalEmbedder:
    """Local profile: nomic-embed-text via Ollama (768 dims).

    Talks to Ollama through its OpenAI-compatible endpoint so the call
    shape matches AzureEmbedder exactly. Dimensions differ from cloud,
    so cloud and local must use separate ChromaDB collections.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
        self.model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.max_retries = 3

    def embed(self, text):
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[text]
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Local embedding failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    raise
        return None

    def embed_batch(self, texts):
        # Ollama's OpenAI-compatible embeddings endpoint accepts a list,
        # but throughput is single-threaded on the daemon side. Kept the
        # same interface as AzureEmbedder so callers don't change.
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"Local batch embedding failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    raise
        return []


def get_embedder():
    """Factory: returns the embedder for the active LLM_PROFILE.

    cloud -> AzureEmbedder (1536 dims)
    local -> LocalEmbedder (768 dims)
    """
    profile = os.getenv("LLM_PROFILE", "cloud").lower()
    if profile == "local":
        return LocalEmbedder()
    return AzureEmbedder()
