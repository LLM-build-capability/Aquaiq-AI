import os
import time
from functools import lru_cache
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

class AzureEmbedder:
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