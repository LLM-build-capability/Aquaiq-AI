"""Profile resolution for LLM_PROFILE=local|cloud.

Single source of truth for which LLM client, model, embedder, and ChromaDB
collection the rest of the code uses. Everything outside this module is
profile-agnostic.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_profile() -> str:
    profile = os.getenv("LLM_PROFILE", "cloud").strip().lower()
    if profile not in ("cloud", "local"):
        raise ValueError(
            f"LLM_PROFILE must be 'cloud' or 'local', got: {profile!r}"
        )
    return profile


def get_llm_client():
    """Return an OpenAI-compatible client for the active profile."""
    from openai import AzureOpenAI, OpenAI

    if get_profile() == "local":
        return OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )


def get_llm_model() -> str:
    if get_profile() == "local":
        return os.getenv("OLLAMA_LLM_MODEL", "gemma4:e4b")
    return os.getenv("AZURE_OPENAI_DEPLOYMENT")


def get_embedder():
    """Return an embedder with .embed(text) and .embed_batch(texts) methods."""
    from aquaiq_ai.embedding_helper import AzureEmbedder, OllamaEmbedder

    if get_profile() == "local":
        return OllamaEmbedder()
    return AzureEmbedder()


def get_collection_name() -> str:
    if get_profile() == "local":
        return os.getenv("LOCAL_COLLECTION_NAME", "water_rag_local")
    return os.getenv("CLOUD_COLLECTION_NAME", "water_rag")
