import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# -------- Paths --------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DB_PATH = os.path.join(BASE_DIR, "db")

# -------- Azure Client --------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# -------- Chroma --------
chroma = chromadb.Client(
    Settings(
        persist_directory=DB_PATH,
        is_persistent=True
    )
)

collection = chroma.get_or_create_collection(name="water_rag")


# -------- Keyword Search (simple) --------
def keyword_search(query, documents, top_k=2):
    scores = []

    query_words = set(query.lower().split())

    for doc in documents:
        doc_words = set(doc.lower().split())
        score = len(query_words & doc_words)  # overlap count
        scores.append(score)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked[:top_k]]


# -------- Hybrid Retrieve --------
def retrieve(query, k=4):
    try:
        # 🔹 Step 1: Vector search
        embedding = client.embeddings.create(
            model=os.getenv("AZURE_OPENAI_EMBEDDING"),
            input=query
        ).data[0].embedding

        results = collection.query(
            query_embeddings=[embedding],
            n_results=k
        )

        vector_docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not vector_docs:
            return "No relevant context found."

        # 🔹 Step 2: Keyword re-ranking
        keyword_docs = keyword_search(query, vector_docs, top_k=2)

        # 🔹 Step 3: Merge results
        final_docs = list(dict.fromkeys(keyword_docs + vector_docs))

        # 🔹 Step 4: Format output
        formatted = []
        for doc, meta in zip(final_docs, metadatas):
            source = meta.get("source", "unknown")
            formatted.append(f"[Source: {source}]\n{doc[:400]}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Retrieval error: {str(e)}"
