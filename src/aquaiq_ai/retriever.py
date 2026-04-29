import os
import sys
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from src.aquaiq_ai.embedding_helper import AzureEmbedder

# getting same path problem like ingest file. so added 3 dirname again.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
DB_PATH = os.path.join(BASE_DIR, os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))  # I increased from 3 to 5 for better results


class WaterDocRetriever:
    def __init__(self):
        self.embedder = AzureEmbedder()
        self.client = chromadb.Client(Settings(
            persist_directory=DB_PATH,
            is_persistent=True
        ))
        try:
            self.collection = self.client.get_collection("water_rag")
            num_chunks = self.collection.count()
            self.available = num_chunks > 0
            print(f"Retriever ready. Found {num_chunks} chunks in database.")
        except Exception as e:
            self.available = False
            print(f"Retriever error: {e}")
            print("Run ingest.py first to build the database.")

    def _expand_query(self, query):
        # Added some related terms
        query_lower = query.lower()

        # I added these after testing because the model kept missing certain terms
        expansions = {
            "clean": "treatment purification disinfection",
            "chlorine": "chlorination disinfection",
            "filter": "filtration membrane separation",
            "how": "process method steps procedure",
            "what": "explanation definition overview",
        }

        words_to_add = []
        for key, expansion in expansions.items():
            if key in query_lower:
                words_to_add.append(expansion)

        if words_to_add:
            return query + " " + " ".join(words_to_add)
        return query

    def get_context(self, query):
        # Get relevant text chunks for the user's question.
        if not self.available:
            return ""

        # Try expanding the query - helps a lot
        expanded = self._expand_query(query)
        if expanded != query:
            print(f"  Expanded query: {expanded[:100]}...")

        try:
            # Convert query to embedding
            query_vec = self.embedder.embed(expanded)

            # Search for similar chunks
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=TOP_K
            )
        except Exception as e:
            print(f"Search failed: {e}")
            return ""

        # Check if we got anything back
        if not results.get('documents') or not results['documents'][0]:
            print("  No results found.")
            return ""

        # Format the results
        context_parts = []
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            source = meta.get('source', 'Unknown')
            chunk_num = meta.get('chunk_index', '?')

            # Show where each chunk came from
            header = f"[From: {source} | Part {chunk_num + 1}]"
            context_parts.append(f"{header}\n{doc}\n")

        return "\n---\n".join(context_parts)