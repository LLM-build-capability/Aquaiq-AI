import chromadb

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection(name="rag_docs")


def get_embedding(text):
    return model.encode(text).tolist()


def retrieve(query, k=4):
    query_embedding = get_embedding(query)

    results = collection.query(

        query_embeddings=[query_embedding],

        n_results=k

    )

    docs = results["documents"][0]

    metas = results["metadatas"][0]

    combined = []

    for doc, meta in zip(docs, metas):
        combined.append({

            "text": doc,

            "source": meta.get("source")

        })

    return combined