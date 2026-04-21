import os
from dotenv import load_dotenv
import chromadb
from openai import AzureOpenAI
load_dotenv()
client = AzureOpenAI(
   api_version="2024-12-01-preview",
   azure_endpoint="https://cds-ds-openai-001-x.openai.azure.com/",
   api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
   name="aquaiq_ai_corpus",
   metadata={"hnsw:space": "cosine"},
)

def retrieve(query: str, top_k: int = 4) -> list[dict]:
   """
   Find the most relevant document chunks for a user's question.
   Args:
       query:  The user's question as a string
               Example: "What is the safe pH range for drinking water?"
       top_k:  How many chunks to return (default = 4)
               Why 4? It gives enough context without overwhelming GPT.
               Testing showed scores drop below 0.5 after the 4th result,
               meaning extra chunks add noise more than useful context.
   Returns:
       A list of dictionaries, each containing:
       - "text":   the actual chunk text from the PDF
       - "source": which PDF file the chunk came from
       - "score":  relevance score from 0.0 to 1.0 (higher = more relevant)
       Sorted by relevance (most relevant first)
   Example return value:
       [
           {
               "text": "The WHO recommends a pH range of 6.5 to 8.5...",
               "source": "who_water_guidelines.pdf",
               "score": 0.89
           },
           {
               "text": "pH affects the effectiveness of chlorination...",
               "source": "epa_water_treatment.pdf",
               "score": 0.76
           },
           ...
       ]
   """
   response = client.embeddings.create(
       model="text-embedding-3-small",
       input=[query],
   )
   query_vector = response.data[0].embedding
   results = collection.query(
       query_embeddings=[query_vector],
       n_results=top_k,
       include=["documents", "metadatas", "distances"],
   )
   chunks = []
   for doc, meta, distance in zip(
       results["documents"][0],   # list of chunk texts
       results["metadatas"][0],   # list of metadata dicts
       results["distances"][0],   # list of distance values
   ):
       similarity = round(1 - distance, 3)
       chunks.append({
           "text": doc,
           "source": meta.get("source", "unknown"),
           "score": similarity,
       })
   return chunks