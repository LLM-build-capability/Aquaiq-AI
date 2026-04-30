# Design and Implementation of the Water Treatment AI Assistant
 
## 1. Purpose
 
This document explains the key design choices, trade offs, control flow, limitations, and future improvements for the Water Treatment AI Assistant. The assistant answers user questions by either searching PDF documents (RAG), calling an external water quality API, or combining both.
 
---
 
## 2. Key Design Decisions
 
### 2.1 Chunking Strategy
 
**Decision:** Split text into chunks at sentence boundaries (periods, question marks, exclamation marks). Target chunk size: 800 characters. Overlap: the last 2 sentences of the previous chunk are repeated in the next chunk.
 
**Reason:**  
- Cutting text at a fixed character count often breaks a sentence in half, making the chunk hard to understand.  
- Keeping whole sentences preserves meaning.  
- Overlap ensures that information that falls near a chunk boundary appears in two chunks, reducing the chance of missing it during retrieval.
 
### 2.2 Number of Chunks to Retrieve (k)
 
**Decision:** Retrieve the top 5 most similar chunks for each user query.
 
**Reason:**  
- With only 3 chunks, relevant information is sometimes spread across more chunks and gets missed.  
- With 10 chunks, the LLM receives too much text and may produce less accurate answers.  
- Testing with the provided PDFs (EPA guidelines, water treatment manuals) showed that 5 chunks gave the best balance between completeness and conciseness.
 
### 2.3 How Retrieved Chunks Are Given to the LLM
 
**Decision:** Insert the chunks as a system message with a short instruction:
 
> “Here is information from the documents. Use it if it helps.”
 
**Reason:**  
- System messages are separate from the conversation history and do not confuse the user.  
- The instruction is permissive (“if it helps”), so the LLM can still use its own knowledge when the retrieved text is not relevant.
 
### 2.4 When the Agent Decides to Call the Tool
 
**Decision:** The agent classifies each query into one of three types - rag, tool, or both - by comparing the query’s embedding with pre defined example questions.
 
**How it works:**  
- At startup, the agent creates embeddings for a set of example questions that represent RAG queries (like “What is chlorination?”) and tool queries (like “Water quality in Travis County”).  
- For a new user question, the agent computes its embedding and measures cosine similarity to the example embeddings.  
- If the average similarity to RAG examples is higher than to tool examples (and the difference is at least 0.08), the query is classified as rag.  
- If tool similarity is higher, it is classified as tool.  
- If the two averages are very close (difference less than 0.08), the query is classified as both.
 
**Reason for this approach:**  
- It is fast and requires no extra API call.  
- It is fully local and deterministic.  
- It works well for a focused domain like water treatment and water quality.
 
**Why a threshold of 0.08?**  
This value was found by testing several ambiguous queries (e.g., “Tell me about water quality and treatment”) and choosing a value that consistently resulted in `both` when both sources were relevant.
 
---
 
## 3. Trade offs Made
 
| Area | Choice | Why I Accepted This Trade off                                                                                                                                 |
|------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Vector database** | Local ChromaDB | As per the instructions we are using ChromaDB. ChromaDB is simple, persistent, and works offline.                                                             |
| **Chunking method** | Sentence boundary chunks | Slightly slower than fixed size chunking, but much better retrieval accuracy.                                                                                 |
| **Similarity aggregation** | Average of all example similarities | Using the maximum similarity could misclassify if a query accidentally matches one example. Average is more stable.                                           |
| **County support** | Hardcoded list of 8 counties | Dynamic lookup would require another API call (geocoding) and more code. For a demo, hardcoding is acceptable.                                                |
| **USGS API format** | CSV instead of JSON | The JSON endpoint gave 406 errors. CSV is reliable and easy to parse.                                                                                         |
| **Query expansion** | Simple keyword based expansion | A full synonym database or LLM based expansion would be heavier. The current expansion improves recall for common terms (“clean” → “treatment purification”). |
 
---
 
## 4. Control Flow: How RAG and Tool Calling Are Combined
 
The agent’s chat() function follows this sequence:
 
1. **Add user message** to the conversation history.
2. **Classify the query** (rag / tool / both) using embedding similarity.
3. **If classification is rag or both** and the document retriever is available:
   - Retrieve relevant chunks from ChromaDB.
   - If chunks are found, add them as a system message.
4. **If classification is tool or both**, the agent will include the tool definition (get_water_quality) in the next LLM call.
5. **Enter the tool call loop** (max 3 iterations):
   - Call the LLM with the current messages and optionally the tool definition.
   - If the LLM returns a tool_call, execute the tool (call the USGS API) and append the result as a tool role message.
   - Continue the loop to let the LLM produce the final answer.
   - If the LLM returns a normal answer (no tool call), return that answer to the user.
 
### Example of a combined query (RAG + tool)
 
> **User:** “According to EPA guidelines, is the current nitrate level in Williamson County safe?”
 
- The query is classified as both (similarity to RAG and tool examples is close).
- The agent retrieves a chunk from the PDF that states the EPA nitrate limit and adds it as system context.
- In the first LLM call, the model sees the question, the EPA limit, and the tool. It decides to call get_water_quality with county = “Williamson County Texas” and characteristic = “Nitrogen”.
- The agent calls the USGS API and receives a value. It appends this result as a tool message.
- In the second LLM call, the model combines the EPA limit and the real data to answer: “The EPA limit is 10mg/L. The current level is 3.2mg/L, which is safe.”
 
---
 
## 5. Known Limitations
 
| Limitation | Why It Happens | What Would Fix It |
|------------|----------------|-------------------|
| **Only works for the example question types** | The agent uses fixed example questions for classification. New domains would need new examples. | Replace with LLM based classification or use a larger, more diverse set of examples. |
| **Only 8 counties are supported** | County names are hardcoded to FIPS codes in tools.py. | Add a geocoding API (like Nominatim) to convert any US county name on the fly. |
| **Adding new PDFs requires deleting the database** | ChromaDB collection is recreated from scratch each time ingestion runs. | Implement incremental ingestion: check for new files and embed only the new chunks. |
| **Only one external tool** | The agent only defines the water quality tool. | The tools list can be extended with more tool schemas without changing the core loop. |
| **Conversation resets when the Streamlit app restarts** | Messages are stored only in memory. | Save conversation history to a file or a lightweight database (SQLite). |
| **Query expansion is basic** | It only handles a few hand written synonyms. | Use a small thesaurus API or a local NLP library for better expansion. |
 
---
 
## 6. Future Enhancements
 
### 6.1 Short term
 
- **Add more tests** – Cover edge cases like empty queries, missing API keys, or malformed PDFs.
- **Replace print statements with logging** – Use Pythons logging module for better control over output.
- **Add a second tool** – For example, a weather API to show how multiple tools can be integrated.
 
### 6.2 Medium term
 
- **Dynamic county lookup** – Use the USGS domain lookup service or a geocoding API to support any US county.
- **Hybrid search** – Combine vector similarity with keyword matching to improve retrieval for rare terms.
- **Persistent conversations** – Store chat history in a file or database so users can return to previous sessions.
 
### 6.3 Longterm
 
- **Containerise the application** – Write a Dockerfile to package the agent for easy deployment.
- **Deploy on a cloud platform** – Use Render or any other platform to make the assistant accessible online (as required in Exercise 2).
- **Add monitoring and tracing** – Use OpenTelemetry to track each agent step (classification, retrieval, tool call) for debugging.
 
---
 
## 7. Conclusion
 
The Water Treatment AI Assistant meets assignment requirements: it performs RAG over PDF documents, calls a public API, and decides when to use each source. The design choices (sentence boundary chunking, embedding based classification, explicit tool call loop) are deliberate and well justified. The known limitations are understood, and a clear path for future improvements is outlined.

