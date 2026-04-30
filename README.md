<h1 align="center">Water Treatment AI Assistant</h1>
 
<p align="center">
  <font color="gray" size="3">
    RAG‑powered chatbot with real‑time water quality API integration – built with Azure OpenAI, ChromaDB, and Streamlit
  </font>
</p>
 
**Authors:**
- Prem Kumar Reddy K
- Deepak P
- Pavitra P

**Cohort:** LLM Capability

---
## Table of Contents
 
- [About the Project](#about-the-project)
- [Project Objectives](#project-objectives)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Code File Descriptions](#code-file-descriptions)
- [Data Sources](#data-sources)
- [Agent Decision Logic](#agent-decision-logic)
- [Workflow Diagrams](#workflow-diagrams)
- [Installation Guide](#installation-guide)
- [Environment Variables](#environment-variables)
- [How to Run the Project](#how-to-run-the-project)
- [How to test the Project](#how-to-test-the-project)
- [Outputs Generated](#outputs-generated)
- [Results and Example Queries](#results-and-example-queries)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Acknowledgements](#acknowledgements)
 
---
 
## About the Project
 
The **Water Treatment AI Assistant** is an intelligent agent that answers user questions by combining:
 
- **Retrieval‑Augmented Generation (RAG)** – extracts information from uploaded PDF documents (e.g., EPA guidelines, treatment manuals)
- **Real‑time tool calling** – fetches live water quality monitoring data from the USGS Water Quality Portal
 
The agent automatically decides whether to search the documents, call the external API, or use **both** sources to produce accurate, evidence‑based answers. It supports multi‑turn conversations through a clean Streamlit chat interface.
 
---
 
## Project Objectives
 
- Implement a local, framework‑free agent using raw OpenAI function calling
- Index unstructured PDF documents with semantic chunking and store embeddings in ChromaDB
- Provide a public API tool (USGS water quality) that the agent can call on demand
- Enable the agent to reason when to use RAG, when to call the tool, or when to combine both
- Deliver a user‑friendly chat interface (Streamlit) with dark mode
 
---
 
## Tech Stack
 
| Component               | Technology                               | Why we use it                                     |
|------------------------|------------------------------------------|---------------------------------------------------|
| **Language**           | Python 3.11+                             | rich ecosystem for AI and data processing.       |
| **LLM + Tool Calling** | Azure OpenAI (GPT‑5.4‑nano) via `AzureOpenAI` client | Provided by the bootcamp; no orchestration frameworks allowed, raw SDK use. |
| **Embeddings**         | Azure OpenAI `text-embedding-3-small`    | Same Azure resource; consistent embedding dimension (1536) and semantic search quality. |
| **Vector Store**       | ChromaDB (local, persistent)             | Local, lightweight, persistent, and easy to reset – fits the “no cloud DB” requirement. |
| **Document Processing**| PyPDF + sentence‑boundary chunking       | PyPDF is simple and dependency‑free; sentence‑boundary chunking preserves meaning better than fixed‑length splitting. |
| **Interface**          | Streamlit (dark mode)                    | Fastest way to build an interactive chat UI with minimal boilerplate code. |
| **External API**       | USGS Water Quality Portal (CSV endpoint) | Public, free, REST API providing real‑world water monitoring data. |
| **Dependency Manager** | Poetry                                   | ensures reproducible environments and lock files. |
---
 
## Project Structure
```
aqua-project/
├── data/
│   └── *.pdf
├── images/
│   └── *.png
├── src/
│   └── aqua_project/
│       ├── __init__.py
│       ├── ingest.py
│       ├── retriever.py
│       ├── embedding_helper.py
│       ├── agent.py
│       ├── tools.py
├── tests/
│   ├── __init__.py
│   ├── complete_tests.py
├── docs/
│   └── writeup.md
├── application.py
├── chroma_db/     # Vector Database
├── .env.example
├── .gitignore
├── README.md
├── poetry.lock
└── myproject.toml

```
---
 
## Code File Descriptions
 
- **`ingest.py`** – Reads your PDF files, splits the text into chunks (without cutting sentences), creates vector embeddings using Azure OpenAI, and stores everything in a local database (ChromaDB).
 
- **`retriever.py`** – When you ask a question, this file converts your question into a vector, finds the most similar chunks from the database, and returns the relevant text with source information.
 
- **`embedding_helper.py`** – A helper that talks to Azure OpenAI to turn text into numbers (vectors). Used by both ingestion and retrieval.
 
- **`agent.py`** – The brain of the assistant. It decides whether to search the documents, call the water quality API, or do both. It also manages the conversation memory and handles tool calling.
 
- **`tools.py`** – Defines the water quality tool. It knows how to convert county names like “Travis County Texas” into FIPS codes, calls the USGS water quality API, and returns monitoring site data.
 
- **`application.py`** – The chat interface you see. It runs with Streamlit, shows the conversation, takes your questions, and displays the assistant’s answers.
 
## Data Sources
 
| Source | Description |
|--------|-------------|
| **PDF Documents** | User‑provided water treatment manuals, EPA guidelines, and technical reports (stored in `data/` folder) |
| **USGS Water Quality Portal** | Live monitoring site data (location, type, coordinates) for counties in Texas, Arkansas, Maryland, Oklahoma |
 
The tool currently supports 8 counties (Travis, Williamson, Harris, Dallas – Texas; Benton, Baxter – Arkansas; Prince George – Maryland; Oklahoma – Oklahoma).
 
---
 
## Agent Decision Logic
 
The agent uses **cosine similarity between query embeddings and pre‑defined example questions** to decide the action:
 
| Similarity | Decision | Action |
|------------|----------|--------|
| RAG examples score higher | `rag` | Retrieve from PDFs only |
| Tool examples score higher | `tool` | Call USGS API only |
| Scores close (<0.08) | `both` | Retrieve + call API |
 
Example questions are manually curated and embedded once at startup.
 
---
 
## Workflow Diagrams
 
### Overall System Workflow
 
```text
       [ User Input ]
             |
             v
      +--------------+
      | Streamlit UI |
      +--------------+
             |
             v
     +----------------+
     | Agent (agent.py)|
     +----------------+
             |
             v
    +------------------+
    | Query Embedding  |
    +------------------+
             |
             v
  +----------------------+
  | Similarity Comparison|
  +----------------------+
             |
             v
    /------------------\
   <      DECISION      >
    \------------------/
      /      |       \
     /       |        \
  [RAG]    [TOOL]    [BOTH]
    |        |          |
    v        v          v
+-------+ +-------+ +------------+
|Chroma | | USGS  | | Chroma +   |
|DB     | | API   | | USGS API   |
+-------+ +-------+ +------------+
    |        |          |
    \        |         /
     \       v        /
      +--------------+
      |    MERGE     |
      +--------------+
             |
             v
     +----------------+
     | Final Response |
     +----------------+
             |
             v
     +------------------+
     | Streamlit Output |
     +------------------+
```
 
This workflow shows how the system processes a user query and dynamically decides whether to use document retrieval (RAG), external API tools, or both.
 
---
 
### Agent Decision Workflow
 
```text
       [ User Query ]
             |
             v
   +--------------------+
   | Convert to         |
   | Embedding (Vector) |
   +--------------------+
             |
             v
   +--------------------+
   | Compare with:      |
   | - RAG Examples     |
   | - Tool Examples    |
   +--------------------+
             |
             v
   +--------------------+
   | Compute Similarity |
   | Scores             |
   +--------------------+
             |
             v
    /------------------\
   <   COMPARE SCORES   >
    \------------------/
      /      |       \
     /       |        \
    /        |         \
   v         v          v
+-------+ +--------+ +---------+
| RAG > | | Tool > | | Diff <  |
| Tool  | | RAG    | | 0.08    |
+-------+ +--------+ +---------+
    |        |          |
    v         v          v
[SELECT]  [SELECT]   [SELECT]
[ RAG  ]  [ TOOL ]   [ BOTH ]
    |        |          |
    \        |         /
     \       v        /
   +--------------------+
   |  Execute Selected  |
   |       Action       |
   +--------------------+
```
 
 
## Installation Guide
 
### Prerequisites
 
- Python 3.11 or 3.12
- Poetry (install via `pip install poetry` or https://python-poetry.org)
- Git
 
### Setup
 
1. **Clone the repository**
   ```bash
   git clone <repo_url>
   ```
      ```bash
   cd Aquaiq-AI
   ```
 
2. **Install dependencies with Poetry**
    ```bash
   poetry install
   ```
This creates a virtual environment and installs all required packages (see pyproject.toml).
 
3. **Activate the virtual environment**
    ```bash
   poetry shell
   ```
   **Adding new dependencies later**
   ```bash
   poetry add <package_name>          # e.g., poetry add numpy
   poetry add --dev <dev_package>     # for dev dependencies
   ```
After adding, update pyproject.toml and poetry.lock automatically.
 
## Environment Variables
Create a .env file in the project root (same level as pyproject.toml):
  ```bash
   # Azure OpenAI
     AZURE_OPENAI_API_KEY=your_actual_key
     AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
     API_VERSION=2024-12-01-preview
     AZURE_OPENAI_DEPLOYMENT=gpt-5.4-nano
     AZURE_OPENAI_EMBEDDING=text-embedding-3-small
 
   # RAG Settings
     RAG_CHUNK_SIZE=800
     RAG_CHUNK_OVERLAP_SENTENCES=2
     RAG_TOP_K=5
 
   # Vector DB
     CHROMA_PERSIST_DIR=./chroma_db
 
   # Agent
     MAX_TOOL_ITERATIONS=3
     LLM_TEMPERATURE=0.7
  ```
Never commit the .env file. Add it to .gitignore.
 
## How to Run the Project
1. Place your PDF files inside the data/ folder (create it if missing).
2. Ingest the documents (run once)
   ```bash
   poetry run python src/aquaq_AI/ingest.py
   ```
This will chunk the PDFs, create embeddings, and store them in ChromaDB.
 
3. Launch the Streamlit app
   ```bash
   poetry run streamlit run application.py
   ```
4. Interact – type your question in the chat box. The agent will automatically decide whether to search documents, call the water quality API, or do both.

## How to test the Project
   ```bash
   poetry run python tests/complete_tests.py
   ```
* This tests file have comprehensive test suite (no API calls except one optional chat test)
* This file will tests embeddings, county codes, API connectivity, ChromaDB, Chucking, and agent routing.
## Outputs Generated
 
After ingestion, the following are created:
 
- chroma_db/ – persistent vector database (do not delete manually unless re‑ingesting)
- Console logs showing chunk counts and embedding progress
 
During a chat session, the assistant prints internal classification steps (RAG similarity, tool similarity) to the terminal for debugging.
 
## Results and Example Queries
### RAG Query Example:
<p align='center'>
 <img src="images/RAG query.png" width ="800"/>
</p>

- The agent answers from the PDF documents without calling an external API.

### Tool Qery Example:
<p align='center'>
 <img src="images/API tool query.png" width ="800"/>
</p>

- The agent calls the USGS water quality API and returns real monitoring site data.

### Agent Decision Logs (Terminal):
<p align='center'>
 <img src="images/CMD output.png" width ="800"/>
</p>

- The terminal shows similarity scores and the final query type classification for debugging.

## Limitations
- The agent’s routing relies on a fixed set of example questions; new domains may require updating the examples.
- County support in the water quality tool is hardcoded (8 counties). Dynamic lookup would require additional API integration.
- ChromaDB collection is not automatically updated when new PDFs are added – ingestion must be re‑run (after deleting the old database).
- The system currently uses only one external API; multiple tools would need manual schema addition.
## Future Enhancements
- Dynamic county lookup – automatically convert any US county name to FIPS code using a geocoding API.
- Add more tools – e.g., weather API, chemical safety database, real‑time sensor data.
- Hybrid search – combine vector similarity with keyword (BM25) for better retrieval.
- Web deployment – containerise with Docker and deploy on a free tier (Fly.io, Render).
- Observability – integrate logging and monitoring for production readiness.
## Acknowledgements
- USGS Water Quality Portal for open access to water quality monitoring data
- ChromaDB and Streamlit open‑source communities
- Python ecosystem (PyPDF, NumPy, pandas, requests, etc.)
- bootcamp leader for guidance and infrastructure support
