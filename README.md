<h1 align="center">Water Treatment AI Assistant</h1>
 
<p align="center">
  <font color="gray" size="3">
    RAG‑powered chatbot with real‑time water quality API integration – built with Azure OpenAI, ChromaDB, and Streamlit · Local mode via Ollama + Gemma
  </font>
</p>

> **Two profiles, one switch.** Set `LLM_PROFILE=cloud` to run against Azure OpenAI + cloud embeddings, or `LLM_PROFILE=local` to run fully offline against Ollama (`gemma4:e4b` + `nomic-embed-text`). The two profiles use **separate ChromaDB collections** (1536‑dim vs 768‑dim) and must be ingested independently.
 
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
 
| Component               | Cloud profile                            | Local profile                                    | Why we use it                                     |
|------------------------|------------------------------------------|--------------------------------------------------|---------------------------------------------------|
| **Language**           | Python 3.11+                             | Python 3.11+                                     | rich ecosystem for AI and data processing.       |
| **LLM + Tool Calling** | Azure OpenAI (GPT‑5.4‑nano) via `AzureOpenAI` | Ollama (`gemma4:e4b`) via OpenAI‑compatible `/v1` endpoint at `http://localhost:11434` | Both speak the OpenAI Chat Completions API, so only client construction changes. |
| **Embeddings**         | Azure OpenAI `text-embedding-3-small` (1536‑dim) | Ollama `nomic-embed-text` (768‑dim)        | Local embedder runs offline; dimension differs, so collections must not be mixed. |
| **Vector Store**       | ChromaDB collection `water_rag`          | ChromaDB collection `water_rag_local`            | One collection per profile to keep dimensions consistent; same on‑disk store. |
| **Document Processing**| PyPDF + sentence‑boundary chunking       | PyPDF + sentence‑boundary chunking               | PyPDF is simple and dependency‑free; sentence‑boundary chunking preserves meaning better than fixed‑length splitting. |
| **Interface**          | Streamlit (dark mode)                    | Streamlit (dark mode)                            | Fastest way to build an interactive chat UI with minimal boilerplate code. |
| **External API**       | USGS Water Quality Portal (CSV endpoint) | USGS Water Quality Portal (CSV endpoint)         | Public, free REST API; identical for both profiles. |
| **Dependency Manager** | Poetry                                   | Poetry                                           | ensures reproducible environments and lock files. |
---
 
## Project Structure
```
Aquaiq-AI/
├── data/
│   ├── *.pdf                       # source corpus
│   └── comparison-results/         # benchmark outputs (gitignored except rubric-scores.csv)
├── images/
│   └── *.png
├── src/
│   └── aquaiq_ai/
│       ├── __init__.py
│       ├── config.py               # LLM_PROFILE switch — single source of truth
│       ├── ingest.py
│       ├── retriever.py
│       ├── embedding_helper.py     # AzureEmbedder + OllamaEmbedder
│       ├── agent.py
│       └── tools.py
├── scripts/
│   ├── smoke_test.py               # one-shot canned query against the active profile
│   ├── queries.json                # 20-query benchmark prompt set
│   ├── benchmark.py                # runs the active profile over queries.json
│   ├── score.py                    # generates rubric-template.csv from raw results
│   └── aggregate.py                # cross-profile p50/p95 + correctness summary
├── tests/
│   ├── __init__.py
│   └── complete_tests.py
├── docs/
│   ├── writeup.md
│   ├── transcripts/                # one combined-query transcript per profile
│   ├── local-mode-comparison.md    # cloud vs local report
│   └── when-to-go-local.md         # decision doc
├── application.py
├── chroma_db/                      # ChromaDB store (water_rag + water_rag_local)
├── .env.example
├── .gitignore
├── README.md
├── poetry.lock
└── pyproject.toml

```
---
 
## Code File Descriptions
 
- **`config.py`** – Single source of truth for the active profile. Reads `LLM_PROFILE` and exposes `get_llm_client()`, `get_llm_model()`, `get_embedder()`, and `get_collection_name()`. Every other module asks `config.py` for its dependencies, so flipping profiles is one env var, not a code change.

- **`ingest.py`** – Reads your PDF files, splits the text into chunks (without cutting sentences), creates vector embeddings via the active profile's embedder, and stores them in ChromaDB. Refuses to overwrite an existing collection, so cloud and local ingests stay isolated.
 
- **`retriever.py`** – When you ask a question, converts your question into a vector using the active profile's embedder, finds the most similar chunks from the profile's collection, and returns the relevant text with source information.
 
- **`embedding_helper.py`** – Embedding clients. `AzureEmbedder` talks to Azure OpenAI (cloud, 1536‑dim). `OllamaEmbedder` talks to Ollama's OpenAI‑compatible `/v1/embeddings` endpoint (local, 768‑dim) and includes a per‑item fallback for chunks that exceed `nomic-embed-text`'s 8K‑token context window.
 
- **`agent.py`** – The brain of the assistant. It decides whether to search the documents, call the water quality API, or do both. It also manages the conversation memory and handles tool calling. Profile‑agnostic — receives its LLM client and embedder from `config.py`.
 
- **`tools.py`** – Defines the water quality tool. It knows how to convert county names like "Travis County Texas" into FIPS codes, calls the USGS water quality API, and returns monitoring site data.
 
- **`application.py`** – The chat interface you see. It runs with Streamlit, shows the conversation, takes your questions, and displays the assistant's answers.

### Scripts (cloud vs local benchmarking)

- **`scripts/smoke_test.py`** – One‑shot canned query against whichever profile `LLM_PROFILE` points at. Useful sanity check after switching profiles or re‑ingesting.
- **`scripts/queries.json`** – Fixed 20‑query prompt set (8 RAG, 6 tool, 4 hybrid, 2 out‑of‑scope) used for both profiles.
- **`scripts/benchmark.py`** – Runs the active profile over `queries.json`. Captures latency, routing similarities, tool‑call success, retrieved sources, and per‑query token counts (via a `UsageRecorder` context manager — no edits to `agent.py`). Writes raw JSON + appends to `summary.csv`.
- **`scripts/score.py`** – Reads the latest cloud + local result JSONs and produces `rubric-template.csv` for offline rubric scoring.
- **`scripts/aggregate.py`** – Cross‑profile aggregation: p50/p95 latency, tokens/sec, per‑category correctness, retrieval‑source overlap.
 
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
 
### Prerequisites (both profiles)
 
- Python 3.11, 3.12, or 3.13 (3.14 has known Poetry issues; if forced onto 3.14, reuse the existing venv `aquaiq-ai-p3ifKeiw-py3.14` rather than reinstalling)
- Poetry (install via `pip install poetry` or https://python-poetry.org)
- Git

### Additional prerequisites for the **local** profile

- **Ollama** ≥ 0.3.0 — install the macOS app from [ollama.com/download](https://ollama.com/download) (or `brew install ollama` on supported platforms). The daemon listens on `http://localhost:11434`.
- **Local models** — pulled once:
  ```bash
  ollama pull nomic-embed-text       # 274 MB — local embedder (768‑dim)
  ollama pull gemma3:e4b             # ~9.6 GB — local LLM
  # Or, if the public registry is blocked, sideload from the AI Fest mirror
  # (see Session 3 in CLAUDE.md for the bundle + manifest merge procedure).
  ```
- ~12 GB free disk space for the models. Model weights live under `~/.ollama/` and **must not be committed** — `.gitignore` already excludes them.
 
### Setup
 
1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd Aquaiq-AI
   ```
 
2. **Install dependencies with Poetry**
    ```bash
   poetry install
   ```
This creates a virtual environment and installs all required packages (see `pyproject.toml`).
 
3. **Activate the virtual environment**
    ```bash
   poetry shell
   ```
   **Adding new dependencies later**
   ```bash
   poetry add <package_name>          # e.g., poetry add numpy
   poetry add --dev <dev_package>     # for dev dependencies
   ```
After adding, update `pyproject.toml` and `poetry.lock` automatically.

4. **Copy and fill `.env`**
   ```bash
   cp .env.example .env
   ```
   Edit it per your chosen profile (see [Environment Variables](#environment-variables)).
 
## Environment Variables
Create a `.env` file in the project root (same level as `pyproject.toml`). Only one variable controls which profile runs: **`LLM_PROFILE=cloud`** or **`LLM_PROFILE=local`**.

```bash
# ── Profile switch (required) ─────────────────────────────────
LLM_PROFILE=cloud                        # cloud | local

# ── Cloud profile (Azure OpenAI) ──────────────────────────────
AZURE_OPENAI_API_KEY=your_actual_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-5.4-nano
AZURE_OPENAI_EMBEDDING=text-embedding-3-small
CLOUD_COLLECTION_NAME=water_rag

# ── Local profile (Ollama) ────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_LLM_MODEL=gemma3:e4b              # or fest-agent if sideloaded
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
LOCAL_COLLECTION_NAME=water_rag_local

# ── RAG Settings ──────────────────────────────────────────────
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP_SENTENCES=2
RAG_TOP_K=5

# ── Vector DB ─────────────────────────────────────────────────
CHROMA_PERSIST_DIR=./chroma_db

# ── Agent ─────────────────────────────────────────────────────
MAX_TOOL_ITERATIONS=3
LLM_TEMPERATURE=0.7
```

**Never commit the `.env` file.** It is already in `.gitignore`. Use `.env.example` as the canonical template.

> **Why two collections?** Cloud embeddings are 1536‑dim, local embeddings are 768‑dim. Mixing them silently corrupts retrieval. Each profile gets its own ChromaDB collection and the two coexist on disk.
 
## How to Run the Project

Place your PDF files inside the `data/` folder (create it if missing). The rest depends on which profile you want.

### Cloud profile (Azure OpenAI)

```bash
# 1. Set the profile
echo "LLM_PROFILE=cloud" >> .env

# 2. Ingest into the cloud collection (water_rag, 1536‑dim) — run once
poetry run python src/aquaiq_ai/ingest.py

# 3. Smoke‑test
poetry run python scripts/smoke_test.py

# 4. Launch the chat UI
poetry run streamlit run application.py
```

### Local profile (Ollama + Gemma)

```bash
# 0. Make sure the Ollama daemon is running
ollama serve &     # or: launch the Ollama menu‑bar app

# 1. Set the profile
echo "LLM_PROFILE=local" >> .env

# 2. Ingest into the local collection (water_rag_local, 768‑dim) — run once.
#    This is independent of the cloud ingest; both collections coexist.
poetry run python src/aquaiq_ai/ingest.py

# 3. Smoke‑test
poetry run python scripts/smoke_test.py

# 4. Launch the chat UI (same UI, different brain)
poetry run streamlit run application.py
```

### Switching profiles after first install

Once both collections are populated, switching profiles is **one line in `.env`** — no re‑ingest, no code changes. The app and the benchmarking scripts all read `LLM_PROFILE` at startup.

> **First run on local profile is slow.** ChromaDB lazily loads the collection and Ollama lazily loads model weights into memory. The first query after a cold start can take 30+ seconds; subsequent queries settle into the steady‑state latency reported in the benchmark.

## How to test the Project

### Unit / integration tests (profile‑agnostic)
```bash
poetry run python tests/complete_tests.py
```
Comprehensive test suite (no API calls except one optional chat test). Covers embeddings, county codes, API connectivity, ChromaDB, chunking, and agent routing.

### Single‑query smoke test (active profile)
```bash
poetry run python scripts/smoke_test.py
```
Fires one canned query against whichever profile `.env` points at. Prints answer + latency.

### 20‑query benchmark (one profile at a time)
```bash
LLM_PROFILE=cloud poetry run python scripts/benchmark.py
LLM_PROFILE=local poetry run python scripts/benchmark.py
```
Each run writes a structured JSON to `data/comparison-results/results-{profile}-{timestamp}.json` and appends a row to `summary.csv`. Per‑query timeout is 120 s; no retries.

### Cross‑profile aggregation
```bash
poetry run python scripts/aggregate.py
```
Reads the most recent cloud + local result JSONs plus `rubric-scores.csv` and prints p50/p95 latency, tokens/sec, per‑category correctness, and retrieval‑source overlap.

### Hardware used for the reported benchmark
- **Machine:** MacBook (Apple Silicon, macOS 26 / Tahoe)
- **Ollama:** v0.3.x, daemon at `localhost:11434`
- **Local LLM:** `gemma3:e4b` (~9.6 GB)
- **Local embedder:** `nomic-embed-text` (274 MB, 768‑dim)
- **Cloud:** Azure OpenAI `gpt-5.4-nano` + `text-embedding-3-small` (1536‑dim) over corporate network
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

**Routing & corpus**
- The agent's routing relies on a fixed set of example questions; new domains may require updating the examples.
- County support in the water quality tool is hardcoded (8 counties). Dynamic lookup would require additional API integration.
- ChromaDB collection is not automatically updated when new PDFs are added — ingestion must be re‑run (after deleting the old collection).
- The system currently uses only one external API; multiple tools would need manual schema addition.

**Local profile (Gemma + Ollama)** — see `docs/local-mode-comparison.md` for the full breakdown.
- **Tool‑call drift.** `gemma3:e4b` is not reliably post‑trained for OpenAI‑style tool calling. On the 20‑query benchmark, only 5/10 tool‑seeking queries actually emitted a `tool_calls` payload — the rest silently fell through to a clarification question. No exception is raised.
- **Latency penalty.** End‑to‑end latency is ~2.5–2.7× cloud on this hardware. Per‑token throughput is actually slightly *higher* locally; the gap comes from Gemma producing ~3.3× more output tokens per query (verbose summaries).
- **Embedder‑driven scope leakage.** `nomic-embed-text` retrieves a different mix of corpus chunks than `text-embedding-3-small`, occasionally surfacing the wrong source for scoped queries (e.g. a query about EPA limits retrieving WHO docs only).
- **Context‑window drops.** A handful of corpus chunks (~6 of ~6,460) exceed `nomic-embed-text`'s 8K‑token context limit and are skipped during local ingest. The per‑item fallback in `OllamaEmbedder` keeps a single oversized chunk from poisoning the whole batch.
- **OOS over‑confidence.** Both profiles answered out‑of‑scope queries factually instead of refusing; the routing classifier surfaced low‑similarity false‑positive sources.
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
