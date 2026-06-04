# Exercise C — A2A Tech Radar Concierge
# Usage:
#   make install     install node deps
#   make ingest      embed the radar corpus into exercise-c/rag-agent/chroma_db/ (run once)
#   make up          start all 4 processes
#   make down        kill all 4 processes
#   make demo        send a sample explain query
#   make demo-change send a sample change query
#   make check       typecheck all TypeScript
#   make health      check all agents
#   make test        run all tests (TypeScript + Python)

NODE := /opt/homebrew/opt/node@22/bin/node
NPM  := /opt/homebrew/opt/node@22/bin/npm

# Exercise A venv — used by the RAG agent (Python)
PYTHON := /Users/kothapr/Library/Caches/pypoetry/virtualenvs/aquaiq-ai-p3ifKeiw-py3.14/bin/python

# Stagger startup so registry is ready before agents register
.PHONY: up
up:
	@echo "Starting registry on :8083..."
	PATH="/opt/homebrew/opt/node@22/bin:$$PATH" $(NPM) run --prefix . registry &
	@sleep 2
	@echo "Starting rag-agent on :8081 (LLM_PROFILE=local, corpus=radar_local)..."
	cd $(CURDIR) && LLM_PROFILE=local CHROMA_PERSIST_DIR=$(CURDIR)/exercise-c/rag-agent/chroma_db LOCAL_COLLECTION_NAME=radar_local CLOUD_COLLECTION_NAME=radar_local $(PYTHON) -m uvicorn main:app --host 0.0.0.0 --port 8081 --app-dir exercise-c/rag-agent &
	@sleep 3
	@echo "Starting mcp-agent on :8082..."
	PATH="/opt/homebrew/opt/node@22/bin:$$PATH" $(NPM) run --prefix . mcp-agent &
	@sleep 1
	@echo "Starting orchestrator on :8080..."
	PATH="/opt/homebrew/opt/node@22/bin:$$PATH" $(NPM) run --prefix . orchestrator &
	@echo ""
	@echo "All processes started. Try: make demo"

.PHONY: down
down:
	@echo "Stopping all exercise-c processes..."
	@pkill -f "registry/main.ts"     2>/dev/null || true
	@pkill -f "rag-agent/main.py"    2>/dev/null || true
	@pkill -f "mcp-agent/main.ts"    2>/dev/null || true
	@pkill -f "orchestrator/main.ts" 2>/dev/null || true
	@echo "Done."

.PHONY: install
install:
	PATH="/opt/homebrew/opt/node@22/bin:$$PATH" $(NPM) install

# Ingest radar corpus using Exercise A's embedder helpers.
# Writes to exercise-c/rag-agent/chroma_db/ — never touches root chroma_db/.
.PHONY: ingest
ingest:
	@echo "Ingesting radar corpus into exercise-c/rag-agent/chroma_db/ ..."
	LLM_PROFILE=local \
	CHROMA_PERSIST_DIR=$(CURDIR)/exercise-c/rag-agent/chroma_db \
	LOCAL_COLLECTION_NAME=radar_local \
	$(PYTHON) exercise-c/rag-agent/ingest_radar.py

.PHONY: demo
demo:
	@echo "Sending explain query (tech radar)..."
	curl -s -X POST http://localhost:8080/request \
	  -H "Content-Type: application/json" \
	  -d '{"query": "Why is Temporal in the ADOPT ring on the tech radar?"}' \
	  | jq .

.PHONY: demo-change
demo-change:
	@echo "Sending change query (tech radar mutation)..."
	curl -s -X POST http://localhost:8080/request \
	  -H "Content-Type: application/json" \
	  -d '{"query": "Move LangGraph from TRIAL to ADOPT"}' \
	  | jq .

.PHONY: check
check:
	PATH="/opt/homebrew/opt/node@22/bin:$$PATH" $(NPM) run --prefix . check

.PHONY: test
test:
	@echo "=== TypeScript tests ==="
	PATH="/opt/homebrew/opt/node@22/bin:$$PATH" $(NPM) run --prefix . test
	@echo ""
	@echo "=== Python tests ==="
	$(PYTHON) -m pytest $(CURDIR)/tests/test_rag_agent.py -v

.PHONY: health
health:
	@echo "=== Registry :8083 ==="
	@curl -s http://localhost:8083/health | jq .
	@echo "=== RAG Agent :8081 ==="
	@curl -s http://localhost:8081/health | jq .
	@echo "=== MCP Agent :8082 ==="
	@curl -s http://localhost:8082/health | jq . 2>/dev/null || echo "(not running)"
	@echo "=== Orchestrator :8080 ==="
	@curl -s http://localhost:8080/health | jq .
