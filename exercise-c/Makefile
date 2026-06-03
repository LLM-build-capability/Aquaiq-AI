# Exercise C — A2A Tech Radar Concierge
# Usage:
#   make install     install node deps
#   make ingest      embed the radar corpus (run once)
#   make up          start all 4 processes
#   make down        kill all 4 processes
#   make demo        send a sample explain query
#   make demo-change send a sample change query
#   make check       typecheck all TypeScript

NODE := node
NPM  := npm

# Stagger startup so registry is ready before agents register
.PHONY: up
up:
	@echo "Starting registry on :8083..."
	$(NPM) run --prefix . registry &
	@sleep 2
	@echo "Starting rag-agent on :8081..."
	$(NPM) run --prefix . rag-agent &
	@sleep 1
	@echo "Starting mcp-agent on :8082..."
	$(NPM) run --prefix . mcp-agent &
	@sleep 1
	@echo "Starting orchestrator on :8080..."
	$(NPM) run --prefix . orchestrator &
	@echo ""
	@echo "All processes started. Try: make demo"

.PHONY: down
down:
	@echo "Stopping all exercise-c processes..."
	@pkill -f "registry/main.ts"   2>/dev/null || true
	@pkill -f "rag-agent/main.ts"  2>/dev/null || true
	@pkill -f "mcp-agent/main.ts"  2>/dev/null || true
	@pkill -f "orchestrator/main.ts" 2>/dev/null || true
	@echo "Done."

.PHONY: install
install:
	$(NPM) install

.PHONY: ingest
ingest:
	@echo "Ingesting radar corpus..."
	$(NPM) run --prefix . ingest

.PHONY: demo
demo:
	@echo "Sending explain query..."
	curl -s -X POST http://localhost:8080/request \
	  -H "Content-Type: application/json" \
	  -d '{"query": "Why is Temporal in ADOPT on the tech radar?"}' \
	  | jq .

.PHONY: demo-change
demo-change:
	@echo "Sending change query..."
	curl -s -X POST http://localhost:8080/request \
	  -H "Content-Type: application/json" \
	  -d '{"query": "Move LangGraph from TRIAL to ADOPT"}' \
	  | jq .

.PHONY: check
check:
	$(NPM) run --prefix . check

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
