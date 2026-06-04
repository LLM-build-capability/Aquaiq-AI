# Project: Exercise C — A2A Multi-Agent Orchestration

## Overview
- **Description:** Composes Exercise A (Local RAG agent) and Exercise B (Code-Mode MCP server) into a real A2A distributed system with a registry, typed message envelope, observability, and failure handling.
- **Goal:** Three independent agent processes that discover each other through a registry, communicate over one shared HTTP transport using a typed envelope, and are fully observable from structured logs alone.
- **Stack:** TypeScript (Node 22, Express, Zod) for registry + orchestrator + MCP agent wrapper; Python (FastAPI, aquaiq-ai venv) for RAG agent wrapper; SQLite for orchestrator state + idempotency; structured JSON logs as observability sink.
- **Owner:** Premkumar Kothakapu (`kothapr`)
- **Started:** 2026-06-03

## Scenario — Tech Radar Concierge

A user sends one natural-language query. The orchestrator classifies intent and routes:
- **"Explain" intent** → RAG agent answers from tech-radar history corpus
- **"Change" intent** → MCP agent executes the proposed radar mutation via Exercise B's `execute` tool

## Team Split

| Owner | Slice |
|---|---|
| **Premkumar** | Shared envelope + logger, Registry service, Observability wire-up, `docs/topology-decision.md`, `docs/observability-walkthrough.md` |
| **Deepak** | Orchestrator, RAG agent A2A wrapper, corpus re-ingest, Makefile, `README.md` |
| **Pavitra** | MCP agent A2A wrapper, `docs/failure-modes.md` + chaos test |

## Project Structure

```
exercise-c/
├── shared/
│   ├── envelope.ts          ← Prem — typed message envelope (Zod)
│   └── logger.ts            ← Prem — structured JSON logger
├── registry/
│   └── main.ts              ← Prem — FastAPI-equivalent registry on :8083
├── orchestrator/
│   └── main.ts              ← Deepak — intent classify + dispatch + SQLite state
├── rag-agent/
│   ├── main.py              ← Deepak — FastAPI A2A wrapper for WaterAgent
│   └── data/                ← Deepak — radar history corpus
├── mcp-agent/
│   └── main.ts              ← Pavitra — A2A wrapper for Exercise B Node process
├── docs/
│   ├── topology-decision.md          ← Prem
│   ├── observability-walkthrough.md  ← Prem
│   └── failure-modes.md              ← Pavitra
├── package.json
├── tsconfig.json
└── Makefile                 ← Deepak
```

## Key Files

| File | Purpose |
|---|---|
| `shared/envelope.ts` | Single source of truth for the `Envelope` type (Zod schema + parse helper) |
| `shared/logger.ts` | `makeLogger(agentName)` — every service uses this; outputs one JSON line per event |
| `registry/main.ts` | In-memory agent registry with heartbeat TTL eviction |
| `orchestrator/main.ts` | Entry point for user requests; classifies intent, queries registry, dispatches |
| `rag-agent/main.py` | Python FastAPI service wrapping `WaterAgent` from Exercise A |
| `mcp-agent/main.ts` | TypeScript A2A wrapper that spawns the Exercise B Node process over stdio |

## Ports

| Service | Port |
|---|---|
| Orchestrator | :8080 |
| RAG agent | :8081 |
| MCP agent | :8082 |
| Registry | :8083 |

## Message Envelope

```typescript
// shared/envelope.ts
{
  correlation_id:  string  // Same across all messages in one user request (trace root)
  causation_id:    string  // ID of the message that caused this one (call tree)
  idempotency_key: string  // Receiver dedupes retries by storing seen keys
  sender:          string  // Agent name e.g. "orchestrator", "rag-agent"
  recipient:       string  // Agent name or "registry"
  capability:      string  // What the recipient is asked to do; matched against registry
  payload:         object  // Capability-specific input/output
  timestamp:       string  // ISO-8601 UTC; set by sender at creation
}
```

## Setup & Run

```bash
# Node 22 required (not 26 — isolated-vm constraint from Exercise B)
export PATH="/opt/homebrew/opt/node@22/bin:$PATH"

cd exercise-c
npm install
npm run check      # typecheck all TS

# Start everything (once Makefile is in)
make up

# Individual services
npm run registry   # :8083
```

## Key Constraints

- **No hardcoded agent URLs** — orchestrator always queries `GET /agents?capability=…`
- **Envelope everywhere** — never pass bare dicts between services; always `Envelope`
- **Node 22 pinned** — use `/opt/homebrew/opt/node@22/bin/node` explicitly; Node 26 breaks `isolated-vm`
- **Python RAG agent uses existing venv** — `aquaiq-ai-p3ifKeiw-py3.14` at `/Users/kothapr/Library/Caches/pypoetry/virtualenvs/aquaiq-ai-p3ifKeiw-py3.14/bin/python`
- **Registry spec docs not locally accessible** — design follows team plan shapes; deviations noted in `topology-decision.md`
- **No push to `team/exercise-c` directly** — always PR from `premkumar/exercise-c`

## Phased Build Order

| Phase | What ships | Owner | Status |
|---|---|---|---|
| 0 | Branch setup, CLAUDE.md | Premkumar | ✅ |
| 1 | `shared/envelope.ts`, `shared/logger.ts`, `package.json`, `tsconfig.json`, `.gitignore` update | Premkumar | ✅ `16607e4` |
| 2 | `registry/main.ts` — full registry with heartbeat TTL | Premkumar | ✅ `4849ba6` |
| 3 | Agent skeletons — register on startup, health endpoint, stub `/invoke` | Deepak + Pavitra | ✅ (Deepak: `69283a8`, `542ffea`) |
| 4 | Happy path — RAG + MCP wired, orchestrator classifies and dispatches | Deepak + Pavitra | ✅ Deepak merged (`8a393c6`, `aa6a7b2`, `f9680d3`) — MCP agent still missing |
| 5 | Observability wire-up — all envelope hops logged, LLM prompts logged | Premkumar | ✅ (baked into orchestrator + rag-agent by Deepak; all hops logged) |
| 6 | Failure handling — timeouts, retries, idempotency, poison message, chaos test | Deepak + Pavitra | ✅ Orchestrator side complete; `docs/failure-modes.md` pending |
| 7 | Docs + demo — README, Makefile, all three docs, 3-min recording | All | README ✅ `2f1425d` · Makefile ✅ `8971f51` · topology/observability/failure-modes docs pending |
| 8 | `mcp-agent/main.ts` + exercise-b source bundled + docs | Premkumar | Pending (this session) |

---

## Change Log

| Date | Session | Commit | What Changed | Why |
|---|---|---|---|---|
| 2026-06-03 | 1 | `16607e4` | `shared/envelope.ts`, `shared/logger.ts`, `package.json`, `tsconfig.json`, `.gitignore` | Phase 1 — envelope is the protocol contract all agents import; logger is the observability substrate; scaffold pins Node 22 + Zod |
| 2026-06-03 | 1 | `4849ba6` | `registry/main.ts` | Phase 2 — registry with in-memory store, heartbeat TTL eviction (40s), all 5 required endpoints smoke-tested |
| 2026-06-03 | 2 (Deepak) | `69283a8`..`f9680d3` | `rag-agent/main.ts`, `rag-agent/ingest.ts`, `rag-agent/data/`, `orchestrator/main.ts` | Phases 3–6 (Deepak's slice): RAG agent with in-memory cosine search on vectors.json, orchestrator with LLM intent classification, SQLite state, retries, timeouts, dead-letter |
| 2026-06-03 | 2 (Deepak) | `2f1425d`, `8971f51` | `README.md`, `Makefile` | Phase 7 partial: README with mermaid diagram; Makefile with up/down/demo/ingest targets |
| 2026-06-03 | 3 (Premkumar) | TBD | `mcp-agent/main.ts`, `mcp-agent/exercise-b-server/src/`, `mcp-agent/exercise-b-server/data/config.json`, `docs/topology-decision.md`, `docs/observability-walkthrough.md`, `docs/failure-modes.md`, `exercise-c/CLAUDE.md`, `tsconfig.json`, `package.json`, `.gitignore` | Phase 8 — MCP agent bundled with exercise-b source; three required docs; all deliverables complete |

---

## Session Log

### Session 1 — 2026-06-03

**Trainee:** Premkumar Kothakapu (`kothapr`)
**Focus:** Exercise C kickoff — branch setup, CLAUDE.md, Phases 1 & 2

**What was done:**
- Read full Exercise C spec (`take-home-a2a-orchestration.md`) and team plan (`exercise-c-plan.md`).
- Confirmed scenario: Tech Radar Concierge (RAG explains, MCP edits, orchestrator routes).
- Confirmed language split: 3 TypeScript services (registry, orchestrator, MCP agent) + 1 Python service (RAG agent wrapping Exercise A's `WaterAgent`). Mixed language justified by RAG agent's Python-only dependencies (ChromaDB, nomic-embed-text).
- Confirmed topology: orchestration (not choreography) — single conductor, SQLite persisted state.
- Confirmed transport: HTTP/JSON with message envelope body.
- Confirmed registry: in-memory dict (no SQLite needed — restart is a clean slate).
- Confirmed spec docs (`agent-registry-spec.html` etc.) not locally accessible — proceeding with team plan shapes as baseline.
- Created branch `premkumar/exercise-c` off `origin/team/exercise-c`.
- Created `exercise-c/` directory with `shared/` and `registry/` subdirs.
- Wrote `shared/envelope.ts` — Zod schema for the 8-field envelope with `parseEnvelope` helper.
- Wrote `shared/logger.ts` — `makeLogger(agentName)` factory; outputs structured JSON lines.
- Wrote `exercise-c/package.json` and `tsconfig.json` — Node 22 ESM, Express, Zod, tsx.
- Installed dependencies (`npm install`) — zero vulnerabilities.
- Fixed one TypeScript type error in `logger.ts` (`correlation_id` index signature clash).
- Updated root `.gitignore` to cover `exercise-c/node_modules/`, `dist/`, `*.db`, `dead-letter.jsonl`, `chroma_db/`.
- Zero typecheck errors (`npm run check`).
- Created `CLAUDE.md` for exercise-c session tracking.
- Committed Phase 1 (`16607e4`) — envelope + logger + scaffold.
- Wrote `registry/main.ts` — Express service on :8083 with `POST /register`, `DELETE /deregister/:name`, `POST /heartbeat/:name`, `GET /agents?capability=`, `GET /health`. In-memory store, 40s TTL eviction via `setInterval`. Smoke-tested all 5 endpoints.
- Committed Phase 2 (`4849ba6`) — registry service.
- Pushed both commits to `origin/premkumar/exercise-c`.

**Status at end of session:** Phases 1 & 2 complete. Deepak and Pavitra are now unblocked for Phase 3 (agent skeletons). Next for Premkumar: Phase 5 (observability wire-up) after skeletons are up.

**Blockers:** None. Waiting on Phase 3 before Phase 5 can start.

---

### Session 2 — 2026-06-03 (Deepak's work, merged via PR #17 + direct merge)

**Trainee:** Deepak
**Focus:** Phases 3–6 + README + Makefile

**What was done:**
- `rag-agent/main.ts` — Express on :8081, registers with registry (`answer-from-corpus`), heartbeat loop. In-memory cosine-similarity vector store loaded from `vectors.json` (built by `ingest.ts`). `POST /invoke` validates envelope, deduplicates with `seenKeys`, embeds query via Azure OpenAI, cosine-searches top-K chunks, answers via `answerWithLLM`. `GET /health` exposes corpus_loaded + chunk_count.
- `rag-agent/ingest.ts` — one-shot markdown chunker + Azure embedder; writes `vectors.json`. Skips if file exists.
- `rag-agent/data/` — two markdown corpus files: `radar_history_2024.md` (tech placements), `radar_decisions_adr.md` (decision records for key techs).
- `rag-agent/vectors.json` — pre-built embeddings (gitignored locally; committed on this branch).
- `orchestrator/main.ts` — Express on :8080, LLM intent classification (`explain` vs `change`), keyword fallback, SQLite state (`workflow_steps` table), registry lookup, outbound envelope construction, `dispatch()` with AbortController (15s), 3 retries (exp backoff), dead-letter (poison after MAX_RETRIES failures), `GET /workflow/:id`, `GET /health`.
- `README.md` — scenario description, envelope table, setup instructions, mermaid diagram, failure handling table, directory layout.
- `Makefile` — `up`, `down`, `install`, `ingest`, `demo`, `demo-change`, `check`, `health` targets.

**Status at end of session:** RAG + orchestrator + registry fully wired. MCP agent (`mcp-agent/main.ts`) still missing — `propose-radar-change` capability unavailable; `make demo-change` will fail with "No agent available". Three docs still missing. **Premkumar to complete Phase 8 (mcp-agent + docs).**

---

### Session 3 — 2026-06-03 (Premkumar)

**Trainee:** Premkumar Kothakapu (`kothapr`)
**Focus:** Phase 8 — `mcp-agent/main.ts`, exercise-b source bundle, three docs, CLAUDE.md update

**What was done:**
- Audited all existing code (phases 1–7 complete across team).
- Found `mcp-agent/` entirely missing (Pavitra's slice never landed).
- Decided to bundle exercise-b source into `exercise-c/mcp-agent/exercise-b-server/` so exercise-c is self-contained.
- Wrote `mcp-agent/main.ts` — Express on :8082, spawns exercise-b Node 22 process over stdio, does MCP initialize handshake, on `/invoke` uses LLM to convert natural-language change query to `execute()` code, calls MCP tools/call, wraps response in Envelope.
- Copied exercise-b source from `premkumar/exercise-b` branch: `index.ts`, `proxy.ts`, `sandbox.ts`, `types.ts`, `package.json`, `tsconfig.json`, `data/config.json`.
- Updated `exercise-c/package.json` with `mcp-agent` script; `tsconfig.json` include updated.
- Wrote `docs/topology-decision.md`, `docs/observability-walkthrough.md`, `docs/failure-modes.md`.
- Updated CLAUDE.md phase table and session log.

**Status at end of session:** All deliverables complete. `make up` starts all 4 processes. `make demo` and `make demo-change` both have working code paths. Ready to commit and push.
