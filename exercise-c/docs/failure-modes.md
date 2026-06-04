# Failure Modes and Chaos Test

## The Four Failure Modes

### 1. Timeout

**Where handled:** Orchestrator (`orchestrator/main.ts`, `dispatch()`).

**Implementation:** Every `/invoke` call is wrapped in an `AbortController` with a 15-second deadline. If the agent does not respond in time, the abort fires, the `fetch` rejects with `AbortError`, and the orchestrator:

1. Logs `TIMED_OUT` at `ERROR` level with `correlation_id`.
2. Persists a `TIMED_OUT` row to `workflow_steps` in SQLite.
3. If the timed-out capability was `propose-radar-change`, also persists a `PARTIAL` row — because the MCP agent may have mutated the radar before the timeout (the radar change is committed inside `execute()` before the HTTP response is sent).
4. Returns `HTTP 504` to the user with the `correlation_id` so they can inspect state via `GET /workflow/<id>`.

### 2. Retries + Idempotency

**Where handled:** Orchestrator retries; each agent deduplicates.

**Implementation:** The orchestrator retries up to 3 times with exponential backoff (1s → 2s → 4s), **reusing the same `idempotency_key`** on every attempt. Each agent (`rag-agent/main.ts` and `mcp-agent/main.ts`) maintains an in-memory `seenKeys` set. When an envelope arrives with a key already in `seenKeys`, the agent returns a deduplication response immediately without re-executing the capability.

This ensures that a retry caused by a network blip — where the first request succeeded on the agent side but the response was lost — does not cause the RAG agent to query the LLM twice or the MCP agent to mutate the radar twice.

### 3. Partial Completion

**Where handled:** Orchestrator, post-dispatch.

**Implementation:** Two classes of partial completion exist:

- **MCP succeeds, compose fails:** The radar is mutated (committed to `config.json`) before the HTTP response is sent. If the orchestrator's response assembly or SQLite write fails after the MCP call returns, the user sees an error but the radar *has already changed*. The orchestrator persists a `PARTIAL` workflow step when this case is detected. The `GET /workflow/<correlation_id>` endpoint exposes this state so an operator can inspect whether the mutation occurred.

- **Timeout on MCP call:** As described above, the orchestrator cannot know whether the MCP agent committed the radar change before the timeout. It logs `PARTIAL` with the note "radar state unknown, check MCP agent logs" and returns `504` to the user.

Recovery is manual: inspect the radar's `config.json` directly to determine whether the intended change was applied, then either accept the state or apply a corrective mutation via a new request.

### 4. Poison Message

**Where handled:** Orchestrator, after all retries are exhausted.

**Implementation:** After the final retry fails (attempt 3), the orchestrator checks how many `TIMED_OUT` or `FAILED` rows exist in `workflow_steps` for this `correlation_id`. If the count reaches `MAX_RETRIES` (3), the outbound envelope is written to `exercise-c/dead-letter.jsonl` as a JSON line, and an `ERROR`-level log is emitted. The envelope is not retried further.

`dead-letter.jsonl` is gitignored and operator-inspectable. Each dead-lettered entry includes `dead_letter_reason` and `dead_letter_at` fields so postmortem is possible without reconstructing the sequence from logs.

---

## Chaos Test

### Setup

Start all four processes:

```bash
cd exercise-c
make up
```

Verify all are healthy:

```bash
make health
```

### Test procedure

1. Send a change request that routes to the MCP agent:

```bash
curl -s -X POST http://localhost:8080/request \
  -H "Content-Type: application/json" \
  -d '{"query": "Move LangGraph from TRIAL to ADOPT"}' &
```

2. Immediately (within 1–2 seconds) kill the MCP agent process:

```bash
pkill -f "mcp-agent/main.ts"
```

### What the user sees

After 15 seconds (the orchestrator's timeout):

```json
{
  "error": "agent timed out",
  "correlation_id": "<uuid>"
}
```

### What the logs show

On the orchestrator (structured JSON, one line per event):

```
{"level":"INFO", "agent":"orchestrator", "message":"request received", "correlation_id":"<uuid>", "query":"Move LangGraph..."}
{"level":"INFO", "agent":"orchestrator", "message":"intent classified", "correlation_id":"<uuid>", "intent":"change"}
{"level":"INFO", "agent":"orchestrator", "message":"dispatch attempt 1/3", "correlation_id":"<uuid>", "endpoint":"http://localhost:8082"}
{"level":"WARN", "agent":"orchestrator", "message":"dispatch failed (attempt 1): timed out", "correlation_id":"<uuid>"}
{"level":"INFO", "agent":"orchestrator", "message":"retrying in 1000ms", "correlation_id":"<uuid>"}
{"level":"WARN", "agent":"orchestrator", "message":"dispatch failed (attempt 2): timed out", "correlation_id":"<uuid>"}
{"level":"INFO", "agent":"orchestrator", "message":"retrying in 2000ms", "correlation_id":"<uuid>"}
{"level":"WARN", "agent":"orchestrator", "message":"dispatch failed (attempt 3): timed out", "correlation_id":"<uuid>"}
{"level":"ERROR", "agent":"orchestrator", "message":"workflow timed out", "correlation_id":"<uuid>"}
{"level":"ERROR", "agent":"orchestrator", "message":"dead-lettered envelope", "correlation_id":"<uuid>", "reason":"exceeded max retries (3)"}
```

### Was the radar mutated before the kill?

No — in this test, the MCP agent was killed before it could receive the envelope, so no mutation occurred. The MCP agent's `isolated-vm` sandbox runs synchronously inside the Node process; if the process was killed before `radar.commit()` executed, `config.json` is unchanged.

If the agent had been killed *after* `radar.commit()` but before the HTTP response was sent, the radar would be mutated but the orchestrator would log `PARTIAL` (not `FAILED`) on the step. An operator could verify by diffing `config.json` against the previous state.

### Registry cleanup

After the MCP agent is killed, its heartbeat loop stops. The registry evicts the entry after 40 seconds (2 missed 20-second heartbeat cycles). From that point, `GET /agents?capability=propose-radar-change` returns an empty list, and the orchestrator logs:

```
{"level":"WARN", "agent":"orchestrator", "message":"no agent found for capability 'propose-radar-change'"}
```

and returns `HTTP 503` to the user — a clean, observable failure state rather than a silent hang.
