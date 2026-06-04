# Observability Walkthrough

## Overview

Every log line across all four processes is a structured JSON object with at minimum:

```json
{
  "timestamp": "2026-06-03T14:22:01.832Z",
  "level": "INFO",
  "agent": "orchestrator",
  "correlation_id": "b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
  "message": "..."
}
```

To filter all log lines for a single user request, pipe any process's stdout through:

```bash
jq 'select(.correlation_id == "<your-id>")' < orchestrator.log
```

---

## Sample Trace — Explain Request

The following is a representative trace for the query **"Why is Temporal in ADOPT on the tech radar?"** (an "explain" intent, routed to the RAG agent).

Log lines are shown in wall-clock order across all processes. In production, centralise them with a single `jq` filter on `correlation_id`.

```
correlation_id: b4e9f1a2-3c7d-4e8f-9012-abcdef123456
```

### Step 1 — Orchestrator receives request

```json
{"timestamp":"2026-06-03T14:22:01.832Z","level":"INFO","agent":"orchestrator",
 "correlation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "message":"request received","query":"Why is Temporal in ADOPT on the tech radar?"}
```

**What this tells us:** The user request arrived. The `correlation_id` is assigned here — it is the root of the trace.

### Step 2 — Intent classification prompt logged

```json
{"timestamp":"2026-06-03T14:22:01.836Z","level":"INFO","agent":"orchestrator",
 "correlation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "message":"intent classification prompt",
 "system_prompt":"You are an intent classifier for a Tech Radar assistant...",
 "user_prompt":"Query: \"Why is Temporal in ADOPT on the tech radar?\""}
```

**What this tells us:** The exact prompt sent to the LLM is logged at `INFO` before the call. This is the primary observability requirement: *without re-running the workflow*, we can see exactly what the orchestrator asked the LLM.

### Step 3 — Intent classified

```json
{"timestamp":"2026-06-03T14:22:02.144Z","level":"INFO","agent":"orchestrator",
 "correlation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "message":"intent classified","raw_response":"explain","intent":"explain"}
```

**What this tells us:** The LLM returned `"explain"`. The orchestrator records both the raw LLM output (`raw_response`) and the resolved intent. If the LLM had returned an unexpected value, we would see the fallback keyword logic here instead.

**Why did the orchestrator call the RAG agent?** Because `intent === "explain"` → `capability = "answer-from-corpus"`. This decision is fully auditable from this single log line.

### Step 4 — Registry lookup

```json
{"timestamp":"2026-06-03T14:22:02.145Z","level":"INFO","agent":"orchestrator",
 "correlation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "message":"registry lookup","capability":"answer-from-corpus",
 "url":"http://localhost:8083/agents?capability=answer-from-corpus"}
```

```json
{"timestamp":"2026-06-03T14:22:02.148Z","level":"INFO","agent":"orchestrator",
 "correlation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "message":"resolved agent 'rag-agent' for capability 'answer-from-corpus'"}
```

**What this tells us:** The orchestrator did not hardcode the RAG agent URL. It queried the registry and got back `{ name: "rag-agent", endpoint: "http://localhost:8081" }`. If the registry had returned an empty list, we would see a `WARN` here.

### Step 5 — Dispatch attempt (orchestrator side)

```json
{"timestamp":"2026-06-03T14:22:02.149Z","level":"INFO","agent":"orchestrator",
 "correlation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "causation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "capability":"answer-from-corpus",
 "message":"dispatch attempt 1/3","endpoint":"http://localhost:8081"}
```

**What this tells us:** The `causation_id` equals the `correlation_id` here because this is the first hop — the orchestrator is the root sender. On subsequent hops (e.g., if the RAG agent called another service), the `causation_id` would point to the envelope that triggered it.

### Step 6 — RAG agent receives invoke

```json
{"timestamp":"2026-06-03T14:22:02.155Z","level":"INFO","agent":"rag-agent",
 "correlation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "causation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "capability":"answer-from-corpus",
 "message":"invoke received"}
```

**What this tells us:** The envelope arrived at the RAG agent with the same `correlation_id`. The `causation_id` confirms this was triggered by the orchestrator's root message.

### Step 7 — RAG agent calls LLM

```json
{"timestamp":"2026-06-03T14:22:02.421Z","level":"INFO","agent":"rag-agent",
 "correlation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "capability":"answer-from-corpus",
 "message":"llm call",
 "prompt_preview":"Context:\n[radar_history_2024.md | chunk 2]\nTemporal entered TRIAL..."}
```

**What this tells us:** The context passed to the LLM is logged (first 200 chars). We can see which corpus chunks were retrieved and confirm that the answer came from the radar history documents, not from the model's pretraining.

### Step 8 — RAG agent returns

```json
{"timestamp":"2026-06-03T14:22:03.012Z","level":"INFO","agent":"rag-agent",
 "correlation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "message":"invoke complete","answer_length":342}
```

### Step 9 — Orchestrator persists COMPLETE

```json
{"timestamp":"2026-06-03T14:22:03.016Z","level":"INFO","agent":"orchestrator",
 "correlation_id":"b4e9f1a2-3c7d-4e8f-9012-abcdef123456",
 "message":"workflow complete","intent":"explain",
 "answer_preview":"Temporal entered TRIAL in Q3 2025 based on the engineering team's..."}
```

---

## Answering the Spec's Observability Questions

Given only the log lines above (no re-run):

| Question | Answer (from logs) |
|---|---|
| **Why did the orchestrator call the RAG agent?** | Step 2–3: classification prompt logged verbatim; LLM returned `"explain"` → `capability = "answer-from-corpus"` |
| **Why did it NOT call the MCP agent?** | Step 3: `intent = "explain"` maps to `answer-from-corpus`, not `propose-radar-change` |
| **Which registry entry resolved the endpoint?** | Step 4: `resolved agent 'rag-agent' for capability 'answer-from-corpus'` |
| **How many retries occurred?** | Step 5: `dispatch attempt 1/3` — no further retry lines → zero retries needed |
| **Did the RAG agent answer from corpus?** | Step 7: `prompt_preview` shows corpus chunks were included in the LLM context |
| **What was the full agent order?** | Steps: orchestrator → registry → rag-agent → orchestrator |

---

## Querying Logs in Practice

All four processes write structured JSON to stdout. To centralise:

```bash
# Run all processes, tee logs to files
npm run registry   2>&1 | tee registry.log   &
npm run rag-agent  2>&1 | tee rag-agent.log  &
npm run mcp-agent  2>&1 | tee mcp-agent.log  &
npm run orchestrator 2>&1 | tee orchestrator.log &

# After a run, reconstruct the full trace for one correlation_id
cat registry.log rag-agent.log mcp-agent.log orchestrator.log \
  | jq -c 'select(.correlation_id == "b4e9f1a2-3c7d-4e8f-9012-abcdef123456")' \
  | jq -s 'sort_by(.timestamp)[]'
```

This returns all log lines for that correlation_id in chronological order, spanning all four processes — the complete workflow audit trail in one command.
