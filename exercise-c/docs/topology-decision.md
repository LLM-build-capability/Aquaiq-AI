# Topology Decision — Orchestration vs Choreography

## Decision

**Orchestration.** A single conductor (`orchestrator/main.ts`) owns every workflow step: it classifies intent, queries the registry, dispatches to the chosen agent, retries on failure, and persists state to SQLite. No agent fires a message without being asked.

---

## The Five Axes

### 1. Cognitive load (new engineer)

**Orchestration wins.** The entire flow for one user request is traceable by reading a single file (`orchestrator/main.ts`) and one SQLite table (`workflow_steps`). A new engineer can answer "what happened to request X" by running:

```bash
curl http://localhost:8080/workflow/<correlation_id> | jq .
```

and sees every step in timestamp order, including which agent was called, what capability was requested, and what status it ended in.

With choreography, the same question would require reading event logs from three separate processes and reconstructing the order from timestamps — nontrivial when retries and timeouts overlap.

### 2. Blast radius

**Orchestration wins on containment.** If the orchestrator crashes, in-flight requests fail, but the RAG agent and MCP agent are unaffected. When the orchestrator restarts, it re-reads `state.db` and can resume any workflow that reached `DISPATCHED` status but never got a `COMPLETE`. Partial-completion state is a first-class concept (the `PARTIAL` status on the workflow step).

With choreography, a failure in the event bus or in any agent's consumer means messages pile up in the queue — the blast radius extends to all downstream consumers until the queue is drained or messages time out.

### 3. Debuggability

**Orchestration wins decisively.** The spec requires: *"given a correlation_id, reconstruct agent order, retries, and the prompt behind every LLM-driven routing decision — without re-running the workflow."*

The orchestrator logs:
- The exact `system_prompt` and `user_prompt` for every intent classification call (logged at `INFO` level with `correlation_id`)
- Every registry lookup, including the URL and the resolved agent endpoint
- Every dispatch attempt number, timeout event, and backoff interval
- Every SQLite step transition

This means the full audit trail lives in structured JSON logs, queryable with `jq` on `correlation_id`. No cross-process reconstruction required.

### 4. Latency

**Choreography has an edge, but it is irrelevant at this scale.** The orchestrator adds one extra HTTP hop per dispatch (~1–5ms on localhost). For a workflow that already spends 500ms–30s on LLM calls and vector search, this is noise. The real latency in this system is dominated by the Azure OpenAI call inside the orchestrator (intent classification) and the RAG/MCP agent's own LLM calls — both outside the orchestrator's control regardless of topology.

If this system were to scale to high concurrency (hundreds of concurrent requests), the orchestrator would become a bottleneck. We note this honestly as a scale-out risk but accept it given the scope (localhost, 3 agents, exercise context).

### 5. Team-size scaling

**Choreography would win at >10 agents.** With only 3 agents, each with a single capability, the orchestrator's routing logic is a trivial 2-branch conditional. The complexity that choreography manages well — decoupling producers and consumers when N is large and teams are independent — does not appear in this system.

If the team were to grow to 5+ agents with overlapping capabilities and independent release cycles, we would revisit this decision. We document this explicitly to show awareness, not because it is a current constraint.

---

## Summary Table

| Axis | Orchestration | Choreography | Winner |
|---|---|---|---|
| Cognitive load | One file, one table | 3+ event topics, cross-process join | Orchestration |
| Blast radius | Contained to one process | Can cascade to all consumers | Orchestration |
| Debuggability | Full audit trail per correlation_id | Requires cross-process reconstruction | Orchestration |
| Latency | +1 hop per step | Near-zero broker overhead | Choreography (irrelevant at scale) |
| Team-size scaling | Bottleneck at high N | Natural fan-out | Choreography (N/A here) |

**Decision stands: orchestration for this system.**
