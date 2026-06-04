# Failure Modes — Exercise C

## Overview

This document describes the four failure modes handled by the Tech Radar Concierge A2A system and the chaos test used to validate them. The behaviour described here is consistent with the orchestrator design defined in exercise-c-plan.md and the implementation delivered in deepak-exercise-c-handoff.md.

The MCP agent participates in the following orchestrated workflow:

User → Orchestrator (:8080) → Registry (:8083) → MCP Agent (:8082) → Node Subprocess (Exercise B)

The orchestrator owns workflow state in SQLite (state.db) and is responsible for retries, timeouts, and dead-lettering. The MCP agent is responsible for idempotent execution at its boundary.

## 1. Timeout

### Description

If the MCP agent does not return a response within 15 seconds, the orchestrator treats the request as failed.

### Implementation

The orchestrator enforces a 15-second timeout on every /invoke call using AbortController.
On timeout, the orchestrator:
Logs the workflow as TIMED_OUT with the associated correlation_id.
Updates the corresponding workflow row in state.db.
Returns a structured error response to the user.
### Outcome

The user receives a clear timeout error.
The orchestrator makes no assumption about whether the radar was modified.
The workflow remains inspectable via GET /workflow/:correlation_id.
## 2. Retries and Idempotency

### Description

If a call to the MCP agent fails, the orchestrator retries the request up to three times before giving up.

### Implementation

Retry strategy: exponential backoff (1s, 2s, 4s).
The same idempotency_key is reused across all retry attempts for a given workflow step.
The MCP agent maintains an in-memory map of processed idempotency_key values.
### MCP Agent Behaviour

When the same idempotency_key is received again, the MCP agent:

Returns the cached response envelope.
Does not re-invoke the Node subprocess.
### Outcome

Duplicate radar mutations are prevented.
Retries are safe and deterministic.
Transient failures are recovered without user-visible impact.
## 3. Partial Completion

### Description

The MCP agent successfully mutates the radar via the Node subprocess, but the orchestrator fails before the response is fully recorded — for example, due to a timeout during response handling or a crash mid-workflow.

### Implementation

The MCP agent executes the Node subprocess independently of the orchestrator's response handling.
The orchestrator persists every workflow step transition to state.db, including STARTED, COMPLETED, TIMED_OUT, and PARTIAL.
On MCP timeout, the orchestrator marks the step as PARTIAL and flags radar state as unknown.
### Outcome

The radar may have been mutated even when the user receives an error.
The workflow row in state.db records exactly which steps completed.
Recovery from the last successful step is possible by inspecting persisted state via GET /workflow/:correlation_id.
## 4. Poison Message

### Description

A request that consistently fails — due to invalid input, a persistent Node error, or repeated agent failure — must not be retried indefinitely.

### Implementation

After three failed retries, the orchestrator:
Writes the full envelope to dead-letter.jsonl.
Logs the failure at ERROR level with the associated correlation_id.
Marks the workflow as failed in state.db.
Stops further processing of the message.
### Outcome

Infinite retry loops are prevented.
Problematic envelopes are isolated for offline inspection and debugging.
System resources remain available for healthy traffic.
## Chaos Test: Killing the MCP Agent Mid-Workflow

### Setup

Start all four services using make up. ### Outcome

Infinite retry loops are prevented.

Problematic envelopes are isolated for offline inspection and debugging.
System resources remain available for healthy traffic.
## Chaos Test: Killing the MCP Agent Mid-Workflow

### Setup

Start all four services using make up.
Send a change request that routes to the MCP agent: make demo-change
While the orchestrator is awaiting the MCP agent's response, terminate the MCP agent process: kill -9
### Observed Behaviour

#### Orchestrator

Continues waiting for a response from the MCP agent.
Hits the 15-second timeout.
Logs the workflow step as TIMED_OUT.
Persists the failure in state.db.
Returns a structured failure response to the user.
#### Registry

Detects the missing heartbeat from the MCP agent.
Evicts the MCP agent after the TTL expires (two missed heartbeats, ~40 seconds).
#### User

Receives a failure response indicating the request did not complete.
#### Radar State

The final radar state depends on timing:

If the Node subprocess had begun execution before the kill, a partial mutation may have occurred.
If the Node subprocess had not yet started, the radar remains unchanged.
In either case, the workflow row in state.db accurately reflects the last completed step, allowing post-incident reconstruction.

## Summary

Failure Mode	Handling Mechanism
Timeout	15-second AbortController with TIMED_OUT SQLite state
Retries	Three attempts with exponential backoff (1s, 2s, 4s)
Idempotency	Deduplication via idempotency_key in the MCP agent
Partial Completion	Workflow steps persisted to state.db; flagged as PARTIAL
Poison Message	Dead-letter queue (dead-letter.jsonl) with ERROR log
## Conclusion

The Tech Radar Concierge system is designed to:

Remain resilient to transient and partial failures.
Prevent duplicate radar mutations through idempotent execution.
Maintain full observability via structured logs and persisted workflow state.
Provide deterministic recovery paths through state.db and the dead-letter queue.