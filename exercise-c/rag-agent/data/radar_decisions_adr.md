# Architecture Decision Records — Tech Radar Entries

## ADR-042: Move Temporal from TRIAL to ADOPT

**Date:** 2024-09-15
**Status:** Accepted
**Deciders:** Platform team, Architecture guild

### Context

Temporal entered TRIAL in 2023 Q2. Since then:
- 3 production services migrated from home-grown retry loops
- 18 months of operational data
- Zero unrecoverable workflow failures (vs ~4/month before)
- The Eva Everywhere pilot used Temporal for its agent orchestration layer

### Decision

Move Temporal from TRIAL to ADOPT.

### Consequences

All new durable-execution use cases default to Temporal. The previous pattern (SQS + DynamoDB + Lambda retry logic) is deprecated for new work.

---

## ADR-038: Standardise on Zod for schema validation

**Date:** 2024-06-01
**Status:** Accepted

### Context

Three different validation approaches existed across services: joi, yup, and hand-rolled validators. This created inconsistent error messages and made shared schema libraries impossible.

### Decision

Zod is the standard. joi and yup are on HOLD for new code.

### Why Zod over alternatives

- TypeScript-first: the inferred type IS the schema, no duplication
- Runtime + compile-time safety in one library
- Excellent error messages out of the box
- zod.infer<typeof Schema> pattern eliminates the "schema says one thing, TS type says another" class of bugs

---

## ADR-044: Assess A2A Protocol for Eva Everywhere

**Date:** 2024-11-20
**Status:** Proposed
**Deciders:** LLM Capability team

### Context

Epic 1018668 (Eva Everywhere) requires multiple AI agents to coordinate. Current approach is direct HTTP calls with no standard envelope, no capability discovery, and no correlation tracking. This works for two agents but breaks down at scale.

### Decision

Place A2A Protocol in ASSESS. Run a proof-of-concept with 3-4 agents (orchestrator, RAG, MCP, registry) to validate:
1. Capability-based routing via a registry
2. Typed envelope with correlation_id / causation_id
3. Heartbeat-based health eviction
4. Observable logs (structured JSON, one envelope per log line)

### Why now

The LLM Capability team (Aquaiq-AI bootcamp) is building Exercises A, B, C which map exactly to these requirements. The bootcamp output will be the first real A2A evidence for the radar.

---

## ADR-033: LangGraph TRIAL — constraints and exit criteria

**Date:** 2024-03-10
**Status:** Active

### Context

LangGraph promises declarative multi-agent graphs with built-in state management. Two teams are piloting it.

### Known issues during TRIAL

1. Memory leak in long-running graphs — StateGraph accumulates intermediate state; must explicitly clear after each run
2. Custom tool serialization — non-JSON-serialisable tool outputs cause cryptic errors; wrap all tools in a serialization layer
3. Abstraction mismatch — when you need to inject a message mid-graph, LangGraph fights you; low-level code is sometimes cleaner

### Exit criteria for ADOPT

- At least 2 production services running LangGraph in production for 3+ months
- Memory leak issue resolved upstream or documented workaround proven stable
- Both pilot teams report positive DX

### Exit criteria for HOLD

- Either pilot team abandons LangGraph in favour of direct orchestration code
- Memory leak causes a production incident

---

## ADR-029: Qdrant TRIAL — replacing Chroma in production

**Date:** 2024-01-15
**Status:** Active

### Context

Chroma works well locally but has a weak operational story: no native auth, no replication, no horizontal scaling. Services moving to production need a more robust vector store.

### Why not keep Chroma everywhere

- Chroma PersistentClient does not support concurrent write access
- No Kubernetes operator available
- Limited filtering compared to Qdrant's payload filters

### Decision

Chroma stays in ADOPT for local development and testing. Qdrant enters TRIAL for production RAG workloads.

### TRIAL success criteria

- Migrate 2 services from Chroma (local) to Qdrant (prod) by 2025 Q1
- Document the local-Chroma to prod-Qdrant migration playbook
