# Tech Radar Snapshot — 2024 Q4

## ADOPT

### Temporal
**Ring:** ADOPT
**Quadrant:** Platforms
**Moved from:** TRIAL (2023 Q2)

Temporal was moved to ADOPT after 18 months of production use across three internal platforms. The workflow engine provides durable execution guarantees that replaced a fragile hand-rolled retry system in the data pipeline. Teams reported a 60% reduction in pipeline-related incidents after migration. The developer experience is excellent — workflows are plain code, and the replay debugger eliminates a whole class of distributed-system bugs.

**Why ADOPT:** Battle-tested in production, strong community, clear operational story on Kubernetes. The Ecolab platform team has deep expertise after the Eva Everywhere pilot.

---

### TypeScript (strict mode)
**Ring:** ADOPT
**Quadrant:** Languages & Frameworks

Strict TypeScript is now mandatory for all new Node.js services. Teams that migrated legacy JS caught an average of 12 type errors per 1000 lines during the migration. The investment in type discipline pays back quickly during code review and refactoring.

---

### Zod
**Ring:** ADOPT
**Quadrant:** Libraries

Zod is the standard schema validation library for TypeScript services. It replaces ad-hoc validation with composable, type-safe schemas that serve as both runtime validators and TypeScript type sources.

---

## TRIAL

### LangGraph
**Ring:** TRIAL
**Quadrant:** Libraries

LangGraph is under evaluation for multi-agent orchestration. Two teams are running pilots. Early results show promise for complex agent topologies, but the learning curve is steep and the abstraction sometimes fights you when you need low-level control. Decision on ADOPT expected 2025 Q1.

**Why TRIAL and not ADOPT:** Not enough production evidence yet. Two known issues: memory management in long-running graphs, and serialization edge cases with custom tool outputs.

---

### Qdrant
**Ring:** TRIAL
**Quadrant:** Platforms

Qdrant is being evaluated as a replacement for Chroma in production RAG workloads. Chroma works well for development but its operational story at scale is weak. Qdrant offers a proper server mode, native filtering, and better performance at >1M vectors.

**Why TRIAL:** The team is mid-migration on two services. Chroma stays in ADOPT for local dev. Qdrant targets production deployments.

---

## ASSESS

### A2A Protocol
**Ring:** ASSESS
**Quadrant:** Techniques

Google's Agent-to-Agent (A2A) protocol is under assessment. The standardised envelope format and capability registry pattern address real pain points in our current ad-hoc agent wiring. We plan a proof-of-concept with the Eva Everywhere platform in Q1 2025.

**Why ASSESS:** Protocol is still evolving. No production implementations in our stack yet. The concept maps well to Epic 1018668 requirements.

---

### Deno 2
**Ring:** ASSESS
**Quadrant:** Platforms

Deno 2's Node compatibility layer is mature enough to warrant assessment. The built-in TypeScript support, permission model, and native web APIs are appealing. Blocked on some npm packages we rely on.

---

## HOLD

### JavaScript (untyped)
**Ring:** HOLD
**Quadrant:** Languages & Frameworks

New untyped JavaScript services are on HOLD. All new Node.js work must use TypeScript strict mode. Existing JS services should be migrated opportunistically.

---

### REST without OpenAPI spec
**Ring:** HOLD

Any new REST API must ship with an OpenAPI 3.1 spec. Undocumented REST endpoints are a maintenance burden and block automated testing. This was escalated after the Q3 incident where a breaking API change wasn't caught before deploy.
