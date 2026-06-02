# Proxy Design — Code Mode MCP (Exercise B)

## Why This Proxy Shape?

The Tech Radar domain is **Variant 2 — a constrained metamodel**: a small number of fixed types (`Quadrant`, `Ring`, `Moved`) and rigid governance rules (no ADOPT→HOLD skip, no duplicate assignments, kebab-case IDs). This structure tells you exactly what the proxy surface should look like before you write a line of code.

### The Interface

```typescript
interface RadarReadProxy {
  listTechnologies(filter?: { quadrant?: Quadrant }): Technology[]
  listTeams(): Team[]
  listAssignments(teamId: string): Assignment[]
  getAssignment(teamId: string, techId: string): Assignment | undefined
  validate(op: PendingOp): ValidationResult
}

interface RadarProxy extends RadarReadProxy {
  addTechnology(id: string, label: string, quadrant: Quadrant): Technology
  assign(teamId: string, techId: string, ring: Ring, moved?: Moved): Assignment
  move(teamId: string, techId: string, newRing: Ring): Assignment
  removeAssignment(teamId: string, techId: string): void
  commit(message: string): void
}
```

The split between `RadarReadProxy` and `RadarProxy` is what enforces the query/execute boundary structurally — the `search()` sandbox receives only a `RadarReadProxy`. If a model writes `radar.commit(...)` inside `search()`, it gets `TypeError: radar.commit is not a function`, not a governance violation. The sandbox enforces it, not a runtime check in the method.

---

## DSL Alternatives Considered

### Option A — JSON Schema in tool descriptions (rejected)

The naive approach: embed the full JSON Schema for each operation in the tool description. This is what the N-tool baseline does. Every field gets a `description` string that re-explains `0=ADOPT, 1=TRIAL...` across five different tool definitions. Token cost: **1,417**. The metamodel is scattered; a schema change requires updating all tools.

### Option B — Pydantic / Zod models (rejected for bootstrap)

Pydantic (Python) and Zod (TypeScript) are good proxy-implementation choices but terrible bootstrap DSLs. Their validator syntax is verbose (`z.object({ id: z.string().regex(...) })`), and the validation error messages are mechanical — they tell you what failed but not what to do instead. The self-correcting error requirement means validation logic belongs in the proxy methods, not in a schema validator.

### Option C — Raw prose description (rejected)

Describe the domain in natural language inside the tool description. Cheap in tokens, but the model has no types to reason against — it will guess field names, invent rings that don't exist, and submit un-validateable calls. The DSL exists to give the model a concrete target, not as documentation for a human.

### Option D — TypeScript type aliases as the DSL (chosen)

Three advantages for Variant 2:

1. **Token-efficient**: the full metamodel in compact type alias form is **243 tokens** — 17% of the N-tool baseline's full cost. Strips JSDoc verbosity while keeping enough structure that a pretrained model (which knows TypeScript) can reason about valid calls.
2. **Single source of truth**: the DSL block in `execute()`'s description is the only place the metamodel lives in model context. The types in `src/types.ts` are authoritative for implementation; the DSL is a projection of them trimmed for tokens.
3. **Pretraining leverage**: models are trained on TypeScript. `type Ring = 0|1|2|3` is immediately understood without a definition of what `integer` means or an explanation of `enum`.

---

## Validation in the Proxy, Not the Tools

Every write method throws a `ProxyError` when called incorrectly. The error message names the valid alternative — it does not just say what went wrong:

```
ProxyError: technology 'this-tech-does-not-exist' not found in radar.
Use radar.addTechnology('this-tech-does-not-exist', label, quadrant) to add it first,
or pick from existing: 'claude-haiku-4-5-databricks', ... (use radar.listTechnologies() to see all).
```

```
ProxyError: demoting 'mcp-model-context-protocol' directly from ADOPT to HOLD is forbidden by governance.
Step down incrementally: radar.move('rde', 'mcp-model-context-protocol', 1) to move to TRIAL first, then ASSESS, then HOLD.
```

Putting this logic in the tool schema (as `description` strings) doesn't work — the model sees the constraint once at bootstrap but the error at runtime contains no context about what was actually called. Putting it in the proxy method means the error is always contextual: it names the specific tech, team, and current ring.

---

## Why Not a Flat Function List Instead of a Class?

The proxy is a class (`RadarProxyImpl`) rather than a set of exported functions because state must accumulate across multiple `execute()` calls within a session. A model might call `addTechnology`, then `assign`, then `commit` across three separate `execute()` invocations. A stateless function set would need the caller to thread state through; the class holds the in-memory `Radar` object across calls and only writes it on `commit()`.

---

## Scorecard Tradeoffs

The proxy does not implement `validate()` as a pre-flight check that the tool layer calls before every write — `validate()` is exposed to the model as a method it can call explicitly in `search()` before running a write in `execute()`. This was a deliberate choice: forcing a pre-flight check would hide validation from the model and prevent the self-correcting round-trip the spec requires. The model sees the error and can reason about it.
