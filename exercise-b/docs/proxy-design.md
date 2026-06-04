# Proxy Design — Code Mode MCP (Exercise B)

## The Interface

```typescript
// Read surface — present in both search() and execute() sandboxes
interface RadarReadProxy {
  listTechnologies(filter?: { quadrant?: Quadrant }): Technology[]
  listTeams(): Team[]
  listAssignments(teamId: string): Assignment[]
  getAssignment(teamId: string, techId: string): Assignment | undefined
  validate(op: PendingOp): ValidationResult
}

// Write surface — execute() only
interface RadarProxy extends RadarReadProxy {
  addTechnology(id: string, label: string, quadrant: Quadrant): Technology
  assign(teamId: string, techId: string, ring: Ring, moved?: Moved): Assignment
  move(teamId: string, techId: string, newRing: Ring): Assignment
  removeAssignment(teamId: string, techId: string): void
  commit(message: string): void
}
```

The split between `RadarReadProxy` and `RadarProxy` enforces the query/execute boundary structurally. The `search()` sandbox receives only a `RadarReadProxy`, so if a model writes `radar.commit(...)` inside `search()` it gets `TypeError: radar.commit is not a function`. The sandbox enforces it — not a runtime check in the method.

---

## Why This Proxy Shape, Not Another

### A — N tools per CRUD operation (the baseline, rejected)

One MCP tool per operation: `listTechnologies`, `getTechnology`, `listTeams`, `listAssignments`, `getAssignment`, `addTechnology`, `assignTechnology`, `moveTechnology`, `removeAssignment`, `commitChanges`. Each carries its own JSON Schema with inline descriptions for every field. The ring/quadrant enum descriptions repeat across five tools. Bootstrap cost: **1,417 tokens**. Every multi-step workflow is also N round-trips through the model.

### B — Single dispatch tool (rejected)

One `radar` tool that accepts an `op` discriminator and an `args` payload, with a giant discriminated-union JSON Schema. The schema is nearly as large as the N-tool case, the model loses TypeScript pretraining priors, and multi-step workflows still cost one tool call per step.

### C — Raw query language over `data/config.json` (rejected)

Expose the config and let the model write JSONPath or JMESPath. No place to enforce governance rules, no structural read/write split, and the model has weak priors on bespoke query languages.

### D — One tool per high-level workflow (rejected)

Verbs like `promoteTechnology` or `onboardTeam`. Every new workflow requires a server change, and the model cannot compose. The MCP server becomes an application, not a substrate.

### E — Two tools with a TypeScript proxy (chosen)

`search()` (read-only) and `execute()` (read + write), backed by a typed `radar` object. Multi-step workflows run inside a single `execute()` call with no per-step round-trip. The metamodel is declared once as TypeScript type aliases and inlined into `execute()`'s description. This is the only shape that satisfies all four of: structural read/write enforcement, single-call multi-step composition, single metamodel placement, and governance validation co-located with mutation.

---

## DSL Alternatives Considered

### Option A — JSON Schema in tool descriptions (rejected)

The naive approach: embed the full JSON Schema for each operation in the tool description. This is exactly what the N-tool baseline does — every field gets a `description` string that re-explains `0=ADOPT, 1=TRIAL...` across five different tool definitions. Token cost: **1,417**. The metamodel is scattered; a schema change requires updating every tool.

### Option B — Pydantic / Zod validators (rejected for bootstrap)

Pydantic (Python) and Zod (TypeScript) are good proxy-implementation choices but terrible bootstrap DSLs. Their validator syntax is verbose (`z.object({ id: z.string().regex(...) })`), and the validation error messages are mechanical — they tell you what failed but not what to do instead. The self-correcting error requirement means validation logic belongs in the proxy methods, not in a schema validator.

### Option C — Raw prose description (rejected)

Describe the domain in natural language inside the tool description. Cheap in tokens, but the model has no types to reason against — it will guess field names, invent rings that don't exist, and submit un-validatable calls. The DSL exists to give the model a concrete target, not documentation for a human.

### Option D — TypeScript type aliases as the DSL (chosen)

Three advantages for Variant 2:

1. **Token-efficient**: the full metamodel in compact type alias form is **243 tokens** — 17% of the N-tool baseline's total cost. JSDoc verbosity is stripped while keeping enough structure for a pretrained model to reason about valid calls.
2. **Single source of truth**: the DSL block in `execute()`'s description is the only place the metamodel appears in model context. The types in `src/types.ts` are authoritative for implementation; the DSL is a token-trimmed projection of them.
3. **Pretraining leverage**: frontier models have seen vastly more TypeScript than JSON Schema. `type Ring = 0|1|2|3` is immediately understood without explaining what `enum` means.

---

## Validation in the Proxy, Not the Tools

Every write method throws a `ProxyError` when called incorrectly. The error message names the valid alternative — it does not just describe what went wrong:

```
ProxyError: technology 'this-tech-does-not-exist' not found in radar.
Use radar.addTechnology('this-tech-does-not-exist', label, quadrant) to add it first,
or pick from existing: 'claude-haiku-4-5-databricks', ... (use radar.listTechnologies() to see all).
```

```
ProxyError: demoting 'mcp-model-context-protocol' directly from ADOPT to HOLD is forbidden by governance.
Step down incrementally: radar.move('rde', 'mcp-model-context-protocol', 1) to move to TRIAL first,
then ASSESS, then HOLD.
```

Putting this logic in the tool schema (`description` strings) doesn't work — the model sees the constraint once at bootstrap but the runtime error carries no context about what was actually called. Putting it in the proxy method means the error is always contextual: it names the specific tech, team, and current ring.

Five guards are implemented in `src/proxy.ts`:

| Guard | Error message contains |
|---|---|
| `assign` with unknown techId | Suggested `addTechnology` call + 5 existing id examples |
| `assign` on already-assigned tech | Current ring name + suggested `move` call |
| `move` on unassigned tech | Suggested `assign` call |
| `move` from ADOPT to HOLD in one step | Governance rule + incremental next step |
| `addTechnology` with non-kebab-case id | Regex rule + concrete valid example |

---

## Why a Class, Not a Flat Function List

`RadarProxyImpl` is a class because state must accumulate across multiple `execute()` calls within a session. A model might call `addTechnology`, then `assign`, then `commit` across three separate invocations. A stateless function set would require the caller to thread state through; the class holds the in-memory `Radar` object across calls and only writes it on `commit()`.

---

## One Remaining Tradeoff

`validate()` is exposed to the model as a method it can call explicitly in `search()` before running a write in `execute()` — it is not a mandatory pre-flight that the tool layer calls automatically before every write. Forcing a pre-flight would hide validation from the model and remove the self-correcting round-trip the spec requires. The model sees the error from the failed write, reads the corrective message, and rewrites the call. That feedback loop is the point.
