 # Proxy Design — Tech Radar MCP Server (Variant 2)

  **Author:** Pavitra
  **Branch:** pavitra/exercise-b
  **Variant:** V2 — Constrained domain metamodel (Stack.TechRadar)

  This document specifies the proxy layer that backs the Code-Mode MCP server
  for Stack.TechRadar (Variant 2). It records the proxy's shape, the reasoning
  behind it, the validation contract it enforces, and the boundaries it
  deliberately does not cross.

  ---

  ## Overview

  The Code-Mode server exposes a single `execute` tool. The model writes short
  TypeScript or JavaScript programs that call a bound `radar` object inside an
  isolated-vm sandbox. That `radar` object is a proxy, not the data — every
  method on it is a thin, validated bridge into the `RadarProxy` implementation
  running on the host. The proxy is the only surface that touches
  `data/config.json`; the sandbox cannot reach the filesystem, the network, or
  the host process.

  The proxy's job is therefore narrow and load-bearing: present a typed,
  governance-aware API to the model, reject misuse with self-correcting error
  messages, and defer all persistence to an explicit `commit` call.

  ---

  ## Design Goals

  - Present one tool to the model instead of N CRUD tools, to cut
  tool-definition token cost and let the model compose multi-step workflows in a
  single call.
  - Encode the Tech Radar metamodel — rings, quadrants, kebab-case ids,
  governance rules — once in TypeScript, and reuse those types as the
  model-facing contract.
  - Enforce a structural read/write split at the sandbox boundary so read-only
  programs are statically incapable of mutating state.
  - Keep mutations in memory until `commit`, so a failing program leaves
  `data/config.json` untouched.
  - Throw errors that tell the model what to do next, not just what went wrong.

  ---

  ## Architecture

  ```text
              +-------------------------------+
     MCP <--> |   MCP Server  (src/index.ts)  |
              |   - registers execute() tool  |
              +---------------+---------------+
                              |
                              v
              +-------------------------------+
              | isolated-vm sandbox           |
              | (src/sandbox.ts)              |
              |   - compiles model code       |
              |   - injects bound radar       |
              |   - read isolate: RadarRead   |
              |   - write isolate: RadarProxy |
              +---------------+---------------+
                              |  bridged calls
                              v
              +-------------------------------+
              |   RadarProxy  (src/proxy.ts)  |
              |   - validation                |
              |   - in-memory mutations       |
              |   - commit() -> file write    |
              +---------------+---------------+
                              |
                              v
                      data/config.json

  The sandbox binding in sandbox.ts constructs the radar object differently
  depending on mode. In read mode it binds only the RadarReadProxy surface —
  listTechnologies, listTeams, listAssignments, getAssignment, and validate. The
  write methods — addTechnology, assign, move, removeAssignment, and commit —
  are absent from the isolate's global, so typeof radar.commit returns undefined
  inside a read program. The split is structural rather than a runtime check:
  there is no write method to call, regardless of what the model attempts.

  RadarProxy extends RadarReadProxy on the host, so the write isolate sees the
  full surface and the read isolate sees a strict subset, both backed by the
  same underlying state object.

  ---
  Interfaces

  // Read surface — present in both sandboxes
  export interface RadarReadProxy {
    listTechnologies(filter?: { quadrant?: Quadrant }): Technology[];
    listTeams(): Team[];
    listAssignments(teamId: string): Assignment[];
    getAssignment(teamId: string, techId: string): Assignment | undefined;
    validate(op: PendingOp): ValidationResult;
  }

  // Write surface — execute() only
  export interface RadarProxy extends RadarReadProxy {
    addTechnology(id: string, label: string, quadrant: Quadrant): Technology;
    assign(teamId: string, techId: string, ring: Ring, moved?: Moved):
  Assignment;
    move(teamId: string, techId: string, newRing: Ring): Assignment;
    removeAssignment(teamId: string, techId: string): void;
    commit(message: string): void;
  }

  // Structured error — message always names what to try instead
  export class ProxyError extends Error {
    constructor(message: string) {
      super(message);
      this.name = "ProxyError";
    }
  }

  ---
  Flow

  1. The model writes JavaScript inside an execute call — typically a short
  program that reads current state, derives a change, and commits.
  2. index.ts forwards the code to sandbox.ts, which compiles it inside an
  isolated-vm isolate and injects the bound radar global.
  3. The program runs. Each call on radar is bridged across the isolate boundary
  to the corresponding method on RadarProxy in proxy.ts.
  4. Reads return plain data snapshots. Mutations are validated, then applied to
  an in-memory copy of the config. Nothing touches disk yet.
  5. When the program calls commit, RadarProxy runs a final validation pass and
  writes the in-memory config back to data/config.json. If the program throws or
  returns without committing, the on-disk file is unchanged.

  A typical model-authored program looks like this:

  const teams = radar.listTeams();
  const target = teams.find((t) => t.id === "rde");

  radar.addTechnology("claude-haiku-4-5", "Claude Haiku 4.5", 0);
  radar.assign(target.id, "claude-haiku-4-5", 1);
  radar.commit("Add claude-haiku-4-5 to RDE at TRIAL");

  ---
  Why TypeScript Types, Not JSON Schema

  Token cost. The N-tool baseline costs 1,417 tokens for the JSON Schema
  definitions of the CRUD tools alone, before any conversation has happened. The
  Code Mode target is no more than 1,200 tokens for the combined execute tool
  definition plus the inlined TypeScript interface block. TypeScript interfaces
  are denser than the equivalent JSON Schema: no object-type wrapper, no
  properties wrapper, no required array — just one line per field.

  ---
  Flow

  1. The model writes JavaScript inside an execute call — typically a short program that reads current state, derives a change, and commits.
  2. index.ts forwards the code to sandbox.ts, which compiles it inside an isolated-vm isolate and injects the bound radar global.
  3. The program runs. Each call on radar is bridged across the isolate boundary to the corresponding method on RadarProxy in proxy.ts.
  4. Reads return plain data snapshots. Mutations are validated, then applied to an in-memory copy of the config. Nothing touches disk yet.
  5. When the program calls commit, RadarProxy runs a final validation pass and writes the in-memory config back to data/config.json. If the program throws or returns without committing, the on-disk file is unchanged.

  A typical model-authored program looks like this:

  const teams = radar.listTeams();
  const target = teams.find((t) => t.id === "rde");

  radar.addTechnology("claude-haiku-4-5", "Claude Haiku 4.5", 0);
  radar.assign(target.id, "claude-haiku-4-5", 1);
  radar.commit("Add claude-haiku-4-5 to RDE at TRIAL");

  ---
  Why TypeScript Types, Not JSON Schema

  Token cost. The N-tool baseline costs 1,417 tokens for the JSON Schema definitions of the CRUD tools alone, before any conversation has happened. The Code Mode target is no more than 1,200 tokens for the combined execute tool definition plus
  the inlined TypeScript interface block. TypeScript interfaces are denser than the equivalent JSON Schema: no object-type wrapper, no properties wrapper, no required array — just one line per field.

  TypeScript priors. Frontier models have seen vastly more TypeScript than JSON Schema in training. A signature like assign(techId, teamId, ring) is interpreted faster and more reliably than the equivalent schema, and discriminated union types
  carry the enum constraint inline.

  Single source of truth. The same interfaces in types.ts are imported by proxy.ts for the host implementation, by sandbox.ts for the isolate binding, and serialized into the execute tool description for the model. One edit propagates
  everywhere; there is no schema-versus-implementation drift.

  Validation surface. Types catch shape errors — missing field, wrong primitive. They do not catch domain errors such as an unknown technology id, an illegal ring transition, or a malformed identifier. Those live in RadarProxy and surface via
  ProxyError with corrective messages. The split is intentional: types fence off the easy mistakes so error messages can focus on the hard ones.

  ---
  Why This Proxy Shape, Not Another

  A. N tools per CRUD operation (the baseline)

  Shape: addTechnology, assignTechnology, moveTechnology, removeAssignment, listTechnologies, listTeams, listAssignments, getAssignment, validate, and commit — each as its own MCP tool with its own JSON Schema.

  Rejected because: 1,417 tokens of tool definitions before the first user turn, and every multi-step workflow becomes N round-trips through the model.

  B. Single dispatch tool

  Shape: one radar tool that accepts an op discriminator and an args payload, with a giant discriminated-union schema.

  Rejected because: the schema is nearly as large as the N-tool case, the model loses TypeScript priors, and multi-step workflows still cost one tool call per step.

  C. Raw query language over data/config.json

  Shape: expose the config and let the model write JSONPath, JMESPath, or a SQL-like DSL.

  Rejected because: there is no place to enforce governance rules, no read/write split, and the model has weak priors on bespoke query languages.

  D. One tool per workflow

  Shape: high-level verbs such as promoteTechnology or onboardTeam.

  Rejected because: every new workflow requires a server change, and the model cannot compose. The MCP server stops being a substrate and becomes an application.

  What we picked, and why it wins

  1. Structural read/write split. RadarReadProxy and RadarProxy are distinct interfaces, and the sandbox binding in sandbox.ts chooses which to inject. Read programs cannot mutate, by construction.
  2. Multi-step workflows in one execute call. The model reads, branches, mutates, validates, and commits in a single program — no per-step round trip.
  3. Validation next to mutation. RadarProxy owns both the data and the rules. There is no path that writes without passing through validation.
  4. Metamodel declared once. Ring, Quadrant, Technology, and Assignment live in types.ts and are reused by the host, the sandbox binding, and the model-facing tool description.

  ---
  Self-Correcting Validation
  
  ProxyError is thrown by RadarProxy with messages that name the offending input, explain the rule, and suggest the next call. The sandbox surfaces the message verbatim to the model, so the model's next program can correct itself without a human
  in the loop.

  ┌──────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────┐
  │                    Misuse                    │                                                                ProxyError message (abridged)                                                                 │     Source location     │
  ├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────┤
  │ radar.assign with an unknown techId          │ technology … not found in radar. Use radar.addTechnology(…) to add it first, or pick from existing: …                                                        │ proxy.ts, lines 190–202 │
  ├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────┤
  │ radar.move from ADOPT to HOLD in one step    │ demoting … directly from ADOPT to HOLD is forbidden by governance. Step down incrementally: radar.move(…, 1) to move to TRIAL first, then ASSESS, then HOLD. │ proxy.ts, lines 226–239 │
  ├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────┤
  │ radar.addTechnology with a non-kebab-case id │ id 'Not Kebab Case' is not valid kebab-case. Use only lowercase letters, digits, and hyphens, e.g. 'my-new-tool'.                                            │ proxy.ts, lines 253–260 │
  └──────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────┘

  A representative guard, lifted from proxy.ts:

  private checkDemotionRule(teamId: string, techId: string, newRing: Ring): void {
    const existing = this.radar.assignments[teamId]?.find((a) => a.tech === techId);
    if (!existing) return;

    // Governance: jumping directly from ADOPT (0) to HOLD (3) is forbidden.
    // Must step through TRIAL (1) or ASSESS (2) first.
    if (existing.ring === 0 && newRing === 3) {
      throw new ProxyError(
        `demoting '${techId}' directly from ADOPT to HOLD is forbidden by governance. ` +
          `Step down incrementally: radar.move('${teamId}', '${techId}', 1) ` +
          `to move to TRIAL first, then ASSESS, then HOLD.`,
      );
    }
  }

  ---
  Benefits
  
  - Prevents misuse. The read/write split is structural rather than advisory; kebab-case and ring-transition rules live in the same file as the mutation, so no path bypasses them.
  - Improves modularity. index.ts handles MCP wiring, sandbox.ts handles isolation and binding, proxy.ts handles domain rules, and types.ts defines the metamodel. Each file has one reason to change.
  - Enhances maintainability. Adding a field to Technology is one edit in types.ts plus its validation in proxy.ts; the model-facing tool description regenerates from the types, so there is no schema to keep in sync.
  - Reduces token cost. The combined execute tool definition plus inlined interfaces stays under the 1,200-token target, against the 1,417-token baseline of the N-tool design.
  - Enables composable workflows. A single execute call can read state, compute a change, mutate, validate, and commit, eliminating per-step model round trips.

  ---
  What the Proxy Does Not Do
  
  - No auto-push to a remote. The commit method writes data/config.json on the local filesystem only. Pushing to a git remote, a registry, or a deployed radar is out of scope.
  - No streaming or pagination. The list methods return full arrays. The dataset is small enough that paging would add complexity without benefit.
  - No transactions across execute calls. All mutations within a single program are held in memory; commit is the atomic write boundary. There is no rollback log, no two-phase commit, and no concurrent-writer coordination.

  ---
  Connection to the Scorecard
  
  ┌────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │  Scorecard dimension   │                                                    How the proxy earns it                                                     │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Token efficiency       │ Single execute tool plus TypeScript interfaces stays under the 1,200-token target against the 1,417-token N-tool baseline.    │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Code Mode fidelity     │ The model writes real JavaScript against a typed radar object inside isolated-vm; no JSON-RPC simulation.                     │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Sandbox safety         │ The isolated-vm isolate has no filesystem, network, or process access; only the bound radar proxy crosses the boundary.       │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Read/write separation  │ RadarReadProxy and RadarProxy are distinct interfaces; read isolates have no commit method bound at all.                      │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Governance enforcement │ Kebab-case ids, ring-transition rules, and referential integrity are validated in proxy.ts before any mutation lands.         │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Self-correction        │ ProxyError messages name the offending input and suggest the next call, enabling the model to recover without operator input. │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Maintainability        │ types.ts is the single source of truth; index.ts, sandbox.ts, and proxy.ts each own one concern.                              │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Atomicity              │ Mutations are in memory until commit; failed programs leave data/config.json untouched.                                       │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ ```                    │                                                                                                                               │
  └────────────────────────┴──────────────────────────────────────────────────────────────────────