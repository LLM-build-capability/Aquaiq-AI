 # Worked Trace 2 — Multi-Step Workflow & No-Schema-Bloat Win

  **Author:** Pavitra
  **Branch:** pavitra/exercise-b
  **Variant:** V2 — Constrained domain metamodel (Stack.TechRadar)
  **Source:** Captured from `npm run smoke` against `exercise-b/scripts/smoke.ts` on Node 22.

  ---

  ## Purpose

  This trace demonstrates the multi-step workflow advantage of Code Mode. A single `execute` call composes three domain operations — `addTechnology`, `assign`, and `commit` — that the N-tool baseline would have to issue as three separate tool
  calls across three model turns. The trace is captured terminal output, not a prose summary.

  The same scenario also evidences the **no-schema-bloat** property: the Tech Radar metamodel (`Quadrant`, `Ring`, `Moved`, `Technology`, `Team`, `Assignment`) is described to the model exactly once, in the `execute` tool description, rather
  than re-stated across ten per-operation JSON Schemas.

  ---

  ## Setup

  ```bash
  $ export PATH="/opt/homebrew/opt/node@22/bin:$PATH"
  $ node --version
  v22.11.0

  $ cd exercise-b
  $ npm run smoke
  ```

  Each scenario runs against a fresh tmp copy of `data/config.json`, so the on-disk snapshot under version control is untouched.

  ---
  
  ## Workflow Overview

  | Step | Actor | Action |
  |---|---|---|
  | 1 | Model | Issues a single `execute` program: add a new technology, assign it to a team at TRIAL, commit. |
  | 2 | `index.ts` | Forwards the code to `sandbox.ts` in write mode. |
  | 3 | `sandbox.ts` | Compiles the script in a fresh V8 isolate and binds the full `RadarProxy`. |
  | 4 | `proxy.ts` | `addTechnology` validates the kebab-case id and quadrant, applies in memory. |
  | 5 | `proxy.ts` | `assign` validates referential integrity and applies in memory. |
  | 6 | `proxy.ts` | `commit` writes `data/config.json` atomically. |
  | 7 | `sandbox.ts` | Returns the resulting `Assignment` to the model. |

  The host emits exactly **one** tool result. The model spends exactly **one** turn. The N-tool equivalent would be three results across three turns.

  ---
  
  ## Step 1 — Single `execute` Program

  The model issues this `execute` payload once:

  ```js
  // Step 1: confirm the team exists (read).
  const target = radar.listTeams().find(t => t.id === "rde");
  if (!target) throw new Error("RDE team not found");

  // Step 2: add a brand-new technology to quadrant 0 (Models & Providers).
  const tech = radar.addTechnology(
    "claude-haiku-4-5",
    "Claude Haiku 4.5",
    0,
  );

  // Step 3: place it on the RDE radar at TRIAL (ring=1), marked as newly moved in.
  const assignment = radar.assign(target.id, tech.id, 1, 1);

  // Step 4: persist all three operations atomically.
  radar.commit("Add claude-haiku-4-5 to RDE at TRIAL");

  return assignment;
  ```

  The program reads, writes, and commits inside a single isolate run. The host sees one `tools/call` request and returns one result.

  ---
  
  ## Step 2 — Captured Terminal Output

  ```text
  [smoke] scenario 3: execute() runs add + assign + commit in one call
  [sandbox] mode=write  isolate=created  memoryLimit=128MB  timeout=5000ms
  [proxy]   listTeams                    -> 3 teams
  [proxy]   addTechnology
              id=claude-haiku-4-5
              label="Claude Haiku 4.5"
              quadrant=0
              checkKebabCase: ok
              checkUnique: ok
              applied (in-memory)
  [proxy]   assign
              team=rde  tech=claude-haiku-4-5  ring=1  moved=1
              checkTechExists: ok
              checkTeamExists: ok
              checkNotAlreadyAssigned: ok
              applied (in-memory)
  [proxy]   commit
              message="Add claude-haiku-4-5 to RDE at TRIAL"
              wrote /tmp/smoke-config-3.json  bytes=4291
  [sandbox] script returned: {"tech":"claude-haiku-4-5","ring":1,"moved":1}
  [smoke]   result: ✅ assignment persisted in one round-trip
  ```

  ### What Happened

  - The isolate executed the entire program in a single run; the model paid one round-trip.
  - Each domain method passed through its validation guards in `proxy.ts` before mutating the in-memory state.
  - No write touched disk until `commit` ran, so a failure at any earlier step would have left the file unchanged.
  - The returned `Assignment` is the value of the final expression in the script, surfaced verbatim by `sandbox.ts`.

  ---

  ## Round-Trip Confirmation

  ```text
  [smoke] post-run check
  [smoke]   getTechnology("claude-haiku-4-5") -> {"id":"claude-haiku-4-5","label":"Claude Haiku 4.5","quadrant":0}
  [smoke]   getAssignment("rde", "claude-haiku-4-5") -> {"tech":"claude-haiku-4-5","ring":1,"moved":1}
  [smoke]   diff vs starting snapshot:
              new technology entry: claude-haiku-4-5
              new assignment under team "rde": ring=1, moved=1
  [smoke] all scenarios: 8/8 ok
  ```

  ---

  ## N-Tool Baseline — What This Replaces

  The same outcome under the ten-tool baseline would require three independent tool calls and three model turns:

  | Turn | Tool call | Notes |
  |---|---|---|
  | 1 | `addTechnology({ id, label, quadrant })` | Schema repeats `Quadrant` enum. |
  | 2 | `assignTechnology({ teamId, techId, ring, moved })` | Schema repeats `Ring` and `Moved` enums. |
  | 3 | `commitChanges({ message })` | Separate atomic-write call. |

  Each call carries its own JSON Schema, its own description block, and its own context round-trip. The Code-Mode equivalent issues one `execute` call against a typed `radar` object instead.

  ---
  
  ## No-Schema-Bloat Evidence

  | Configuration | Tools | Bootstrap tokens (cl100k_base) | Where the metamodel lives |
  |---|---|---|---|
  | N-tool baseline | 10 | **1,417** | Repeated across 5 tool descriptions (`Quadrant` × 3, `Ring` × 2). |
  | Code Mode (`search` + `execute`) | 2 | **≤ 1,200** (target) | Declared once as a TypeScript interface block in `execute`. |

  ### Why the Bloat Disappears

  - **Single tool description carries the DSL.** `index.ts` inlines the metamodel into the `execute` description as a TypeScript interface block. `search` does not repeat it.
  - **Enums are declared once.** `Quadrant`, `Ring`, and `Moved` appear in `types.ts` and are referenced from the interface block. The N-tool baseline restates each enum inline in every tool description that uses it.
  - **Multi-step workflows do not multiply schema cost.** Adding a new domain method (say `archiveTechnology`) is one method on `RadarProxy` and one extra signature in the inlined block — not a new tool with its own JSON Schema wrapper, 
  properties block, and required array.
  - **Validation rules are not in the schema at all.** Kebab-case enforcement, ring-transition rules, and referential integrity live in `proxy.ts`. They cost zero bootstrap tokens because the model learns them from `ProxyError` messages at 
  runtime, only when relevant.

  ---
  
  ## Why This Is the Multi-Step Win

  - **One tool call replaces three.** `addTechnology` → `assign` → `commit` runs inside one isolate, returning one result.
  - **Atomicity is explicit.** `commit` is the single write boundary; a failure before it leaves `data/config.json` untouched.
  - **The model composes naturally.** It can read first, branch on the result, then mutate, all in the same script — exactly the pattern shown above where `listTeams().find(...)` gates the subsequent writes.
  - **Per-step round trips are eliminated.** No serialization, no re-prompting, no schema re-parsing between steps. The model writes a small program; the host runs it.
  - **The DSL is described once.** Adding a fourth or fifth step to the workflow does not grow the bootstrap cost — only the script body grows, and that is paid only on the turn that uses it.

  ---

  ## Reproducing the Trace

  ```bash
  $ cd exercise-b
  $ npm run smoke
  # Scenario 3 reproduces this multi-step workflow.
  ```

  The terminal output above is taken directly from a single run of the smoke harness. Rerunning the script reproduces the same sequence because the per-scenario tmp config is reseeded from the committed snapshot before each run, and the program 
  itself is deterministic.

