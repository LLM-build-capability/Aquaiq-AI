# Exercise B — Deepak's Handoff & Team Status

> **Audience:** Premkumar, Pavitra, bootcamp lead.
> **Branch:** `deepak/exercise-b` (now in sync with `team/exercise-b` after merges).
> **Status:** Prem's and Deepak's slices are merged into `team/exercise-b`. Pavitra's slice is in progress on her branch (not yet merged).

---

## 1. TL;DR

| Owner | Slice | Status |
|---|---|---|
| **Premkumar** | Domain types + RadarProxy + validation + config snapshot | ✅ Merged to `team/exercise-b` (PRs #13, #16) |
| **Deepak** | `isolated-vm` sandbox + MCP server (stdio) + smoke test | ✅ Merged to `team/exercise-b` (PR #15) |
| **Pavitra** | Token measurement + 3 design docs + 2 worked traces + README + scorecard | 🛠️ In progress on `pavitra/exercise-b` (commit `f21d15a`); not yet merged |

**Exercise B overall completion: ~75–80% by deliverable count.** Implementation + infrastructure are merged; Pavitra's measurement + write-up layer is drafted on her branch and pending PR/review.

**My personal slice: 100% — and now merged to `team/exercise-b` via PR #15.** sandbox.ts, index.ts, smoke test, and the setup that makes them runnable.

---

## 2. What I Shipped (Deepak's commits)

Five logical commits on top of Prem's scaffold (`a76884d feat(exercise-b): add project scaffold + RadarProxy implementation`):

```
8f9fa73 test(exercise-b): add end-to-end smoke test for sandbox + proxy
e768bff feat(exercise-b): add MCP server (stdio) with search() and execute() tools
06532c3 feat(exercise-b): add isolated-vm sandbox runner with read/write modes
d8950c0 fix(exercise-b): set tsconfig rootDir to "." so scripts/* typechecks
9383d18 chore(exercise-b): gitignore node_modules + dist; commit package-lock.json
```

### Files added

| File | Purpose | Lines |
|---|---|---|
| `exercise-b/src/sandbox.ts` | `isolated-vm` V8-isolate runner. Two modes: read (RadarReadProxy methods only) and write (full RadarProxy). Bridges proxy methods into the isolate via `ivm.Reference` and exposes them as a frozen `radar` object. Captures `console.log`, enforces 5s timeout / 128MB memory. | ~160 |
| `exercise-b/src/index.ts` | MCP server over stdio using `@modelcontextprotocol/sdk` 1.29.0. Registers two tools: `search` (read-only) and `execute` (read+write). Carries the DSL bootstrap **once** in `execute`'s description (per the framework rule that the metamodel appears in one place, not per tool). | ~175 |
| `exercise-b/scripts/smoke.ts` | End-to-end test that exercises the sandbox + proxy directly. 8 scenarios covering happy path, write-isolation, multi-step workflows, all 3 self-correcting error cases, corrected retry, and timeout. | ~150 |
| `exercise-b/.gitignore` | Excludes `node_modules/` (~100 MB) and `dist/`. | 2 |

### Files modified

| File | Why |
|---|---|
| `exercise-b/tsconfig.json` | Changed `rootDir` from `"src"` to `"."` so `scripts/*.ts` (count-tokens.ts, smoke.ts) typechecks alongside `src/`. Was a latent bug — would have hit Pavitra when she ran typecheck on count-tokens.ts. |
| `exercise-b/package.json` | Added `npm run smoke`. |
| `exercise-b/package-lock.json` | Newly committed for reproducible installs. |

---

## 3. How the Pieces Fit Together (For the Whole Team)

```
                       ┌────────────────────────────────────────────────────┐
                       │           MCP host (Claude Code, Cursor)           │
                       └───────────────────────┬────────────────────────────┘
                                               │ JSON-RPC over stdio
                       ┌───────────────────────▼────────────────────────────┐
                       │     index.ts        — MCP server                   │
                       │       • lists 2 tools: search / execute            │
                       │       • DSL bootstrap lives once in execute desc   │
                       │       • shares one RadarProxyImpl across calls     │
                       └───────────────┬───────────────────────┬────────────┘
                            mode=read  │                       │  mode=write
                       ┌───────────────▼─────────┐   ┌─────────▼──────────────┐
                       │  sandbox.ts             │   │  sandbox.ts            │
                       │  isolated-vm (V8)       │   │  isolated-vm (V8)      │
                       │  exposes only:          │   │  exposes:              │
                       │  listTechnologies, …    │   │  reads + addTechnology │
                       │  validate               │   │  + assign + move +     │
                       │                         │   │  removeAssignment +    │
                       │                         │   │  commit                │
                       └───────────────┬─────────┘   └─────────┬──────────────┘
                                       │   ivm.Reference       │
                                       │   (call host fn,      │
                                       │    deep-copy args)    │
                       ┌───────────────▼───────────────────────▼────────────┐
                       │  proxy.ts — RadarProxyImpl  (Prem)                 │
                       │    • reads data/config.json on construction        │
                       │    • mutates in memory                             │
                       │    • commit(msg) writes data/config.json back      │
                       │    • throws ProxyError with self-correcting text   │
                       └────────────────────────────────────────────────────┘
                                                │
                       ┌────────────────────────▼───────────────────────────┐
                       │  data/config.json — Tech Radar snapshot (Prem)     │
                       └────────────────────────────────────────────────────┘
```

**The core idea (in plain English):** instead of giving the LLM 10 separate tools (one per CRUD operation), we give it 2 tools that take *JavaScript code* as input. The model writes code that calls a `radar` object — and that object's methods are bridged into Prem's RadarProxy on the host side. So the LLM is doing one tool call ("execute this code") instead of 10 round-trips, and the metamodel (quadrants, rings, etc.) is described once instead of repeated in every tool's schema.

**Why two tools instead of one:** `search` is structurally read-only — its sandbox literally does not have `radar.commit` (`typeof radar.commit === 'undefined'` in read mode). That prevents accidental mutations during introspection. The split is enforced by `sandbox.ts`, not by trust.

---

## 4. How to Run It Locally

### Prerequisite — Node 22

`isolated-vm` does not build on newer Node (26+). You **must** use Node 22.

```bash
brew install node@22
# Always prepend Node 22 to PATH when working in exercise-b/:
export PATH="/opt/homebrew/opt/node@22/bin:$PATH"
node --version   # should print v22.x
```

### Install

```bash
cd exercise-b
npm install        # installs @modelcontextprotocol/sdk, isolated-vm (native build), tiktoken, tsx
```

### Verify everything works

```bash
npm run typecheck  # should print nothing (clean)
npm run smoke      # runs 8 end-to-end scenarios; all 8 should be ok
```

The smoke test creates a fresh tmp copy of `data/config.json` for each scenario, so re-runs are idempotent and never touch the committed snapshot.

### Run the actual MCP server

```bash
npm start          # boots the server on stdio
# or with a custom config:
RADAR_CONFIG_PATH=/tmp/my-radar.json npm start
```

The server only logs `[tech-radar-mcp] connected — config: …` to stderr (stdout is reserved for MCP framing). Connect Claude Code / Cursor / MCP Inspector to the spawned process.

---

## 5. Verified Working — Evidence

From `npm run smoke` (full output reproducible — re-run any time):

| # | Scenario | Result |
|---|---|---|
| 1 | `search()` returns a 3-element projection of quadrant-0 ids | ✅ Returns just the ids, not the full Technology objects |
| 2 | `typeof radar.commit` inside `search()` | ✅ Returns `"undefined"` — write methods are structurally absent |
| 3 | `execute()` runs add + assign + commit in one call | ✅ Returns the assignment; `data/config.json` updated on disk |
| 4 | `execute()` calls `radar.assign('rde', 'this-tech-does-not-exist', 1)` | ✅ Errors with: *"technology 'this-tech-does-not-exist' not found in radar. Use radar.addTechnology(…) to add it first, or pick from existing: …"* |
| 5 | `execute()` tries to demote ADOPT → HOLD in one step | ✅ Errors with: *"demoting … directly from ADOPT to HOLD is forbidden by governance. Step down incrementally: radar.move(…, 1) to move to TRIAL first, then ASSESS, then HOLD."* |
| 6 | Corrected retry — ADOPT → TRIAL | ✅ Returns `{ tech, ring: 1, moved: -1 }`, persisted |
| 7 | `radar.addTechnology('Not Kebab Case', …)` | ✅ Errors with: *"id 'Not Kebab Case' is not valid kebab-case. Use only lowercase letters, digits, and hyphens, e.g. 'my-new-tool'."* |
| 8 | Infinite loop with 250ms budget | ✅ Errors with: *"Script execution timed out."* |

Also verified directly via the MCP protocol (separate manual test): the server answers `initialize`, `tools/list`, and `tools/call` correctly over stdio JSON-RPC.

---

## 6. What's Left — Pavitra's Slice

> **Update (2026-05-29):** Pavitra has drafted this layer on `pavitra/exercise-b` (commit `f21d15a` — `proxy-design.md`, `sandbox-choice.md`, `scorecard.md`, `trace-1.md`, `trace-2.md`, README, scripts). It is **not yet merged** — pending a PR into `team/exercise-b` and review. ⚠️ Her commit also accidentally tracked `exercise-b/node_modules/`; that should be removed (`git rm -r --cached exercise-b/node_modules`) before her PR merges, otherwise the expanded `.gitignore` won't retroactively untrack it.

Pavitra owns the *measurement, docs, and traces* layer. The smoke test output (section 5) is intentionally structured so it can be lifted directly into the worked-trace docs — no need to re-run scenarios by hand.

### Concrete TODO list

1. **`scripts/count-tokens.ts`** — measure bootstrap token cost for the Code Mode server. The cleanest source of truth is the JSON of the `tools/list` response from `index.ts`. Use `tiktoken` cl100k_base. Compare against the 1,417 tokens already documented in `docs/n-tool-baseline.md`. Target is **≤ 1,200** (per the framework's V2 rule).

   ```ts
   // sketch
   import { get_encoding } from 'tiktoken';
   import { TOOLS } from '../src/index.js';   // export TOOLS from index.ts if helpful
   const enc = get_encoding('cl100k_base');
   const tokens = enc.encode(JSON.stringify(TOOLS, null, 2));
   console.log(tokens.length);
   enc.free();
   ```

   *Note for Deepak:* if you need `TOOLS` exported from `index.ts`, ping me — it's a 1-line change.

2. **`docs/proxy-design.md`** — one page on *why this proxy shape, not another*. Cover:
   - Why TypeScript types and not JSON Schema (see assignment FAQ).
   - Why the read/write split is on the proxy, not the tool schema.
   - Alternatives considered: raw `search` over the config file, Pydantic, single `dispatch(op)` style.

3. **`docs/sandbox-choice.md`** — one paragraph defending **`isolated-vm`** against alternatives. Address security isolation explicitly (V8 isolates *do* provide it; `node:vm` does not). Bullet against the framework's table:
   - `isolated-vm` ✅ true V8 isolation, native build cost
   - `node:vm` ❌ context isolation only
   - `RestrictedPython` ❌ wrong language for our DSL
   - subprocess+timeout — heavier; lose the proxy bridge ergonomics

4. **Worked trace 1 — self-correcting validation round-trip.** Lift scenarios 4–6 from the smoke test (deliberate misuse → structured error with named alternatives → corrected retry succeeds → state persisted).

5. **Worked trace 2 — multi-step workflow / constraint-validation win.** Lift scenario 3 (add + assign + commit in one round-trip vs. 3+ tool calls in the N-tool world). Side-by-side token counts: from the n-tool-baseline + Pavitra's count-tokens.ts.

6. **`docs/scorecard.md` (or in README)** — fill the framework's 7-dimension scorecard:

   | Dimension | Notes for filling in |
   |---|---|
   | Tool count | 2 (down from 10). Score 3. |
   | Bootstrap token cost | Measure with count-tokens.ts. Target ≤ 1,200; below 1,000 is great. |
   | Metamodel location | DSL types appear once in `execute`'s description. Score 2–3. |
   | Credential exposure | N/A for V2 (no creds). Note this honestly — don't claim 3 if it doesn't apply. |
   | Multi-step workflows | Single `execute()` does add+assign+commit (scenario 3). Score 3. |
   | Validation error quality | Three self-correcting cases demonstrated (scenarios 4, 5, 7). Score 3. |
   | Result verbosity control | Model controls projection via `return …slice(0,3).map(…)` etc. (scenario 1). Score 2–3. |

   Aim for **12–17 total** (the framework's expected range). If you score 21, double-check.

7. **`README.md`** — bring it all together:
   - Variant chosen (V2) + one-paragraph justification.
   - Proxy interface (just paste `RadarReadProxy` + `RadarProxy` from `proxy.ts`).
   - How to run (link to / copy section 4 here).
   - Before/after token table.
   - Filled scorecard.

---

## 7. Completion Percentage Breakdown

By deliverable count (from the assignment's `## Deliverables` section + the team plan):

| Owner | Deliverables | Done | % of personal slice | % of total Exercise B |
|---|---|---|---|---|
| Premkumar | types.ts, proxy.ts (with validation), config.json snapshot, n-tool-baseline.md | 4 / 4 | **100%** | ~30% (merged) |
| **Deepak (me)** | sandbox.ts, index.ts, smoke test (incl. infra: gitignore, tsconfig fix, lockfile) | 3 / 3 | **100%** | ~30% (merged) |
| Pavitra | count-tokens.ts, proxy-design.md, sandbox-choice.md, trace 1, trace 2, README, scorecard | drafted on branch | **~85%** | ~40% (pending PR) |

**Total Exercise B: ~75–80% complete.** Prem's and Deepak's slices are merged to `team/exercise-b`. Pavitra's writeup + measurement layer is drafted on `pavitra/exercise-b` and awaiting a reviewed PR. None of her items were ever blocked — the proxy, sandbox, and server are all running, and the smoke test gives her trace data on demand.

> **Note on percentages:** these are by *deliverable count*, not lines of code or hours of work. They're meant to give the team a rough split, not to grade individual contributions. Final evaluation is the bootcamp lead's scorecard.

---

## 8. Team Notes

### Branch & PR rules
- Nobody pushes directly to `main` or `team/exercise-b` / `team/exercise-c`. Always feature branch → PR.
- **As of 2026-05-29 this is enforced by GitHub branch rulesets** on `main` and `team/exercise-b`: direct commits are rejected server-side, and every change must go through a PR with at least one review.
- My branch `deepak/exercise-b` is merged into `team/exercise-b` (PR #15) and is in sync after pulling the latest team merges.

### Merge status / order
1. ✅ **Prem's PRs #13 + #16** (`premkumar/exercise-b` → `team/exercise-b`) merged — types + proxy + n-tool-baseline + scaffold + expanded `.gitignore` are on `team/exercise-b`.
2. ✅ **My PR #15** (`deepak/exercise-b` → `team/exercise-b`) merged — sandbox + MCP server + smoke test on top.
3. ⏳ **Pavitra's PR** (`pavitra/exercise-b` → `team/exercise-b`) still to come — docs, traces, scorecard, README. (Needs the `node_modules` removal noted in §6 before merging.)

### Hard runtime constraint
**Always use Node 22** in `exercise-b/`. `isolated-vm` does not build on Node 26. Set `PATH="/opt/homebrew/opt/node@22/bin:$PATH"` in your shell or wrap commands.

### Where I'd help next
- Wiring `count-tokens.ts` against `index.ts`'s tool definitions (5-min change to export `TOOLS`).
- Pairing on the scorecard if Pavitra wants a sanity check.
- Hooking the server into Exercise C's MCP-agent wrapper when we get there.

---

*Generated 2026-05-27, updated 2026-05-29 to reflect merges (PRs #13/#15/#16), Pavitra's drafted slice, and the new branch rulesets. Ping me on Teams if anything in here is unclear.*
— Deepak
