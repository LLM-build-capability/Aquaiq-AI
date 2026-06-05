<h1 align="center">Tech Radar — Code-Mode MCP Server</h1>

<p align="center">
  <font color="gray" size="3">
    Token-efficient MCP server for Stack.TechRadar — replacing 10 naive tools with 2 using the Code Mode pattern
  </font>
</p>

**Authors:**
- Prem Kumar Reddy K
- Deepak P
- Pavitra P

**Cohort:** LLM Capability

---

## Table of Contents

- [About the Project](#about-the-project)
- [Learning Objectives](#learning-objectives)
- [Variant Classification](#variant-classification)
- [Proxy Interface](#proxy-interface)
- [Domain Metamodel](#domain-metamodel)
- [Validation Rules](#validation-rules)
- [Required Stack](#required-stack)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Code File Descriptions](#code-file-descriptions)
- [Data Sources](#data-sources)
- [Tool Routing Logic](#tool-routing-logic)
- [Workflow Diagrams](#workflow-diagrams)
- [Installation Guide](#installation-guide)
- [Environment Variables](#environment-variables)
- [How to Run the Project](#how-to-run-the-project)
- [How to Test the Project](#how-to-test-the-project)
- [Outputs Generated](#outputs-generated)
- [Results and Token Numbers](#results-and-token-numbers)
- [Evaluation Scorecard](#evaluation-scorecard)
- [Evaluation Rubric](#evaluation-rubric)
- [Explicit Non-Goals](#explicit-non-goals)
- [Version Control Expectations](#version-control-expectations)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Resources](#resources)
- [FAQ / Common Pitfalls](#faq--common-pitfalls)
- [Acknowledgements](#acknowledgements)

---

## About the Project

Native MCP servers leak tokens. A 2,500-endpoint API exposed as MCP tools costs ~1.17M tokens of bootstrap context — before any reasoning has happened. Even a small domain server, designed naively, repeats the same metamodel across seven tool schemas.

**Code Mode** is a structural rewrite: replace N tools with two (`search` and `execute`) and let the model write code against a server-side proxy. The savings depend on what kind of MCP server you have.

The **Tech Radar Code-Mode MCP Server** applies this pattern to the [Stack.TechRadar](https://github.com/LLM-build-capability/Stack.TechRadar_MultiTeam) — a shared technology radar tracking 89 technologies across 6 teams. Instead of exposing one MCP tool per domain operation (the naive approach costs 1,417 bootstrap tokens), it exposes exactly **two tools**:

- **`search(code)`** — runs read-only JavaScript in a sandboxed V8 isolate. The model writes code against a `radar` proxy object to query technologies, teams, and assignments.
- **`execute(code)`** — runs read + write JavaScript in the same sandbox. The model writes multi-step workflows (add → assign → commit) in a single call instead of N round-trips.

The entire domain metamodel is declared once as a compact TypeScript DSL bootstrap (~243 tokens) in `execute()`'s tool description. All validation — including self-correcting error messages that name the exact corrective call — lives in the proxy, not the tool schema.

**Bootstrap token reduction: 1,417 → 949 (33% cheaper).**

> **The proxy is what we designed, not the tool schema.** That is the core lesson this exercise teaches.

---

## Learning Objectives

By the end of Exercise B you can:

1. Read an existing MCP server, **count its bootstrap token cost** with `tiktoken` (cl100k_base), and classify it as **Variant 1 / 2 / 3** using the framework's decision heuristic.
2. Design a **proxy interface in TypeScript** that captures the domain abstraction. The tool schema collapses to almost nothing once the proxy is right.
3. Implement **`search()` + `execute()`** with a sandbox runtime appropriate to your stack — and *defend* the choice. Security isolation and context isolation are not the same thing.
4. Write **self-correcting validation errors** — every error must tell the model what to try instead. Demonstrated with a deliberate misuse and the corrected retry in `docs/trace-1.md`.
5. Separate **`query`-path** from **`execute`-path** to prevent accidental mutation during introspection. Demonstrated in `docs/trace-2.md`.
6. Measure **before / after** bootstrap token cost and fill in the framework's evaluation scorecard (7 dimensions × 0–3). Scored honestly — the framework is not your friend if you cherry-pick.
7. Defend your DSL choices against alternatives — *why TypeScript types and not JSON Schema; why this proxy shape and not that one.* See `docs/proxy-design.md`.

---

## Variant Classification

The framework defines three structural variants. We classified our server before picking one.

| Variant | What drives token cost | Decision heuristic |
|---|---|---|
| **V1 — Large External API** | Schema bloat: hundreds of endpoints, each needing a description | API surface > 50 endpoints OR OpenAPI spec > 100KB |
| **V2 — Constrained Domain Metamodel** | Repeated type definitions: same enums/interfaces duplicated across tool schemas | Fixed types + hard governance rules |
| **V3 — Stateful Context-Rich Domain** | Result verbosity: a single `read_file` response dwarfs all tool definitions | Result size > tool definition size |

**We are Variant 2.** Measurement confirmed this before we claimed it:

- The N-tool baseline (10 tools, `docs/n-tool-baseline.md`) costs **1,417 tokens** — driven entirely by the ring/quadrant enum descriptions repeating across 5 of the 10 tool schemas.
- The domain has a fixed, constrained metamodel: exactly 4 quadrants, 4 rings, and hard governance rules (kebab-case IDs, no duplicate assignments, no ADOPT→HOLD skip).
- Result size is small — a full `listTechnologies()` response is ~3 KB, far smaller than the 1,417-token tool cost. So V3's result-verbosity optimisation is not the primary lever here.
- The API surface is 10 operations, not hundreds. V1's concern (schema bloat from a huge API) does not apply.

**V2 is correct**: the domain's value is in its constraints, not in a large API surface (V1) or result verbosity (V3).

---

## Proxy Interface

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

`RadarReadProxy` is exposed in both `search()` and `execute()`. `RadarProxy` (which extends it with write methods) is exposed in `execute()` only. The sandbox enforces this structurally — `commit`, `assign`, `move`, `addTechnology`, and `removeAssignment` are absent from the `search()` isolate's global. There is no runtime check to bypass.

See `docs/proxy-design.md` for the full defence of this shape against alternatives (N-tool CRUD, single-dispatch, raw query language, per-workflow verbs).

---

## Domain Metamodel

The full metamodel from `Stack.TechRadar/docs/config.json`, typed in `src/types.ts`:

```typescript
// 4 quadrants, fixed
type Quadrant = 0 | 1 | 2 | 3
//   0 = Models & Providers
//   1 = Infrastructure & Cloud
//   2 = Frameworks & Libraries
//   3 = Techniques & Patterns

// 4 rings, fixed
type Ring = 0 | 1 | 2 | 3
//   0 = ADOPT, 1 = TRIAL, 2 = ASSESS, 3 = HOLD

// movement annotation since the previous radar
type Moved = -1 | 0 | 1

interface Technology {
  id: string          // kebab-case, unique across the radar
  label: string       // human-readable, ~30 chars
  quadrant: Quadrant
}

interface Team {
  id: string          // e.g. "llm-capability-office"
  name: string
  date: string        // "YYYY.MM"
}

interface Assignment {
  tech: string        // FK to Technology.id
  ring: Ring
  moved: Moved
}

// the full radar config:
interface Radar {
  date: string        // "YYYY.MM"
  default_team: string
  teams: Team[]
  technologies: Technology[]
  assignments: Record<string /* team id */, Assignment[]>
}
```

A token-trimmed projection of these types is embedded once as the DSL bootstrap in `execute()`'s tool description (~243 tokens). The full types in `src/types.ts` are the authoritative implementation; the DSL is a compact projection of them for model context.

---

## Validation Rules

The spec defines 5 validation rules for the Tech Radar domain. All 5 are implemented in `src/proxy.ts` as `ProxyError` guards with self-correcting error messages:

| Rule | Status | Guard in proxy.ts | Error tells the model |
|---|---|---|---|
| 1. `assignments[team][i].tech` must reference an existing `technologies[].id` | ✅ Implemented | `requireTech()` | Suggests `addTechnology(id, label, quadrant)` + lists 5 existing ids |
| 2. A tech may not be assigned twice to the same team | ✅ Implemented | `checkNotAssigned()` | Names current ring + suggests `radar.move(teamId, techId, ring)` |
| 3. `Moved` is only meaningful relative to the previous radar | ✅ Implemented | `move()` auto-computes `moved` from old vs new ring — model never sets it manually |
| 4. Demoting from ADOPT to HOLD in one step is forbidden — must pass through TRIAL or ASSESS first | ✅ Implemented | `checkDemotionRule()` | Names the specific tech + gives the exact incremental next step |
| 5. There is always exactly one `default_team` | ✅ Enforced | `config.json` schema — proxy never touches `default_team` field |

Additionally, a **kebab-case ID** guard is enforced on `addTechnology()` (not in the original spec but required by the schema): the error gives a concrete valid example (`'my-new-tool'`).

---

## Required Stack

Per the exercise specification:

| Layer | We use | Not allowed |
|---|---|---|
| MCP server runtime | Node.js 22 LTS | Anything you can't sandbox |
| Sandbox runtime | `isolated-vm` 6.1.2 (V8 isolate — security isolation) | Letting the model run unrestricted code in the server process |
| MCP transport | **stdio** — see justification below | A custom protocol that defeats the point of MCP |
| Token counting | `tiktoken` cl100k_base | Hand-waving "looks smaller" — show the number |
| Validation | Inside the proxy methods, with self-correcting errors | Validation in the tool schema only |

**Why stdio over HTTP:** This server runs locally as a development tool. stdio is the simplest transport for Claude Code and MCP Inspector integration — no port allocation, no auth headers, no CORS. HTTP would add value for a multi-client production deployment (Exercise C), not here.

**Why `isolated-vm` over `node:vm`:** `node:vm` provides context isolation, not engine isolation — it runs in the same V8 heap as the host process. The documented escape is `this.constructor.constructor('return process')()`. `isolated-vm` runs each script in a separate V8 isolate with its own heap; the escape path does not exist. See `docs/sandbox-choice.md` for the full 7-threat threat model.

---

## Tech Stack

| Component | Technology | Why we use it |
|---|---|---|
| **Language** | TypeScript (Node.js 22 LTS) | Strong typing for the domain proxy; `isolated-vm` requires Node 22 — Node 23+ breaks the native build |
| **MCP Protocol** | `@modelcontextprotocol/sdk` v1.29.0 | Official SDK for stdio MCP transport; handles framing and request routing |
| **Sandbox Runtime** | `isolated-vm` v6.1.2 | Separate V8 isolate — own heap, own microtask queue, no host built-ins reachable from model code |
| **Token Counting** | `tiktoken` (cl100k_base) | Spec-mandated tool for before/after bootstrap token measurement |
| **Transport** | stdio | Local development server; simplest path to Claude Code and MCP Inspector |
| **Data** | `data/config.json` | Development snapshot of the Tech Radar config (89 techs, 6 teams, 269 assignments) |
| **Type Checking** | TypeScript 5.5 + `tsx` | Zero-compile-step dev loop via `tsx`; `tsc --noEmit` for type checking |

---

## Project Structure

```
exercise-b/
├── data/
│   └── config.json              # Development snapshot of the radar (do not push to main)
├── docs/
│   ├── n-tool-baseline.md       # Before: 10 tools, 1,417 tokens (committed before Code Mode)
│   ├── proxy-design.md          # Why this proxy shape, not another (DSL alternatives)
│   ├── sandbox-choice.md        # Why isolated-vm: 7-threat threat model
│   ├── trace-1.md               # Self-correcting validation round-trip (3 errors → 3 retries)
│   └── trace-2.md               # Multi-step workflow + search() pre-flight token comparison
├── scripts/
│   ├── count-tokens.ts          # tiktoken before/after measurement (imports live TOOLS)
│   └── smoke.ts                 # End-to-end proxy + sandbox test (8 scenarios, idempotent)
├── src/
│   ├── types.ts                 # Domain types (Quadrant, Ring, Moved, Technology, Team, etc.)
│   ├── proxy.ts                 # RadarProxy + RadarProxyImpl with 5 self-correcting guards
│   ├── sandbox.ts               # isolated-vm runner (read/write modes, timeout, memory cap)
│   └── index.ts                 # MCP server — search() + execute() tools, stdio transport
├── package.json
└── tsconfig.json
```

---

## Code File Descriptions

- **`src/types.ts`** — Domain type definitions: `Quadrant`, `Ring`, `Moved`, `Technology`, `Team`, `Assignment`, `Radar`, `PendingOp`, `ValidationResult`, plus human-readable display-name constants (`QUADRANT_NAMES`, `RING_NAMES`) used in self-correcting error messages.

- **`src/proxy.ts`** — The core of the server. Defines `RadarReadProxy` (read surface) and `RadarProxy` (read + write surface) interfaces, and `RadarProxyImpl` which implements all 5 validation guards. Reads `data/config.json` on startup; `commit(message)` writes changes back to the file.

- **`src/sandbox.ts`** — The `isolated-vm` runner. Creates a fresh V8 isolate per call, bridges exactly the methods for the requested mode (`read` → read methods only; `write` → read + write methods), captures `console.log` output, enforces wall-clock timeout (5 s) and memory cap (128 MB), and disposes the isolate in a `finally` block.

- **`src/index.ts`** — MCP server wiring. Exports `DSL_BOOTSTRAP` and `TOOLS` (the two tool definitions) so `count-tokens.ts` can import them without starting the server. Wraps server startup in an `isMain` guard. Handles `search` and `execute` tool calls by delegating to `runInSandbox`.

- **`scripts/smoke.ts`** — End-to-end test that bypasses the MCP protocol and exercises the proxy + sandbox directly. Runs 8 scenarios: projection, write-method blocking, multi-step add+assign+commit, 3 self-correcting error round-trips, and sandbox timeout kill. Each run uses a fresh copy of `data/config.json` in a temp directory — repeated runs are idempotent and never touch the committed snapshot.

- **`scripts/count-tokens.ts`** — Measures bootstrap token cost of both tools using `tiktoken` cl100k_base. Imports `TOOLS` directly from `src/index.ts` so the count is always in sync with the live server.

- **`docs/n-tool-baseline.md`** — The *before* measurement: 10 naive tools, full JSON Schema definitions, 1,417 tokens. Committed before any Code Mode implementation, as required by the spec.

---

## Data Sources

| Source | Description |
|---|---|
| **`data/config.json`** | Development snapshot of `Stack.TechRadar_MultiTeam/docs/config.json` — 89 technologies, 6 teams, 269 assignments, 4 quadrants, 4 rings |
| **Stack.TechRadar repo** | Source of truth; `commit()` writes to the local snapshot only. Changes are proposed via PR — never auto-pushed to `main` |

The radar schema is fully typed in `src/types.ts`. The four quadrants are `Models & Providers`, `Infrastructure & Cloud`, `Frameworks & Libraries`, and `Techniques & Patterns`. The four rings are `ADOPT`, `TRIAL`, `ASSESS`, and `HOLD`.

---

## Tool Routing Logic

The model chooses between `search()` and `execute()` based on whether it needs to mutate state:

| Situation | Tool | Why |
|---|---|---|
| Read technologies, teams, or assignments | `search(code)` | Read-only sandbox; write methods structurally absent — `typeof radar.commit === 'undefined'` |
| Add, assign, move, remove, or commit | `execute(code)` | Read + write sandbox; full `RadarProxy` exposed |
| Dry-run check before a write | `search(code)` with `radar.validate(op)` | Validates a pending operation without touching state — catches governance violations before any mutation |

The read/write split is **structural, not conditional**. In the `search()` sandbox, `addTechnology`, `assign`, `move`, `removeAssignment`, and `commit` are never bridged into the isolate.

---

## Workflow Diagrams

### Overall Code Mode Flow

```text
        [ Model generates JavaScript code ]
                       |
                       v
            +---------------------+
            | MCP tool call:      |
            | search(code)  OR    |
            | execute(code)       |
            +---------------------+
                       |
                       v
            +---------------------+
            | src/index.ts        |
            | routes by tool name |
            +---------------------+
                  /         \
                 /           \
                v             v
     +--------------+   +--------------+
     | runInSandbox |   | runInSandbox |
     | mode='read'  |   | mode='write' |
     +--------------+   +--------------+
            |                   |
            v                   v
     +------------+      +------------+
     | V8 Isolate |      | V8 Isolate |
     | radar =    |      | radar =    |
     | READ only  |      | READ+WRITE |
     +------------+      +------------+
            |                   |
            v                   v
     +-------------------+  +-------------------+
     | Result / Error    |  | Result / Error    |
     | returned to model |  | returned to model |
     +-------------------+  +-------------------+
```

### Self-Correcting Validation Round-Trip

```text
   [ Model writes code with a mistake ]
                  |
                  v
     +---------------------------+
     | isolated-vm executes      |
     | model's code              |
     +---------------------------+
                  |
                  v
     +---------------------------+
     | ProxyError thrown:        |
     | 1. What failed            |
     | 2. Why it failed          |
     | 3. Exact corrective call  |
     +---------------------------+
                  |
                  v
     +---------------------------+
     | Model reads error,        |
     | rewrites code             |
     +---------------------------+
                  |
                  v
     +---------------------------+
     | Second execute() call     |
     | succeeds on first retry   |
     +---------------------------+
```

### Multi-Step Workflow (N-tool vs Code Mode)

```text
N-tool — 3 round-trips:                Code Mode — 1 execute() call:

 Model                                   Model
   |                                       |
   |-- addTechnology() -------> Server     |-- execute(`                
   |<-- { id, label, quad } ---            |     addTechnology(...)     
   |                                       |     assign(...)            
   |-- assignTechnology() ----> Server     |     commit(...)            
   |<-- { tech, ring, moved } -            |     return getAssignment() 
   |                                       |   `) --------------------> Server
   |-- commitChanges() -------> Server     |<-- { tech, ring, moved } -
   |<-- { status: ok } --------
```

---

## Installation Guide

### Prerequisites

- **Node 22 LTS** — Node 23+ breaks `isolated-vm`
  ```bash
  brew install node@22
  ```
- **Xcode CLI tools** — required to compile the `isolated-vm` native module
  ```bash
  xcode-select --install
  ```
  If already installed, this is a no-op.

> **Intel Mac?** Replace `/opt/homebrew` with `/usr/local` in every command below.

### Setup

1. **Clone the repository and check out the branch**
   ```bash
   git clone https://github.com/LLM-build-capability/Aquaiq-AI.git
   cd Aquaiq-AI
   git checkout premkumar/exercise-b
   ```

2. **Enter the exercise folder**
   ```bash
   cd exercise-b
   ```

3. **Install dependencies** (compiles `isolated-vm` from C++ — takes ~30 s on first run)
   ```bash
   PATH="/opt/homebrew/opt/node@22/bin:$PATH" npm install
   ```

> **Note on the PATH prefix:** Node 22 installed via Homebrew is not on the system PATH by default. Every command that runs Node must be prefixed with `PATH="/opt/homebrew/opt/node@22/bin:$PATH"`. To make this permanent, add `export PATH="/opt/homebrew/opt/node@22/bin:$PATH"` to your `~/.zshrc`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RADAR_CONFIG_PATH` | `exercise-b/data/config.json` | Absolute path to the Tech Radar config file the proxy reads and writes |

No `.env` file is needed for local development — the default path works out of the box. There are no API keys, credentials, or secrets. The server operates entirely on a local JSON file.

```bash
# Optional — override the config path
export RADAR_CONFIG_PATH="/absolute/path/to/Aquaiq-AI/exercise-b/data/config.json"
```

---

## How to Run the Project

### 1. Run the smoke test (recommended first step — no MCP host required)

```bash
PATH="/opt/homebrew/opt/node@22/bin:$PATH" npm run smoke
```

Runs 8 scenarios: projection, write-method blocking, multi-step add+assign+commit, 3 self-correcting error round-trips, and sandbox timeout kill. Expected last lines:

```
  smoke test complete
════════════════════════════════════════════════════════════════════════
```

### 2. Connect to Claude Code

Add to your MCP config (`~/.claude/mcp_settings.json` or `.claude/settings.json`). Replace `/absolute/path/to/Aquaiq-AI` with your actual path (run `pwd` from the repo root):

```json
{
  "mcpServers": {
    "tech-radar": {
      "command": "/opt/homebrew/opt/node@22/bin/node",
      "args": ["--import=tsx/esm", "src/index.ts"],
      "cwd": "/absolute/path/to/Aquaiq-AI/exercise-b",
      "env": {
        "RADAR_CONFIG_PATH": "/absolute/path/to/Aquaiq-AI/exercise-b/data/config.json"
      }
    }
  }
}
```

### 3. Connect to MCP Inspector

```bash
cd exercise-b
PATH="/opt/homebrew/opt/node@22/bin:$PATH" npx @modelcontextprotocol/inspector \
  node --import=tsx/esm src/index.ts
```

---

## How to Test the Project

### Type check (no output = pass)

```bash
PATH="/opt/homebrew/opt/node@22/bin:$PATH" npm run typecheck
```

### Count bootstrap tokens

```bash
PATH="/opt/homebrew/opt/node@22/bin:$PATH" npm run count-tokens
```

Expected output:

```
=== Bootstrap Token Count (cl100k_base) ===

N-tool baseline (10 tools):  1417 tokens
Code Mode  — search():              274 tokens
Code Mode  — execute() + DSL:       669 tokens
Code Mode  — total (2 tools):        949 tokens

Reduction:  468 tokens  (33%)
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `npm install` fails with `node-gyp` / `isolated-vm` build errors | Run `xcode-select --install` first, then retry |
| `command not found: node` | Missing `PATH=` prefix, or Node 22 not installed — run `brew install node@22` |
| `Cannot find module '../src/index.js'` | Wrong directory — run `cd exercise-b` first |
| MCP Inspector shows no tools | Check that `cwd` in your MCP config is the absolute path to `exercise-b/` |

---

## Outputs Generated

After `npm run smoke`:
- 8 scenario results printed to terminal (ok/error + result value per scenario)
- Temporary config copies written to system `tmp/` and auto-cleaned — `data/config.json` is never modified

After `npm run count-tokens`:
- Bootstrap token counts for N-tool baseline, `search()`, `execute()` + DSL, and total Code Mode cost

After connecting to Claude Code or MCP Inspector and calling `execute()` with `radar.commit(message)`:
- `exercise-b/data/config.json` updated with pending changes
- `[radar] committed: <message>` printed to stderr

---

## Results and Token Numbers

Measured with `tiktoken` cl100k_base. Run `npm run count-tokens` to reproduce.

### Bootstrap token comparison

| | Tools | Bootstrap tokens |
|---|---|---|
| N-tool baseline | 10 | **1,417** |
| Code Mode | 2 | **949** |
| Reduction | | **−468 (33%)** |

Breakdown of the Code Mode cost:

| Tool | Tokens |
|---|---|
| `search()` description + schema | 274 |
| `execute()` description + DSL + schema | 669 |
| **Total** | **949** |

The DSL bootstrap block accounts for **243 tokens** of the `execute()` cost and is the only place the metamodel appears in model context.

### Runtime token comparison (multi-step workflow)

| | Round-trips | Estimated runtime tokens |
|---|---|---|
| N-tool (add + assign + commit = 3 calls + results) | 3 | ~150 |
| Code Mode (1 `execute()` call + result) | 1 | ~132 |

A session with 10 such workflows saves approximately `468 (bootstrap) + 10 × 18 (runtime) = 648 tokens` vs the N-tool server.

### tiktoken snippet

```ts
// From scripts/count-tokens.ts — imports the live tool definitions from src/index.ts
import { get_encoding } from 'tiktoken'
import { TOOLS } from '../src/index.js'

const enc = get_encoding('cl100k_base')
console.log('search:', enc.encode(JSON.stringify(TOOLS[0], null, 2)).length)
console.log('execute+DSL:', enc.encode(JSON.stringify(TOOLS[1], null, 2)).length)
console.log('total:', enc.encode(JSON.stringify(TOOLS, null, 2)).length)
enc.free()
```

---

## Evaluation Scorecard

Scored honestly per the framework's 7 dimensions (0–3 each, max 21). Expected range for a first build: **12–17**.

| Dimension | N-tool baseline state | Code Mode state | Score |
|---|---|---|---|
| Tool count | 10 tools | 2 tools (`search` + `execute`) | **3** |
| Bootstrap token cost | 1,417 tokens | 949 tokens (within V2 ≤1,200 target) | **2** |
| Metamodel location | Repeated across 5 tool descriptions | TypeScript DSL placed once in `execute()`; read method signatures repeated in both tool descriptions | **2** |
| Credential exposure | N/A — no credentials in this domain | File path injected via `RADAR_CONFIG_PATH` env var, never in model context; dimension is V1-primary | **2** |
| Multi-step workflows | 3 round-trips (add + assign + commit) | Single `execute()` call | **3** |
| Validation error quality | No validation — opaque errors only | 5 guards, each naming the exact corrective call and current state | **3** |
| Result verbosity control | Full object returned per tool call | Model writes `.map`/`.filter` projections in code; no built-in `project()` operator | **1** |
| **Total** | | | **16 / 21** |

**Score: 16/21** — within the spec's expected first-build range of 12–17.

- **Bootstrap token cost (2):** 949 tokens meets the V2 ≤1,200 target but not the V1 ≤1,000 bar. The DSL block adds ~243 tokens a V1 server wouldn't carry.
- **Metamodel location (2):** TypeScript type aliases appear once in `execute()`, which is correct. The 5 read method signatures are repeated in both `SEARCH_DESC` and `EXECUTE_DESC` — partial repetition.
- **Credential exposure (2):** The only secret is the config file path, kept out of model context via env var. Scoring 3 for a dimension that wasn't a real challenge here would be dishonest.
- **Result verbosity control (1):** The proxy returns full objects. The model can write its own projection in code (e.g. `.map(t => t.id).slice(0, 5)`), but there is no built-in `project()` operator. `commit()` also only writes to the local snapshot; PR automation is a TODO.

---

## Evaluation Rubric

| Criterion | What we look for | Weight | Our assessment |
|---|---|---|---|
| **Variant classification** | Defend why this server is V1, V2, or V3 — measured before claiming | 10% | V2 confirmed by token measurement in `docs/n-tool-baseline.md` before any Code Mode code was written |
| **Proxy quality** | Interface fits on a page, names map to domain concepts, query/execute split is honest, validation lives in the proxy not the tools | 25% | `RadarReadProxy` / `RadarProxy` split enforced structurally in `src/sandbox.ts`; all validation in `src/proxy.ts` |
| **Code Mode mechanics** | `search()` and `execute()` work with a real sandbox; credentials never leak; bootstrap token measurement is reproducible | 20% | `isolated-vm` sandbox with `npm run count-tokens` reproducible measurement |
| **Self-correcting errors** | At least 3 deliberate-misuse cases with structured errors + named alternatives; one full round-trip recorded | 15% | 5 guards in `src/proxy.ts`; full round-trips in `docs/trace-1.md` |
| **Honest scorecard** | Filled in, defended, total in 12–17 range | 15% | 16/21 with per-dimension justification above |
| **Engineering discipline** | Modular code; types; secrets handled; commits readable; README runnable in 15 min | 15% | TypeScript throughout; no secrets; `isMain` guard; this README |

**Total: 100%. Passing bar: ≥ 70%.**

---

## Explicit Non-Goals

| Not in scope | Why |
|---|---|
| Production deployment of the MCP server | Local-only is fine for this exercise |
| Authentication of MCP clients | Stub it; the registry/auth story is part of Exercise C |
| Multi-tenant credential isolation | Stretch goal; not required for the base submission |
| Full migration of `Stack.TechRadar` to this server | Changes are proposed locally via `data/config.json`; do not auto-push to `main` |
| Reimplementing the wrapped domain | We wrap `config.json`; we do not reinvent the radar |
| Re-using LangChain / LlamaIndex / FastMCP magic that hides the proxy/tool split | The whole point is to design the split deliberately |

---

## Version Control Expectations

- **N-tool baseline was committed first** (`docs/n-tool-baseline.md`, commit `d8950c0`) — the *before* state is in git before the *after* state, as required.
- **No API specs committed** — the metamodel is typed inline; no external specs are cached in the repo.
- **Never push to `main` of `Stack.TechRadar`** — `commit()` writes to `exercise-b/data/config.json` only. Any change to the real radar must go through a PR from a topic branch.
- **No model weights or large binaries committed** — `*.gguf`, `node_modules/`, `dist/` are all gitignored.
- Commits are small and readable; each covers one logical change (proxy types, sandbox runner, MCP wiring, docs).

---

## Limitations

- **No built-in result projection operator.** The proxy returns full objects. The model must write its own `.map`/`.filter` projections in code. A `project()` helper would reduce runtime token cost further.
- **`commit()` is local-file only.** Changes are persisted to `data/config.json` in the development snapshot. Opening a PR against the real `Stack.TechRadar` repo is a TODO — `commit()` currently only calls `writeFileSync`.
- **Read method signatures duplicated.** The 5 read methods appear in both the `search()` and `execute()` tool descriptions. Factoring them into a shared reference would save ~60 tokens and eliminate the repetition.
- **Node 22 pin.** `isolated-vm` 6.x does not build against Node 23+. The server is pinned to Node 22 LTS until `isolated-vm` publishes Node 24+ support.
- **Single governance rule hardcoded.** Only the ADOPT→HOLD demotion rule is enforced. Additional governance rules (e.g. "ADOPT requires a `link` field") would need to be added manually to `proxy.ts`.

---

## Future Enhancements

These map to the spec's Stretch Challenges:

1. **Automatic DSL generation** *(Stretch 5)* — Generate the DSL bootstrap block from `src/types.ts` at build time so the tool description can never drift from the proxy.
2. **PR automation** — Wire `commit(message)` to `gh pr create` — create a branch in `Stack.TechRadar`, write the config, and open a PR for human review automatically.
3. **Result `project()` operator** — Add a built-in `project(fields)` method so the model can request a projection without writing `.map` code every time.
4. **Per-call observability** *(Stretch 2)* — Emit a structured event per `execute()` — input code, sandbox runtime, time, token deltas, mutations — so "why did the model write that code?" can be answered from logs alone.
5. **Adversarial harness** *(Stretch 4)* — 10 prompts that try to break out of the sandbox or trick the proxy into mutating during a `search()`. Document what each one does and which the design rejects.
6. **Cross-runtime build** *(Stretch 1)* — Implement the same server in Python (`RestrictedPython`). Compare ergonomics, security posture, dev-loop speed.
7. **Multi-tenant support** *(Stretch 3)* — One MCP server instance per team, with per-session credentials and an audit trail.
8. **Plug into Exercise A** *(Stretch 6)* — Let the local-RAG agent call this server through the MCP protocol. Note any tool-call drift in Gemma.

---

## Resources

**MCP basics**
- [Model Context Protocol spec](https://modelcontextprotocol.io/)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
- Cloudflare's *Code Mode* announcement (the source pattern) — search "Cloudflare Code Mode MCP"

**Sandboxes**
- [`isolated-vm`](https://www.npmjs.com/package/isolated-vm) — V8 isolate, suitable for security-relevant servers
- [`node:vm`](https://nodejs.org/api/vm.html) — context isolation only; **not** security isolation
- [`RestrictedPython`](https://pypi.org/project/RestrictedPython/) — Python sandbox; review limits
- Subprocess + timeout — OS-level isolation; highest overhead

**Token measurement**
- [`tiktoken`](https://www.npmjs.com/package/tiktoken) (Node) — use `cl100k_base` for OpenAI-family models

**Tech Radar**
- Source-of-truth repo: `Stack.TechRadar_MultiTeam` — schema is in `docs/config.json`

**Ecolab MCP context**
- `docs/strategies/mcp-strategy.md` — deployment-tier model the server eventually fits into
- `docs/designs/mcp-registry-spec.html` — registry contract for when the server is published

---

## FAQ / Common Pitfalls

**Q: Why is the proxy interface "on the first page" of the README?**
The spec requires it there. The proxy interface is the core deliverable — it should be readable before anything else.

**Q: The model keeps trying to write Python in the sandbox.**
Be explicit in `execute()`'s description: *"You will write JavaScript. The proxy `radar` is the only way to interact with the domain. Do not import Node APIs."* Smaller models drift more — you saw this in Exercise A with Gemma.

**Q: My tool returns an error and the model just gives up.**
That is the framework's Mistake 5. The error needs to name the valid alternatives. Compare: `Error: invalid` (unhelpful) vs `technology 'gpt-5-nano' not found. Use radar.addTechnology('gpt-5-nano', label, quadrant) to add it first, or pick from: ...` (model retries deterministically).

**Q: Why not score 21/21 on the scorecard?**
Either you are lying or you have miscounted. Both lose points. Our score is 16/21 with per-dimension justification.

**Q: Should `commit()` actually edit `docs/config.json` in the real Stack.TechRadar repo?**
Not during the exercise. `commit()` writes to `exercise-b/data/config.json` (our development snapshot). Any proposed change to the real radar must go through a PR from a topic branch — never auto-pushed to `main`.

**Q: Can we skip the N-tool baseline since "obviously" Code Mode is cheaper?**
No. The before/after delta is half of what is being evaluated. A scorecard without the *before* is a brochure. See `docs/n-tool-baseline.md`.

**Q: The DSL block is over 600 tokens — is that too much for V2?**
Yes. Target ≤ 600 tokens for V2. If over, strip JSDoc, collapse unions, factor out repeated structures. The model has pretraining knowledge — you do not need to define `string` for it. Our DSL block is **243 tokens**.

---

## Acknowledgements

- [Model Context Protocol](https://modelcontextprotocol.io/) specification and SDK
- [Cloudflare Code Mode](https://blog.cloudflare.com/introducing-mcp-server-cloudflare/) — the original pattern this exercise generalises from
- [`isolated-vm`](https://github.com/laverdet/isolated-vm) — V8 isolate sandbox for Node.js
- [`tiktoken`](https://github.com/openai/tiktoken) — token counting for cl100k_base
- Stack.TechRadar team for the open metamodel
- Bootcamp lead for guidance and the exercise framework
