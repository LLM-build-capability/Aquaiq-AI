# Exercise B — Token-Efficient Code-Mode MCP Server

**Variant:** V2 — Constrained Domain Metamodel (Stack.TechRadar)

## Variant Justification

The Tech Radar has a fixed metamodel: 4 quadrants, 4 rings, typed identifiers, and hard governance rules (no ADOPT→HOLD skip, no duplicate assignments, kebab-case IDs). This is Variant 2 by definition — the domain's value is in its constraints, not in a large API surface (V1) or result verbosity (V3). The proxy can express the entire metamodel as a compact TypeScript DSL bootstrap (~243 tokens) placed once in the `execute()` tool description. A naive N-tool implementation would repeat the ring/quadrant enums across five separate tool schemas for a total bootstrap cost of 1,417 tokens. The Code Mode version brings this to **949 tokens** — a 33% reduction — while adding self-correcting validation that the N-tool server doesn't have.

---

## Proxy Interface

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

`RadarReadProxy` is exposed in both `search()` and `execute()`. `RadarProxy` (which extends it with write methods) is exposed in `execute()` only. The sandbox enforces this structurally — `commit`, `assign`, `move`, `addTechnology`, and `removeAssignment` are simply absent from the `search()` isolate.

---

## Setup

### Prerequisites

- Node 22 LTS (`brew install node@22`)
- `PATH="/opt/homebrew/opt/node@22/bin:$PATH"` prefix for all commands (Node 26 breaks `isolated-vm`)

### Install

```bash
cd exercise-b
PATH="/opt/homebrew/opt/node@22/bin:$PATH" npm install
```

### Run the smoke test (proxy + sandbox, no MCP host required)

```bash
PATH="/opt/homebrew/opt/node@22/bin:$PATH" npx tsx scripts/smoke.ts
```

### Count bootstrap tokens

```bash
PATH="/opt/homebrew/opt/node@22/bin:$PATH" npx tsx scripts/count-tokens.ts
```

### Connect to Claude Code

Add this to your Claude Code MCP config (`~/.claude/mcp_settings.json` or `.claude/settings.json`):

```json
{
  "mcpServers": {
    "tech-radar": {
      "command": "/opt/homebrew/opt/node@22/bin/node",
      "args": ["--import=tsx/esm", "src/index.ts"],
      "cwd": "/path/to/Aquaiq-AI/exercise-b",
      "env": {
        "RADAR_CONFIG_PATH": "/path/to/Aquaiq-AI/exercise-b/data/config.json"
      }
    }
  }
}
```

### Connect to MCP Inspector

```bash
PATH="/opt/homebrew/opt/node@22/bin:$PATH" npx @modelcontextprotocol/inspector \
  node --import=tsx/esm src/index.ts
```

---

## Before / After Token Numbers

Measured with `tiktoken` cl100k_base. Run `npx tsx scripts/count-tokens.ts` to reproduce.

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

The DSL bootstrap block (`type Quadrant = ...` through `type PendingOp = ...`) accounts for **243 tokens** of the `execute()` cost and is the only place the metamodel appears in model context.

```ts
// Snippet that produced these numbers (from scripts/count-tokens.ts):
const enc = get_encoding('cl100k_base');
enc.encode(JSON.stringify(tools, null, 2)).length
```

---

## Evaluation Scorecard

Scored honestly per the framework's 7 dimensions (0–3 each, max 21).

| Dimension | N-tool baseline state | Code Mode state | Score |
|---|---|---|---|
| Tool count | 10 tools | 2 tools (`search` + `execute`) | **3** |
| Bootstrap token cost | 1,417 tokens | 949 tokens (within V2 ≤1,200 target) | **2** |
| Metamodel location | Repeated across 5 tool descriptions | TypeScript DSL placed once in `execute()`; read method signatures repeated in both tool descriptions | **2** |
| Credential exposure | N/A — no credentials in this domain | File path injected via env var, never in model context; dimension is V1-primary | **2** |
| Multi-step workflows | 3 round-trips (add + assign + commit) | Single `execute()` call | **3** |
| Validation error quality | No validation — opaque errors only | 5 guards, each naming the exact corrective call and current state | **3** |
| Result verbosity control | Full object returned per tool call | Model writes `.map`/`.filter` projections in code; no built-in `project()` operator; `commit()` is a local-file stub (PR automation not implemented) | **1** |
| **Total** | | | **16 / 21** |

**Score: 16/21** — within the spec's expected first-build range of 12–17.

- **Bootstrap token cost (2):** 949 tokens meets the V2 ≤1,200 target but does not meet the V1 ≤1,000 bar. The DSL block adds ~243 tokens that a V1 server wouldn't carry.
- **Metamodel location (2):** The TypeScript type aliases appear once in `execute()`, which is correct. However, the 5 read method signatures (`listTechnologies`, `listTeams`, `listAssignments`, `getAssignment`, `validate`) are repeated in both `SEARCH_DESC` and `EXECUTE_DESC`. Partial repetition — not as bad as the N-tool baseline, but not a clean single placement.
- **Credential exposure (2):** This dimension is most relevant to V1 servers with API keys. For V2 the only secret is the config file path, which is correctly kept out of model context via `RADAR_CONFIG_PATH` env var. Scoring 3 for a dimension that wasn't a real challenge would be dishonest.
- **Result verbosity control (1):** The proxy returns full objects on every call — the model must write its own projection in code (e.g. `.map(t => t.id).slice(0, 5)`). There is no built-in `project()` operator. Additionally, `commit()` writes only to the local `data/config.json` snapshot; the PR-opening step is a TODO comment in the code, not implemented.

---

## Project Structure

```
exercise-b/
├── data/
│   └── config.json              # Development snapshot of the radar (do not push to main)
├── docs/
│   ├── n-tool-baseline.md       # Before: 10 tools, 1,417 tokens
│   ├── proxy-design.md          # Why this proxy shape
│   ├── sandbox-choice.md        # Why isolated-vm
│   ├── trace-1.md               # Self-correcting validation round-trip
│   └── trace-2.md               # Multi-step workflow efficiency trace
├── scripts/
│   ├── count-tokens.ts          # tiktoken before/after measurement
│   └── smoke.ts                 # End-to-end proxy + sandbox test
├── src/
│   ├── types.ts                 # Domain types (Quadrant, Ring, Moved, etc.)
│   ├── proxy.ts                 # RadarProxy + RadarProxyImpl with self-correcting errors
│   ├── sandbox.ts               # isolated-vm runner (read/write modes)
│   └── index.ts                 # MCP server — search() + execute() tools, stdio transport
├── package.json
└── tsconfig.json
```
