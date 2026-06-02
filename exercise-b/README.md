# Exercise B — Code-Mode MCP Server (Tech Radar, Variant 2)

  **Variant:** V2 — Constrained domain metamodel (Stack.TechRadar)
  **Branch:** team/exercise-b
  **Runtime:** Node.js 22 (pinned — `isolated-vm` does not build on Node 26+)

  This project replaces a naive ten-tool MCP server with a **two-tool
  Code-Mode** server. Instead of exposing one MCP tool per CRUD operation, it
  exposes a `search` tool and an `execute` tool. The model writes short
  JavaScript programs that call a typed `radar` object inside an `isolated-vm`
  sandbox; that object is bridged into a host-side `RadarProxy` that owns
  validation and persistence.

  The result: fewer tools, lower bootstrap token cost, multi-step workflows in a
  single round-trip, and self-correcting validation messages that name the next
  call to make.

  ---

  ## Project Overview

  The Tech Radar is a small domain: technologies grouped by quadrant (Models &
  Providers, Infrastructure & Cloud, Frameworks & Libraries, Techniques &
  Patterns), placed by teams onto rings (ADOPT, TRIAL, ASSESS, HOLD). The naive
  MCP design hands the model ten CRUD tools. Each tool restates the same
  `Quadrant`/`Ring` enums in its JSON Schema, and any non-trivial change (add a
  technology, assign it to a team, persist) costs three model turns.

  Code Mode replaces that surface with a typed proxy and an executable DSL:

  - **Two tools, not ten.** `search` (read-only) and `execute` (read + write).
  - **One DSL declaration, not five repeated enums.** The metamodel lives once
  in `types.ts` and is inlined into the `execute` tool description.
  - **One round-trip, not three.** A single `execute` call composes
  `addTechnology` → `assign` → `commit`.
  - **Structural read/write split.** The read sandbox does not bind write
  methods at all — `typeof radar.commit` is `undefined` inside a `search`
  script.
  - **Self-correcting validation.** `RadarProxy` throws errors that name the
  offending input, the rule, and the next call to try.

  ---

  ## Architecture / Components

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
                │  binds RadarReadProxy   │   │  binds full RadarProxy │
                └───────────────┬─────────┘   └─────────┬──────────────┘
                                │   ivm.Reference       │
                                │   (deep-copy args)    │
                ┌───────────────▼───────────────────────▼────────────┐
                │  proxy.ts — RadarProxyImpl                         │
                │    • reads data/config.json on construction        │
                │    • mutates in memory                             │
                │    • commit(msg) writes data/config.json           │
                │    • throws ProxyError with self-correcting text   │
                └────────────────────────┬───────────────────────────┘
                                         ▼
                                data/config.json
  ```

  ### Components

  - **`src/types.ts`** — Domain types: `Quadrant`, `Ring`, `Moved`,
  `Technology`, `Team`, `Assignment`, `Radar`. Single source of truth for the
  metamodel.
  - **`src/proxy.ts`** — `RadarReadProxy` (read surface) and `RadarProxy` (read
  + write). Owns validation, in-memory mutations, and the only call path that
  writes to disk (`commit`). Throws `ProxyError` with corrective messages.
  - **`src/sandbox.ts`** — `isolated-vm` runner with two modes. Compiles
  model-authored code in a fresh V8 isolate, binds the appropriate proxy surface
  as a frozen `radar` global, enforces a 5-second timeout and 128 MB memory
  cap.
  - **`src/index.ts`** — MCP server over stdio using
  `@modelcontextprotocol/sdk`. Registers exactly two tools; the DSL bootstrap is
  carried once in the `execute` description.
  - **`data/config.json`** — Snapshot of the Tech Radar state. The only file the
  proxy ever writes.
  - **`scripts/smoke.ts`** — End-to-end harness covering happy-path,
  write-isolation, multi-step workflow, three self-correcting error cases,
  corrected retry, and timeout.
  - **`scripts/count-tokens.ts`** — Measures bootstrap token cost of the
  Code-Mode `tools/list` response with `tiktoken` (`cl100k_base`).

  ---

  ## Folder Structure

  ```
  exercise-b/
  ├── src/
  │   ├── types.ts          # domain types (Quadrant, Ring, Technology, ...)
  │   ├── proxy.ts          # RadarReadProxy + RadarProxy + ProxyError
  │   ├── sandbox.ts        # isolated-vm runner (read / write modes)
  │   └── index.ts          # MCP server (stdio) — search + execute tools
  ├── data/
  │   └── config.json       # Tech Radar snapshot
  ├── scripts/
  │   ├── smoke.ts          # 8-scenario end-to-end harness
  │   └── count-tokens.ts   # bootstrap token measurement
  ├── docs/
  │   ├── n-tool-baseline.md
  │   ├── proxy-design.md
  │   ├── sandbox-choice.md
  │   ├── worked-trace-1.md
  │   ├── worked-trace-2.md
  │   └── scorecard.md
  ├── package.json
  ├── tsconfig.json
  ├── .gitignore
  └── README.md             # this file
  ```

  ---

  ## Setup Instructions

  ### Prerequisite — Node 22

  `isolated-vm` ships as a native module and does not build on Node 26+. Use
  Node 22.

  ```bash
  brew install node@22

  # Always prepend Node 22 to PATH when working in exercise-b/
  export PATH="/opt/homebrew/opt/node@22/bin:$PATH"
  node --version   # expect v22.x
  ```

  ### Install

  ```bash
  cd exercise-b
  npm install      # @modelcontextprotocol/sdk, isolated-vm, tiktoken, tsx
  ```

  ### Verify

  ```bash
  npm run typecheck    # clean (no output)
  npm run smoke        # runs 8 end-to-end scenarios; all 8 should pass
  npm run count-tokens # measures Code-Mode bootstrap tokens
  ```

  The smoke harness reseeds a per-scenario tmp copy of `data/config.json` before
  each run, so the committed snapshot is never mutated.

  ### Run the MCP Server

  ```bash
  npm start
  # or with a custom config path:
  RADAR_CONFIG_PATH=/tmp/my-radar.json npm start
  ```

  The server logs `[tech-radar-mcp] connected — config: …` to **stderr** (stdout
  is reserved for MCP framing). Connect Claude Code, Cursor, or the MCP
  Inspector to the spawned process.

  ---

  ## How It Works — MCP → Proxy → Sandbox Flow

  1. **MCP host calls `tools/call`** with either `search` (read-only) or
  `execute` (read + write) and a JavaScript snippet as the argument.
  2. **`index.ts`** receives the call, picks the corresponding sandbox mode, and
  forwards the code.
  3. **`sandbox.ts`** creates a fresh `isolated-vm` isolate with a 128 MB memory
  cap, compiles the script, and injects the `radar` global. In read mode, only
  `RadarReadProxy` methods are bound; in write mode, the full `RadarProxy` is
  bound.
  4. **The script runs.** Each method call on `radar` is bridged across the
  isolate boundary via `ivm.Reference`, with arguments deep-copied so the
  isolate cannot retain references to host objects.
  5. **`proxy.ts`** validates each call (kebab-case ids, ring transitions,
  referential integrity), mutates an in-memory copy of the config, and throws
  `ProxyError` with corrective text on misuse.
  6. **`commit(message)`** is the only write path to disk. If the script throws
  or returns without committing, `data/config.json` is unchanged.
  7. **The script's return value** is surfaced back to the MCP host as the tool
  result.

  ### Example — Multi-Step Workflow in One Call

  ```js
  // Single execute() call composes three operations atomically.
  const target = radar.listTeams().find(t => t.id === "rde");

  const tech = radar.addTechnology("claude-haiku-4-5", "Claude Haiku 4.5", 0);
  const assignment = radar.assign(target.id, tech.id, 1, 1);

  radar.commit("Add claude-haiku-4-5 to RDE at TRIAL");
  return assignment;
  ```

  ### Example — Self-Correcting Validation

  ```text
  [execute] radar.move("rde", "gpt-4o-azure-openai", 3)   // ADOPT -> HOLD
  [error]   ProxyError: demoting 'gpt-4o-azure-openai' directly from ADOPT to
            HOLD is forbidden by governance. Step down incrementally:
            radar.move('rde', 'gpt-4o-azure-openai', 1) to move to TRIAL first,
            then ASSESS, then HOLD.
  [retry]   radar.move("rde", "gpt-4o-azure-openai", 1)   // ADOPT -> TRIAL
  [ok]      { tech: "gpt-4o-azure-openai", ring: 1, moved: -1 } persisted
  ```

  The error message names the offending input, the rule, and the literal next
  call to make. The model rewrites its own program from that text alone — no
  human in the loop.

  ---

  ## Key Features

  - **2 tools, not 10.** `search` and `execute` cover the full CRUD surface.
  - **Bootstrap token cost.** N-tool baseline is **1,417 tokens** (cl100k_base,
  measured); the Code-Mode bootstrap is reproducible via `npm run count-tokens`
  and targets ≤ 1,200 tokens. Real measurement should be lifted from the script
  output, not hard-coded here.
  - **Structural read/write split.** Enforced by `sandbox.ts` binding choices,
  not by trust. `typeof radar.commit === "undefined"` inside a `search` script.
  - **Sandbox isolation.** True V8-isolate isolation: separate heap, separate
  microtask queue, no `require`/`process`/`fs`/network. Wall-clock timeout (5 s)
  and memory cap (128 MB) enforced by V8 itself.
  - **Self-correcting validation.** Three implemented and exercised cases:
    - Unknown techId on `assign` → suggests `addTechnology` and lists existing
  ids.
    - ADOPT → HOLD on `move` → suggests stepping through TRIAL.
    - Non-kebab-case id on `addTechnology` → suggests a valid example.
  - **Atomicity.** `commit` is the sole write boundary. Failed scripts leave
  `data/config.json` untouched.
  - **Single source of truth for the metamodel.** `types.ts` is imported by
  host, sandbox binding, and the inlined DSL block in the `execute` description.

  ---

  ## Roles and Responsibilities

  | Owner | Area | Deliverables |
  |---|---|---|
  | Premkumar | Domain & Proxy | `src/types.ts`, `src/proxy.ts` (read + write
  surfaces, validation, `ProxyError`), `data/config.json` snapshot. |
  | Deepak | Sandbox & MCP Server | `src/sandbox.ts` (isolated-vm runner,
  read/write modes), `src/index.ts` (stdio MCP server, `search` + `execute` with
  inlined DSL bootstrap), `scripts/smoke.ts` end-to-end harness, infra
  (`.gitignore`, `tsconfig` rootDir, lockfile). |
  | Pavitra | Measurement, Docs & Traces | `scripts/count-tokens.ts`,
  `docs/proxy-design.md`, `docs/sandbox-choice.md`, `docs/worked-trace-1.md`,
  `docs/worked-trace-2.md`, `docs/scorecard.md`, this README. |

  ### Branch and PR Rules

  - Nobody commits directly to `team/exercise-b`. Always feature branch → PR.
    - Non-kebab-case id on `addTechnology` → suggests a valid example.
  - **Atomicity.** `commit` is the sole write boundary. Failed scripts leave `data/config.json` untouched.
  - **Single source of truth for the metamodel.** `types.ts` is imported by host, sandbox binding, and the inlined DSL block in the `execute` description.

  ---

  ## Roles and Responsibilities

  | Owner | Area | Deliverables |
  |---|---|---|
  | Premkumar | Domain & Proxy | `src/types.ts`, `src/proxy.ts` (read + write surfaces, validation, `ProxyError`), `data/config.json` snapshot. |
  | Deepak | Sandbox & MCP Server | `src/sandbox.ts` (isolated-vm runner, read/write modes), `src/index.ts` (stdio MCP server, `search` + `execute` with inlined DSL bootstrap), `scripts/smoke.ts` end-to-end harness, infra (`.gitignore`,
  `tsconfig` rootDir, lockfile). |
  | Pavitra | Measurement, Docs & Traces | `scripts/count-tokens.ts`, `docs/proxy-design.md`, `docs/sandbox-choice.md`, `docs/worked-trace-1.md`, `docs/worked-trace-2.md`, `docs/scorecard.md`, this README. |

  ### Branch and PR Rules

  - Nobody commits directly to `team/exercise-b`. Always feature branch → PR.
  - Recommended merge order: `premkumar/exercise-b` → `deepak/exercise-b` → `pavitra/exercise-b`.
  - Always run on Node 22 in `exercise-b/`. `isolated-vm` does not build on Node 26.
  - Do not commit secrets, model weights, or `node_modules/`.

  ---

  ## Reproducible Measurement

  Bootstrap token cost is measured, not asserted. To reproduce:

  ```bash
  cd exercise-b
  npm run count-tokens
  ```

  The script encodes the JSON of the `tools/list` response from `index.ts` with `tiktoken` `cl100k_base` and prints the count. Compare against the baseline of **1,417 tokens** documented in `docs/n-tool-baseline.md`. Do not hard-code the
  after-number in this README — read it from the script output. If a reader sees a number here, it should match `npm run count-tokens` on the current branch.

  ---

  ## Learnings / Observations

  - **The biggest wins are structural, not numeric.** The token reduction is real, but the more durable gains are the read/write split being enforced by the sandbox binding and validation living next to mutation in `RadarProxy`.
  - **Threat model drives runtime choice.** `node:vm` provides context isolation but shares a V8 isolate with the host — unsafe for adversarial input. `isolated-vm` provides true engine isolation. The native-build cost and Node 22 pin are
  acceptable prices.
  - **Self-correcting messages depend on the next call being literal.** Errors that say "use `radar.move('rde', '...', 1)`" recover faster than errors that say "ring transition invalid." The model can paste the next call.
  - **Atomicity falls out of the proxy shape.** Because `commit` is the only path that touches disk, any failure before it leaves the file unchanged. There is no rollback log because there is nothing to roll back.
  - **The DSL is described once.** Adding a fourth or fifth operation does not multiply the bootstrap cost — only the script body grows, and only on the turn that uses it.
  - **Worked traces should be captured terminal output, not prose.** The smoke harness emits structured logs that lift cleanly into trace docs without rewriting.

  ---

  ## Quick Reference

  ```bash
  # Setup
  export PATH="/opt/homebrew/opt/node@22/bin:$PATH"
  cd exercise-b && npm install

  # Verify
  npm run typecheck
  npm run smoke

  # Measure
  npm run count-tokens

  # Run server
  npm start
  RADAR_CONFIG_PATH=/tmp/my-radar.json npm start
  ```
