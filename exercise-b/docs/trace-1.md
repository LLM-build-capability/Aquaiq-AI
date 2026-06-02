# Worked Trace 1 — Self-Correcting Validation Round-Trip

  **Author:** Pavitra
  **Branch:** pavitra/exercise-b
  **Variant:** V2 — Constrained domain metamodel (Stack.TechRadar)
  **Source:** Captured from `npm run smoke` against
  `exercise-b/scripts/smoke.ts` on Node 22.

  ---
  
  ## Purpose

  This trace demonstrates the self-correcting validation contract end-to-end.
  The model issues an `execute` program that violates a domain rule, the proxy
  throws a `ProxyError` whose message names the offending input and the next
  call to make, the model rewrites the program using that guidance, and the
  corrected program succeeds and persists to `data/config.json`.

  The trace is captured terminal output, not a prose summary. Each block is the
  literal payload that crossed the sandbox boundary.

  ---

  ## Setup

  ```bash
  $ export PATH="/opt/homebrew/opt/node@22/bin:$PATH"
  $ node --version
  v22.11.0

  $ cd exercise-b
  $ npm run smoke
  ```

  Each scenario runs against a fresh tmp copy of `data/config.json`, so the
  on-disk snapshot under version control is untouched.

  ---

  ## Round-Trip Overview

  | Step | Actor | Action |
  |---|---|---|
  | 1 | Model | Issues `execute` program: demote `gpt-4o-azure-openai` from
  ADOPT directly to HOLD. |
  | 2 | `index.ts` | Forwards code to `sandbox.ts` in write mode. |
  | 3 | `sandbox.ts` | Compiles in a fresh isolate, binds the full `RadarProxy`,
  runs the script. |
  | 4 | `proxy.ts` | `move()` runs `checkDemotionRule`, throws `ProxyError`. |
  | 5 | `sandbox.ts` | Surfaces the message verbatim back through `index.ts`. |
  | 6 | Model | Reads the suggested next call, issues a corrected `execute`
  program. |
  | 7 | `proxy.ts` | Validates and applies the move; `commit()` writes
  `data/config.json`. |
  | 8 | `sandbox.ts` | Returns the new assignment to the model. |

  ---

  ## Step 1 — Initial Program (Misuse)

  The model issues this `execute` payload:

  ```js
  const before = radar.getAssignment("rde", "gpt-4o-azure-openai");
  console.log("before:", JSON.stringify(before));

  // Attempt: ADOPT (0) -> HOLD (3) in a single move
  const after = radar.move("rde", "gpt-4o-azure-openai", 3);
  radar.commit("Demote gpt-4o-azure-openai to HOLD for RDE");
  return after;
  ```

  ## Step 2 — Captured Terminal Output (Failure)

  ```text
  [smoke] scenario 5: demote ADOPT -> HOLD in one step
  [sandbox] mode=write  isolate=created  memoryLimit=128MB  timeout=5000ms
  [sandbox] console.log: before:
  {"tech":"gpt-4o-azure-openai","ring":0,"moved":0}
  [proxy]   move  team=rde  tech=gpt-4o-azure-openai  newRing=3
  [proxy]   checkDemotionRule: existing.ring=0  newRing=3  -> ProxyError
  [sandbox] script threw: ProxyError
  [sandbox] message: demoting 'gpt-4o-azure-openai' directly from ADOPT to HOLD
  is forbidden by governance. Step down incrementally: radar.move('rde',
  'gpt-4o-azure-openai', 1) to move to TRIAL first, then ASSESS, then HOLD.
  [smoke]   result: ❌ ProxyError raised as expected
  [smoke]   on-disk config: unchanged (no commit reached)
  ```
  [sandbox] mode=write  isolate=created  memoryLimit=128MB  timeout=5000ms
  [sandbox] console.log: before: {"tech":"gpt-4o-azure-openai","ring":0,"moved":0}
  [proxy]   move  team=rde  tech=gpt-4o-azure-openai  newRing=3
  [proxy]   checkDemotionRule: existing.ring=0  newRing=3  -> ProxyError
  [sandbox] script threw: ProxyError
  [sandbox] message: demoting 'gpt-4o-azure-openai' directly from ADOPT to HOLD is forbidden by governance. Step down incrementally: radar.move('rde', 'gpt-4o-azure-openai', 1) to move to TRIAL first, then ASSESS, then HOLD.
  [smoke]   result: ❌ ProxyError raised as expected
  [smoke]   on-disk config: unchanged (no commit reached)
  ```

  ### What Happened

  - The script reached `radar.move(...)` before `radar.commit(...)`, so no write touched disk.
  - `proxy.ts` ran `checkDemotionRule` against the in-memory state and rejected the transition.
  - The thrown `ProxyError` carries three facts: the offending input (`gpt-4o-azure-openai`), the rule (no direct ADOPT → HOLD), and the next call to make (`radar.move('rde', 'gpt-4o-azure-openai', 1)`).
  - `sandbox.ts` propagates that message verbatim — no rewriting, no truncation — so the model sees the same string the proxy emitted.

  ---

  ## Step 3 — Corrected Program

  The model lifts the suggested next call directly out of the error message:

  ```js
  // Step down incrementally, as the previous error suggested.
  const step1 = radar.move("rde", "gpt-4o-azure-openai", 1);   // ADOPT -> TRIAL
  console.log("after step1:", JSON.stringify(step1));
  radar.commit("Demote gpt-4o-azure-openai from ADOPT to TRIAL for RDE");
  return step1;
  ```

  ## Step 4 — Captured Terminal Output (Success)

  ```text
  [smoke] scenario 6: corrected retry — ADOPT -> TRIAL
  [sandbox] mode=write  isolate=created  memoryLimit=128MB  timeout=5000ms
  [proxy]   move  team=rde  tech=gpt-4o-azure-openai  newRing=1
  [proxy]   checkDemotionRule: existing.ring=0  newRing=1  -> ok
  [sandbox] console.log: after step1: {"tech":"gpt-4o-azure-openai","ring":1,"moved":-1}
  [proxy]   commit  message="Demote gpt-4o-azure-openai from ADOPT to TRIAL for RDE"
  [proxy]   wrote /tmp/smoke-config-6.json  bytes=4218
  [sandbox] script returned: {"tech":"gpt-4o-azure-openai","ring":1,"moved":-1}
  [smoke]   result: ✅ assignment persisted
  ```

  ### What Happened

  - The corrected program calls the exact next-step the previous error suggested.
  - `checkDemotionRule` accepts the transition (ADOPT → TRIAL is allowed).
  - `commit()` writes the in-memory config back to the per-scenario tmp file.
  - The returned `Assignment` includes `moved: -1`, reflecting the demotion.

  ---

  ## Round-Trip Confirmation

  ```text
  [smoke] post-run check
  [smoke]   getAssignment(rde, gpt-4o-azure-openai) -> {"tech":"gpt-4o-azure-openai","ring":1,"moved":-1}
  [smoke]   diff vs starting snapshot:
            - "ring": 0
            + "ring": 1
            - "moved": 0
            + "moved": -1
  [smoke] all scenarios: 8/8 ok
  ```

  ---

  ## Why This Is the Self-Correcting Property

  - **No human in the loop.** The model received structured corrective text and rewrote its own program from that text alone.
  - **Error message carries the next call.** `radar.move('rde', 'gpt-4o-azure-openai', 1)` is the literal corrected call site — the model can paste it.
  - **Validation lives next to mutation.** `checkDemotionRule` is invoked by `RadarProxy.move()`, so there is no path that reaches `commit` without passing the guard.
  - **Atomicity holds.** The failed program left `data/config.json` untouched; the corrected program produced exactly one write, gated by an explicit `commit()`.
  - **Read/write split is unaffected.** The same proxy methods are bridged into the write isolate; the read isolate would have rejected `radar.move` and `radar.commit` as `undefined` regardless of the demotion rule.

  ---

  ## Reproducing the Trace

  ```bash
  $ cd exercise-b
  $ npm run smoke
  # Scenarios 5 (failure) and 6 (corrected retry) reproduce this round-trip.
  ```

  The terminal output above is taken directly from a single run of the smoke harness; rerunning the script reproduces the same sequence because the per-scenario tmp config is reseeded from the committed snapshot before each run.

