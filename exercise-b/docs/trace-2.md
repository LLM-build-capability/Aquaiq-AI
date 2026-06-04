# Worked Trace 2 — Multi-Step Workflow and Constraint-Validation Win

This trace demonstrates two things: (1) a multi-step workflow completed in a single `execute()` call instead of N round-trips, and (2) the constraint-validation win — the model learns about a governance rule via `validate()` in a `search()` call before touching any write.

All output is captured from `scripts/smoke.ts` against a fresh copy of `data/config.json`.

## Reproduce

```bash
cd exercise-b
PATH="/opt/homebrew/opt/node@22/bin:$PATH" npx tsx scripts/smoke.ts
```

---

## Part 1 — Idempotent Add + Assign + Commit in One Call

**Task:** Add `gpt-5-nano` to quadrant 0 (Models & Providers) if it doesn't already exist, assign it to team `rde` at ring 1 (TRIAL), and persist the change.

With a naive N-tool server this requires **3 sequential tool calls** with 3 model round-trips:

```
call 1: addTechnology({ id: "gpt-5-nano", label: "GPT-5 nano", quadrant: 0 })
← result: { id: "gpt-5-nano", label: "GPT-5 nano", quadrant: 0 }

call 2: assignTechnology({ teamId: "rde", techId: "gpt-5-nano", ring: 1 })
← result: { tech: "gpt-5-nano", ring: 1, moved: 0 }

call 3: commitChanges({ message: "add gpt-5-nano to RDE at TRIAL" })
← result: { status: "ok" }
```

**N-tool runtime token cost (3 calls + results): ~150 tokens**

With Code Mode the entire workflow is **1 `execute()` call**:

```js
// execute() call
const techId = 'gpt-5-nano-smoketest';
if (!radar.listTechnologies({ quadrant: 0 }).some(t => t.id === techId)) {
  radar.addTechnology(techId, 'GPT-5 nano (smoke)', 0);
}
radar.assign('rde', techId, 1);
radar.commit('smoke: add ' + techId + ' to RDE at TRIAL');
return radar.getAssignment('rde', techId);
```

**Result:**

```
ok=true
result={"tech":"gpt-5-nano-smoketest","ring":1,"moved":0}
persisted to file=true
```

**Code Mode runtime token cost (1 call + result): ~132 tokens**

The idempotency check (`if (!...some(t => t.id === techId))`) runs inside the sandbox with zero extra round-trips. An equivalent guard in the N-tool version would require a 4th call (`listTechnologies`) before the add — raising the N-tool cost to ~190 tokens.

---

## Part 2 — `search()` Pre-Flight Before a Write

**Task:** Before moving a tech from ADOPT to a lower ring, use `validate()` in a read-only `search()` call to check whether the governance demotion rule applies — before any write happens.

**Step 1 — `search()` pre-flight:**

```js
// search() call — read-only; no mutation possible
const adoptTech = radar.listAssignments('rde').find(a => a.ring === 0)?.tech;
return radar.validate({ type: 'move', teamId: 'rde', techId: adoptTech, newRing: 3 });
```

**Result:**

```
ok=true
result={
  "valid": false,
  "error": "demoting 'mcp-model-context-protocol' directly from ADOPT to HOLD is forbidden by governance.
            Step down incrementally: radar.move('rde', 'mcp-model-context-protocol', 1) to move to TRIAL first,
            then ASSESS, then HOLD."
}
```

The model now knows — from a read-only call, before any write — that `newRing: 3` is invalid and what the correct next step is. It rewrites the `execute()` call:

**Step 2 — corrected `execute()` call:**

```js
// execute() call
const adoptTech = radar.listAssignments('rde').find(a => a.ring === 0)?.tech;
radar.move('rde', adoptTech, 1);  // ADOPT -> TRIAL is allowed
radar.commit('demote ' + adoptTech + ' to TRIAL');
return radar.getAssignment('rde', adoptTech);
```

**Result:**

```
ok=true
result={"tech":"mcp-model-context-protocol","ring":1,"moved":-1}
```

In the N-tool server there is no equivalent pre-flight — the model would call `moveTechnology` with `newRing: 3`, receive an error, and have to guess the correction. Here, `validate()` surfaces the governance rule with the exact corrective call before any disk write occurs.

---

## Token Summary

| | Tokens |
|---|---|
| N-tool baseline bootstrap | 1,417 |
| Code Mode bootstrap (`search` + `execute`) | 949 |
| **Bootstrap reduction** | **−468 (33%)** |
| N-tool multi-step runtime (3 calls + results) | ~150 |
| Code Mode multi-step runtime (1 call + result) | ~132 |
| **Runtime reduction per workflow** | **~−18 (12%)** |

The bootstrap saving dominates for any session that runs multiple workflows. A session with 10 workflows saves approximately `468 (bootstrap) + 10 × 18 (runtime) = 648 tokens` vs the N-tool server.
