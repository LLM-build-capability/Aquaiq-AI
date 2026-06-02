# Worked Trace 1 — Self-Correcting Validation Round-Trip

This trace demonstrates the three-error round-trip: the model emits broken code, the proxy throws a structured error naming what to do instead, and the model rewrites the call to succeed.

All output below is captured directly from `scripts/smoke.ts` running against a fresh copy of `data/config.json`.

---

## Error 1 — Assign a Technology That Doesn't Exist

**Model's first attempt** (referencing a tech id it invented):

```js
// execute() call — attempt 1
radar.assign('rde', 'this-tech-does-not-exist', 1);
return 'unreached';
```

**Sandbox result:**

```
ok=false
error.name=Error
error.message=technology 'this-tech-does-not-exist' not found in radar.
  Use radar.addTechnology('this-tech-does-not-exist', label, quadrant) to add it first,
  or pick from existing: 'claude-haiku-4-5-databricks', 'claude-opus-4-6-azure-ai-foundry',
  'claude-opus-4-6-databricks', 'claude-opus-4-7-azure-ai-foundry',
  'claude-opus-4-7-databricks' (use radar.listTechnologies() to see all).
```

**Model reads the error message.** It now knows two things: (a) the tech doesn't exist yet, and (b) it should call `addTechnology` first. It rewrites:

```js
// execute() call — corrected retry
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

---

## Error 2 — Forbidden ADOPT → HOLD Demotion (Governance Rule)

**Model's first attempt** (trying to skip governance steps):

```js
// execute() call — attempt 1
const adoptTech = radar.listAssignments('rde').find(a => a.ring === 0)?.tech;
radar.move('rde', adoptTech, 3);  // ADOPT -> HOLD, not allowed
return 'unreached';
```

**Sandbox result:**

```
ok=false
error.name=Error
error.message=demoting 'mcp-model-context-protocol' directly from ADOPT to HOLD is forbidden by governance.
  Step down incrementally: radar.move('rde', 'mcp-model-context-protocol', 1) to move to TRIAL first,
  then ASSESS, then HOLD.
```

**Model reads the error.** The error names the specific tech (`mcp-model-context-protocol`), the rule that was violated, and the exact next valid call. It rewrites:

```js
// execute() call — corrected retry (step 1 of incremental demotion)
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

---

## Error 3 — Invalid Technology ID (Not Kebab-Case)

**Model's first attempt** (using a display name as the id):

```js
// execute() call — attempt 1
radar.addTechnology('Not Kebab Case', 'Bad id', 0);
return 'unreached';
```

**Sandbox result:**

```
ok=false
error.name=Error
error.message=id 'Not Kebab Case' is not valid kebab-case.
  Use only lowercase letters, digits, and hyphens, e.g. 'my-new-tool'.
```

**Model reads the error.** The error gives a concrete example of a valid id. The model corrects:

```js
// execute() call — corrected retry
radar.addTechnology('not-kebab-case', 'Bad id', 0);
return radar.listTechnologies({ quadrant: 0 }).find(t => t.id === 'not-kebab-case');
```

**Result:**

```
ok=true
result={"id":"not-kebab-case","label":"Bad id","quadrant":0}
```

---

## What Makes This Round-Trip Work

Each `ProxyError` carries three pieces of information:
1. **What failed** — the specific id, team, or ring that was invalid.
2. **Why it failed** — the rule or constraint that was violated.
3. **What to do instead** — a concrete alternative call the model can execute immediately.

Without point 3, the model has no signal to recover from and typically either gives up or hallucinates a new attempt. With it, the retry is deterministic — the model copies the suggested call from the error message and succeeds on the first retry.
