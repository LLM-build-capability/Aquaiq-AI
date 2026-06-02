# Sandbox Choice — Code Mode MCP (Exercise B)

## Decision: `isolated-vm`

We use [`isolated-vm`](https://www.npmjs.com/package/isolated-vm) (v6.x), which creates a true V8 isolate — a separate heap with no shared object references to the host Node.js process.

## Why `isolated-vm` Over the Alternatives

The four sandbox options differ on one axis that matters here: **security isolation vs context isolation**.

| Sandbox | Isolation type | Can model code reach host memory? | Notes |
|---|---|---|---|
| `isolated-vm` | Security — separate V8 heap | No — objects cross the boundary only via explicit `.copy` transfer | Chosen |
| `node:vm` | Context — same V8 heap, different global | Yes — prototype chain attacks, shared ArrayBuffers, `this.constructor` escapes | Only suitable for low-risk, trusted code |
| `RestrictedPython` | AST rewrite + limited builtins | Partially — depends on `safe_globals` configuration; bypass CVEs have existed | Python only; weaker than a VM boundary |
| subprocess + timeout | OS process boundary | No — strongest isolation | Highest overhead; IPC serialisation cost per call; overkill for local Dev |

**The threat model for this server:** the model writes JavaScript and we run it. Even in a controlled bootcamp setting, the model may hallucinate code that tries `process.exit()`, reads `__dirname`, or walks the prototype chain to reach host objects. `node:vm` is specifically documented as *not* a security sandbox — Node.js's own docs say "Do not use `node:vm` to run untrusted code." The spec requires that we *defend* our choice; that note alone disqualifies `node:vm` for a server where model-generated code is the input.

`isolated-vm` enforces the boundary at the V8 engine level. Every value passed between the isolate and the host is deep-copied via `{ copy: true }` transfer options; no reference to a host object ever enters the isolate. The `radar` proxy methods are bridged as `ivm.Reference` handles and then deleted from `globalThis` inside the bootstrap script, so model code can only call them through the frozen `radar` object — it cannot reach the raw handle or inspect the host closure.

`subprocess + timeout` would give stronger OS-level isolation but adds ~20–50ms IPC overhead per call and requires a full serialise/deserialise cycle for every proxy method invocation. For a local Dev server making dozens of synchronous proxy calls per `execute()`, that overhead compounds. `isolated-vm`'s synchronous `applySync` call through a Reference has sub-millisecond overhead per method.

## Memory and Timeout Budget

Each isolate is created with `memoryLimit: 128 MB` and the user code runs with a `timeout: 5,000 ms`. Both are configurable via `SandboxOptions`. The isolate is `dispose()`d in a `finally` block after every call, so memory is released even if the code throws or times out.

## Known Limitation

`isolated-vm` requires a native build against the Node.js V8 API. Node 26 changed the V8 ABI in a way that breaks `isolated-vm` 6.x. The server is pinned to **Node 22 LTS**, which is stable against this module. This constraint is documented in `package.json` (`engines: { node: ">=22 <24" }`) and in the README setup steps.
