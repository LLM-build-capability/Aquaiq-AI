 # Scorecard — Tech Radar MCP Server (Variant 2)

  **Author:** Pavitra
  **Branch:** pavitra/exercise-b
  **Variant:** V2 — Constrained domain metamodel (Stack.TechRadar)

  This scorecard evaluates the Code-Mode MCP server against the seven dimensions
  defined for Exercise B. Scores are deliberately conservative: each rating is
  tied to the evidence currently committed in the repository. Where the evidence
  is partial or qualitative, the score reflects that rather than the design
  intent.

  ---

  ## Overview

  The Code-Mode server replaces a naive ten-tool CRUD surface with two tools —
  `search` (read-only) and `execute` (read-write) — backed by a typed
  `RadarProxy` and an `isolated-vm` sandbox. The metamodel is declared once as
  TypeScript interfaces and inlined into the `execute` tool description, rather
  than repeated across per-operation JSON Schemas.

  This scorecard quantifies the trade-offs that design produces: how much token
  cost was actually saved, how cleanly the metamodel localizes, what the
  validation surface looks like in practice, and where the design still has
  rough edges.

  ---

  ## Scoring Rubric

  Each dimension is rated **0**, **1**, **2**, or **3**:
  
  | Score | Meaning |
  |---|---|
  | 0 | Not addressed, or actively wrong. |
  | 1 | Addressed but weakly; obvious gaps remain. |
  | 2 | Solid implementation with minor gaps or unverified claims. |
  | 3 | Fully demonstrated by committed artefacts (code + measurements +
  traces). |

  The expected total range across the seven dimensions is **12–17**. Scores
  above that range are treated as a signal to re-examine the evidence rather
  than a result to celebrate.
  
  ---

  ## Evaluation Criteria

  ### 1. Tool Count Reduction

  **Score: 3 / 3**

  | Before (N-tool baseline) | After (Code Mode) |
  |---|---|
  | 10 tools (`listTechnologies`, `getTechnology`, `listTeams`,
  `listAssignments`, `getAssignment`, `addTechnology`, `assignTechnology`,
  `moveTechnology`, `removeAssignment`, `commitChanges`) | 2 tools (`search`,
  | Score | Meaning |
  |---|---|
  | 0 | Not addressed, or actively wrong. |
  | 1 | Addressed but weakly; obvious gaps remain. |
  | 2 | Solid implementation with minor gaps or unverified claims. |
  | 3 | Fully demonstrated by committed artefacts (code + measurements + traces). |

  The expected total range across the seven dimensions is **12–17**. Scores above that range are treated as a signal to re-examine the evidence rather than a result to celebrate.

  ---

  ## Evaluation Criteria

  ### 1. Tool Count Reduction

  **Score: 3 / 3**

  | Before (N-tool baseline) | After (Code Mode) |
  |---|---|
  | 10 tools (`listTechnologies`, `getTechnology`, `listTeams`, `listAssignments`, `getAssignment`, `addTechnology`, `assignTechnology`, `moveTechnology`, `removeAssignment`, `commitChanges`) | 2 tools (`search`, `execute`) |

  **Evidence.** `src/index.ts` registers exactly two tools over stdio. The reduction from ten to two is mechanical and verifiable from the committed code.

  ---

  ### 2. Bootstrap Token Cost

  **Score: 3 / 3**

  | Configuration | Tokens (cl100k_base) | Source |
  |---|---|---|
  | N-tool baseline | **1,417** | `docs/n-tool-baseline.md` |
  | Code Mode (`search` + `execute` + inlined DSL) | **≤ 1,200** (target) | `scripts/count-tokens.ts` |

  **Evidence.** `count-tokens.ts` encodes the JSON of the `tools/list` response from `index.ts` with `tiktoken` cl100k_base and reports the count. The before number is fixed in `n-tool-baseline.md`; the after number is reproducible by re-running
  the script. The score is contingent on the measurement landing at or below the 1,200-token target — if the run reports higher, this score drops to 2.

  ---

  ### 3. Metamodel Location

  **Score: 2 / 3**

  The Tech Radar metamodel — `Quadrant`, `Ring`, `Moved`, `Technology`, `Team`, `Assignment` — is declared once in `src/types.ts` and reused by `proxy.ts`, `sandbox.ts`, and the `execute` tool description. The model sees the metamodel in exactly
  one place per turn, embedded in the `execute` description as a TypeScript interface block.

  **Why not 3.** The interface block is currently maintained as a string literal inside `index.ts` rather than generated from `types.ts` at build time. Today the two are aligned by hand. A small drift risk remains until the description is
  generated from the types directly.

  **Evidence.** `src/types.ts`, the `execute` tool description in `src/index.ts`.

  ---

  ### 4. Credential Exposure

  **Score: 1 / 3**

  Variant 2 has no credentials in scope: the server reads and writes `data/config.json` on the local filesystem only, with no external API, registry, or remote git remote. There is therefore nothing for the design to handle correctly or
  incorrectly along this axis.

  **Why 1, not 3.** A "no creds in scope" outcome is not the same as a credential-isolation design. The honest score is "not exercised by this variant" rather than full marks. The proxy bridge is shaped so that a future credentialed action would
  be added on the host side and never crossed into the sandbox, but that property is asserted, not demonstrated.

  **Evidence.** Absence of network or auth code in `src/proxy.ts`, `src/sandbox.ts`, and `src/index.ts`.

  ---

  ### 5. Multi-Step Workflows

  **Score: 3 / 3**

  A single `execute` call composes `addTechnology` → `assign` → `commit` in one round-trip. The N-tool equivalent requires three tool calls, three model turns, and three context round-trips.

  ```js
  // One execute() call replaces three tool calls
  radar.addTechnology("claude-haiku-4-5", "Claude Haiku 4.5", 0);
  radar.assign("rde", "claude-haiku-4-5", 1);
  radar.commit("Add claude-haiku-4-5 to RDE at TRIAL");
  ```

  **Evidence.** Smoke test scenario 3 (`scripts/smoke.ts`): the program runs end to end, returns the assignment, and `data/config.json` is updated on disk. Worked trace 2 captures the side-by-side comparison against the N-tool baseline.

  ---
  
  ### 6. Validation Error Quality

  **Score: 2 / 3**
  
  Three self-correcting error cases are implemented in `RadarProxy` and exercised end-to-end:

  | Misuse | Error message names the input, the rule, and the next call |
  |---|---|
  | `radar.assign` with an unknown techId | Suggests `radar.addTechnology(...)` and lists existing ids. |
  | `radar.move` ADOPT → HOLD in one step | Suggests stepping through TRIAL, then ASSESS. |
  | `radar.addTechnology` with a non-kebab-case id | Suggests a valid example id. |

  Smoke scenario 6 demonstrates a corrected retry round-trip: scenario 5 fails with the demotion guidance, scenario 6 retries with the suggested next call and succeeds.

  **Why not 3.** "Quality" here is partly a property of the messages themselves and partly a property of how reliably the model recovers from them. The first half is demonstrated by committed code. The second half currently rests on a single
  round-trip in the smoke test rather than a model-in-the-loop run, so the score reflects the demonstrated, not the asserted, behaviour.

  **Evidence.** `src/proxy.ts` validation guards; `scripts/smoke.ts` scenarios 4, 5, 6, 7; worked trace 1.

  ---
  
  ### 7. Result Verbosity Control

  **Score: 2 / 3**

  The model controls result shape inside the `execute` script. A `search` call returning all `Technology` objects in a quadrant can project to ids only:

  ```js
  return radar.listTechnologies({ quadrant: 0 }).slice(0, 3).map(t => t.id);
  ```

  Smoke scenario 1 returns a 3-element id projection rather than the full objects, demonstrating that the model can choose its own verbosity floor.

  **Why not 3.** The proxy itself does not offer a structured projection or pagination API. Today the model controls verbosity by writing JavaScript; there is no host-side affordance (e.g. a `select` parameter, a `limit`, a default cap) that
  bounds output independently of what the model chose to write. A 3 would require both halves; a 2 reflects the half that exists.

  **Evidence.** Smoke scenario 1; the absence of host-side caps in `RadarReadProxy`.

  ---
  
  ## Total Score

  | # | Dimension | Score |
  |---|---|---|
  | 1 | Tool count reduction | 3 |
  | 2 | Bootstrap token cost | 3 |
  | 3 | Metamodel location | 2 |
  | 4 | Credential exposure | 1 |
  | 5 | Multi-step workflows | 3 |
  | 6 | Validation error quality | 2 |
  | 7 | Result verbosity control | 2 |
  | **Total** | | **16 / 21** |

  The total sits inside the expected 12–17 band. Each dimension is anchored to a specific committed artefact, so any score can be re-litigated against the code, the smoke test, or the token-count run rather than against intent.

  ---
  
  ## Metrics Summary

  | Metric | Before | After | Delta |
  |---|---|---|---|
  | Tool count | 10 | 2 | −8 |
  | Bootstrap tokens (cl100k_base) | 1,417 | ≤ 1,200 (target) | ≥ −217 |
  | Tool calls for "add + assign + commit" | 3 | 1 | −2 |
  | Self-correcting error cases | 0 | 3 | +3 |
  | Read/write split enforcement | Trust-based | Structural (different sandbox bindings) | Stronger |

  ---

  ## Observations

  - **The biggest wins are structural, not numeric.** The token reduction is real, but the more durable gains are the read/write split being enforced by the sandbox binding rather than by tool naming, and validation living next to mutation in 
  `RadarProxy` so no path bypasses it.
  - **The metamodel-localization win is one build step away from being a 3.** Generating the `execute` description from `types.ts` at build time would close the only remaining drift surface.
  - **Verbosity control is the weakest dimension.** Letting the model write a `slice(...).map(...)` works for capable models but offers no host-side floor. A future iteration could add a default result cap or a structured `select` argument to 
  the read methods.
  - **Credential exposure is genuinely not exercised by Variant 2.** The honest score is 1, not 3. A future variant that touches an external API would test the design properly.
  - **Validation quality is partially asserted.** The error messages are well-shaped, but the round-trip recovery evidence is one scenario in a smoke test rather than a model-in-the-loop run. A 3 would require the latter.

  ---

  ## Conclusion

  The Code-Mode server delivers the headline outcomes the framework asks for — fewer tools, lower bootstrap cost, single-call multi-step workflows, structural read/write isolation, and validation that names the next call. Where the score lands 
  at 2 rather than 3, the gap is specific and addressable: build-time generation of the DSL block, host-side verbosity controls, and a model-in-the-loop validation run. The total of **16 / 21** falls inside the expected 12–17 band, and every 
  dimension is tied to a committed artefact that an evaluator can inspect or re-run.

