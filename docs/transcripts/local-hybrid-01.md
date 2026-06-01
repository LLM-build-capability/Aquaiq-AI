# Transcript — `hybrid-01` (local profile)

| Field | Value |
|---|---|
| **Profile** | `local` |
| **Model** | `gemma4:e4b` (Ollama, OpenAI-compatible endpoint) |
| **Embedder** | `nomic-embed-text` (768-dim) |
| **Collection** | `water_rag_local` |
| **Run timestamp** | `20260530T145249Z` |
| **Latency** | 16.27 s |
| **Prompt tokens** | 3,442 |
| **Completion tokens** | 781 |
| **Tokens/sec** | 49.35 |
| **Sources retrieved** | `who_water_guidelines.pdf` |

## User query

> What is the EPA limit for nitrate in drinking water, and what are the current nitrate levels in Williamson County?

## Routing

```
RAG similarity:  0.584
Tool similarity: 0.566
Query type:      both
```

Local routing — same query, same example bank — landed on **`both`** instead of `tool`. The two scores differ by only 0.018, well inside the 0.08 "both" threshold. Under `nomic-embed-text` the similarity geometry is denser, so RAG and tool sides routinely score within the threshold.

## Execution

```
Expanded query: What is the EPA limit for nitrate in drinking water, ...
Fetching water quality data for Williamson County Texas...
```

The query was expanded for retrieval and the tool was invoked. The retriever returned one chunk from `who_water_guidelines.pdf` — a **scope leak** for an EPA-specific question.

## Final answer

```
(empty)
```

The recorded answer is an empty string. The agent emitted a tool call and the tool returned (with an SSL failure as on cloud), but Gemma did not produce a final assistant message after consuming the tool result. **Total round-trip succeeded; the user-visible payload is blank.**

## Notes

- **The most damaging failure mode in the benchmark.** Cloud got a partial-but-honest answer. Local got nothing — same query, same agent loop, same corpus.
- **Two compounding local-mode issues at once:**
  1. **Embedder scope leak.** `nomic-embed-text` retrieved WHO instead of EPA. Even if Gemma had emitted a final message, the citation would have been wrong for an EPA-scoped query.
  2. **Final-message generation glitch.** Gemma silently dropped the assistant turn after the tool call. No exception, no partial text — a clean empty string.
- **Token mismatch.** Local consumed ~5× the prompt tokens of cloud (3,442 vs 685) and ~3× the completion tokens (781 vs 264) — yet produced no visible answer. Tokens went into intermediate reasoning / tool-result formatting that never surfaced.
- **Rubric:** correctness 0, citation 1 (a source was surfaced, even if wrong-scoped), refused_correctly n/a (per `rubric-scores.csv`).
- **Carry-forward for Exercise C.** This is exactly the kind of failure that needs to show up in `failure-modes.md` for the orchestration writeup: a downstream agent that "succeeds" at the tool layer can still hand the orchestrator an empty payload. The protocol envelope must distinguish "no answer" from "answer = empty string".
