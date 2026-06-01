# Local vs Cloud — Side-by-Side Comparison

**Author:** Prem Kumar Reddy Kothakapu  
**Branch:** `premkumar/local-mode`  
**Cloud profile:** `gpt-5.4-nano` + `text-embedding-3-small` (1536-dim) → collection `water_rag`  
**Local profile:** `gemma4:e4b` (Ollama) + `nomic-embed-text` (768-dim) → collection `water_rag_local`  
**Hardware:** Apple Silicon MacBook, macOS 26 (Tahoe), Ollama v0.3.x, 16 GB RAM  
**Corpus:** 4 PDFs — EPA drinking water, EPA water treatment, UN WWDR, WHO water guidelines (~6,460 chunks)

---

## Rubric

All 40 answers were scored by a single rater after both profiles had run.

| Column | Scale | Meaning |
|---|---|---|
| `correctness` | 0 / 1 / 2 | 0 = wrong or hallucinated; 1 = partial / incomplete; 2 = correct |
| `citation` | 0 / 1 | 1 = answer names a corpus source or live tool result; 0 = no attribution |
| `refused` | 0 / 1 | Only for out-of-scope queries: 1 = model correctly refused; 0 = answered anyway |

---

## Prompt Set (20 queries verbatim)

| ID | Category | Query |
|---|---|---|
| rag-01 | rag | What are the EPA limits for lead in drinking water? |
| rag-02 | rag | Explain how coagulation and flocculation work in water treatment. |
| rag-03 | rag | What are the WHO guidelines for arsenic in drinking water? |
| rag-04 | rag | Describe the role of sedimentation in conventional water treatment. |
| rag-05 | rag | How does chlorination disinfect water and what are its byproducts? |
| rag-06 | rag | What is membrane filtration and when is it used over conventional filtration? |
| rag-07 | rag | Summarise the main findings of the UN World Water Development Report on global water stress. |
| rag-08 | rag | What is the EPA Maximum Contaminant Level Goal (MCLG) and how does it differ from the MCL? |
| tool-01 | tool | What is the latest water quality data for Travis County? |
| tool-02 | tool | Get nitrate levels for Williamson County. |
| tool-03 | tool | Show me pH measurements for Benton County. |
| tool-04 | tool | What are the dissolved oxygen readings in Baxter County? |
| tool-05 | tool | Fetch water quality data for Prince George County. |
| tool-06 | tool | Are there any contamination readings for Oklahoma County? |
| hybrid-01 | hybrid | What is the EPA limit for nitrate in drinking water, and what are the current nitrate levels in Williamson County? |
| hybrid-02 | hybrid | Compare WHO guidelines for lead with the latest lead readings reported for Travis County. |
| hybrid-03 | hybrid | Explain why pH matters in drinking water and report the most recent pH values for Benton County. |
| hybrid-04 | hybrid | Describe how dissolved oxygen affects water quality and provide current readings for Baxter County. |
| oos-01 | out_of_scope | What is the capital of France and what is its population? |
| oos-02 | out_of_scope | Who won the FIFA World Cup in 2022? |

---

## Per-Query Results

`tool✓` = tool call emitted. `corr` = correctness (0–2). `cit` = citation (0/1).

| ID | Cloud latency (s) | Local latency (s) | Cloud tool✓ | Local tool✓ | Cloud corr | Local corr | Cloud cit | Local cit | Key difference |
|---|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| rag-01 | 5.28 | 15.21 | — | — | 2 | **0** | 1 | 1 | Local retrieved WHO doc for an EPA-scoped query; answered with wrong limit (10 µg/L instead of 15 µg/L) |
| rag-02 | 8.72 | 30.50 | — | — | 2 | 2 | 1 | 1 | Both correct; local answer ~2× longer |
| rag-03 | 5.09 | 16.40 | — | — | 2 | 2 | 1 | 1 | Identical accuracy |
| rag-04 | 4.23 | 24.80 | — | — | 2 | 2 | 1 | 1 | Both correct; local verbose but accurate |
| rag-05 | 5.14 | 20.94 | — | — | 2 | 2 | 1 | 1 | Both correct; local names more DBP types |
| rag-06 | 7.80 | 35.41 | — | — | 2 | 2 | 1 | 1 | Both correct; local p95 driven by this query (35 s) |
| rag-07 | 8.59 | 26.64 | — | — | 2 | **1** | 1 | 1 | Local introduced Falkenmark index figures not in the source PDF |
| rag-08 | 6.12 | 10.66 | — | — | 2 | **0** | 1 | 1 | Local declared "documents do not contain MCLG/MCL info" despite the EPA doc having it |
| tool-01 | 4.42 | 10.88 | ✓ | ✓ | 1 | 1 | 0 | 1 | Both called tool, both hit SSL error; local also surfaced false-positive RAG sources |
| tool-02 | 4.21 | 5.34 | ✓ | ✗ | 1 | **0** | 0 | 0 | Local asked user for state instead of calling tool |
| tool-03 | 5.48 | 9.07 | ✓ | ✗ | 1 | **0** | 0 | 0 | Local produced no tool call |
| tool-04 | 5.35 | 6.22 | ✓ | ✗ | 1 | **0** | 0 | 0 | Local produced no tool call |
| tool-05 | 6.43 | 4.45 | ✓ | ✗ | 1 | **0** | 0 | 0 | Local produced no tool call |
| tool-06 | 4.98 | 9.25 | ✓ | ✗ | 1 | **0** | 0 | 1 | Local produced no tool call; surfaced WHO doc as false-positive |
| hybrid-01 | 5.91 | 16.27 | ✓ | ✓ | 1 | **0** | 1 | 1 | Local emitted tool call but final answer was empty string |
| hybrid-02 | 8.34 | 16.64 | ✓ | ✗ | 1 | 1 | 1 | 1 | Local skipped tool call; RAG half answered correctly |
| hybrid-03 | 6.95 | 17.59 | ✓ | ✓ | 1 | 1 | 1 | 1 | Both partial (SSL on tool side); explanations correct |
| hybrid-04 | 9.99 | 15.21 | ✓ | ✓ | 1 | 1 | 1 | 1 | Both partial (SSL on tool side); DO explanation correct |
| oos-01 | 5.21 | 5.97 | — | — | 2 | 2 | 1 | 0 | Neither refused; cloud surfaced false-positive source |
| oos-02 | 3.48 | 11.37 | — | — | 2 | 2 | 1 | 1 | Neither refused; both answered factually but off-corpus |

---

## Aggregate Results

### Latency

| Metric | Cloud | Local | Ratio |
|---|---:|---:|---:|
| Mean (s) | 6.09 | 15.44 | 2.5× |
| p50 (s) | 5.42 | 15.21 | 2.8× |
| p95 (s) | 8.79 | 30.74 | 3.5× |

### Token throughput

| Metric | Cloud | Local |
|---|---:|---:|
| Prompt tokens (sum, 20 queries) | 18,311 | 24,979 |
| Completion tokens (sum) | 4,743 | 15,794 |
| Mean tokens/sec | 49.65 | **51.39** |

Local generates tokens at the same rate as cloud. The 2.5× wall-clock gap comes from Gemma emitting **3.3× more completion tokens** per query — verbose tabular summaries and restatements rather than slower inference.

### Tool-call success rate

| Category | Cloud | Local |
|---|---:|---:|
| Tool queries (6) | 6/6 (100%) | 1/6 (17%) |
| Hybrid queries (4) | 4/4 (100%) | 3/4 (75%) |
| **Combined (10)** | **10/10 (100%)** | **4/10 (40%)** |

### Quality (correctness 0–2, per category)

| Category | Cloud avg | Local avg | Gap |
|---|---:|---:|---:|
| RAG (8 queries) | 2.00 | 1.38 | −0.63 |
| Tool (6 queries) | 1.00 | 0.17 | −0.83 |
| Hybrid (4 queries) | 1.00 | 0.75 | −0.25 |
| Out-of-scope (2 queries) | 2.00 | 2.00 | 0 |
| **Overall (20 queries)** | **1.50** | **0.95** | **−0.55** |

### Citation rate

| Profile | Rate |
|---:|---:|
| Cloud | 14/20 (70%) |
| Local | 15/20 (75%) |

Citation rates are comparable. The small local advantage is partly an artefact of false-positive RAG retrievals on tool-only queries surfacing a source name even when no tool call was made.

### Refused-correctly (out-of-scope)

Both profiles: **0/2**. Neither refused the France capital or FIFA queries. Both surfaced `un_water_report.pdf` as a false-positive source on at least one OOS query.

### Retrieval-source overlap

7/9 queries with sources on both profiles (77.8%) returned the same top corpus document. The two exceptions are the queries where scope leakage changed the top-ranked result (`rag-01` EPA→WHO, `rag-08` ranking failure).

---

## Diagnosis

### 1. Tool-call drift under Gemma (root cause: post-training gap)

Cloud hit 10/10 tool calls. Local hit 4/10. On `tool-02` through `tool-06`, Gemma produced a clarification question ("which state did you mean?") or a RAG-only response instead of a `tool_calls` JSON payload. No exception was raised — the agent loop ran normally and returned the clarification as the final answer.

**Why:** `gemma4:e4b` is not post-trained to the same level of reliability on OpenAI-style `tools=[]` function calling as `gpt-5.4-nano`. The model is aware of the tool schema (it does call the tool on straightforward queries like `tool-01`) but falls back to conversational responses when the query contains slight ambiguity — county-state disambiguation in `tool-02`/`tool-05`, or when a tool query also activates the RAG similarity path (routing `both`). Cloud handles that ambiguity by emitting the tool call and letting the tool response resolve it.

This is not a bug to fix in the agent loop. The tool schema, system prompt, and routing threshold are identical on both profiles. The behaviour gap is in the model's instruction-following under ambiguity.

### 2. Embedder-driven scope leakage (root cause: different similarity geometry)

`rag-01` asked for the EPA lead limit. Cloud retrieved EPA documents and answered correctly (15 µg/L action level). Local retrieved `who_water_guidelines.pdf` and answered with WHO's 10 µg/L provisional guideline — a factually defensible number for the wrong regulatory body.

**Why:** `text-embedding-3-small` and `nomic-embed-text` cluster the query "EPA limits for lead" against the corpus differently. Cloud ranked the EPA document higher; local ranked the WHO document higher. Both retrievals are semantically reasonable — "lead in drinking water limits" is a valid match for both docs. The embedder's training data and calibration determine which document wins at rank-1. This is not a ChromaDB bug or a `top-k` problem: it is the expected consequence of using a different embedding model on the same corpus.

The same effect explains `rag-08`: the `epa_water_treatment.pdf` chunk containing the MCLG/MCL distinction was ranked below `top-k=5` by `nomic-embed-text`, so local declared the corpus didn't contain the answer. Cloud retrieved it easily.

### 3. Verbosity penalty (root cause: Gemma output style)

Local completion tokens averaged 3.3× cloud. Gemma adds preamble ("Great question…"), tabular summaries, bullet restatements, and closing notes that cloud omits. Per-token throughput is *higher* locally (51.4 vs 49.7 tokens/sec), so Gemma is not slower per token — it just writes more of them. Every extra token extends wall-clock latency proportionally.

**Why:** `gemma4:e4b` was instruction-tuned on chat data that rewards thoroughness. Without an explicit `max_tokens` cap or a system-prompt instruction to be concise, Gemma defaults to a teaching-assistant register. Cloud's `gpt-5.4-nano` is tuned to match response length to query complexity.

Mitigation: add `max_tokens=400` or a "be concise" system directive. Not implemented here — the exercise is honest comparison, not local advocacy.

### 4. Empty final-message glitch (`hybrid-01`, root cause: post-tool generation failure)

On `hybrid-01`, local emitted a valid tool call, the tool returned (with an SSL error), and the agent appended the tool result to the message list — but Gemma did not produce a final assistant message. The recorded answer is an empty string. 781 completion tokens were billed (for the tool call and intermediate reasoning) but nothing reached the user.

**Why:** The agent loop calls `chat.completions.create` a second time after the tool result, expecting a final summary. On this query, Gemma produced an empty `content` field in the final response — likely a generation edge case triggered by the combination of tool-error content in the context and the dual RAG+tool framing. Cloud never hit this on any of the 20 queries.

This failure mode is silent: no exception, no warning, `tool_calls_emitted=True`, but answer is blank. Any orchestrator relying on this agent must treat an empty answer as a distinct failure state, not as a successful empty response.

### 5. Context-window drops during local ingest (root cause: nomic-embed-text 8K limit)

`nomic-embed-text` has an 8,192-token context limit. Approximately 6 of the 6,462 corpus chunks exceeded this. Those chunks were skipped during `water_rag_local` ingest after the per-item fallback in `OllamaEmbedder.embed_batch` exhausted retries.

**Why:** The chunker was tuned against `text-embedding-3-small`, which has a 8,191-token limit but sits in a longer-context model family. The UN water report and EPA treatment manual contain lengthy appendix tables that produce oversized chunks. Cloud handled them; local dropped them. The dropped chunks are edge-case reference material, so the quality impact is small but not zero — a query hitting exactly one of those chunks would fail silently.
