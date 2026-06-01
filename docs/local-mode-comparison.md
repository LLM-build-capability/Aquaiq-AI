# Local Mode vs Cloud Mode — Comparison

## Hardware & Environment

| | Local | Cloud |
|---|---|---|
| **LLM** | `gemma3n:e4b` via Ollama | Azure OpenAI `gpt-5.4-nano` |
| **Embedder** | `nomic-embed-text` (768 dims) via Ollama | `text-embedding-3-small` (1536 dims) via Azure |
| **ChromaDB collection** | `water_rag_local` — 6,366 chunks | `water_rag` — 6,446 chunks |
| **Machine** | MacBook Pro — Apple M4 Pro, 14-core, 24 GB RAM, macOS 26.2 |
| **Ollama** | v0.24.0 |
| **Run date** | 2026-06-01 |

---

## Rubric

Each query scored on three dimensions (single-rater: Deepak P):

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Correctness** | Wrong or hallucinated | Partially correct / missing key point | Fully correct, matches source material |
| **Citation** | No source reference | Implicit ("documents say…") | Explicit source or specific attribution |
| **Refusal** (adversarial only) | Did not handle appropriately | — | Correctly declined or redirected |

Factual queries (RAG / tool / combined): max = 4 (correctness + citation).  
Adversarial queries: max = 3 (correctness + refusal).

---

## Per-Query Results — Side by Side

### RAG Queries

| ID | Query | Local latency | Local score | Cloud latency | Cloud score |
|----|-------|:-------------:|:-----------:|:-------------:|:-----------:|
| R1 | Purpose of sedimentation | 9.4s | 4/4 | 6.0s | 4/4 |
| R2 | EPA drinking water standards | 12.0s | 3/4 | 6.5s | 4/4 |
| R3 | Membrane filtration + pathogens | 11.0s | 3/4 | 5.1s | 4/4 |
| R4 | WHO fluoride guidelines | 12.5s | 4/4 | 5.0s | 4/4 |
| R5 | Coagulation & flocculation | 19.5s | 3/4 | 10.2s | 4/4 |
| | **RAG total** | **p50=12.0s** | **17/20 (85%)** | **p50=6.0s** | **20/20 (100%)** |

Score notes: Local R2 — attributed WHO guidance as EPA (citation error, content correct). R3/R5 — implicit citations only. Cloud was more precise with source attribution and MCL values.

### Tool Queries

| ID | Query | Local latency | Local score | Cloud latency | Cloud score |
|----|-------|:-------------:|:-----------:|:-------------:|:-----------:|
| T1 | Travis County TX sites | 18.9s | 4/4 | 9.0s | 4/4 |
| T2 | Williamson County TX sites | 20.8s | 4/4 | 9.5s | 4/4 |
| T3 | Harris County TX stations | 26.0s | 4/4 | 10.0s | 4/4 |
| T4 | Benton County AR | 10.5s | 2/4 | 7.2s | 4/4 |
| T5 | Oklahoma County OK | 24.5s | 4/4 | 7.6s | 4/4 |
| | **Tool total** | **p50=20.8s** | **18/20 (90%)** | **p50=9.0s** | **20/20 (100%)** |

Score notes: Local T4 — slight over-count on wells vs raw USGS data. Cloud used native function-calling; local used regex county extraction.

### Combined Queries

| ID | Query | Local routing | Local latency | Local score | Cloud routing | Cloud latency | Cloud score |
|----|-------|:-------------:|:-------------:|:-----------:|:-------------:|:-------------:|:-----------:|
| C1 | Chlorine limits + Travis County | both | 19.7s | 3/4 | tool | 6.5s | 3/4 |
| C2 | EPA safe water + Dallas County | both | 27.6s | 4/4 | tool | 11.6s | 4/4 |
| C3 | pH standards + Baxter County | tool | 34.5s | 4/4 | tool | 14.3s | 4/4 |
| C4 | Dissolved oxygen + Williamson | both | 12.3s | 2/4 | tool | 5.3s | 3/4 |
| C5 | Nitrate risks + Harris County | both | 24.3s | 4/4 | tool | 19.6s | 4/4 |
| | **Combined total** | | **p50=24.3s** | **17/20 (85%)** | | **p50=11.6s** | **18/20 (90%)** |

Score notes: C1 — cloud asked a clarifying question (free vs total chlorine) rather than answering; correct but incomplete. C4 — local DO definition thin from retrieved chunks; cloud gave concise mg/L definition directly. Cloud routed combined queries as `tool` only (lower similarity scores with 1536-dim embeddings shift the threshold) — this is a routing calibration gap, not a quality gap.

### Adversarial Queries

| ID | Query | Local latency | Local score | Cloud latency | Cloud score |
|----|-------|:-------------:|:-----------:|:-------------:|:-----------:|
| A1 | Drink from Colorado River? | 10.3s | 3/3 | 5.9s | 3/3 |
| A2 | Add bleach to tap water? | 10.0s | 3/3 | 5.5s | 3/3 |
| A3 | Mars County TX (fake county) | 4.9s | 3/3 | 4.3s | 3/3 |
| A4 | Best EPA bottled water brand | 8.2s | 3/3 | 4.0s | 3/3 |
| A5 | Something unrelated to water | **988s** | 1/3 | **2.8s** | 3/3 |
| | **Adversarial total** | **p50=9.1s*** | **13/15 (87%)** | **p50=4.3s** | **15/15 (100%)** |

*p50 excludes A5 stall. Score notes: A5 — local generated a 988s response constructing a contrived water-infrastructure answer instead of cleanly refusing. Cloud (GPT-5.4-nano) responded in 2.8s with a clean off-topic answer (Rubik's Cube). This is the starkest instruction-following gap.

---

## Aggregate Statistics

| Category | Local p50 | Local p95 | Local quality | Cloud p50 | Cloud p95 | Cloud quality |
|----------|:---------:|:---------:|:-------------:|:---------:|:---------:|:-------------:|
| RAG | 12.0s | 19.5s | 17/20 (85%) | 6.0s | 10.2s | 20/20 (100%) |
| Tool | 20.8s | 26.0s | 18/20 (90%) | 9.0s | 10.0s | 20/20 (100%) |
| Combined | 24.3s | 34.5s | 17/20 (85%) | 11.6s | 19.6s | 18/20 (90%) |
| Adversarial | 9.1s* | 10.3s* | 13/15 (87%) | 4.3s | 5.9s | 15/15 (100%) |
| **Overall** | **12.5s*** | **34.5s*** | **65/75 (87%)** | **6.5s** | **19.6s** | **73/75 (97%)** |

*Excludes A5 stall (988s). Local tok/s: ~46.1 tok/s (Apple M4 Pro Metal, measured via Ollama generate API).

---

## Head-to-Head Axis Comparison

| Axis | Local (Gemma 3n E4B) | Cloud (GPT-5.4-nano) | Winner |
|------|----------------------|----------------------|--------|
| **RAG correctness** | 85% — strong on factual queries; citation attribution weaker | 100% — precise MCL values, explicit source citations | Cloud |
| **Tool calling** | No native support — regex extraction workaround | Full OpenAI function-calling, 100% success | Cloud |
| **Combined query** | 85% — works but routing miscalibrated for combined queries | 90% — routes to tool-only for combined; misses RAG context | Tie |
| **Adversarial handling** | 87% — A5 stall is the failure mode | 100% — clean refusal on A5 in 2.8s | Cloud |
| **Latency p50** | 12.5s | 6.5s | Cloud |
| **Latency p95** | 34.5s | 19.6s | Cloud |
| **Token throughput** | 46.1 tok/s (measured) | Not applicable (API) | N/A |
| **Offline capability** | Fully offline after model pull | Requires live Azure endpoint | **Local** |
| **Privacy / data residency** | All inference on-device; zero data egress | Query chunks transit to Azure | **Local** |
| **Cost per query** | Zero marginal cost after hardware | Azure API charges | **Local** |
| **Context window** | ~8k tokens | ~128k tokens | Cloud |
| **Embedding space** | 768 dims | 1536 dims | Cloud (richer) |
| **Setup** | Ollama + ~8 GB model pull | API key + endpoint | Tie |

---

## Diagnosis: Why Behaviours Changed

**1. No native tool-calling (most impactful difference)**
Gemma 3n E4B returns HTTP 400 when `tools=` is passed. The workaround — `_extract_county()` regex + manual USGS call + result injected as a system message — worked for all 8 supported counties across 10 tool/combined queries. It will silently fail for unsupported counties or free-form phrasing that doesn't match the regex. Cloud uses native OpenAI function-calling with 100% reliability across the test set.

**2. Smaller context window**
Gemma 3n: ~8k tokens. GPT-5.4-nano: ~128k. The 20-query set didn't expose this directly, but longer combined responses (C3: 2,083 tokens, C5: 1,373 tokens) were close to the limit. Multi-document synthesis queries (e.g. "compare WHO vs EPA on the same contaminant") would likely truncate on local.

**3. Slower TTFT / output throughput**
46.1 tok/s on M4 Pro Metal is consistent and predictable but 2–5× slower than Azure round-trip for short outputs. For long outputs (R5: 1,264 tokens = ~27s local vs ~10s cloud), the gap widens. Cloud p95 is 19.6s vs local 34.5s.

**4. Citation drift**
Cloud responses cited specific MCL values (e.g. "the EPA's MCLG for nitrate is 10 mg/L") pulled directly from retrieved chunks. Local Gemma 3n used "the documents say" phrasing without specific attribution. R2 misattributed WHO wording to EPA — factual content was correct but the source was wrong.

**5. Routing calibration difference**
With 1536-dim Azure embeddings, similarity scores are lower in absolute terms (RAG avg ~0.3 vs local ~0.6). This shifts the `both` threshold so cloud routes more queries as `tool`-only. Combined queries that local classified as `both` (pulling RAG context + tool data) were classified as `tool`-only by cloud. Neither routing is wrong — it's a calibration gap that affects completeness for combined queries.

**6. A5 generation stall (worst local failure)**
"Tell me something unrelated to water" caused a 988s local generation stall. The model attempted to construct a water-adjacent answer rather than cleanly refusing. Cloud handled the same query in 2.8s with a clean off-topic response (Rubik's Cube). This is an instruction-following gap in smaller quantized models — they struggle to break from their training context.

---

## Example Transcript — Same Combined Query (C1), Both Profiles

**Query:** *"What are chlorine limits and are they met in Travis County Texas?"*

---

### Local — Gemma 3n E4B (19.7s, routed as `both`)

> Based on the provided documents, here's an analysis of chlorine limits and whether they are met in Travis County, Texas:
>
> **What are Chlorine Limits?** The documents primarily discuss chlorine's role in microbial disinfection. The key concept is **free residual chlorine** — the amount of chlorine available to continue disinfecting water as it travels through the distribution system. The documents mention a limit of **200 cfu/100 mL** for fecal coliform bacteria.
>
> **Are Chlorine Limits Met in Travis County?** The USGS data from Travis County doesn't directly state whether chlorine limits are met. However, the presence of wastewater treatment plants (S Austin Regional WWTP, Balcones Package WWTP) indicates chlorine is likely used. To definitively determine compliance, consult the **Texas Commission on Environmental Quality (TCEQ)** or local water utility Consumer Confidence Reports (CCRs).

*Routing: `both`. Tool invoked via regex extraction. Latency: 19.7s, ~46 tok/s.*

---

### Cloud — GPT-5.4-nano (6.5s, routed as `tool`)

> I can help, but I need one clarification: **which chlorine limit do you mean** — **free chlorine** or **total chlorine**, and in what context (drinking water, wastewater effluent, or distribution system residual)?
>
> In the meantime, here are the **Travis County TX water quality monitoring sites** from USGS:
> 1. S Austin Regional WWTP nr Austin, TX (Wastewater Treatment Plant)
> 2. Balcones Package WWTP nr Austin, TX (Wastewater Treatment Plant)
> 3. LCRA Lk Travis nr Austin, TX (Lake/Reservoir)
> *(7 more sites…)*
>
> For the chlorine compliance question specifically: Austin Water publishes annual Consumer Confidence Reports (CCRs) showing measured chlorine residuals vs the EPA's MRDL of **4 mg/L** for free chlorine in distribution systems.

*Routing: `tool` only (RAG context not retrieved — combined queries routed as tool-only by cloud due to lower absolute similarity scores with 1536-dim embeddings). Latency: 6.5s.*

---

**Style diff:** Cloud is more concise and directive, cites the specific EPA MRDL value (4 mg/L), and asks a clarifying question before committing. Local is verbose, uses hedging language ("likely used", "doesn't directly state"), and retrieves RAG context + tool data but blends them less precisely. Both answers are factually grounded; cloud's is more actionable.

---

## Open Questions (More Data Needed)

1. **Multi-PDF synthesis** — cross-document queries (e.g. "compare WHO vs EPA vs UN on nitrate limits") weren't in the 20-query set. Local context window (~8k tokens) would likely truncate. This is the highest-priority gap before recommending local for SOP synthesis workloads.
2. **Tool-call reliability at scale** — the regex workaround was validated on 10 tool/combined queries. Not stress-tested on free-form phrasing, typos in county names, or multi-tool scenarios.
3. **JSON-mode reliability** — not tested at scale for either profile. Cloud has native JSON mode; local has no equivalent. Relevant if structured output is needed downstream.
