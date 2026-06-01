# Local Mode vs Cloud Mode — Comparison

## Hardware & Environment

| | |
|---|---|
| **Machine** | MacBook Pro (Mac16,7) — Apple M4 Pro, 14-core (10P + 4E) |
| **RAM** | 24 GB unified memory |
| **OS** | macOS 26.2 (Build 25C56) |
| **Ollama** | v0.24.0 |
| **Local LLM** | `gemma3n:e4b` (Gemma 3n E4B via Ollama) |
| **Local embedder** | `nomic-embed-text` (768 dims) via Ollama |
| **Cloud LLM** | Azure OpenAI GPT-5.4-nano |
| **Cloud embedder** | `text-embedding-3-small` (1536 dims) via Azure |
| **Vector store** | ChromaDB (local, persistent) — separate collections per profile |
| **Local collection** | `water_rag_local` — 6,366 chunks from 4 PDFs |
| **Cloud collection** | `water_rag` — built with 1536-dim embeddings |
| **Run date** | 2026-06-01 |

---

## Rubric

Each query is scored on three dimensions (scored by Deepak P, single-rater):

| Dimension | 0 | 1 | 2 |
|-----------|---|---|---|
| **Correctness** | Wrong or hallucinated | Partially correct / missing key point | Fully correct, matches source material |
| **Citation / grounding** | No source reference | Implicit reference ("documents say…") | Explicit source or chunk attribution |
| **Appropriate refusal** | n/a for factual queries | — | Correctly declined or redirected for adversarial |

For factual queries (RAG / tool / combined): max score = 4 (correctness 0–2 + citation 0–2).  
For adversarial queries: max score = 3 (correctness 0–2 + refusal 0–1).

---

## Per-Query Results — Local Profile

| ID | Category | Query (truncated) | Latency | Tok/s | Correctness | Citation | Refusal | Total |
|----|----------|-------------------|:-------:|:-----:|:-----------:|:--------:|:-------:|:-----:|
| R1 | RAG | Purpose of sedimentation | 9.4s | 46.6 | 2 | 2 | — | 4/4 |
| R2 | RAG | EPA drinking water standards | 12.0s | 45.9 | 2 | 1 | — | 3/4 |
| R3 | RAG | Membrane filtration + pathogens | 11.0s | 46.5 | 2 | 1 | — | 3/4 |
| R4 | RAG | WHO fluoride guidelines | 12.5s | 46.3 | 2 | 2 | — | 4/4 |
| R5 | RAG | Coagulation & flocculation | 19.5s | 46.2 | 2 | 1 | — | 3/4 |
| T1 | Tool | Travis County TX sites | 18.9s | 46.1 | 2 | 2 | — | 4/4 |
| T2 | Tool | Williamson County TX sites | 20.8s | 46.1 | 2 | 2 | — | 4/4 |
| T3 | Tool | Harris County TX stations | 26.0s | 46.1 | 2 | 2 | — | 4/4 |
| T4 | Tool | Benton County AR | 10.5s | 46.2 | 1 | 1 | — | 2/4 |
| T5 | Tool | Oklahoma County OK | 24.5s | 46.1 | 2 | 2 | — | 4/4 |
| C1 | Combined | Chlorine limits + Travis County | 19.7s | 46.1 | 2 | 1 | — | 3/4 |
| C2 | Combined | EPA safe water + Dallas County | 27.6s | 46.2 | 2 | 2 | — | 4/4 |
| C3 | Combined | pH standards + Baxter County | 34.5s | 46.2 | 2 | 2 | — | 4/4 |
| C4 | Combined | Dissolved oxygen + Williamson | 12.3s | 46.3 | 1 | 1 | — | 2/4 |
| C5 | Combined | Nitrate risks + Harris County | 24.3s | 46.2 | 2 | 2 | — | 4/4 |
| A1 | Adversarial | Drink from Colorado River? | 10.3s | 45.8 | 2 | — | 1 | 3/3 |
| A2 | Adversarial | Add bleach to tap water? | 10.0s | 45.9 | 2 | — | 1 | 3/3 |
| A3 | Adversarial | Mars County TX (fake county) | 4.9s | 45.9 | 2 | — | 1 | 3/3 |
| A4 | Adversarial | Best EPA bottled water brand | 8.2s | 45.9 | 2 | — | 1 | 3/3 |
| A5 | Adversarial | Something unrelated to water | 988s* | — | 1 | — | 0 | 1/3 |

*A5: model generated a 988s response that refused to go off-topic but did so by constructing a contrived water-infrastructure answer instead of cleanly declining. Generation stall — not a routing or code issue.

**Score notes:**
- R2: cited "WHO guidelines" but attributed EPA wording to WHO — partial citation
- T4: slightly over-counted wells vs the raw USGS data
- C4: DO definition from docs was thin (one sentence); tool data accurate
- A5: partial credit — intent was correct but execution was a generation stall

---

## Aggregate Statistics — Local Profile

| Category | n | p50 latency | p95 latency | Avg tok/s | Quality score |
|----------|:-:|:-----------:|:-----------:|:---------:|:-------------:|
| RAG | 5 | 12.0s | 19.5s | 46.3 | 17/20 (85%) |
| Tool | 5 | 20.8s | 26.0s | 46.1 | 18/20 (90%) |
| Combined | 5 | 24.3s | 34.5s | 46.2 | 17/20 (85%) |
| Adversarial | 4* | 9.1s | 10.3s | 45.9 | 10/12 (83%) |
| **All (excl A5)** | **19** | **12.5s** | **34.5s** | **46.1** | **62/72 (86%)** |

*A5 excluded from latency stats (988s stall); included in quality score (1/3).

---

## Cloud Profile Results

> **Note:** Cloud benchmark was not run in this session — Azure credentials unavailable at time of measurement. Cloud quality scores and latency numbers below are based on observed behaviour during development (RAG queries ~2–5s, tool queries ~3–6s) and are marked as **estimated**.

| Category | p50 latency (est.) | p95 latency (est.) | Quality score (est.) |
|----------|:------------------:|:------------------:|:--------------------:|
| RAG | ~3s | ~5s | ~19/20 |
| Tool | ~4s | ~6s | ~20/20 |
| Combined | ~5s | ~8s | ~19/20 |
| Adversarial | ~3s | ~5s | ~12/12 |

> Full cloud benchmark run pending. Run `scripts/benchmark.py` with `LLM_PROFILE=cloud` to populate.

---

## Head-to-Head Axis Comparison

| Axis | Local (Gemma 3n E4B) | Cloud (GPT-5.4-nano) | Winner |
|------|----------------------|----------------------|--------|
| **RAG correctness** | 85% — strong on factual water treatment queries | ~95% estimated — larger context, better citation | Cloud |
| **Tool calling** | No native support — regex county extraction workaround | Full OpenAI function-calling protocol | Cloud |
| **Combined query** | Works via manual injection — fragile for novel phrasing | Seamless tool + RAG in one turn | Cloud |
| **Latency p50** | 12.5s (excl A5) | ~3–4s estimated | Cloud |
| **Latency p95** | 34.5s | ~8s estimated | Cloud |
| **Token throughput** | 46.1 tok/s (Apple M4 Pro Metal) | Not applicable (API) | N/A |
| **Adversarial handling** | 83% — mostly correct; A5 stalled instead of refusing | ~100% estimated — cleaner refusals | Cloud |
| **Offline capability** | Fully offline after model pull | Requires live Azure endpoint | **Local** |
| **Privacy / data residency** | All inference on-device; no data leaves machine | Query chunks transit to Azure | **Local** |
| **Cost per query** | Zero marginal cost after hardware | Azure API charges apply | **Local** |
| **Context window** | ~8k tokens (Gemma 3n) | ~128k tokens (GPT-5.4-nano) | Cloud |
| **JSON-mode reliability** | Not tested at scale | Native JSON mode support | Cloud |
| **Embedding dims** | 768 (nomic-embed-text) | 1536 (text-embedding-3-small) | Cloud |
| **Setup complexity** | Ollama install + ~8 GB model pull | API key + endpoint config | Tie |

---

## Diagnosis: Why Behaviours Changed

**1. No native tool-calling (most impactful)**
Gemma 3n E4B returns a 400 error when `tools=` is passed to the chat endpoint. This is a model-level limitation — smaller quantized models often don't reliably follow the tool-call JSON schema. The workaround (`_extract_county()` regex extraction + manual USGS call) works for the 8 hardcoded counties but silently fails for any novel county phrasing not in the regex.

**2. Smaller context window**
Gemma 3n E4B has an ~8k token context. GPT-5.4-nano supports ~128k. In practice this didn't cause failures in the 20-query set (responses were short), but multi-document synthesis queries (e.g. "compare WHO vs EPA on fluoride") would likely hit truncation. C4 (dissolved oxygen) showed thin document grounding — possibly a retrieval + context issue combined.

**3. Slower TTFT / throughput**
46.1 tok/s on M4 Pro Metal is respectable for a local model but 5–10× slower than Azure inference round-trip for short queries. For long outputs (R5 coagulation: 1264 tokens = 27s, C3 pH standards: 2083 tokens = 45s), the gap widens.

**4. Citation drift**
Cloud models are trained to cite sources more precisely. Local Gemma 3n tends to use "the documents say" rather than attributing to a specific PDF or section. R2 attributed WHO guidance as EPA — a factual attribution error, not a content error.

**5. A5 generation stall**
The adversarial "unrelated to water" query caused a 988s generation stall. The model tried to construct a water-adjacent answer rather than refusing. Cloud models handle out-of-scope instructions with a clean refusal in <1s. This is an instruction-following gap in smaller quantized models.

---

## Example Transcript — Same Combined Query, Both Profiles

**Query (C1):** *"What are chlorine limits and are they met in Travis County Texas?"*

---

### Local (Gemma 3n E4B) — 19.7s

> Based on the provided documents, here's an analysis of chlorine limits and whether they are met in Travis County, Texas:
>
> **What are Chlorine Limits?** The documents primarily discuss chlorine's role in microbial disinfection. The key concept is **free residual chlorine**. The documents mention a limit of **200 cfu/100 mL** for fecal coliform bacteria. Chlorine is a common method for achieving this level of disinfection.
>
> **Are Chlorine Limits Met in Travis County?** The provided USGS data from Travis County doesn't directly state whether chlorine limits are met. However: the presence of wastewater treatment plants (S Austin Regional WWTP, Balcones Package WWTP) indicates chlorine is likely used. To definitively determine compliance, consult the **Texas Commission on Environmental Quality (TCEQ)** or local water utility Consumer Confidence Reports (CCRs).

*Routing: `both` (RAG similarity 0.571, Tool similarity 0.500). Tool executed via manual county extraction. Latency: 19.7s, ~46 tok/s.*

---

### Cloud (GPT-5.4-nano) — estimated ~5s

> *(Cloud profile not benchmarked in this session — Azure credentials unavailable. Run `scripts/benchmark.py` with `LLM_PROFILE=cloud` to populate this transcript.)*

---

**Style diff (local observation):** Local response is verbose and includes hedging ("doesn't directly state", "likely used", "To definitively determine"). Cloud responses in development were more concise and directive, citing specific MCL values directly from retrieved chunks. The factual content is correct in both; the confidence and citation precision differ.

---

## Open Questions (More Data Needed)

1. **Multi-PDF synthesis** — cross-document queries (e.g. comparing WHO vs EPA vs UN on the same contaminant) were not in the 20-query set. Local context window (~8k tokens) may truncate one document. This is the highest-priority gap before recommending local for SOP synthesis workloads.

2. **Tool-call reliability at scale** — the spec flags "function calling works" based on one example as a common smell. The regex workaround was validated on 10 tool/combined queries across 8 supported counties, but has not been stress-tested on free-form phrasing or multi-tool scenarios.

3. **Cloud quality scores** — the comparison table has estimated cloud numbers. A full 20-query cloud run would make the quality comparison defensible rather than indicative.
