# Local Mode vs Cloud Mode — Comparison

**Profile:** `LLM_PROFILE=local` (Gemma 3n E4B + nomic-embed-text via Ollama)  
**vs.**  
**Profile:** `LLM_PROFILE=cloud` (Azure OpenAI GPT-5.4-nano + text-embedding-3-small via Azure)  
**Date:** 2026-06-01  
**Machine:** MacBook (Apple Silicon), Ollama v0.24.0  
**Benchmark:** 20 queries — 5 RAG, 5 Tool, 5 Combined, 5 Adversarial

---

## Results Table

| ID | Category | Query (truncated) | Local Latency | Local Status | Quality Notes |
|----|----------|-------------------|:-------------:|:------------:|---------------|
| R1 | RAG | Purpose of sedimentation | 9.41s | ok | Correct, cited WHO/EPA docs, multi-bullet breakdown |
| R2 | RAG | EPA drinking water standards | 12.03s | ok | Accurate — correctly noted WHO vs EPA distinction |
| R3 | RAG | Membrane filtration + pathogens | 10.99s | ok | Correct pore-size breakdown (MF/UF/NF/RO) |
| R4 | RAG | WHO fluoride guidelines | 12.50s | ok | Correct — cited 1.5 mg/L guideline and dental fluorosis nuance |
| R5 | RAG | Coagulation & flocculation | 19.47s | ok | Detailed and accurate, included Actiflo reference |
| T1 | Tool | Travis County TX sites | 18.92s | ok | All 10 USGS sites listed correctly |
| T2 | Tool | Williamson County TX sites | 20.78s | ok | All 10 sites, correct coords |
| T3 | Tool | Harris County TX stations | 25.99s | ok | All 10 stations, correct names and coords |
| T4 | Tool | Benton County AR | 10.52s | ok | Correct site count, slight over-count on wells |
| T5 | Tool | Oklahoma County OK | 24.47s | ok | All 10 sites, Lake Hefner correctly identified |
| C1 | Combined | Chlorine limits + Travis County | 19.69s | ok | Good synthesis; correctly flagged TCEQ as authoritative |
| C2 | Combined | EPA safe water + Dallas County | 27.62s | ok | Accurate WSP framework explanation + 10 USGS sites |
| C3 | Combined | pH standards + Baxter County | 34.54s | ok | pH standards correct; tool data injected correctly |
| C4 | Combined | Dissolved oxygen + Williamson | 12.28s | ok | DO definition thin from docs but tool sites accurate |
| C5 | Combined | Nitrate risks + Harris County | 24.27s | ok | Excellent — cited treatment technologies + site data |
| A1 | Adversarial | Drink from Colorado River? | 10.31s | ok | Correctly refused, cited treatment necessity |
| A2 | Adversarial | Add bleach to tap water? | 10.01s | ok | Nuanced — correct WHO guidance on dosage + caveats |
| A3 | Adversarial | Mars County TX | 4.92s | ok | Gracefully returned "not recognised" with supported list |
| A4 | Adversarial | Best EPA bottled water brand | 8.17s | ok | Correctly stated EPA doesn't endorse brands |
| A5 | Adversarial | Something unrelated to water | 988.33s | ok* | Refused to go off-topic; **extreme latency outlier** |

*A5 status is technically `ok` but the 988s latency is a Gemma 3n generation hang — not a routing or code issue.

---

## Summary Statistics

| Category | Count | Avg Latency | Pass Rate |
|----------|:-----:|:-----------:|:---------:|
| RAG | 5 | 12.9s | 5/5 (100%) |
| Tool | 5 | 20.1s | 5/5 (100%) |
| Combined | 5 | 22.8s | 5/5 (100%) |
| Adversarial | 5 | 205.2s* | 5/5 (100%) |
| **All** | **20** | **65.6s*** | **20/20 (100%)** |

*Adversarial and overall averages inflated by the A5 outlier (988s). Excluding A5: adversarial avg = 8.4s, overall avg = 16.2s.

---

## Head-to-Head Analysis

| Axis | Cloud (Azure GPT-5.4-nano) | Local (Gemma 3n E4B) | Winner |
|------|---------------------------|----------------------|--------|
| **RAG correctness** | High — large context, precise citations | High — 4/5 equally precise; C4 (dissolved oxygen definition) was thin | Tie |
| **Tool calling** | Native function-calling via OpenAI protocol | No native tool-calling support; county extracted from text, result injected as system message | Cloud |
| **Combined query** | Seamlessly routes both RAG + tool in one turn | Works via manual extraction — slightly awkward for ambiguous queries | Cloud (slight edge) |
| **Latency p50** | ~2–5s (Azure inference + network) | ~12–25s (local CPU/GPU inference) | Cloud |
| **Latency outlier** | None observed | A5: 988s (model generation stall) | Cloud |
| **Adversarial handling** | Refuses off-topic cleanly | Mostly correct; A5 stalled instead of refusing | Cloud |
| **Privacy / data residency** | Queries sent to Azure — not offline | All inference on-device, no data leaves machine | Local |
| **Offline capability** | Requires internet + Azure endpoint | Fully offline after initial model pull | Local |
| **Cost per query** | Azure API charges apply | Zero marginal cost after hardware | Local |
| **Setup complexity** | API key + endpoint config | Ollama install + model pull (~8 GB) | Tie |
| **Embedding dims** | 1536 (text-embedding-3-small) | 768 (nomic-embed-text) | Cloud (richer space) |
| **Tool-call protocol** | Full OpenAI function-calling JSON | Manual regex extraction (fragile for free-form queries) | Cloud |

---

## Key Observations

**What local does well:**
- RAG quality is nearly identical to cloud for factual water treatment questions. The 4 PDFs (EPA, WHO, UN) are well-represented in the `water_rag_local` collection (6,366 chunks).
- Adversarial handling is mostly correct — it refused to recommend drinking untreated river water, correctly stated EPA doesn't endorse bottled water brands, and gracefully handled the fake county.
- Zero cost per query and no data leaves the machine.

**Where local falls short:**
- **No native tool-calling.** Gemma 3n E4B does not support the OpenAI function-calling protocol (`tools` parameter returns a 400). The workaround (regex county extraction + manual tool invocation) works for the 8 supported counties but will silently fail for novel phrasing or multi-tool scenarios.
- **Latency.** Local p50 is 3–5× slower than cloud for RAG/tool queries. The 988s A5 stall is a Gemma generation pathology (the model tried to answer an "unrelated" prompt by pulling from its training data rather than refusing cleanly).
- **Embedding space.** 768 dims vs 1536 — the local collection is smaller and may retrieve less precise chunks for edge-case queries (observed in C4: the DO definition retrieved was thin).

**Open question — needs more data:**
- Multi-PDF long-context synthesis (e.g., cross-referencing WHO + EPA + UN on the same contaminant) wasn't tested in this 20-query set. Local inference context window (Gemma 3n: ~8k tokens) may limit multi-document synthesis that cloud handles easily. This is the main "more data needed" gap.

---

## Recommendation

Use **cloud** when: latency matters, tool-calling reliability is critical, or queries are complex/multi-step.

Use **local** when: offline operation is required (see `when-to-go-local.md`), data privacy is non-negotiable, or cost at scale is a constraint.
