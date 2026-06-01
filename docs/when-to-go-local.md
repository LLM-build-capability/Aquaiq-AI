# When to Go Local — Decision Document

**Author:** Prem Kumar Reddy Kothakapu  
**Audience:** peer architect  
**Data source:** 20-query benchmark — `docs/local-mode-comparison.md`

---

## Recommended scenario

**Air-gapped water treatment plant — field technician tablet querying operational SOPs offline.**

A field technician at a municipal water treatment facility needs to look up chemical dosing procedures, equipment maintenance steps, and regulatory limits while standing on the plant floor. The facility network is air-gapped (no internet egress from the operations zone), the tablet has no persistent cloud connection, and the queries are repetitive and narrow — the same 50–200 SOP questions account for 90% of usage.

This is the scenario where the local profile's weaknesses matter least and its strengths are load-bearing.

---

## Hard constraints — scenarios where local is disqualifying

Before evaluating axes, rule out local entirely if any of these apply:

| Constraint | Why local fails | Benchmark evidence |
|---|---|---|
| **Workflow requires reliable tool / API calls** | `gemma4:e4b` emits tool calls on only 4/10 tool-seeking queries. The other 6 silently fall through to a clarification question or RAG-only answer. No exception, no warning. This is a post-training gap, not a prompt-engineering problem — tighter tool descriptions and temperature reduction did not fix it. | tool-02 through tool-06: 0/5 tool calls emitted; hybrid-01: tool called but empty final message |
| **Queries span multiple regulatory bodies or document sources** | The local embedder (`nomic-embed-text`) returns a different corpus ranking than the cloud embedder (`text-embedding-3-small`) for the same query. An EPA-scoped query retrieved a WHO document and answered with the wrong limit. If the corpus mixes regulatory sources, scope leakage is likely. | rag-01: cloud correct (EPA 15 µg/L), local wrong (WHO 10 µg/L) |
| **Out-of-scope refusal is required** | Neither profile refused off-corpus questions. Both answered factually from parametric knowledge and surfaced false-positive sources. This is a routing classifier limitation shared by both profiles, but if hard refusal is a safety requirement, neither profile is ready without an additional guard layer. | oos-01, oos-02: 0/2 refused on both profiles |
| **Corpus contains long technical appendices (>8K tokens per chunk)** | `nomic-embed-text` has an 8,192-token context limit. Chunks exceeding this are silently dropped at ingest. The per-item fallback recovers most batches but cannot recover genuinely oversized chunks. | 6 chunks dropped from 6,460 attempted at local ingest |

If none of these apply, proceed to axis scoring.

---

## Named axes

Scores are derived from the benchmark data. Scale: ✓ (acceptable/win), ~ (acceptable with caveats), ✗ (loss accepted).

| Axis | Cloud | Local | Winner | Notes from benchmark |
|---|:---:|:---:|:---:|---|
| **Offline / air-gap capability** | ✗ | ✓ | **Local** | Cloud requires Azure endpoint access; local runs entirely on-device after install |
| **Data privacy / no egress** | ✗ | ✓ | **Local** | Cloud sends all queries and retrieved chunks to Azure; local never leaves the device |
| **RAG correctness (general)** | 2.00/2 | 1.38/2 | Cloud | Local drops ~0.6 on RAG; 2 outright failures in 8 queries |
| **Tool-call reliability** | 10/10 | 4/10 | Cloud | Local fails 60% of tool-seeking queries silently |
| **Latency p50** | 5.4 s | 15.2 s | Cloud | Local is ~2.8× slower at median |
| **Latency p95** | 8.8 s | 30.7 s | Cloud | Local worst-case is 30+ s for verbose RAG answers |
| **Cost per 1 000 queries** | ~$2–5 (Azure token pricing) | ~$0 (marginal) | **Local** | After hardware, local queries are free; no per-token billing |
| **Hallucination risk** | Low | Moderate | Cloud | Local introduced Falkenmark figures not in source; wrong regulatory body on rag-01 |
| **Corpus coverage** | 6,460 chunks (full) | 6,454 chunks (−6) | Cloud | 6 oversized chunks dropped at ingest; minor for SOPs |

---

## Where local lost — and why I accepted it

**RAG correctness and retrieval precision** are worse locally. The benchmark showed two outright wrong answers on RAG queries (rag-01, rag-08) and one hallucination (rag-07 Falkenmark figures). For the SOP scenario, this loss is acceptable for two reasons:

1. The SOP corpus is narrower and more homogeneous than the four-document benchmark corpus. A corpus containing only one regulatory body's documents removes the embedder scope-leakage problem (the rag-01 failure) entirely. There is no WHO vs EPA ambiguity if the corpus is purely internal SOPs.
2. The field technician's queries are high-frequency and low-variance. The first week of deployment surfaces the handful of queries where the model fails; those can be flagged and a verified fallback answer cached. In a RAG-over-diverse-corpus scenario (research, compliance audit), this remediation path doesn't scale.

**Tool-call reliability** is a clear and unmitigated loss (4/10 local vs 10/10 cloud). This is not a configuration issue — the tool schema, system prompt, and temperature were identical on both profiles. Gemma simply does not follow the OpenAI `tools=[]` calling convention reliably. For the SOP use case I am recommending, this loss is **irrelevant because the scenario has no tool calls** — the field technician queries documents only. The recommended scenario was selected in part because it sidesteps this disqualifying constraint. If live sensor data from a plant historian or any external API were needed, local would be ruled out regardless of the air-gap argument.

**Latency** (p50 15.2 s, p95 30.7 s) is a real user-experience cost. A field technician waiting 30 seconds for a response is suboptimal. Mitigation: apply a `max_tokens` cap (400 tokens covers all SOP answers in the benchmark) which would cut the worst-case latency roughly in proportion to token reduction; or stream the response so the technician reads as the model writes.

---

## Where local won — and that win is load-bearing

**Offline and air-gap capability is the load-bearing win.**

The facility network policy prohibits egress from the operations zone. Cloud is not a viable option — not slower, not more expensive, simply not permitted. Any solution that requires calling Azure OpenAI would need a network exception, a security review, and ongoing compliance overhead. Local eliminates that dependency entirely.

**Data privacy** compounds this. Queries about chemical dosing concentrations, equipment failure thresholds, and incident response procedures are operationally sensitive. Sending them to a cloud provider raises data classification questions that a self-hosted model avoids by design.

For this specific scenario, both axes are binary: either the system works air-gapped or it does not. Local passes; cloud fails.

---

## One axis where I need more data

**Hallucination rate on narrow SOP corpora.**

The benchmark showed local hallucinating Falkenmark index figures on a UN report summary. That failure mode — model introducing plausible-but-unsourced numbers — is concerning for a domain where wrong dosing quantities can have safety consequences.

The benchmark corpus (4 diverse PDFs, ~6,460 chunks) is not representative of a narrow SOP corpus (50–100 documents, shorter, more repetitive). Hallucination rates in RAG systems vary significantly with corpus homogeneity and chunk density. I do not have data on whether `gemma4:e4b` hallucinates more or less on narrow technical corpora vs the diverse benchmark corpus.

Before recommending production deployment, I would run a targeted evaluation: 50 queries against a pilot SOP corpus, with every answer manually verified against the source document, measuring hallucinated-fact rate rather than overall correctness score. If the hallucination rate on the narrow corpus drops below 5%, local is ready. If it stays at the benchmark level (~10% of RAG answers), a post-generation grounding check (verify every number against retrieved chunks) is needed before go-live.
