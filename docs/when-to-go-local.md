# When to Go Local — Decision Guide

## Scenario: Plant-Floor SOP Retrieval (Offline Water Treatment)

**Context:** A water treatment plant operator on the plant floor needs to query Standard Operating Procedures (SOPs) and treatment guidelines. The plant has unreliable or no internet connectivity (e.g., remote facility, air-gapped industrial network, or Zscaler-class MITM proxies that break Python SSL).

---

## Decision Framework

### Go Local when ALL of these are true:

| Condition | Why it matters |
|-----------|---------------|
| **Offline / unreliable internet** | Cloud inference requires a live Azure endpoint. One dropped connection mid-query = no answer. Local Ollama runs entirely on-device. |
| **Data residency / privacy required** | SOP documents may contain proprietary process parameters or regulated chemical handling procedures. Sending chunks to Azure means those fragments transit and are processed outside the facility. Local mode: data never leaves the machine. |
| **Query type is primarily RAG** | The 20-query benchmark shows local RAG quality is near-identical to cloud for factual water treatment questions. Plant-floor SOP queries (e.g., "what is the chlorination dose for X flow rate?") fall squarely in the RAG bucket. |
| **Latency tolerance is >10s** | Local p50 is 12–25s on Apple Silicon. Operators waiting for a procedure reference can tolerate this. Real-time control loops cannot. |
| **Hardware available** | Gemma 3n E4B requires ~8 GB VRAM/RAM. A modern workstation or ruggedised laptop at the plant is sufficient. |

### Stick with Cloud when ANY of these are true:

| Condition | Why it forces cloud |
|-----------|---------------------|
| **Tool-calling required** | USGS API lookups and any future structured tool use require native function-calling. Gemma 3n E4B does not support this; the workaround (regex county extraction) is fragile. |
| **Sub-5s latency required** | Real-time dashboards, alert systems, or interactive chat with impatient users need cloud inference speed. |
| **Multi-document synthesis across many PDFs** | Local context window (~8k tokens) limits how much cross-document content the model can reason over in a single turn. Cloud models handle much larger contexts. |
| **Query complexity is unpredictable** | If users will ask open-ended, multi-step, or adversarial questions, cloud's stronger instruction-following and refusal behaviour is more reliable. |

---

## Applied to Plant-Floor SOP Retrieval

**Verdict: Local wins for this scenario.**

| Axis | Assessment |
|------|------------|
| Offline capability | Load-bearing. Remote water treatment plants frequently have no reliable internet. Local is the only viable option. |
| Privacy / data residency | SOP documents often contain proprietary chemical dosing, equipment specs, and emergency procedures. Keeping them on-device eliminates a data exfiltration risk vector. |
| RAG correctness | Benchmark shows 5/5 correct on factual water treatment RAG queries (sedimentation, membrane filtration, chlorination, coagulation, fluoride guidelines). Plant-floor SOP queries are the same shape. |
| Latency | 12–25s is acceptable for an operator looking up a procedure. Not acceptable for a control loop. |
| Tool calling | USGS real-time water quality data is NOT needed for SOP retrieval — operators need document context, not live API data. The missing tool-calling capability is irrelevant for this scenario. |
| Hardware footprint | A ruggedised laptop with 16 GB RAM running Ollama + Gemma 3n is realistic for a plant floor deployment. |

**Where local still needs validation:**
- **Multi-PDF SOP synthesis.** If an operator asks "compare the chlorination procedure in the EPA doc vs the WHO guidelines," this requires the model to reason across two large documents simultaneously. The 20-query benchmark didn't test this pattern. The local context window may truncate one source. This needs a dedicated test before deploying in a scenario where cross-document synthesis is common.
- **Embedding quality for technical terminology.** `nomic-embed-text` (768 dims) may underperform `text-embedding-3-small` (1536 dims) on highly domain-specific chemical or process terminology not well-represented in its training data. Monitor retrieval precision on plant-specific SOP vocabulary.

---

## Deployment Checklist (Local Mode)

- [ ] Install Ollama on plant workstation: `brew install ollama` (or Linux equivalent)
- [ ] Pull models: `ollama pull gemma3n:e4b && ollama pull nomic-embed-text`
- [ ] Copy SOP PDFs to `data/` directory
- [ ] Set `LLM_PROFILE=local` in `.env`
- [ ] Run ingest: `python -m src.aquaiq_ai.ingest`
- [ ] Verify collection: check `water_rag_local` chunk count > 0
- [ ] Start app: `streamlit run application.py`
- [ ] No internet connection required after this point
