# Exercise 3 – Decision Document: Teams Bot with Entra ID SSO and RAG over SharePoint

## 1. Problem Statement

**Business unit:** Water  
**Content:** ~8,000 SharePoint documents (SOPs, safety data sheets, playbooks, training material)  
**Users:** Frontline colleagues  
**Pain point:** SharePoint search alone is insufficient; users ask constant questions.

**Must-have features:**
- Entra ID SSO (no separate login)
- Respect SharePoint permissions (never show content a user cannot access)
- Natural‑language Q&A with citations back to source documents
- Multi‑turn follow‑ups
- Deployed inside the Ecolab Azure tenant

---

## 2. Options Considered

| Variant | Core Idea | Typical Pull |
|---------|-----------|---------------|
| **A – Azure-native, hand-built** | Bot Framework + Azure OpenAI + Azure AI Search (with SharePoint indexer + ACL trimming) + App Service + Cosmos DB | Maximum control, best extensibility, highest ownership cost |
| **B – Microsoft Copilot Studio / declarative agent** (interpreted as AKS‑based microservices in your implementation) | Custom containerised solution on AKS with API gateway, ACL engine, tool routing, Redis, and Azure OpenAI | Fastest to launch in theory, but  AKS design increases complexity |


---

## 3. Winner: **Variant A**

**Justification:**  
Variant A gives us the fastest time‑to‑first‑user, lower monthly cost, and simpler operational requirements while fully satisfying mandatory compliance axes (permission fidelity, EU data residency). The higher vendor lock‑in and moderate extensibility are acceptable given the Water BU’s immediate need for a reliable, auditable, and cost‑contained solution. Variant B adds unnecessary complexity (AKS, custom ACL engine, tool routing) without improving core compliance.

---

## 4. Trade‑off Axes – Scored (High / Medium / Low)

| Axis | Variant A | Variant B | Notes                                                                                               |
|------|-----------|-----------|-----------------------------------------------------------------------------------------------------|
| **Time‑to‑first‑user** (weeks) | **Medium** (6‑8) | **Low** (12+) | AKS setup, Helm charts, and custom ACL engine takes weeks to do                                     |
| **Permission fidelity** | **High** | **High** | Both can achieve full ACL enforcement; A uses native indexer, B uses custom engine (more risk of bugs) |
| **Data residency** (EU only) | **High** | **High** | Both can be deployed entirely in EU region                                                          |
| **Cost per active user/month** (assume 10 queries/user/day) | **Medium** (~€0.30‑0.50) | **High** (~€0.80‑1.20) | AKS control plane + API Management + Redis + larger compute footprint                               |
| **Extensibility** (add SAP, Snowflake, etc.) | **Medium** (code in App Service) | **High** (tool routing engine + API gateway) | Variant B designed for plug‑in tools, but not required now                                          |
| **Vendor lock‑in** | **High** | **Medium** | A depends on Azure AI Search, Cosmos DB; B uses AKS which is more portable, but still OpenAI        |
| **Observability** | **High** (App Insights + Log Analytics) | **High** (same + custom telemetry) | Both good                                                                                           |
| **Skills required** | **Medium** (Azure PaaS, .NET/JS, Bot Framework) | **High** (Kubernetes, Helm, API gateway, plus AI/retrieval) | Variant B requires dedicated K8s expertise                                                          |

---

## 5. Accepted Trade‑offs for Winner (Variant A)

- **Higher vendor lock‑in** → acceptable because Ecolab is already using Azure alot
- **Moderate extensibility** → adding a new tool (e.g., SAP) requires code changes and redeployment of App Service, but that is manageable for the expected 1‑2 tools per year.
- **Cost not absolute lowest** → but still well under €5k/month budget.

---

## 6. What Would Change My Mind

Switch from Variant A to Variant B if:

1. **Extensibility becomes critical** – e.g., the business requires integrating >5 external APIs within 6 months.
2. **Legal mandates absolute data sovereignty with no Azure OpenAI** – then Variant C (open‑source LLM) would be required, not B.
3. **The Water BU acquires a dedicated Kubernetes team** – reducing the skills gap.

Switch to a different variant (e.g., self‑hosted) if:
- **Budget cap drops below €3k/month** *and* user volume exceeds 20k – then consider open‑source LLM on spot instances.

---

## 7. Mid‑Sprint Twists – Resilience Check

| Twist | Variant A survives? | Explanation |
|-------|-------------|-------------|
| **Legal: no document content may leave EU tenant** | Yes | Deploy all resources (AI Search, OpenAI, Cosmos DB, App Service) to EU region; no data leaves EU. |
| **Scale from 500 to 40,000 users** | Yes | App Service Premium plan + AI Search autoscale + Cosmos DB autoscale. Breakpoint: OpenAI TPM limits – would need multiple instances or regional distribution. |
| **Add custom tool calling internal API** | Moderate effort | Add new endpoint to App Service and orchestration logic; ~2‑3 sprints. Not a design breaker. |
| **Multi‑document comparison questions** | Yes | Standard RAG with prompt engineering can handle; for complex agentic retrieval, extend orchestrator – still within A's architecture. |
| **Budget cap: €5,000/month all‑in** | Yes | Variant A fits comfortably; Variant B would exceed due to AKS and API Management costs. |

---

## 8. Required Views – Summary (per exercise)

The following ArchiMate views have been created for each variant (files provided separately). For the winning variant (A), the **Layered View SVG** is attached.

### Variant A – Views completed
- `Exercise-3-variant-a_Motivation View.svg`
- `Exercise-3-variant-a_Application Cooperation View.svg`
- `Exercise-3-variant-a_Technology usage view.svg`
- `Exercise-3-variant-a_Layered View.svg`

### Variant B – Views completed
- `Exercise-3-variant-b_Motivation View.svg`
- `Exercise-3-variant-b_Application Cooperation View.svg`
- `Exercise-3-variant-b_Technology usage view.svg`
- `Exercise-3-variant-b_Layered View.svg`

### Winning variant’s Layered View
See attached: `Exercise-3-variant-a_Layered View.svg`

---

## 9. Conclusion

**Variant A** is selected for implementation. It meets all functional and non‑functional requirements, stays within budget, respects EU data residency and SharePoint permissions, and can be delivered faster than the more complex AKS‑based alternative. The trade‑offs are understood and accepted by the project team.
