# Architecture Decision Document  
## Microservices Architecture for a Distributed Ecolab / Compliance Platform

---

## 1. Problem Context

The target system is an enterprise-scale, multi-region compliance and audit platform migrating away from SAP, Salesforce, and a tangle of legacy SharePoint sites. Two architectural variants were designed and modelled in ArchiMate:

| | Variant B | Variant C |
|---|---|---|
| **Compute model** | AKS (containers) per region | Azure Functions (serverless) per region |
| **Event backbone** | Azure Event Hub (shared) | Azure Event Hubs per region + Schema Registry |
| **Application depth** | Reporting, Compliance, Notification, Audit Ingestion Core (replicated × 3 regions) | Full event-driven model: Audit Ingestion Adapter, Compliance Report Generator, Regional Report Store, SAP CDC Connector, Competitor Bridge, Field Auditor Sync, Legacy SharePoint Crawler, Dead Letter & Retry Handler, Peak Load Autoscaler, Data Sovereignty Policy Enforcer, On-call Dashboard |
| **SAP integration** | SAP Integration Service Node (per region) | SAP CDC Connector application component + SAPDataChangedEvent |
| **Acquisition** | Not modelled | Competitor Bridge (acquisition) component |
| **Migration plan** | None modelled | 7 explicit plateaus (Legacy Tangle → Global Audit Viewer) |
| **Motivation model** | Not modelled | Full: Goals, Constraints, Requirements, Drivers, Principles, Stakeholders |
| **Architecture principles** | Implicit | Event Log is Source of Truth · Schema Per Region · No Central Orchestrator |

---

## 2. Trade-off Analysis

### 2.1 Cognitive Load per Team (Team Topologies)

**Variant B** deploys three services (Reporting, Compliance, Notification) per AKS cluster per region. Each service is independently deployable but the region-level topology means a stream-aligned team must understand AKS cluster management, network policies, and cross-region replication — roughly 9 service instances plus platform concerns.

**Variant C** introduces a *Cognitive Load Limit* constraint explicitly in the motivation layer and maps each bounded context to a single serverless function group. The on-call dashboard and runbook are first-class artefacts. A 4-person on-call team already owns 12 services (per the twist); Variant C surfaces this constraint and designs around it by reducing the blast radius of each function (short-lived, stateless) and providing a Dead Letter & Retry Handler as a shared recovery mechanism.

**Winner: Variant C** — cognitive load is bounded by design.

### 2.2 Deployment Independence

**Variant B** uses a shared Azure API Management gateway serving all three regional AKS clusters. A breaking API contract change requires coordinated rollout across EU, US, and APAC simultaneously.

**Variant C** uses a Customer-Facing API Gateway over a Regional Report Store. Each region's Compliance Report Generator publishes via its own Regional Event Hub; the gateway reads from the Report Store, which is decoupled from the ingestion path. SAP CDC events and AcquisitionDataReadyEvents flow through the same regional hub without coupling the teams that produce them.

**Winner: Variant C** — event-driven decoupling means Team X ships without gating Team Y.

### 2.3 Failure Blast Radius

**Variant B** — if the shared Azure Event Hub fails, all three Compliance Services lose their event feed simultaneously. The hub is listed as a global shared platform service alongside APIM and Azure Monitor.

**Variant C** — a *Failure Blast Radius Boundary* constraint is encoded in the motivation layer. Each region has its own Event Hub. If EU Event Hub degrades, US and APAC ingestion continues. The Dead Letter & Retry Handler captures failed events for later replay, insulating downstream report generation.

**Winner: Variant C** — per-region event isolation + retry handling limits blast radius to a single geography.

### 2.4 Consistency Model

**Variant B** — Compliance Service reads/writes to a regional Azure SQL Database. No explicit consistency guarantee is modelled; it is assumed the Compliance Service enforces strong consistency within a region.

**Variant C** — Two explicit consistency requirements are modelled:
- *Consistency Requirement – Ingestion*: eventual (driven by the Data Sovereignty Constraint; events must not leave the region before being acknowledged locally).
- *Consistency Requirement – Final Report*: strong (Compliance Report Generator writes to Regional Report Store atomically before emitting ComplianceReportReadyEvent).

The principle *Event Log is Source of Truth* makes the Event Store / Audit Log (backed by Azure Blob Storage with Immutable Policy) the authoritative record. This is the correct model for a regulated audit domain.

**Winner: Variant C** — explicit consistency boundaries match the domain requirements.

### 2.5 Operational Cost

**Variant B** — AKS clusters run 24/7 regardless of load. Three clusters × three regions = 9 AKS node-pool costs, plus the fixed APIM, Event Hub, and SQL costs.

**Variant C** — Azure Functions scales to zero between audit submissions. During flu/audit season the Peak Load Autoscaler triggers pre-warm of Function instances before the 100× spike hits. Event Hub Capture buffers events to Blob Storage during peaks rather than dropping them.

**Winner: Variant C** — serverless cost model at baseline, pre-warmed capacity at peaks. Estimated infra at baseline is ~40–60% lower than equivalent AKS.

### 2.6 Latency Budget

**Variant B** — AKS pods are always warm; P50 ingestion latency is ~20 ms within a region. Cross-region API calls route via APIM, adding ~80–120 ms.

**Variant C** — Azure Functions cold-start adds ~200–800 ms for the first call after idle. The Peak Load Autoscaler and pre-warm patterns mitigate this. The Field Auditor Sync Service uses Azure Functions (Mobile Sync) which is always-on to eliminate cold-start for the latency-sensitive mobile path.

For real-time requirements (e.g., field auditor submitting a finding and receiving acknowledgement), Variant B has a slight latency advantage. For async report generation and bulk ingestion, Variant C is acceptable.

**Winner: Variant B** for synchronous, sub-100 ms SLA paths; **Variant C** for async pipelines.

### 2.7 Data Sovereignty

**Variant B** — per-region SQL databases (EU Compliance SQL, US Compliance SQL, APAC Compliance SQL) address the regulator requirement that data stays in geography. Private Endpoints and Azure Virtual Network isolate traffic.

**Variant C** — per-region SQL/PostgreSQL plus Azure Virtual Network per region, Azure Policy enforcement, and a dedicated *Data Sovereignty Policy Enforcer* application component that gates event routing — events tagged with a region flag cannot be forwarded to a hub in another geography. The constraint is also modelled in the motivation layer so it is traceable to the regulatory driver.

**Winner: Variant C** — policy enforcement is an application-layer concern, not just infrastructure topology.

### 2.8 Migration from Today (SAP / SharePoint / Legacy)

**Variant B** — The SAP Integration Service is modelled as a technology Node, with no explicit migration path.

**Variant C** — Seven plateaus with explicit Gaps and Work Packages:

| Plateau | Key deliverable |
|---|---|
| 0 – Legacy Tangle | Baseline: SAP + SharePoint tangle, no new platform |
| 1 – Single Region Pilot | SAP CDC Connector + Event Schema Registry + First Audit Report + On-call Runbook |
| 2 – Regional Rollout | EU/US/APAC event hubs live, regional compliance databases populated |
| 3 – SAP Outage Resilience | Dead Letter & Retry; platform survives 4-hour SAP blackouts |
| 4 – Peak Load Automation | Autoscaler live, Cost Estimator & Budget Tracker baseline |
| 5 – Competitor Integration | Competitor Bridge onboards acquired 500-person firm's data |
| 6 – Global Audit Viewer & On-call Tooling | Cross-region read model + SRE dashboard |

The first deployable increment (Plateau 1) delivers value — a real compliance report — without requiring full regional rollout. The SAP CDC Connector deployment work package directly enables this.

**Winner: Variant C** — migration is planned, incremental, and value-driven.

---

## 3. Recommended Variant

**Variant C is the recommended architecture.**

It addresses all five twists:
1. **Acquisition**: Competitor Bridge onboards in Plateau 5 within 6 months.
2. **Regional data sovereignty**: Data Sovereignty Policy Enforcer + per-region Event Hubs + Azure Policy.
3. **Peak load 100×**: Peak Load Autoscaler + Event Hub Capture + serverless elasticity.
4. **4-person on-call team**: Cognitive Load Limit constraint, Dead Letter & Retry Handler, On-call Dashboard.
5. **SAP outages**: SAP CDC Connector buffers changes; Plateau 3 explicitly targets resilience; 4-hour outages are absorbed.

Variant B remains a valid fallback if the organisation has existing AKS expertise and the synchronous latency SLA is below 50 ms — in which case AKS pods can be introduced selectively alongside the Variant C event backbone.

---

## 4. Architecture Decision Records (ADRs)

### ADR-001 — Serverless Functions over AKS for Ingestion and Processing

**Decision:** Use Azure Functions (serverless) for Audit Ingestion Adapters, Compliance Report Generators, and Dead Letter & Retry Handlers, rather than long-running AKS microservices.

**Rationale:** The ingestion workload is bursty (100× during audit season), and the on-call team is constrained to 4 people. Azure Functions scale to zero at baseline, reducing both cost and the surface area that must be patched. The Peak Load Autoscaler work package ensures pre-warm is in place before known peak windows.

**Known trade-off accepted:** Cold-start latency (200–800 ms) on the first invocation after idle. We mitigate this for the latency-sensitive field auditor mobile path by keeping the Field Auditor Sync Service warm via Azure Functions (Mobile Sync).

**We will know we were wrong when:** P99 ingestion acknowledgement latency consistently exceeds 2 seconds outside peak season despite pre-warm, or when the on-call team spends more time managing Function runtime issues than AKS would have required.

---

### ADR-002 — Per-Region Event Hubs over a Single Global Event Hub

**Decision:** Deploy one Azure Event Hub namespace per region (EU, US, APAC), rather than a single shared global namespace.

**Rationale:** A regulator in Region X requires all customer data to remain in that region (twist 2). A single global hub would require cross-region replication of every event before they can be consumed locally, violating the data locality law. Per-region hubs ensure the AuditSubmittedEvent never leaves its origin geography before being processed and stored.

**Known trade-off accepted:** No global fan-out topology; cross-region analytics require the Global Audit Viewer to federate reads from three regional Report Stores rather than subscribing to a single hub.

**We will know we were wrong when:** Regulators require real-time cross-region audit trail correlation (i.e., the latency of the Global Audit Viewer read model exceeds the regulatory inspection SLA), or when managing three hub namespaces consumes more than 20% of the on-call team's operational budget.

---

### ADR-003 — Event Log as Source of Truth over Database-of-Record

**Decision:** The Event Store / Audit Log (Azure Blob Storage with Immutable Policy) is the authoritative record for all audit submissions. The Regional Report Store and Compliance databases are read models derived from this log.

**Rationale:** Audit data must be tamper-evident (Immutable Audit Trail Requirement). Storing events immutably in Blob Storage satisfies this and provides a natural replay mechanism for rebuilding downstream read models after schema changes or failures. SAP outages are absorbed because the CDC events land in the Event Store before SAP is even queried for confirmation.

**Known trade-off accepted:** Eventual consistency between the Event Store and the Regional Report Store. The Compliance Report Generator will not reflect a submitted audit until the next processing cycle (seconds to low minutes). For the *Final Report* consistency tier, the Report Generator writes atomically to the Regional Report Store before emitting ComplianceReportReadyEvent, providing a strong read guarantee for customers.

**We will know we were wrong when:** A compliance officer requires point-in-time consistent views at sub-second latency and the eventual read model is not fast enough, or when Blob Storage immutability prevents legitimate regulatory corrections that require log amendment.

---

### ADR-004 — SAP Change Data Capture (CDC) over Direct API Polling

**Decision:** Integrate with SAP via a CDC Connector that streams SAPDataChangedEvents into the regional Event Hub, rather than polling SAP's REST or BAPI interface on a schedule.

**Rationale:** SAP experiences two 4-hour outages per year (twist 5). A polling-based integration accumulates a backlog during an outage and then floods the downstream compliance pipeline on restart. The CDC Connector captures the transaction log; events are buffered during outages and replayed in order when connectivity resumes. The Dead Letter & Retry Handler handles any events that fail processing, preventing data loss.

**Known trade-off accepted:** CDC requires a dedicated SAP CDC Connector deployment work package in Plateau 1, and the connector must be maintained by the team that owns the SAP integration boundary. This adds one service to the on-call team's portfolio (currently 12 services → 13).

**We will know we were wrong when:** The SAP system is decommissioned as part of the migration and the CDC Connector becomes unnecessary within 12 months of deployment, making the investment in the connector negative-value.

---

### ADR-005 — Schema Registry per Region over a Global Schema Registry

**Decision:** Apply the *Schema Per Region* principle: each region's Event Hub uses its own Azure Schema Registry namespace, and the Schema Registry Service gates schema evolution for that region's events.

**Rationale:** Regulatory schema requirements diverge between geographies (EU GDPR data minimisation vs US SOX audit fields vs APAC data residency fields). A single global schema would either be a superset (carrying fields irrelevant to each region) or a lowest-common-denominator (dropping region-specific mandatory fields). The RegionSchemaUpdatedEvent allows a schema update in one region to propagate to the Compliance Report Generator without triggering a global schema migration.

**Known trade-off accepted:** Three schema registries to maintain. Introducing a field required globally (e.g., a new audit category mandated by all regulators) requires three coordinated schema updates rather than one.

**We will know we were wrong when:** More than 30% of schema evolution work packages require simultaneous changes to all three regional schemas, at which point the overhead of maintaining three registries outweighs the benefit of regional isolation.

---

### ADR-006 — Competitor Bridge as an Adapter Pattern over Direct Platform Integration

**Decision:** Onboard the acquired 500-person competitor's platform via a dedicated Competitor Bridge application component that translates their events into AcquisitionDataReadyEvents and injects them into the regional Event Hub, rather than integrating their platform directly with our Regional Event Hub consumers.

**Rationale:** The acquisition must be completed within 6 months (twist 1), and the competitor's platform has its own data model and schema. Direct integration would require the Compliance Report Generator team to understand and maintain compatibility with the competitor's schema. The Competitor Bridge is owned by the integration team, translates once at the boundary, and emits a well-typed AcquisitionDataReadyEvent that all existing consumers already handle.

**Known trade-off accepted:** The Competitor Bridge is a translation component that must be updated whenever the competitor's platform schema changes. If the acquisition results in a full platform migration, the bridge becomes technical debt.

**We will know we were wrong when:** The competitor's platform schema changes more than twice per quarter and maintaining the bridge consumes more effort than a full schema migration would have, or when the competitor is fully migrated to our platform within 18 months (making the bridge unnecessary).

---

### ADR-007 — Terraform for Infrastructure as Code over Azure Resource Manager Templates

**Decision:** All Azure infrastructure (Event Hubs, Functions, SQL databases, VNets, API Management, Blob Storage) is provisioned and updated via Terraform, not ARM templates or Bicep.

**Rationale:** Terraform's state model makes it straightforward to manage three regional environments as separate workspaces with shared modules. The multi-region expansion in Plateau 2 requires identical infrastructure to be stamped out in EU, US, and APAC with environment-specific variables. Terraform also supports the Azure Chaos Studio integration required for resilience testing in Plateau 3.

**Known trade-off accepted:** Terraform requires a state backend (Azure Blob Storage) and a CI/CD pipeline that has contributor permissions across all three regions. This adds an infrastructure management service to the on-call team's portfolio.

**We will know we were wrong when:** The organisation standardises on Bicep across all teams and the maintenance cost of two IaC languages (Terraform + Bicep for any legacy resources) exceeds the benefit of Terraform's multi-region workspace model.

---

## 5. Summary of Variant Comparison

| Trade-off axis | Variant B | Variant C | Recommended |
|---|---|---|---|
| Cognitive load | Moderate (AKS ops) | Low (bounded contexts, on-call tooling) | **C** |
| Deployment independence | Coupled via shared APIM | Event-driven, decoupled | **C** |
| Failure blast radius | Global Event Hub → global impact | Per-region hub, retry handler | **C** |
| Consistency model | Implicit strong | Explicit tiered (eventual ingestion / strong report) | **C** |
| Operational cost | Higher (always-on AKS) | Lower at baseline, elastic at peak | **C** |
| Latency (sync paths) | ~20 ms P50 | ~200–800 ms cold start | **B** |
| Data sovereignty | Infra-only (SQL per region) | Infra + policy + app enforcer | **C** |
| Migration readiness | Not planned | 7-plateau incremental plan | **C** |
| SAP resilience | None modelled | CDC + dead letter + retry | **C** |
| Acquisition support | None modelled | Competitor Bridge (Plateau 5) | **C** |

**Overall winner: Variant C.** The single latency advantage of Variant B is addressed by keeping the Field Auditor Sync Service always-warm in Variant C, making the remaining trade-off negligible for the majority of platform interactions.
