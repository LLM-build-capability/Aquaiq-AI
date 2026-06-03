import express, { Request, Response } from "express";
import { makeLogger } from "../shared/logger.js";

const PORT = 8083;
const HEARTBEAT_TTL_MS = 40_000; // evict after 2 missed 20s heartbeats

const log = makeLogger("registry");

interface AgentRecord {
  name: string;
  capabilities: string[];
  endpoint: string;
  health_url: string;
  registered_at: string;
  last_heartbeat: number; // Date.now()
}

// In-memory store — restart is a clean slate; agents re-register on next heartbeat
const agents = new Map<string, AgentRecord>();

// Evict agents that missed 2 heartbeat cycles
function evictStale() {
  const now = Date.now();
  for (const [name, agent] of agents) {
    if (now - agent.last_heartbeat > HEARTBEAT_TTL_MS) {
      agents.delete(name);
      log.warn("agent evicted — missed heartbeat TTL", { agent: name, correlation_id: "none" });
    }
  }
}

setInterval(evictStale, 10_000);

const app = express();
app.use(express.json());

// POST /register
app.post("/register", (req: Request, res: Response) => {
  const { name, capabilities, endpoint, health_url } = req.body as {
    name?: string;
    capabilities?: string[];
    endpoint?: string;
    health_url?: string;
  };

  if (!name || !capabilities || !endpoint || !health_url) {
    res.status(400).json({ error: "name, capabilities, endpoint, health_url are required" });
    return;
  }

  const record: AgentRecord = {
    name,
    capabilities,
    endpoint,
    health_url,
    registered_at: new Date().toISOString(),
    last_heartbeat: Date.now(),
  };

  agents.set(name, record);
  log.info("agent registered", { agent: name, capabilities: capabilities.join(","), correlation_id: "none" });
  res.status(201).json({ status: "registered", name });
});

// DELETE /deregister/:name
app.delete("/deregister/:name", (req: Request, res: Response) => {
  const { name } = req.params;
  if (!agents.has(name)) {
    res.status(404).json({ error: `agent '${name}' not found` });
    return;
  }
  agents.delete(name);
  log.info("agent deregistered", { agent: name, correlation_id: "none" });
  res.json({ status: "deregistered", name });
});

// POST /heartbeat/:name
app.post("/heartbeat/:name", (req: Request, res: Response) => {
  const { name } = req.params;
  const record = agents.get(name);
  if (!record) {
    res.status(404).json({ error: `agent '${name}' not found — register first` });
    return;
  }
  record.last_heartbeat = Date.now();
  res.json({ status: "ok", name });
});

// GET /agents?capability=answer-from-corpus
app.get("/agents", (req: Request, res: Response) => {
  const capability = req.query["capability"] as string | undefined;
  evictStale();

  const results = [...agents.values()].filter((a) =>
    capability ? a.capabilities.includes(capability) : true
  );

  res.json(results.map(({ name, capabilities, endpoint, health_url, registered_at }) => ({
    name,
    capabilities,
    endpoint,
    health_url,
    registered_at,
  })));
});

// GET /health — registry health + per-agent reachability summary
app.get("/health", (_req: Request, res: Response) => {
  evictStale();
  res.json({
    status: "ok",
    agent_count: agents.size,
    agents: [...agents.values()].map(({ name, capabilities, endpoint, last_heartbeat }) => ({
      name,
      capabilities,
      endpoint,
      last_heartbeat_age_ms: Date.now() - last_heartbeat,
    })),
  });
});

app.listen(PORT, () => {
  log.info(`registry listening on :${PORT}`, { correlation_id: "none" });
});
