import express, { Request, Response } from "express";
import Database from "better-sqlite3";
import path from "path";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";
import { AzureOpenAI } from "openai";
import dotenv from "dotenv";
dotenv.config({ path: path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../.env") });
import { makeLogger } from "../shared/logger.js";
import { parseEnvelope, type Envelope } from "../shared/envelope.js";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const PORT = parseInt(process.env.ORCHESTRATOR_PORT ?? "8080");
const REGISTRY_URL = process.env.REGISTRY_URL ?? "http://localhost:8083";
const AGENT_NAME = "orchestrator";
const HEARTBEAT_INTERVAL_MS = 20_000;
const CALL_TIMEOUT_MS = 15_000;
const MAX_RETRIES = 3;
const DEAD_LETTER_PATH = path.join(__dirname, "..", "dead-letter.jsonl");

const log = makeLogger(AGENT_NAME);

// ---------------------------------------------------------------------------
// SQLite state persistence
// ---------------------------------------------------------------------------
const db = new Database(path.join(__dirname, "state.db"));

db.exec(`
  CREATE TABLE IF NOT EXISTS workflow_steps (
    id           TEXT PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    step         TEXT NOT NULL,
    agent        TEXT,
    capability   TEXT,
    status       TEXT NOT NULL,
    detail       TEXT,
    created_at   TEXT NOT NULL
  );
`);

type StepStatus = "STARTED" | "DISPATCHED" | "COMPLETE" | "TIMED_OUT" | "FAILED" | "PARTIAL";

function persistStep(
  correlationId: string,
  step: string,
  status: StepStatus,
  agent?: string,
  capability?: string,
  detail?: string
): void {
  db.prepare(
    `INSERT OR REPLACE INTO workflow_steps
     (id, correlation_id, step, agent, capability, status, detail, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    randomUUID(),
    correlationId,
    step,
    agent ?? null,
    capability ?? null,
    status,
    detail ?? null,
    new Date().toISOString()
  );
}

// ---------------------------------------------------------------------------
// Dead-letter
// ---------------------------------------------------------------------------
function deadLetter(envelope: Envelope, reason: string): void {
  const entry = JSON.stringify({ ...envelope, dead_letter_reason: reason, dead_letter_at: new Date().toISOString() });
  fs.appendFileSync(DEAD_LETTER_PATH, entry + "\n");
  log.error("dead-lettered envelope", { correlation_id: envelope.correlation_id, reason });
}

// ---------------------------------------------------------------------------
// Registry lookup
// ---------------------------------------------------------------------------
async function lookupAgent(capability: string, correlationId: string): Promise<string | null> {
  const url = `${REGISTRY_URL}/agents?capability=${encodeURIComponent(capability)}`;
  log.info("registry lookup", { correlation_id: correlationId, capability, url });
  try {
    const resp = await fetch(url);
    if (!resp.ok) {
      log.warn(`registry returned ${resp.status}`, { correlation_id: correlationId });
      return null;
    }
    const agents = (await resp.json()) as Array<{ name: string; endpoint: string }>;
    if (agents.length === 0) {
      log.warn(`no agent found for capability '${capability}'`, { correlation_id: correlationId });
      return null;
    }
    log.info(`resolved agent '${agents[0].name}' for capability '${capability}'`, { correlation_id: correlationId });
    return agents[0].endpoint;
  } catch (e) {
    log.error(`registry lookup failed: ${(e as Error).message}`, { correlation_id: correlationId });
    return null;
  }
}

// ---------------------------------------------------------------------------
// Dispatch with timeout + retries
// ---------------------------------------------------------------------------
async function dispatch(
  endpoint: string,
  envelope: Envelope,
  attempt = 1
): Promise<Envelope> {
  const { correlation_id } = envelope;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CALL_TIMEOUT_MS);

  log.info(`dispatch attempt ${attempt}/${MAX_RETRIES}`, {
    correlation_id,
    causation_id: envelope.causation_id,
    capability: envelope.capability,
    endpoint,
  });

  try {
    const resp = await fetch(`${endpoint}/invoke`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(envelope),
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!resp.ok) {
      throw new Error(`agent returned HTTP ${resp.status}`);
    }

    const body = await resp.json();
    return parseEnvelope(body);
  } catch (e) {
    clearTimeout(timer);
    const isTimeout = (e as Error).name === "AbortError";
    const msg = isTimeout ? "timed out" : (e as Error).message;

    log.warn(`dispatch failed (attempt ${attempt}): ${msg}`, { correlation_id });

    if (attempt < MAX_RETRIES) {
      const backoffMs = Math.pow(2, attempt - 1) * 1000; // 1s, 2s, 4s
      log.info(`retrying in ${backoffMs}ms`, { correlation_id });
      await new Promise((r) => setTimeout(r, backoffMs));
      // Reuse same idempotency_key — agent will dedup on retry
      return dispatch(endpoint, envelope, attempt + 1);
    }

    if (isTimeout) {
      persistStep(correlation_id, "dispatch", "TIMED_OUT", envelope.recipient, envelope.capability);
    }
    throw e;
  }
}

// ---------------------------------------------------------------------------
// Intent classification via LLM
// ---------------------------------------------------------------------------
type Intent = "explain" | "change";

async function classifyIntent(query: string, correlationId: string): Promise<Intent> {
  const systemPrompt = `You are an intent classifier for a Tech Radar assistant.
Classify the user's query as either:
- "explain" — user wants to understand why a technology is in a certain ring, its history, rationale, or context.
- "change" — user wants to move, promote, demote, or otherwise mutate a technology's ring on the radar.

Reply with exactly one word: either "explain" or "change". Nothing else.`;

  const userPrompt = `Query: "${query}"`;

  log.info("intent classification prompt", {
    correlation_id: correlationId,
    system_prompt: systemPrompt,
    user_prompt: userPrompt,
  });

  try {
    const client = new AzureOpenAI({
      apiKey: process.env.AZURE_OPENAI_API_KEY ?? "",
      apiVersion: process.env.API_VERSION ?? "2024-12-01-preview",
      endpoint: process.env.AZURE_OPENAI_ENDPOINT ?? "",
    });

    const resp = await client.chat.completions.create({
      model: process.env.AZURE_OPENAI_DEPLOYMENT ?? "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0,
      max_completion_tokens: 5,
    });

    const raw = resp.choices[0].message.content?.trim().toLowerCase() ?? "explain";
    const intent: Intent = raw.includes("change") ? "change" : "explain";

    log.info("intent classified", { correlation_id: correlationId, raw_response: raw, intent });
    return intent;
  } catch (e) {
    // Fall back to keyword heuristic if LLM call fails
    log.warn(`LLM classify failed, using keyword fallback: ${(e as Error).message}`, {
      correlation_id: correlationId,
    });
    // Only match change keywords when used as verbs (e.g. "move X to ADOPT"), not when asking about rings
    const lower = query.toLowerCase();
    const changeKeywords = ["move", "promote", "demote", "change", "update", "set ring", "put in", "place in"];
    return changeKeywords.some((kw) => lower.includes(kw)) ? "change" : "explain";
  }
}

// ---------------------------------------------------------------------------
// Registry self-registration
// ---------------------------------------------------------------------------
async function register(): Promise<void> {
  try {
    const resp = await fetch(`${REGISTRY_URL}/register`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        name: AGENT_NAME,
        capabilities: ["orchestrate"],
        endpoint: `http://localhost:${PORT}`,
        health_url: `http://localhost:${PORT}/health`,
      }),
    });
    if (resp.ok) log.info("registered with registry", { correlation_id: "none" });
    else log.warn(`registry registration returned ${resp.status}`, { correlation_id: "none" });
  } catch (e) {
    log.warn(`registry registration failed: ${(e as Error).message}`, { correlation_id: "none" });
  }
}

async function deregister(): Promise<void> {
  try {
    await fetch(`${REGISTRY_URL}/deregister/${AGENT_NAME}`, { method: "DELETE" });
    log.info("deregistered", { correlation_id: "none" });
  } catch {}
}

function startHeartbeat(): NodeJS.Timeout {
  return setInterval(async () => {
    try {
      await fetch(`${REGISTRY_URL}/heartbeat/${AGENT_NAME}`, { method: "POST" });
    } catch {}
  }, HEARTBEAT_INTERVAL_MS);
}

// ---------------------------------------------------------------------------
// Request model
// ---------------------------------------------------------------------------
interface UserRequest {
  query: string;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
const app = express();
app.use(express.json());

// POST /request — user entry point
app.post("/request", async (req: Request, res: Response) => {
  const { query } = req.body as UserRequest;
  if (!query) {
    res.status(400).json({ error: "query is required" });
    return;
  }

  const correlationId = randomUUID();
  const idempotencyKey = randomUUID();

  log.info("request received", { correlation_id: correlationId, query });
  persistStep(correlationId, "receive", "STARTED");

  // 1. Classify intent
  const intent = await classifyIntent(query, correlationId);
  persistStep(correlationId, "classify", "COMPLETE", undefined, undefined, intent);

  // 2. Determine capability
  const capability = intent === "explain" ? "answer-from-corpus" : "propose-radar-change";
  log.info("routing decision", { correlation_id: correlationId, intent, capability });

  // 3. Registry lookup
  const agentEndpoint = await lookupAgent(capability, correlationId);
  if (!agentEndpoint) {
    const msg = `No agent available for capability '${capability}'`;
    persistStep(correlationId, "lookup", "FAILED", undefined, capability, msg);
    res.status(503).json({ error: msg, correlation_id: correlationId });
    return;
  }
  persistStep(correlationId, "lookup", "COMPLETE", undefined, capability, agentEndpoint);

  // 4. Build outbound envelope
  const outbound: Envelope = {
    correlation_id: correlationId,
    causation_id: correlationId,
    idempotency_key: idempotencyKey,
    sender: AGENT_NAME,
    recipient: capability === "answer-from-corpus" ? "rag-agent" : "mcp-agent",
    capability,
    payload: { query },
    timestamp: new Date().toISOString(),
  };

  persistStep(correlationId, "dispatch", "DISPATCHED", outbound.recipient, capability);

  // 5. Dispatch with retries
  try {
    const responseEnvelope = await dispatch(agentEndpoint, outbound);
    persistStep(correlationId, "dispatch", "COMPLETE", outbound.recipient, capability);

    const answer = (responseEnvelope.payload["answer"] as string | undefined) ?? JSON.stringify(responseEnvelope.payload);
    log.info("workflow complete", { correlation_id: correlationId, intent, answer_preview: answer.slice(0, 100) });

    res.json({
      correlation_id: correlationId,
      intent,
      capability,
      answer,
    });
  } catch (e) {
    const errMsg = (e as Error).message;
    const isTimeout = (e as Error).name === "AbortError" || errMsg.toLowerCase().includes("timed out");

    if (isTimeout) {
      log.error("workflow timed out", { correlation_id: correlationId });
      persistStep(correlationId, "dispatch", "TIMED_OUT", outbound.recipient, capability, errMsg);

      // Check if the radar was mutated before timeout (relevant for MCP calls)
      if (capability === "propose-radar-change") {
        persistStep(correlationId, "dispatch", "PARTIAL", outbound.recipient, capability,
          "MCP call timed out — radar state unknown, check MCP agent logs");
      }

      // Poison message: if this idempotency_key has failed too many times, dead-letter it
      const failCount = (db.prepare(
        `SELECT COUNT(*) as n FROM workflow_steps WHERE correlation_id = ? AND status IN ('TIMED_OUT','FAILED')`
      ).get(correlationId) as { n: number }).n;

      if (failCount >= MAX_RETRIES) {
        deadLetter(outbound, `exceeded max retries (${MAX_RETRIES})`);
      }

      res.status(504).json({ error: "agent timed out", correlation_id: correlationId });
    } else {
      log.error(`workflow failed: ${errMsg}`, { correlation_id: correlationId });
      persistStep(correlationId, "dispatch", "FAILED", outbound.recipient, capability, errMsg);
      res.status(500).json({ error: errMsg, correlation_id: correlationId });
    }
  }
});

// GET /workflow/:id — inspect workflow state from SQLite
app.get("/workflow/:id", (req: Request, res: Response) => {
  const steps = db.prepare(
    `SELECT * FROM workflow_steps WHERE correlation_id = ? ORDER BY created_at`
  ).all(req.params.id);
  if (!steps.length) {
    res.status(404).json({ error: "workflow not found" });
    return;
  }
  res.json(steps);
});

// GET /health
app.get("/health", (_req: Request, res: Response) => {
  const stepCount = (db.prepare("SELECT COUNT(*) as n FROM workflow_steps").get() as { n: number }).n;
  res.json({ status: "ok", agent: AGENT_NAME, capabilities: ["orchestrate"], total_workflow_steps: stepCount });
});

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------
await register();
const heartbeatTimer = startHeartbeat();

const server = app.listen(PORT, () => {
  log.info(`orchestrator listening on :${PORT}`, { correlation_id: "none" });
});

process.on("SIGTERM", async () => {
  clearInterval(heartbeatTimer);
  await deregister();
  db.close();
  server.close(() => process.exit(0));
});
process.on("SIGINT", async () => {
  clearInterval(heartbeatTimer);
  await deregister();
  db.close();
  server.close(() => process.exit(0));
});
