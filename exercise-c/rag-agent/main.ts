import express, { Request, Response } from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";
import dotenv from "dotenv";
dotenv.config({ path: path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../.env") });
import { AzureOpenAI } from "openai";
import { makeLogger } from "../shared/logger.js";
import { parseEnvelope, type Envelope } from "../shared/envelope.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const PORT = parseInt(process.env.RAG_AGENT_PORT ?? "8081");
const REGISTRY_URL = process.env.REGISTRY_URL ?? "http://localhost:8083";
const AGENT_NAME = "rag-agent";
const HEARTBEAT_INTERVAL_MS = 20_000;
const TOP_K = parseInt(process.env.RAG_TOP_K ?? "5");
const VECTORS_PATH = path.join(__dirname, "vectors.json");

const log = makeLogger(AGENT_NAME);

// ---------------------------------------------------------------------------
// Idempotency dedup
// ---------------------------------------------------------------------------
const seenKeys = new Set<string>();

// ---------------------------------------------------------------------------
// In-memory vector store (loaded from vectors.json built by ingest.ts)
// ---------------------------------------------------------------------------
interface Chunk {
  id: string;
  text: string;
  source: string;
  embedding: number[];
}

let chunks: Chunk[] = [];

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function loadVectors(): void {
  if (!fs.existsSync(VECTORS_PATH)) {
    log.warn("vectors.json not found — run: npx tsx exercise-c/rag-agent/ingest.ts", {
      correlation_id: "none",
    });
    return;
  }
  chunks = JSON.parse(fs.readFileSync(VECTORS_PATH, "utf8")) as Chunk[];
  log.info(`loaded ${chunks.length} chunks from vectors.json`, { correlation_id: "none" });
}

function search(queryEmbedding: number[], k: number): Chunk[] {
  return chunks
    .map((c) => ({ chunk: c, score: cosineSimilarity(queryEmbedding, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((r) => r.chunk);
}

// ---------------------------------------------------------------------------
// Azure OpenAI helpers
// ---------------------------------------------------------------------------
function makeClient() {
  return new AzureOpenAI({
    apiKey: process.env.AZURE_OPENAI_API_KEY ?? "",
    apiVersion: process.env.API_VERSION ?? "2024-12-01-preview",
    endpoint: process.env.AZURE_OPENAI_ENDPOINT ?? "",
  });
}

async function embed(text: string): Promise<number[]> {
  const resp = await makeClient().embeddings.create({
    model: process.env.AZURE_OPENAI_EMBEDDING ?? "text-embedding-3-small",
    input: text,
  });
  return resp.data[0].embedding;
}

async function answerWithLLM(query: string, context: string, correlationId: string): Promise<string> {
  const systemPrompt =
    "You are the Tech Radar Concierge. Answer questions about the technology radar " +
    "using ONLY the context provided. Be concise. If the context does not contain the " +
    "answer, say so explicitly.";
  const userPrompt = `Context:\n${context}\n\nQuestion: ${query}`;

  log.info("llm call", {
    correlation_id: correlationId,
    capability: "answer-from-corpus",
    prompt_preview: userPrompt.slice(0, 200),
  });

  const resp = await makeClient().chat.completions.create({
    model: process.env.AZURE_OPENAI_DEPLOYMENT ?? "gpt-4o-mini",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    temperature: 0.3,
    max_completion_tokens: 400,
  });

  const answer = resp.choices[0].message.content?.trim() ?? "";
  log.info("llm response", { correlation_id: correlationId, answer_preview: answer.slice(0, 100) });
  return answer;
}

// ---------------------------------------------------------------------------
// Registry helpers
// ---------------------------------------------------------------------------
async function register(): Promise<void> {
  try {
    const resp = await fetch(`${REGISTRY_URL}/register`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        name: AGENT_NAME,
        capabilities: ["answer-from-corpus"],
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
// App
// ---------------------------------------------------------------------------
const app = express();
app.use(express.json());

app.post("/invoke", async (req: Request, res: Response) => {
  let envelope: Envelope;
  try {
    envelope = parseEnvelope(req.body);
  } catch (e) {
    res.status(400).json({ error: "invalid envelope", detail: String(e) });
    return;
  }

  const { correlation_id, idempotency_key, capability, payload } = envelope;
  log.info("invoke received", { correlation_id, causation_id: envelope.causation_id, capability });

  if (capability !== "answer-from-corpus") {
    res.status(400).json({ error: `unknown capability: ${capability}` });
    return;
  }

  if (seenKeys.has(idempotency_key)) {
    log.info("idempotency deduplicated", { correlation_id, idempotency_key });
    res.json({
      correlation_id,
      causation_id: correlation_id,
      idempotency_key: randomUUID(),
      sender: AGENT_NAME,
      recipient: envelope.sender,
      capability,
      payload: { answer: "(deduplicated — same idempotency_key)", sources_available: false },
      timestamp: new Date().toISOString(),
    });
    return;
  }
  seenKeys.add(idempotency_key);

  const query = payload["query"] as string | undefined;
  if (!query) {
    res.status(400).json({ error: "payload.query is required" });
    return;
  }

  try {
    let answer: string;
    if (chunks.length === 0) {
      answer = "Corpus not available — run ingest.ts first.";
      log.warn("corpus empty", { correlation_id });
    } else {
      const queryVec = await embed(query);
      const topChunks = search(queryVec, TOP_K);
      const context = topChunks
        .map((c, i) => `[${c.source} | chunk ${i}]\n${c.text}`)
        .join("\n\n---\n\n");
      answer = await answerWithLLM(query, context, correlation_id);
    }

    const response: Envelope = {
      correlation_id,
      causation_id: correlation_id,
      idempotency_key: randomUUID(),
      sender: AGENT_NAME,
      recipient: envelope.sender,
      capability,
      payload: { answer, sources_available: chunks.length > 0 },
      timestamp: new Date().toISOString(),
    };

    log.info("invoke complete", { correlation_id, answer_length: answer.length });
    res.json(response);
  } catch (e) {
    log.error("invoke failed", { correlation_id, error: String(e) });
    res.status(500).json({ error: "internal error", detail: String(e) });
  }
});

app.get("/health", (_req: Request, res: Response) => {
  res.json({
    status: "ok",
    agent: AGENT_NAME,
    capabilities: ["answer-from-corpus"],
    corpus_loaded: chunks.length > 0,
    chunk_count: chunks.length,
  });
});

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------
loadVectors();
await register();
const heartbeatTimer = startHeartbeat();

const server = app.listen(PORT, () => {
  log.info(`rag-agent listening on :${PORT}`, { correlation_id: "none" });
});

process.on("SIGTERM", async () => {
  clearInterval(heartbeatTimer);
  await deregister();
  server.close(() => process.exit(0));
});
process.on("SIGINT", async () => {
  clearInterval(heartbeatTimer);
  await deregister();
  server.close(() => process.exit(0));
});
