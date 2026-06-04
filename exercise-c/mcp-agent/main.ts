import express, { Request, Response } from "express";
import { spawn, type ChildProcess } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";
import { AzureOpenAI } from "openai";
import dotenv from "dotenv";
dotenv.config({ path: path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../.env") });
import { makeLogger } from "../shared/logger.js";
import { parseEnvelope, type Envelope } from "../shared/envelope.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const PORT = parseInt(process.env.MCP_AGENT_PORT ?? "8082");
const REGISTRY_URL = process.env.REGISTRY_URL ?? "http://localhost:8083";
const AGENT_NAME = "mcp-agent";
const HEARTBEAT_INTERVAL_MS = 20_000;

// Node 22 required — isolated-vm won't build against Node 26
const NODE_BIN = process.env.NODE_BIN ?? "/opt/homebrew/opt/node@22/bin/node";
const EXERCISE_B_DIR = path.join(__dirname, "exercise-b-server");
const EXERCISE_B_ENTRY = path.join(EXERCISE_B_DIR, "src", "index.ts");
const RADAR_CONFIG_PATH = path.join(EXERCISE_B_DIR, "data", "config.json");

const log = makeLogger(AGENT_NAME);

// ---------------------------------------------------------------------------
// Idempotency dedup
// ---------------------------------------------------------------------------
const seenKeys = new Set<string>();

// ---------------------------------------------------------------------------
// MCP subprocess — one persistent child process per mcp-agent lifetime
// ---------------------------------------------------------------------------
interface McpRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params?: unknown;
}

interface McpResponse {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: { code: number; message: string };
}

class McpClient {
  private proc: ChildProcess | null = null;
  private pending = new Map<number, { resolve: (r: McpResponse) => void; reject: (e: Error) => void }>();
  private nextId = 1;
  private buf = "";
  private initialized = false;

  async start(): Promise<void> {
    this.proc = spawn(
      NODE_BIN,
      ["--import", "tsx/esm", EXERCISE_B_ENTRY],
      {
        env: {
          ...process.env,
          RADAR_CONFIG_PATH,
          PATH: `/opt/homebrew/opt/node@22/bin:${process.env.PATH ?? ""}`,
        },
        cwd: EXERCISE_B_DIR,
        stdio: ["pipe", "pipe", "pipe"],
      }
    );

    if (!this.proc.stdout || !this.proc.stdin) {
      throw new Error("Failed to open stdio pipes to MCP subprocess");
    }

    this.proc.stdout.setEncoding("utf8");
    this.proc.stdout.on("data", (chunk: string) => {
      this.buf += chunk;
      let nl: number;
      while ((nl = this.buf.indexOf("\n")) !== -1) {
        const line = this.buf.slice(0, nl).trim();
        this.buf = this.buf.slice(nl + 1);
        if (!line) continue;
        try {
          const msg = JSON.parse(line) as McpResponse;
          const cb = this.pending.get(msg.id);
          if (cb) {
            this.pending.delete(msg.id);
            cb.resolve(msg);
          }
        } catch {
          // non-JSON lines (e.g. stderr leaked to stdout) — ignore
        }
      }
    });

    this.proc.stderr?.on("data", (d: Buffer) => {
      // exercise-b writes startup message to stderr — surface at DEBUG
      log.debug("mcp-subprocess stderr", { correlation_id: "none", detail: d.toString().trim() });
    });

    this.proc.on("exit", (code) => {
      log.warn(`mcp subprocess exited with code ${code ?? "null"}`, { correlation_id: "none" });
      this.initialized = false;
      this.proc = null;
    });

    await this.initialize();
  }

  private async initialize(): Promise<void> {
    // MCP initialize handshake — required before any tools/call
    const initResp = await this.send("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "mcp-agent-a2a", version: "1.0.0" },
    });

    if (initResp.error) {
      throw new Error(`MCP initialize failed: ${initResp.error.message}`);
    }

    // Send initialized notification (fire-and-forget, no response expected)
    this.write({ jsonrpc: "2.0", method: "notifications/initialized", params: {} } as unknown as McpRequest);
    this.initialized = true;
    log.info("mcp subprocess initialized", { correlation_id: "none" });
  }

  async callTool(name: string, args: Record<string, unknown>, correlationId: string): Promise<string> {
    if (!this.initialized) {
      throw new Error("MCP client not initialized — call start() first");
    }

    log.info("mcp tools/call", { correlation_id: correlationId, tool: name, code_preview: String(args["code"] ?? "").slice(0, 80) });

    const resp = await this.send("tools/call", { name, arguments: args });

    if (resp.error) {
      throw new Error(`MCP error ${resp.error.code}: ${resp.error.message}`);
    }

    // result is { content: Array<{ type: "text", text: string }>, isError?: boolean }
    const result = resp.result as { content: Array<{ type: string; text: string }>; isError?: boolean };
    const text = result.content.map((c) => c.text).join("\n");

    if (result.isError) {
      throw new Error(`MCP tool error: ${text}`);
    }

    log.info("mcp tools/call complete", { correlation_id: correlationId, result_preview: text.slice(0, 100) });
    return text;
  }

  private send(method: string, params: unknown): Promise<McpResponse> {
    const id = this.nextId++;
    const req: McpRequest = { jsonrpc: "2.0", id, method, params };

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP request '${method}' timed out`));
      }, 10_000);

      this.pending.set(id, {
        resolve: (r) => { clearTimeout(timer); resolve(r); },
        reject: (e) => { clearTimeout(timer); reject(e); },
      });

      this.write(req);
    });
  }

  private write(msg: unknown): void {
    if (!this.proc?.stdin?.writable) {
      throw new Error("MCP subprocess stdin not writable");
    }
    this.proc.stdin.write(JSON.stringify(msg) + "\n");
  }

  stop(): void {
    this.proc?.kill("SIGTERM");
    this.proc = null;
  }
}

const mcp = new McpClient();

// ---------------------------------------------------------------------------
// LLM: convert natural-language change query to execute() JavaScript
// ---------------------------------------------------------------------------
function makeLlmClient() {
  return new AzureOpenAI({
    apiKey: process.env.AZURE_OPENAI_API_KEY ?? "",
    apiVersion: process.env.API_VERSION ?? "2024-12-01-preview",
    endpoint: process.env.AZURE_OPENAI_ENDPOINT ?? "",
  });
}

const CODE_GENERATION_SYSTEM = `You are a Tech Radar mutation planner.
Given a natural-language change request, generate a single JavaScript snippet to be
run inside the radar execute() sandbox. The snippet uses a frozen \`radar\` object with
these methods:

  radar.listTechnologies({ quadrant? }): Technology[]
  radar.listTeams(): Team[]
  radar.listAssignments(teamId): Assignment[]
  radar.getAssignment(teamId, techId): Assignment | undefined
  radar.validate(op): { valid: true } | { valid: false, error: string }
  radar.addTechnology(id, label, quadrant): Technology
  radar.assign(teamId, techId, ring, moved=0): Assignment
  radar.move(teamId, techId, newRing): Assignment
  radar.removeAssignment(teamId, techId): void
  radar.commit(message): void

Ring values: 0=ADOPT, 1=TRIAL, 2=ASSESS, 3=HOLD
Quadrant values: 0=Models & Providers, 1=Infrastructure & Cloud, 2=Frameworks & Libraries, 3=Techniques & Patterns
Team id examples: llm-capability-office, internal-functions, commercial, sales-service, supply-chain, rde

Rules:
- Use \`return\` to return a result value.
- Call radar.commit(message) to persist changes.
- Use radar.validate() before mutating if unsure of current state.
- If the technology id is not known, call radar.listTechnologies() to find it.
- Default teamId to "llm-capability-office" when not specified.
- Output ONLY the JavaScript code — no markdown, no explanation, no backticks.`;

async function generateExecuteCode(query: string, correlationId: string): Promise<string> {
  const client = makeLlmClient();

  log.info("generating execute code", {
    correlation_id: correlationId,
    system_prompt: CODE_GENERATION_SYSTEM,
    user_prompt: query,
  });

  const resp = await client.chat.completions.create({
    model: process.env.AZURE_OPENAI_DEPLOYMENT ?? "gpt-4o-mini",
    messages: [
      { role: "system", content: CODE_GENERATION_SYSTEM },
      { role: "user", content: query },
    ],
    temperature: 0,
    max_completion_tokens: 300,
  });

  const code = resp.choices[0].message.content?.trim() ?? "";
  log.info("generated code", { correlation_id: correlationId, code });
  return code;
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
        capabilities: ["propose-radar-change"],
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

  if (capability !== "propose-radar-change") {
    res.status(400).json({ error: `unknown capability: ${capability}` });
    return;
  }

  // Idempotency dedup
  if (seenKeys.has(idempotency_key)) {
    log.info("idempotency deduplicated", { correlation_id, idempotency_key });
    res.json({
      correlation_id,
      causation_id: correlation_id,
      idempotency_key: randomUUID(),
      sender: AGENT_NAME,
      recipient: envelope.sender,
      capability,
      payload: { answer: "(deduplicated — same idempotency_key)", committed: false },
      timestamp: new Date().toISOString(),
    } satisfies Envelope);
    return;
  }
  seenKeys.add(idempotency_key);

  const query = payload["query"] as string | undefined;
  if (!query) {
    res.status(400).json({ error: "payload.query is required" });
    return;
  }

  try {
    // 1. LLM generates JavaScript execute() code from the natural-language query
    const code = await generateExecuteCode(query, correlation_id);

    // 2. Call the MCP server's execute tool over stdio
    const mcpResult = await mcp.callTool("execute", { code }, correlation_id);

    const response: Envelope = {
      correlation_id,
      causation_id: correlation_id,
      idempotency_key: randomUUID(),
      sender: AGENT_NAME,
      recipient: envelope.sender,
      capability,
      payload: {
        answer: `Radar change executed successfully.\n\nGenerated code:\n${code}\n\nResult:\n${mcpResult}`,
        committed: true,
        mcp_result: mcpResult,
        generated_code: code,
      },
      timestamp: new Date().toISOString(),
    };

    log.info("invoke complete", { correlation_id, mcp_result_length: mcpResult.length });
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
    capabilities: ["propose-radar-change"],
    mcp_subprocess_running: true,
  });
});

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------
log.info("starting mcp subprocess...", { correlation_id: "none" });
await mcp.start();

await register();
const heartbeatTimer = startHeartbeat();

const server = app.listen(PORT, () => {
  log.info(`mcp-agent listening on :${PORT}`, { correlation_id: "none" });
});

process.on("SIGTERM", async () => {
  clearInterval(heartbeatTimer);
  mcp.stop();
  await deregister();
  server.close(() => process.exit(0));
});
process.on("SIGINT", async () => {
  clearInterval(heartbeatTimer);
  mcp.stop();
  await deregister();
  server.close(() => process.exit(0));
});
