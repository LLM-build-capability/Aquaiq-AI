import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import request from "supertest";
import { app, agents } from "../registry/main.js";

const VALID_AGENT = {
  name: "test-agent",
  capabilities: ["answer-from-corpus"],
  endpoint: "http://localhost:9999",
  health_url: "http://localhost:9999/health",
};

beforeEach(() => {
  agents.clear();
});

describe("POST /register", () => {
  it("registers a new agent and returns 201", async () => {
    const res = await request(app).post("/register").send(VALID_AGENT);
    assert.equal(res.status, 201);
    assert.equal(res.body.status, "registered");
    assert.equal(res.body.name, "test-agent");
    assert.equal(agents.has("test-agent"), true);
  });

  it("returns 400 when name is missing", async () => {
    const { name: _, ...body } = VALID_AGENT;
    const res = await request(app).post("/register").send(body);
    assert.equal(res.status, 400);
    assert.match(res.body.error, /required/i);
  });

  it("returns 400 when capabilities is missing", async () => {
    const { capabilities: _, ...body } = VALID_AGENT;
    const res = await request(app).post("/register").send(body);
    assert.equal(res.status, 400);
  });

  it("overwrites an existing registration (re-register)", async () => {
    await request(app).post("/register").send(VALID_AGENT);
    const updated = { ...VALID_AGENT, endpoint: "http://localhost:9998" };
    const res = await request(app).post("/register").send(updated);
    assert.equal(res.status, 201);
    assert.equal(agents.get("test-agent")!.endpoint, "http://localhost:9998");
  });
});

describe("DELETE /deregister/:name", () => {
  it("removes a registered agent", async () => {
    await request(app).post("/register").send(VALID_AGENT);
    const res = await request(app).delete("/deregister/test-agent");
    assert.equal(res.status, 200);
    assert.equal(res.body.status, "deregistered");
    assert.equal(agents.has("test-agent"), false);
  });

  it("returns 404 for unknown agent", async () => {
    const res = await request(app).delete("/deregister/ghost");
    assert.equal(res.status, 404);
    assert.match(res.body.error, /not found/i);
  });
});

describe("POST /heartbeat/:name", () => {
  it("updates last_heartbeat and returns ok", async () => {
    await request(app).post("/register").send(VALID_AGENT);
    const before = agents.get("test-agent")!.last_heartbeat;
    await new Promise((r) => setTimeout(r, 5));
    const res = await request(app).post("/heartbeat/test-agent");
    assert.equal(res.status, 200);
    assert.equal(res.body.status, "ok");
    assert.ok(agents.get("test-agent")!.last_heartbeat >= before);
  });

  it("returns 404 for unregistered agent", async () => {
    const res = await request(app).post("/heartbeat/nobody");
    assert.equal(res.status, 404);
    assert.match(res.body.error, /register first/i);
  });
});

describe("GET /agents", () => {
  it("returns all agents when no capability filter", async () => {
    await request(app).post("/register").send(VALID_AGENT);
    await request(app).post("/register").send({
      ...VALID_AGENT,
      name: "agent-b",
      capabilities: ["propose-radar-change"],
    });
    const res = await request(app).get("/agents");
    assert.equal(res.status, 200);
    assert.equal(res.body.length, 2);
  });

  it("filters agents by capability", async () => {
    await request(app).post("/register").send(VALID_AGENT);
    await request(app).post("/register").send({
      ...VALID_AGENT,
      name: "agent-b",
      capabilities: ["propose-radar-change"],
    });
    const res = await request(app).get("/agents?capability=answer-from-corpus");
    assert.equal(res.status, 200);
    assert.equal(res.body.length, 1);
    assert.equal(res.body[0].name, "test-agent");
  });

  it("returns empty array when no match", async () => {
    await request(app).post("/register").send(VALID_AGENT);
    const res = await request(app).get("/agents?capability=no-such-cap");
    assert.equal(res.status, 200);
    assert.deepEqual(res.body, []);
  });

  it("response objects do not expose last_heartbeat", async () => {
    await request(app).post("/register").send(VALID_AGENT);
    const res = await request(app).get("/agents");
    assert.equal("last_heartbeat" in res.body[0], false);
  });
});

describe("GET /health", () => {
  it("returns status ok with agent_count", async () => {
    await request(app).post("/register").send(VALID_AGENT);
    const res = await request(app).get("/health");
    assert.equal(res.status, 200);
    assert.equal(res.body.status, "ok");
    assert.equal(res.body.agent_count, 1);
    assert.equal(res.body.agents[0].name, "test-agent");
  });

  it("returns agent_count 0 when registry is empty", async () => {
    const res = await request(app).get("/health");
    assert.equal(res.status, 200);
    assert.equal(res.body.agent_count, 0);
  });

  it("includes last_heartbeat_age_ms in health response", async () => {
    await request(app).post("/register").send(VALID_AGENT);
    const res = await request(app).get("/health");
    assert.ok(typeof res.body.agents[0].last_heartbeat_age_ms === "number");
    assert.ok(res.body.agents[0].last_heartbeat_age_ms >= 0);
  });
});
