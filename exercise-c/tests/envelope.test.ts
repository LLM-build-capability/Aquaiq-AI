import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { parseEnvelope, EnvelopeSchema } from "../shared/envelope.js";

const VALID: Record<string, unknown> = {
  correlation_id: "550e8400-e29b-41d4-a716-446655440000",
  causation_id: "550e8400-e29b-41d4-a716-446655440001",
  idempotency_key: "550e8400-e29b-41d4-a716-446655440002",
  sender: "orchestrator",
  recipient: "rag-agent",
  capability: "answer-from-corpus",
  payload: { query: "What is water quality?" },
  timestamp: "2024-01-15T12:00:00.000Z",
};

describe("EnvelopeSchema", () => {
  it("parses a valid envelope", () => {
    const env = parseEnvelope(VALID);
    assert.equal(env.sender, "orchestrator");
    assert.equal(env.capability, "answer-from-corpus");
    assert.deepEqual(env.payload, { query: "What is water quality?" });
  });

  it("rejects missing correlation_id", () => {
    const bad = { ...VALID };
    delete bad["correlation_id"];
    assert.throws(() => parseEnvelope(bad), /ZodError|required/i);
  });

  it("rejects non-UUID correlation_id", () => {
    const bad = { ...VALID, correlation_id: "not-a-uuid" };
    assert.throws(() => parseEnvelope(bad), /ZodError|uuid/i);
  });

  it("rejects empty sender", () => {
    const bad = { ...VALID, sender: "" };
    assert.throws(() => parseEnvelope(bad), /at least 1 character/i);
  });

  it("rejects empty capability", () => {
    const bad = { ...VALID, capability: "" };
    assert.throws(() => parseEnvelope(bad), /at least 1 character/i);
  });

  it("rejects non-datetime timestamp", () => {
    const bad = { ...VALID, timestamp: "not-a-date" };
    assert.throws(() => parseEnvelope(bad), /datetime|Invalid/i);
  });

  it("accepts any payload shape", () => {
    const env = parseEnvelope({ ...VALID, payload: { nested: { x: 1 }, arr: [1, 2] } });
    assert.deepEqual(env.payload["arr"], [1, 2]);
  });

  it("rejects null payload", () => {
    const bad = { ...VALID, payload: null };
    assert.throws(() => parseEnvelope(bad), /Expected object|invalid_type/i);
  });

  it("round-trips through zod safeParse", () => {
    const result = EnvelopeSchema.safeParse(VALID);
    assert.equal(result.success, true);
  });
});
