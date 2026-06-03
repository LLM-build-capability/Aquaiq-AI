import { z } from "zod";

export const EnvelopeSchema = z.object({
  correlation_id: z.string().uuid(),
  causation_id: z.string().uuid(),
  idempotency_key: z.string().uuid(),
  sender: z.string().min(1),
  recipient: z.string().min(1),
  capability: z.string().min(1),
  payload: z.record(z.unknown()),
  timestamp: z.string().datetime(),
});

export type Envelope = z.infer<typeof EnvelopeSchema>;

export function parseEnvelope(raw: unknown): Envelope {
  return EnvelopeSchema.parse(raw);
}
