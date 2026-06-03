/**
 * One-shot ingest — builds vectors.json from markdown files in data/.
 * Run once before starting the RAG agent:
 *   npx tsx exercise-c/rag-agent/ingest.ts
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";
import { AzureOpenAI } from "openai";
import dotenv from "dotenv";
import { fileURLToPath as _ftu } from "url";
dotenv.config({ path: path.resolve(path.dirname(_ftu(import.meta.url)), "../../.env") });

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(__dirname, "data");
const VECTORS_PATH = path.join(__dirname, "vectors.json");
const CHUNK_SIZE = 600;

// ---------------------------------------------------------------------------
// Chunker — split on double-newlines, keep chunks under CHUNK_SIZE chars
// ---------------------------------------------------------------------------
function chunkText(text: string, source: string): Array<{ id: string; text: string; source: string }> {
  const paragraphs = text.split(/\n\n+/).map((p) => p.trim()).filter(Boolean);
  const chunks: Array<{ id: string; text: string; source: string }> = [];
  let current = "";

  for (const para of paragraphs) {
    if (current.length + para.length + 2 > CHUNK_SIZE && current) {
      chunks.push({ id: randomUUID(), text: current.trim(), source });
      current = para;
    } else {
      current = current ? `${current}\n\n${para}` : para;
    }
  }
  if (current) chunks.push({ id: randomUUID(), text: current.trim(), source });
  return chunks;
}

// ---------------------------------------------------------------------------
// Embed
// ---------------------------------------------------------------------------
async function embed(text: string): Promise<number[]> {
  const client = new AzureOpenAI({
    apiKey: process.env.AZURE_OPENAI_API_KEY ?? "",
    apiVersion: process.env.API_VERSION ?? "2024-12-01-preview",
    endpoint: process.env.AZURE_OPENAI_ENDPOINT ?? "",
  });
  const resp = await client.embeddings.create({
    model: process.env.AZURE_OPENAI_EMBEDDING ?? "text-embedding-3-small",
    input: text,
  });
  return resp.data[0].embedding;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
if (fs.existsSync(VECTORS_PATH)) {
  const existing = JSON.parse(fs.readFileSync(VECTORS_PATH, "utf8")) as unknown[];
  console.log(`vectors.json already exists with ${existing.length} chunks — skipping.`);
  console.log("Delete exercise-c/rag-agent/vectors.json and re-run to force re-ingest.");
  process.exit(0);
}

const files = fs.readdirSync(DATA_DIR).filter((f) => f.endsWith(".md"));
const allChunks: Array<{ id: string; text: string; source: string }> = [];

for (const file of files.sort()) {
  const text = fs.readFileSync(path.join(DATA_DIR, file), "utf8");
  const chunks = chunkText(text, file);
  allChunks.push(...chunks);
  console.log(`  ${file}: ${chunks.length} chunks`);
}

console.log(`\nEmbedding ${allChunks.length} chunks via Azure OpenAI...`);

const withEmbeddings = [];
for (let i = 0; i < allChunks.length; i++) {
  const chunk = allChunks[i];
  process.stdout.write(`  [${i + 1}/${allChunks.length}] ${chunk.source} chunk ${i}...\r`);
  const embedding = await embed(chunk.text);
  withEmbeddings.push({ ...chunk, embedding });
}

fs.writeFileSync(VECTORS_PATH, JSON.stringify(withEmbeddings, null, 2));
console.log(`\nDone. ${withEmbeddings.length} chunks saved to vectors.json`);
