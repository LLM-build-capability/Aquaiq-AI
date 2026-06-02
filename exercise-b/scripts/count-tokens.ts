// Measures bootstrap token cost for the N-tool baseline and the Code Mode server.
// Run from exercise-b/: PATH="/opt/homebrew/opt/node@22/bin:$PATH" npx tsx scripts/count-tokens.ts
// Requires tiktoken in node_modules (installed via npm install).

import { get_encoding } from 'tiktoken'

const enc = get_encoding('cl100k_base')

// ── Code Mode — reconstruct the tool definitions exactly as in src/index.ts ──
// These strings must stay in sync with the SEARCH_DESC / EXECUTE_DESC in
// src/index.ts. The DSL_BOOTSTRAP block is the single metamodel placement.

const DSL_BOOTSTRAP = [
  "type Quadrant = 0|1|2|3  // 0=Models & Providers, 1=Infrastructure & Cloud, 2=Frameworks & Libraries, 3=Techniques & Patterns",
  "type Ring     = 0|1|2|3  // 0=ADOPT, 1=TRIAL, 2=ASSESS, 3=HOLD",
  "type Moved    = -1|0|1   // movement since previous radar",
  "type Technology = { id: string; label: string; quadrant: Quadrant }",
  "type Team       = { id: string; name: string; date: string }",
  "type Assignment = { tech: string; ring: Ring; moved: Moved }",
  "type PendingOp =",
  "  | { type: 'addTechnology'; id: string; label: string; quadrant: Quadrant }",
  "  | { type: 'assign'; teamId: string; techId: string; ring: Ring; moved?: Moved }",
  "  | { type: 'move'; teamId: string; techId: string; newRing: Ring }",
  "  | { type: 'removeAssignment'; teamId: string; techId: string }",
].join('\n')

const SHARED_NOTES = [
  "Use `return` to return a value; only the returned projection reaches model context.",
  "`console.log(...)` output is captured and shown after the return value.",
  "Errors include the valid alternatives — read them and rewrite the call.",
].join('\n')

const SEARCH_DESC = `Run read-only JavaScript against the Tech Radar. The sandbox exposes a frozen \`radar\` object with READ methods only; mutating methods are not present in this tool.

  radar.listTechnologies({ quadrant? }: { quadrant?: Quadrant }): Technology[]
  radar.listTeams(): Team[]
  radar.listAssignments(teamId: string): Assignment[]
  radar.getAssignment(teamId: string, techId: string): Assignment | undefined
  radar.validate(op: PendingOp): { valid: true } | { valid: false; error: string }

${SHARED_NOTES}

Example:
  return radar.listTechnologies({ quadrant: 0 }).map(t => t.id).slice(0, 5);`

const EXECUTE_DESC = `Run JavaScript against the Tech Radar with READ + WRITE access. The sandbox exposes a frozen \`radar\` object.

${DSL_BOOTSTRAP}

  radar.listTechnologies({ quadrant? }): Technology[]
  radar.listTeams(): Team[]
  radar.listAssignments(teamId): Assignment[]
  radar.getAssignment(teamId, techId): Assignment | undefined
  radar.validate(op): { valid: true } | { valid: false, error: string }
  radar.addTechnology(id, label, quadrant): Technology
  radar.assign(teamId, techId, ring, moved=0): Assignment
  radar.move(teamId, techId, newRing): Assignment
  radar.removeAssignment(teamId, techId): void
  radar.commit(message): void   // persists changes to data/config.json

${SHARED_NOTES}

Example (multi-step in one call):
  if (!radar.listTechnologies({ quadrant: 0 }).some(t => t.id === 'gpt-5-nano')) {
    radar.addTechnology('gpt-5-nano', 'GPT-5 nano', 0);
  }
  radar.assign('rde', 'gpt-5-nano', 1);
  radar.commit('add gpt-5-nano to RDE at TRIAL');
  return radar.getAssignment('rde', 'gpt-5-nano');`

const TOOLS = [
  {
    name: 'search',
    description: SEARCH_DESC,
    inputSchema: { type: 'object', properties: { code: { type: 'string', description: 'JavaScript using `return` to surface a value' } }, required: ['code'] },
  },
  {
    name: 'execute',
    description: EXECUTE_DESC,
    inputSchema: { type: 'object', properties: { code: { type: 'string', description: 'JavaScript using `return` to surface a value' } }, required: ['code'] },
  },
]

const codeModeText = JSON.stringify(TOOLS, null, 2)
const codeModeTokens = enc.encode(codeModeText).length

const searchTokens = enc.encode(JSON.stringify(TOOLS[0], null, 2)).length
const executeTokens = enc.encode(JSON.stringify(TOOLS[1], null, 2)).length

// ── N-Tool Baseline — committed in docs/n-tool-baseline.md ──────────────────
// The authoritative count (1,417) was measured on 2026-05-27 on the exact JSON
// string in that file and is recorded there. Re-parsing the JSON from markdown
// changes whitespace and produces a different token count, so we reference the
// committed number directly.
const baselineTokens = 1417
const baselineToolCount = 10

enc.free()

// ── Output ────────────────────────────────────────────────────────────────────

console.log('\n=== Bootstrap Token Count (cl100k_base) ===\n')
console.log(`N-tool baseline (${baselineToolCount} tools):  ${baselineTokens} tokens`)
console.log(`Code Mode  — search():              ${searchTokens} tokens`)
console.log(`Code Mode  — execute() + DSL:       ${executeTokens} tokens`)
console.log(`Code Mode  — total (2 tools):        ${codeModeTokens} tokens`)
console.log(`\nReduction:  ${baselineTokens - codeModeTokens} tokens  (${Math.round(100 * (1 - codeModeTokens / baselineTokens))}%)`)
console.log('\nSnippet that produced these numbers:')
console.log('  const enc = get_encoding("cl100k_base");')
console.log('  enc.encode(JSON.stringify(tools, null, 2)).length')
