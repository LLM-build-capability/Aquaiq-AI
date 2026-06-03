// Measures bootstrap token cost for the N-tool baseline and the Code Mode server.
// Run from exercise-b/: PATH="/opt/homebrew/opt/node@22/bin:$PATH" npx tsx scripts/count-tokens.ts

import { get_encoding } from 'tiktoken'
import { TOOLS } from '../src/index.js'

const enc = get_encoding('cl100k_base')

const searchTokens  = enc.encode(JSON.stringify(TOOLS[0], null, 2)).length
const executeTokens = enc.encode(JSON.stringify(TOOLS[1], null, 2)).length
const totalTokens   = enc.encode(JSON.stringify(TOOLS,    null, 2)).length

// The authoritative N-tool baseline count (1,417) was measured on 2026-05-27
// against the exact JSON in docs/n-tool-baseline.md.
const baselineTokens    = 1417
const baselineToolCount = 10

enc.free()

console.log('\n=== Bootstrap Token Count (cl100k_base) ===\n')
console.log(`N-tool baseline (${baselineToolCount} tools):  ${baselineTokens} tokens`)
console.log(`Code Mode  — search():              ${searchTokens} tokens`)
console.log(`Code Mode  — execute() + DSL:       ${executeTokens} tokens`)
console.log(`Code Mode  — total (2 tools):        ${totalTokens} tokens`)
console.log(`\nReduction:  ${baselineTokens - totalTokens} tokens  (${Math.round(100 * (1 - totalTokens / baselineTokens))}%)`)
