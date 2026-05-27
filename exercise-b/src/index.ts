import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

import { RadarProxyImpl } from './proxy.js'
import { runInSandbox } from './sandbox.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const CONFIG_PATH =
  process.env.RADAR_CONFIG_PATH ?? join(__dirname, '..', 'data', 'config.json')

// One proxy instance per server lifetime. State accumulates across execute()
// calls until the model calls radar.commit(message), which writes the file.
const proxy = new RadarProxyImpl(CONFIG_PATH)

// ─── DSL bootstrap (placed once in the execute() tool description) ───────────
// This is the *single* place the metamodel appears across both tool schemas.
// Trimmed for tokens — pretrained models already know `string`/`number`.

const DSL_BOOTSTRAP = `
type Quadrant = 0|1|2|3  // 0=Models & Providers, 1=Infrastructure & Cloud, 2=Frameworks & Libraries, 3=Techniques & Patterns
type Ring     = 0|1|2|3  // 0=ADOPT, 1=TRIAL, 2=ASSESS, 3=HOLD
type Moved    = -1|0|1   // movement since previous radar
type Technology = { id: string; label: string; quadrant: Quadrant }
type Team       = { id: string; name: string; date: string }
type Assignment = { tech: string; ring: Ring; moved: Moved }
type PendingOp =
  | { type: 'addTechnology'; id: string; label: string; quadrant: Quadrant }
  | { type: 'assign'; teamId: string; techId: string; ring: Ring; moved?: Moved }
  | { type: 'move'; teamId: string; techId: string; newRing: Ring }
  | { type: 'removeAssignment'; teamId: string; techId: string }`.trim()

const SHARED_NOTES = `
Use \`return\` to return a value; only the returned projection reaches model context.
\`console.log(...)\` output is captured and shown after the return value.
Errors include the valid alternatives — read them and rewrite the call.`.trim()

const SEARCH_DESC = `
Run read-only JavaScript against the Tech Radar. The sandbox exposes a frozen \`radar\` object with READ methods only; mutating methods are not present in this tool.

  radar.listTechnologies({ quadrant? }: { quadrant?: Quadrant }): Technology[]
  radar.listTeams(): Team[]
  radar.listAssignments(teamId: string): Assignment[]
  radar.getAssignment(teamId: string, techId: string): Assignment | undefined
  radar.validate(op: PendingOp): { valid: true } | { valid: false; error: string }

${SHARED_NOTES}

Example:
  return radar.listTechnologies({ quadrant: 0 }).map(t => t.id).slice(0, 5);`.trim()

const EXECUTE_DESC = `
Run JavaScript against the Tech Radar with READ + WRITE access. The sandbox exposes a frozen \`radar\` object.

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
  return radar.getAssignment('rde', 'gpt-5-nano');`.trim()

const TOOLS = [
  {
    name: 'search',
    description: SEARCH_DESC,
    inputSchema: {
      type: 'object' as const,
      properties: {
        code: {
          type: 'string',
          description: 'JavaScript using `return` to surface a value',
        },
      },
      required: ['code'],
    },
  },
  {
    name: 'execute',
    description: EXECUTE_DESC,
    inputSchema: {
      type: 'object' as const,
      properties: {
        code: {
          type: 'string',
          description: 'JavaScript using `return` to surface a value',
        },
      },
      required: ['code'],
    },
  },
]

// ─── Server wiring ───────────────────────────────────────────────────────────

const server = new Server(
  { name: 'tech-radar-mcp', version: '0.1.0' },
  { capabilities: { tools: {} } },
)

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }))

server.setRequestHandler(CallToolRequestSchema, async req => {
  const { name, arguments: args } = req.params
  const code = (args as { code?: string } | undefined)?.code ?? ''

  if (name !== 'search' && name !== 'execute') {
    return {
      content: [
        {
          type: 'text',
          text: `Unknown tool '${name}'. Use 'search' (read-only) or 'execute' (read+write).`,
        },
      ],
      isError: true,
    }
  }

  const mode = name === 'execute' ? 'write' : 'read'
  const r = await runInSandbox(code, proxy, mode)

  if (!r.ok) {
    const text = `${r.error?.name ?? 'Error'}: ${r.error?.message ?? 'unknown failure'}`
    const logs = r.logs.length ? `\n--- console ---\n${r.logs.join('\n')}` : ''
    return {
      content: [{ type: 'text', text: text + logs }],
      isError: true,
    }
  }

  const body =
    r.result === undefined
      ? '(no return value — use `return` to surface a result)'
      : safeJson(r.result)
  const logs = r.logs.length ? `\n--- console ---\n${r.logs.join('\n')}` : ''
  return { content: [{ type: 'text', text: body + logs }] }
})

function safeJson(x: unknown): string {
  try {
    return JSON.stringify(x, null, 2)
  } catch {
    return String(x)
  }
}

const transport = new StdioServerTransport()
await server.connect(transport)
// Errors during stdio go to stderr; stdout is reserved for MCP framing.
process.stderr.write(
  `[tech-radar-mcp] connected — config: ${CONFIG_PATH}\n`,
)
