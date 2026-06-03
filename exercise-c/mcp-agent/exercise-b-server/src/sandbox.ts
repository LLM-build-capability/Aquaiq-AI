import ivm from 'isolated-vm'
import type { RadarProxy, RadarReadProxy } from './proxy.js'

// ─── Method partitioning ─────────────────────────────────────────────────────
// READ_METHODS go into both sandbox modes; WRITE_METHODS are execute()-only.
// This is the structural enforcement of the "query path vs execute path" split.

const READ_METHODS = [
  'listTechnologies',
  'listTeams',
  'listAssignments',
  'getAssignment',
  'validate',
] as const

const WRITE_METHODS = [
  'addTechnology',
  'assign',
  'move',
  'removeAssignment',
  'commit',
] as const

export type SandboxMode = 'read' | 'write'

export interface SandboxOptions {
  timeoutMs?: number
  memoryMb?: number
}

export interface SandboxResult {
  ok: boolean
  result?: unknown
  error?: { name: string; message: string }
  logs: string[]
}

export async function runInSandbox(
  code: string,
  proxy: RadarProxy | RadarReadProxy,
  mode: SandboxMode,
  opts: SandboxOptions = {},
): Promise<SandboxResult> {
  const timeoutMs = opts.timeoutMs ?? 5_000
  const memoryMb = opts.memoryMb ?? 128

  const logs: string[] = []
  let ok = false
  let result: unknown
  let error: { name: string; message: string } | undefined

  const isolate = new ivm.Isolate({ memoryLimit: memoryMb })
  try {
    const context = await isolate.createContext()
    const jail = context.global

    // Methods exposed to the model in this mode.
    // Cast to a shape that lets us index by string; the read sandbox only ever
    // gets a RadarReadProxy and READ_METHODS only references its read surface.
    const methods: readonly string[] =
      mode === 'write'
        ? [...READ_METHODS, ...WRITE_METHODS]
        : [...READ_METHODS]

    const proxyRecord = proxy as unknown as Record<string, (...a: unknown[]) => unknown>

    // Each method is bridged into the isolate as an ivm.Reference.
    // The host fn copies args back out (already deep-copied across the boundary
    // by `arguments: { copy: true }`) and forwards them to the proxy.
    // Errors thrown by the proxy are surfaced to the isolate as plain Errors —
    // their message already carries the self-correcting alternative text.
    for (const name of methods) {
      const bound = proxyRecord[name].bind(proxy)
      await jail.set(
        `__radar_${name}`,
        new ivm.Reference((...args: unknown[]) => {
          try {
            return bound(...args)
          } catch (e) {
            const err = e as Error
            const wrapped = new Error(err.message ?? String(err))
            wrapped.name = err.name ?? 'Error'
            throw wrapped
          }
        }),
      )
    }

    // Console capture — model code can `console.log(...)`; messages are
    // collected on the host and returned alongside the result.
    await jail.set(
      '__hostLog',
      new ivm.Reference((msg: unknown) => {
        logs.push(typeof msg === 'string' ? msg : safeStringify(msg))
      }),
    )

    // Bootstrap: turn the raw References into a friendly `radar` object and
    // a `console` shim. The References are captured in closure variables and
    // then deleted from globalThis, so model code cannot reach the raw handles.
    const methodList = methods.map(m => `'${m}'`).join(', ')
    const bootstrap = `
      ;(function () {
        const callOpts = { arguments: { copy: true }, result: { copy: true } };
        const logOpts  = { arguments: { copy: true } };

        const refs = {};
        for (const m of [${methodList}]) {
          refs[m] = globalThis['__radar_' + m];
          delete globalThis['__radar_' + m];
        }
        const hostLog = globalThis.__hostLog;
        delete globalThis.__hostLog;

        const radar = {};
        for (const m of [${methodList}]) {
          radar[m] = function (...args) {
            return refs[m].applySync(undefined, args, callOpts);
          };
        }
        Object.freeze(radar);
        globalThis.radar = radar;

        const fmt = (x) => {
          if (typeof x === 'string') return x;
          try { return JSON.stringify(x); } catch (_) { return String(x); }
        };
        globalThis.console = Object.freeze({
          log:   (...a) => hostLog.applySync(undefined, [a.map(fmt).join(' ')], logOpts),
          error: (...a) => hostLog.applySync(undefined, [a.map(fmt).join(' ')], logOpts),
        });
      })();
    `

    await context.eval(bootstrap)

    // User code is wrapped in an IIFE so a top-level `return` inside `code`
    // returns the value from the script. Document this in the tool description.
    const wrapped = `(function () {\n${code}\n})()`
    const script = await isolate.compileScript(wrapped)
    result = await script.run(context, { timeout: timeoutMs, copy: true })
    ok = true
  } catch (e) {
    const err = e as Error
    error = {
      name: err.name ?? 'Error',
      message: err.message ?? String(err),
    }
  } finally {
    isolate.dispose()
  }

  return { ok, result, error, logs }
}

function safeStringify(x: unknown): string {
  try {
    return JSON.stringify(x)
  } catch {
    return String(x)
  }
}
