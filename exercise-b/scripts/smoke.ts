// End-to-end smoke test for the Code-Mode MCP server.
// Bypasses the MCP protocol layer and exercises the proxy + sandbox directly,
// so failures are easy to read and the output is reusable as a worked trace.
//
// Each scenario runs against a fresh copy of data/config.json in a tmp dir, so
// repeated runs are idempotent and never touch the committed snapshot.

import { copyFileSync, mkdtempSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { RadarProxyImpl } from '../src/proxy.js'
import { runInSandbox } from '../src/sandbox.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const SOURCE_CONFIG = join(__dirname, '..', 'data', 'config.json')

function freshProxy(): { proxy: RadarProxyImpl; configPath: string } {
  const tmp = mkdtempSync(join(tmpdir(), 'radar-smoke-'))
  const cfg = join(tmp, 'config.json')
  copyFileSync(SOURCE_CONFIG, cfg)
  return { proxy: new RadarProxyImpl(cfg), configPath: cfg }
}

function banner(title: string): void {
  console.log('\n' + '═'.repeat(72))
  console.log(`  ${title}`)
  console.log('═'.repeat(72))
}

function show(label: string, code: string, r: Awaited<ReturnType<typeof runInSandbox>>): void {
  console.log(`\n— ${label} —`)
  console.log('CODE:')
  for (const line of code.trim().split('\n')) console.log('  ' + line)
  console.log('RESULT:')
  console.log('  ok=' + r.ok)
  if (r.ok) console.log('  result=' + JSON.stringify(r.result))
  if (!r.ok) console.log(`  error.name=${r.error?.name}\n  error.message=${r.error?.message}`)
  if (r.logs.length) console.log('  logs=' + JSON.stringify(r.logs))
}

async function main(): Promise<void> {
  // ── Scenario 1: search() returns a small projection ─────────────────────────
  banner('1. search() — projection only what the model needs')
  {
    const { proxy } = freshProxy()
    const code = `return radar.listTechnologies({ quadrant: 0 }).slice(0, 3).map(t => t.id);`
    const r = await runInSandbox(code, proxy, 'read')
    show('quadrant 0, first 3 ids', code, r)
  }

  // ── Scenario 2: search() blocks accidental writes structurally ──────────────
  banner('2. search() — write methods are not present in the read sandbox')
  {
    const { proxy } = freshProxy()
    const code = `return typeof radar.commit;`
    const r = await runInSandbox(code, proxy, 'read')
    show('typeof radar.commit', code, r)
  }

  // ── Scenario 3: execute() runs a multi-step workflow in one call ────────────
  banner("3. execute() — add + assign + commit in a single round-trip")
  {
    const { proxy, configPath } = freshProxy()
    const code = `
      const techId = 'gpt-5-nano-smoketest';
      if (!radar.listTechnologies({ quadrant: 0 }).some(t => t.id === techId)) {
        radar.addTechnology(techId, 'GPT-5 nano (smoke)', 0);
      }
      radar.assign('rde', techId, 1);
      radar.commit('smoke: add ' + techId + ' to RDE at TRIAL');
      return radar.getAssignment('rde', techId);`
    const r = await runInSandbox(code, proxy, 'write')
    show('add + assign + commit', code, r)

    const persisted = JSON.parse(readFileSync(configPath, 'utf-8')) as {
      assignments: Record<string, Array<{ tech: string }>>
    }
    const found = persisted.assignments.rde?.some(
      a => a.tech === 'gpt-5-nano-smoketest',
    )
    console.log(`  persisted to file=${found}`)
  }

  // ── Scenario 4: deliberate misuse — reference a non-existent tech ───────────
  banner('4. execute() — misuse: assign tech that is not in the radar')
  {
    const { proxy } = freshProxy()
    const code = `radar.assign('rde', 'this-tech-does-not-exist', 1); return 'unreached';`
    const r = await runInSandbox(code, proxy, 'write')
    show('assign unknown tech', code, r)
  }

  // ── Scenario 5: deliberate misuse — ADOPT → HOLD demotion ───────────────────
  banner('5. execute() — misuse: forbidden ADOPT → HOLD demotion')
  {
    const { proxy } = freshProxy()
    const code = `
      const adoptTech = radar.listAssignments('rde').find(a => a.ring === 0)?.tech;
      radar.move('rde', adoptTech, 3);  // ADOPT -> HOLD, not allowed
      return 'unreached';`
    const r = await runInSandbox(code, proxy, 'write')
    show('move 0 -> 3 directly', code, r)
  }

  // ── Scenario 6: corrected retry — step through TRIAL ────────────────────────
  banner('6. execute() — corrected retry: ADOPT → TRIAL (step 1 of 2)')
  {
    const { proxy } = freshProxy()
    const code = `
      const adoptTech = radar.listAssignments('rde').find(a => a.ring === 0)?.tech;
      radar.move('rde', adoptTech, 1);  // ADOPT -> TRIAL is allowed
      radar.commit('demote ' + adoptTech + ' to TRIAL');
      return radar.getAssignment('rde', adoptTech);`
    const r = await runInSandbox(code, proxy, 'write')
    show('move 0 -> 1 then commit', code, r)
  }

  // ── Scenario 7: deliberate misuse — kebab-case validation ───────────────────
  banner('7. execute() — misuse: invalid id (not kebab-case)')
  {
    const { proxy } = freshProxy()
    const code = `radar.addTechnology('Not Kebab Case', 'Bad id', 0); return 'unreached';`
    const r = await runInSandbox(code, proxy, 'write')
    show('addTechnology with bad id', code, r)
  }

  // ── Scenario 8: timeout ─────────────────────────────────────────────────────
  banner('8. sandbox — runaway loop is killed by timeout')
  {
    const { proxy } = freshProxy()
    const code = `while (true) {} return 'unreached';`
    const r = await runInSandbox(code, proxy, 'read', { timeoutMs: 250 })
    show('infinite loop, 250ms budget', code, r)
  }

  console.log('\n' + '═'.repeat(72))
  console.log('  smoke test complete')
  console.log('═'.repeat(72))
}

main().catch(e => {
  console.error('smoke test crashed:', e)
  process.exit(1)
})
