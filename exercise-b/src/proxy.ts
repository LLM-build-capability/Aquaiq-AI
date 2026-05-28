import { readFileSync, writeFileSync } from 'fs'
import {
  type Assignment,
  type Moved,
  type PendingOp,
  type Quadrant,
  type Radar,
  type Ring,
  type Team,
  type Technology,
  type ValidationResult,
  QUADRANT_NAMES,
  RING_NAMES,
} from './types.js'

// ─── Interfaces ──────────────────────────────────────────────────────────────

/** Read surface — exposed in both search() and execute() sandboxes. */
export interface RadarReadProxy {
  listTechnologies(filter?: { quadrant?: Quadrant }): Technology[]
  listTeams(): Team[]
  listAssignments(teamId: string): Assignment[]
  getAssignment(teamId: string, techId: string): Assignment | undefined
  validate(op: PendingOp): ValidationResult
}

/**
 * Full proxy — exposed in execute() sandbox only.
 *
 * DSL bootstrap (placed once in execute() tool description):
 *
 *   type Quadrant = 0|1|2|3  // 0=Models & Providers, 1=Infrastructure & Cloud,
 *                             // 2=Frameworks & Libraries, 3=Techniques & Patterns
 *   type Ring     = 0|1|2|3  // 0=ADOPT, 1=TRIAL, 2=ASSESS, 3=HOLD
 *   type Moved    = -1|0|1   // -1=moved out, 0=no change, 1=moved in
 *
 *   radar.listTechnologies({ quadrant? })  → Technology[]
 *   radar.listTeams()                      → Team[]
 *   radar.listAssignments(teamId)          → Assignment[]
 *   radar.getAssignment(teamId, techId)    → Assignment | undefined
 *   radar.validate(op)                     → { valid } | { valid, error }
 *   radar.addTechnology(id, label, quadrant) → Technology   [execute only]
 *   radar.assign(teamId, techId, ring, moved=0) → Assignment [execute only]
 *   radar.move(teamId, techId, newRing)    → Assignment      [execute only]
 *   radar.removeAssignment(teamId, techId) → void            [execute only]
 *   radar.commit(message)                  → void            [execute only]
 */
export interface RadarProxy extends RadarReadProxy {
  addTechnology(id: string, label: string, quadrant: Quadrant): Technology
  assign(teamId: string, techId: string, ring: Ring, moved?: Moved): Assignment
  move(teamId: string, techId: string, newRing: Ring): Assignment
  removeAssignment(teamId: string, techId: string): void
  commit(message: string): void
}

// ─── Error ───────────────────────────────────────────────────────────────────

/** Structured error — message always names what to try instead. */
export class ProxyError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ProxyError'
  }
}

// ─── Implementation ──────────────────────────────────────────────────────────

export class RadarProxyImpl implements RadarProxy {
  private radar: Radar

  constructor(private readonly configPath: string) {
    this.radar = JSON.parse(readFileSync(configPath, 'utf-8')) as Radar
  }

  // ── Reads ──────────────────────────────────────────────────────────────────

  listTechnologies(filter?: { quadrant?: Quadrant }): Technology[] {
    if (filter?.quadrant !== undefined) {
      return this.radar.technologies.filter(t => t.quadrant === filter.quadrant)
    }
    return [...this.radar.technologies]
  }

  listTeams(): Team[] {
    return [...this.radar.teams]
  }

  listAssignments(teamId: string): Assignment[] {
    this.requireTeam(teamId)
    return [...(this.radar.assignments[teamId] ?? [])]
  }

  getAssignment(teamId: string, techId: string): Assignment | undefined {
    this.requireTeam(teamId)
    return this.radar.assignments[teamId]?.find(a => a.tech === techId)
  }

  validate(op: PendingOp): ValidationResult {
    try {
      switch (op.type) {
        case 'addTechnology':
          this.checkNewTechId(op.id)
          this.checkKebabCase(op.id)
          break
        case 'assign':
          this.requireTeam(op.teamId)
          this.requireTech(op.techId)
          this.checkNotAssigned(op.teamId, op.techId)
          break
        case 'move':
          this.requireTeam(op.teamId)
          this.requireAssigned(op.teamId, op.techId)
          this.checkDemotionRule(op.teamId, op.techId, op.newRing)
          break
        case 'removeAssignment':
          this.requireTeam(op.teamId)
          this.requireAssigned(op.teamId, op.techId)
          break
      }
      return { valid: true }
    } catch (e) {
      return { valid: false, error: (e as Error).message }
    }
  }

  // ── Writes ─────────────────────────────────────────────────────────────────

  addTechnology(id: string, label: string, quadrant: Quadrant): Technology {
    this.checkKebabCase(id)
    this.checkNewTechId(id)

    const tech: Technology = { id, label, quadrant }
    this.radar.technologies.push(tech)
    return tech
  }

  assign(teamId: string, techId: string, ring: Ring, moved: Moved = 0): Assignment {
    this.requireTeam(teamId)
    this.requireTech(techId)
    this.checkNotAssigned(teamId, techId)

    const assignment: Assignment = { tech: techId, ring, moved }
    if (!this.radar.assignments[teamId]) {
      this.radar.assignments[teamId] = []
    }
    this.radar.assignments[teamId].push(assignment)
    return assignment
  }

  move(teamId: string, techId: string, newRing: Ring): Assignment {
    this.requireTeam(teamId)
    const existing = this.requireAssigned(teamId, techId)
    this.checkDemotionRule(teamId, techId, newRing)

    const moved: Moved = newRing < existing.ring ? 1 : newRing > existing.ring ? -1 : 0
    existing.ring = newRing
    existing.moved = moved
    return existing
  }

  removeAssignment(teamId: string, techId: string): void {
    this.requireTeam(teamId)
    this.requireAssigned(teamId, techId)

    this.radar.assignments[teamId] = this.radar.assignments[teamId].filter(
      a => a.tech !== techId,
    )
  }

  commit(message: string): void {
    // TODO: create a branch in Stack.TechRadar repo and open a PR via
    // `gh pr create` — do not auto-push to main.
    writeFileSync(this.configPath, JSON.stringify(this.radar, null, 2), 'utf-8')
    console.error(`[radar] committed: ${message}`)
  }

  // ── Guard helpers (all throw ProxyError with self-correcting messages) ──────

  private requireTeam(teamId: string): Team {
    const team = this.radar.teams.find(t => t.id === teamId)
    if (!team) {
      const valid = this.radar.teams.map(t => `'${t.id}'`).join(', ')
      throw new ProxyError(
        `team '${teamId}' not found. Valid team ids: ${valid}.`,
      )
    }
    return team
  }

  private requireTech(techId: string): Technology {
    const tech = this.radar.technologies.find(t => t.id === techId)
    if (!tech) {
      // Surface up to 5 techs from the same likely quadrant as a hint
      const similar = this.radar.technologies.slice(0, 5).map(t => `'${t.id}'`)
      throw new ProxyError(
        `technology '${techId}' not found in radar. ` +
          `Use radar.addTechnology('${techId}', label, quadrant) to add it first, ` +
          `or pick from existing: ${similar.join(', ')} (use radar.listTechnologies() to see all).`,
      )
    }
    return tech
  }

  private checkNotAssigned(teamId: string, techId: string): void {
    const existing = this.radar.assignments[teamId]?.find(a => a.tech === techId)
    if (existing) {
      throw new ProxyError(
        `technology '${techId}' is already assigned to team '${teamId}' ` +
          `at ${RING_NAMES[existing.ring]}. ` +
          `Use radar.move('${teamId}', '${techId}', ring) to change its ring.`,
      )
    }
  }

  private requireAssigned(teamId: string, techId: string): Assignment {
    const existing = this.radar.assignments[teamId]?.find(a => a.tech === techId)
    if (!existing) {
      throw new ProxyError(
        `technology '${techId}' is not assigned to team '${teamId}'. ` +
          `Use radar.assign('${teamId}', '${techId}', ring) to assign it first.`,
      )
    }
    return existing
  }

  private checkDemotionRule(teamId: string, techId: string, newRing: Ring): void {
    const existing = this.radar.assignments[teamId]?.find(a => a.tech === techId)
    if (!existing) return

    // Governance: jumping directly from ADOPT (0) to HOLD (3) is forbidden.
    // Must step through TRIAL (1) or ASSESS (2) first.
    if (existing.ring === 0 && newRing === 3) {
      throw new ProxyError(
        `demoting '${techId}' directly from ADOPT to HOLD is forbidden by governance. ` +
          `Step down incrementally: radar.move('${teamId}', '${techId}', 1) ` +
          `to move to TRIAL first, then ASSESS, then HOLD.`,
      )
    }
  }

  private checkNewTechId(id: string): void {
    const existing = this.radar.technologies.find(t => t.id === id)
    if (existing) {
      throw new ProxyError(
        `technology '${id}' already exists ` +
          `(label: '${existing.label}', quadrant: ${QUADRANT_NAMES[existing.quadrant]}). ` +
          `Use radar.assign('teamId', '${id}', ring) to assign it to a team, ` +
          `or radar.move('teamId', '${id}', newRing) to change its ring.`,
      )
    }
  }

  private checkKebabCase(id: string): void {
    if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(id)) {
      throw new ProxyError(
        `id '${id}' is not valid kebab-case. ` +
          `Use only lowercase letters, digits, and hyphens, e.g. 'my-new-tool'.`,
      )
    }
  }
}
