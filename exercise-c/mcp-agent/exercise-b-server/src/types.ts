// 4 fixed quadrants
export type Quadrant = 0 | 1 | 2 | 3

// 4 fixed rings
export type Ring = 0 | 1 | 2 | 3

// Movement annotation relative to the previous radar
export type Moved = -1 | 0 | 1

export interface Technology {
  id: string       // kebab-case, unique across the radar
  label: string    // human-readable, ~30 chars
  quadrant: Quadrant
}

export interface Team {
  id: string       // e.g. "llm-capability-office"
  name: string
  date: string     // "YYYY.MM"
}

export interface Assignment {
  tech: string     // FK → Technology.id
  ring: Ring
  moved: Moved
}

export interface Radar {
  date: string
  default_team: string
  teams: Team[]
  technologies: Technology[]
  assignments: Record<string, Assignment[]>
}

// Human-readable names used in error messages and DSL descriptions
export const QUADRANT_NAMES: Record<Quadrant, string> = {
  0: 'Models & Providers',
  1: 'Infrastructure & Cloud',
  2: 'Frameworks & Libraries',
  3: 'Techniques & Patterns',
}

export const RING_NAMES: Record<Ring, string> = {
  0: 'ADOPT',
  1: 'TRIAL',
  2: 'ASSESS',
  3: 'HOLD',
}

// Pending operation — used by validate() for dry-run checks
export type PendingOp =
  | { type: 'addTechnology'; id: string; label: string; quadrant: Quadrant }
  | { type: 'assign'; teamId: string; techId: string; ring: Ring; moved?: Moved }
  | { type: 'move'; teamId: string; techId: string; newRing: Ring }
  | { type: 'removeAssignment'; teamId: string; techId: string }

export type ValidationResult =
  | { valid: true }
  | { valid: false; error: string }
