# N-Tool Baseline — Tech Radar MCP Server (Variant 2)

**Measured:** 2026-05-27  
**Variant:** V2 — Constrained domain metamodel (Stack.TechRadar)  
**Purpose:** Establish the *before* token cost. This file must exist in git before any Code Mode implementation is committed.

---

## What a Naive N-Tool Server Would Look Like

A straightforward MCP server for the Tech Radar would expose one tool per domain operation. Below are the 10 tools that cover the full CRUD surface of `docs/config.json`.

### Tool Definitions (JSON Schema)

```json
[
  {
    "name": "listTechnologies",
    "description": "List all technologies in the radar, optionally filtered by quadrant (0=Models & Providers, 1=Infrastructure & Cloud, 2=Frameworks & Libraries, 3=Techniques & Patterns).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "quadrant": {
          "type": "integer",
          "enum": [0, 1, 2, 3],
          "description": "0=Models & Providers, 1=Infrastructure & Cloud, 2=Frameworks & Libraries, 3=Techniques & Patterns"
        }
      }
    }
  },
  {
    "name": "getTechnology",
    "description": "Get a single technology by its kebab-case id. Returns id, label, and quadrant.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Technology id (kebab-case, e.g. gpt-4o-azure-openai)"
        }
      },
      "required": ["id"]
    }
  },
  {
    "name": "listTeams",
    "description": "List all teams registered in the radar. Returns each team id, name, and date.",
    "inputSchema": {
      "type": "object",
      "properties": {}
    }
  },
  {
    "name": "listAssignments",
    "description": "List all ring assignments for a given team. Returns each technology id, ring (0=ADOPT, 1=TRIAL, 2=ASSESS, 3=HOLD), and moved flag.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "teamId": {
          "type": "string",
          "description": "Team id, e.g. llm-capability-office"
        }
      },
      "required": ["teamId"]
    }
  },
  {
    "name": "getAssignment",
    "description": "Get the ring assignment for a specific technology within a team. Returns ring and moved, or null if not assigned.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "teamId": { "type": "string", "description": "Team id" },
        "techId": { "type": "string", "description": "Technology id" }
      },
      "required": ["teamId", "techId"]
    }
  },
  {
    "name": "addTechnology",
    "description": "Add a new technology entry to the radar. The id must be unique and kebab-case. Assign it to a team separately using assignTechnology.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique kebab-case identifier, e.g. my-new-tool"
        },
        "label": {
          "type": "string",
          "description": "Human-readable label, ~30 chars max"
        },
        "quadrant": {
          "type": "integer",
          "enum": [0, 1, 2, 3],
          "description": "0=Models & Providers, 1=Infrastructure & Cloud, 2=Frameworks & Libraries, 3=Techniques & Patterns"
        }
      },
      "required": ["id", "label", "quadrant"]
    }
  },
  {
    "name": "assignTechnology",
    "description": "Assign an existing technology to a team with a ring placement. The technology must already exist in the radar. Fails if already assigned to this team — use moveTechnology to change ring.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "teamId": { "type": "string", "description": "Team id" },
        "techId": {
          "type": "string",
          "description": "Technology id — must already exist in the radar"
        },
        "ring": {
          "type": "integer",
          "enum": [0, 1, 2, 3],
          "description": "0=ADOPT, 1=TRIAL, 2=ASSESS, 3=HOLD"
        },
        "moved": {
          "type": "integer",
          "enum": [-1, 0, 1],
          "description": "Movement since previous radar: -1=moved out, 0=no change, 1=moved in. Defaults to 0."
        }
      },
      "required": ["teamId", "techId", "ring"]
    }
  },
  {
    "name": "moveTechnology",
    "description": "Change the ring for a technology already assigned to a team. Use this to promote or demote a technology. Fails if not yet assigned — use assignTechnology first.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "teamId": { "type": "string", "description": "Team id" },
        "techId": { "type": "string", "description": "Technology id" },
        "newRing": {
          "type": "integer",
          "enum": [0, 1, 2, 3],
          "description": "0=ADOPT, 1=TRIAL, 2=ASSESS, 3=HOLD"
        }
      },
      "required": ["teamId", "techId", "newRing"]
    }
  },
  {
    "name": "removeAssignment",
    "description": "Remove a technology ring assignment from a team. Does not delete the technology from the radar — it just stops being tracked by this team.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "teamId": { "type": "string", "description": "Team id" },
        "techId": { "type": "string", "description": "Technology id" }
      },
      "required": ["teamId", "techId"]
    }
  },
  {
    "name": "commitChanges",
    "description": "Persist all staged changes back to docs/config.json. Must be called after any write operations (addTechnology, assignTechnology, moveTechnology, removeAssignment) for changes to take effect. Provide a short message describing what changed.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "message": {
          "type": "string",
          "description": "Short description of the changes, e.g. Add claude-haiku to RDE team at TRIAL"
        }
      },
      "required": ["message"]
    }
  }
]
```

---

## Token Count

Measured with `tiktoken` cl100k_base on the full JSON above:

| | Count |
|---|---|
| Tools | 10 |
| Bootstrap token cost | **1,417 tokens** |

```js
// Snippet that produced this number (run from Stack.TechRadar_MultiTeam 1/ where tiktoken is installed)
const { get_encoding } = require('tiktoken');
const enc = get_encoding('cl100k_base');
const tokens = enc.encode(JSON.stringify(tools, null, 2));
console.log(tokens.length); // 1417
enc.free();
```

---

## Why This Is the Baseline, Not the Target

The 1,417 tokens are paid on **every request**, before any reasoning has happened. The problems compound:

1. **Quadrant and ring enums are repeated across 5 tools** — the model sees the same description of `0=ADOPT, 1=TRIAL...` four times.
2. **The metamodel is embedded in tool descriptions** — if the schema changes, every tool description must be updated.
3. **Multi-step workflows require N round-trips** — adding a technology and assigning it to a team is two tool calls, two context round-trips.
4. **No validation logic is visible to the model** — the only feedback a failed call gives is an opaque error. The model has no guidance on what to try instead.

The Code Mode version replaces all 10 tools with **2 tools (`search` + `execute`)**, carries the metamodel once as a TypeScript DSL (~400 tokens), and moves all validation into the proxy with self-correcting error messages.

**Target bootstrap cost: ≤ 1,200 tokens (tools + DSL combined).**
