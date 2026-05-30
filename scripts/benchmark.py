"""Phase 6 benchmark harness.

Runs the queries in scripts/queries.json against the active LLM_PROFILE
(cloud or local) and writes:

    data/comparison-results/results-{profile}-{timestamp}.json   (raw, per-query)
    data/comparison-results/summary.csv                          (flat, appended)

Per-query metrics captured: latency, routing similarities + chosen path,
tool_calls_emitted (bool), top_k doc sources (corpus filenames seen in the
RAG-context system message), error string if any.

The harness reuses WaterAgent unchanged — no edits to agent.py. It captures
the agent's printed routing line via stdout redirection and inspects
agent.messages after chat() to detect tool calls and retrieved sources.

Per-query timeout: 120 s (signal.SIGALRM, Unix-only). Errors are recorded
and the run continues; no automatic retry.

Usage:
    LLM_PROFILE=cloud python scripts/benchmark.py
    LLM_PROFILE=local python scripts/benchmark.py
"""

import csv
import io
import json
import os
import re
import signal
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from src.aquaiq_ai.config import get_profile, get_llm_model, get_collection_name
from src.aquaiq_ai.agent import WaterAgent

QUERIES_PATH = os.path.join(ROOT, "scripts", "queries.json")
RESULTS_DIR = os.path.join(ROOT, "data", "comparison-results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")
PER_QUERY_TIMEOUT_S = 120

ROUTING_RE = re.compile(r"RAG similarity:\s*([0-9.]+),\s*Tool similarity:\s*([0-9.]+)")
QUERY_TYPE_RE = re.compile(r"Query type:\s*(\w+)")
SOURCE_RE = re.compile(r"\[From:\s*([^|\]]+?)\s*(?:\||\])")


class QueryTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise QueryTimeout(f"query exceeded {PER_QUERY_TIMEOUT_S}s")


def _extract_sources(messages):
    """Pull corpus filenames out of the RAG-context system message."""
    sources = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if not content or not isinstance(content, str):
            continue
        if "Here's info from the documents:" in content:
            for match in SOURCE_RE.finditer(content):
                src = match.group(1).strip()
                if src not in sources:
                    sources.append(src)
    return sources


def _tool_calls_emitted(messages):
    """True if any assistant message in the conversation requested a tool call."""
    for m in messages:
        # WaterAgent appends raw OpenAI message objects after a tool call.
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            return True
        if isinstance(m, dict) and m.get("tool_calls"):
            return True
    return False


def run_one(agent, q):
    """Run a single query, returning a result dict."""
    captured = io.StringIO()
    record = {
        "id": q["id"],
        "category": q["category"],
        "query": q["query"],
        "answer": None,
        "latency_s": None,
        "rag_similarity": None,
        "tool_similarity": None,
        "query_type": None,
        "tool_calls_emitted": False,
        "sources": [],
        "error": None,
    }

    agent.reset()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(PER_QUERY_TIMEOUT_S)
    t0 = time.time()
    try:
        with redirect_stdout(captured):
            answer = agent.chat(q["query"])
        record["answer"] = answer
    except QueryTimeout as e:
        record["error"] = str(e)
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        record["latency_s"] = round(time.time() - t0, 3)

    # Parse the captured stdout for routing info.
    out = captured.getvalue()
    rmatch = ROUTING_RE.search(out)
    if rmatch:
        record["rag_similarity"] = float(rmatch.group(1))
        record["tool_similarity"] = float(rmatch.group(2))
    qtype = QUERY_TYPE_RE.search(out)
    if qtype:
        record["query_type"] = qtype.group(1)

    # Inspect post-run messages for tool-call evidence and retrieved sources.
    record["tool_calls_emitted"] = _tool_calls_emitted(agent.messages)
    record["sources"] = _extract_sources(agent.messages)
    record["agent_stdout"] = out

    return record


def main():
    profile = get_profile()
    model = get_llm_model()
    collection = get_collection_name()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("=" * 70)
    print(f"Benchmark run — profile={profile} model={model} collection={collection}")
    print(f"Timestamp:     {timestamp}")
    print(f"Per-query cap: {PER_QUERY_TIMEOUT_S}s")
    print("=" * 70)

    with open(QUERIES_PATH) as f:
        spec = json.load(f)
    queries = spec["queries"]
    print(f"Loaded {len(queries)} queries from {QUERIES_PATH}")

    print("\nBuilding agent (one-time embedder + routing setup)...")
    agent = WaterAgent()
    print("Agent ready.\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    for i, q in enumerate(queries, 1):
        print(f"[{i:>2}/{len(queries)}] {q['id']:<10} ({q['category']:<13}) {q['query'][:70]}")
        record = run_one(agent, q)
        results.append(record)
        status = "ERR" if record["error"] else "OK"
        print(f"           -> {status} latency={record['latency_s']}s "
              f"route={record['query_type']} tool_calls={record['tool_calls_emitted']} "
              f"sources={len(record['sources'])}")

    # Raw JSON dump (human-readable, includes full answers + agent_stdout).
    raw_path = os.path.join(RESULTS_DIR, f"results-{profile}-{timestamp}.json")
    with open(raw_path, "w") as f:
        json.dump({
            "profile": profile,
            "model": model,
            "collection": collection,
            "timestamp": timestamp,
            "per_query_timeout_s": PER_QUERY_TIMEOUT_S,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nRaw results: {raw_path}")

    # Flat CSV summary (appended; one row per query per run).
    new_file = not os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "timestamp", "profile", "model", "collection", "id", "category",
                "query", "latency_s", "query_type", "rag_similarity",
                "tool_similarity", "tool_calls_emitted", "num_sources",
                "sources", "error",
            ])
        for r in results:
            w.writerow([
                timestamp, profile, model, collection, r["id"], r["category"],
                r["query"], r["latency_s"], r["query_type"], r["rag_similarity"],
                r["tool_similarity"], r["tool_calls_emitted"], len(r["sources"]),
                ";".join(r["sources"]), r["error"] or "",
            ])
    print(f"Summary CSV: {SUMMARY_CSV}")

    # Quick aggregates printed to stdout.
    successes = [r for r in results if not r["error"]]
    if successes:
        avg = sum(r["latency_s"] for r in successes) / len(successes)
        print(f"\n{len(successes)}/{len(results)} succeeded — avg latency {avg:.2f}s")
    errors = [r for r in results if r["error"]]
    if errors:
        print(f"{len(errors)} errors:")
        for r in errors:
            print(f"  {r['id']}: {r['error']}")


if __name__ == "__main__":
    main()
