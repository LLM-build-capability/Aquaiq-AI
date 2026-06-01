"""Cross-profile aggregation for the 20-query benchmark.

Reads the most recent results-cloud-*.json and results-local-*.json under
data/comparison-results/ plus the manually-scored rubric-scores.csv, then
prints a single side-by-side report covering:

    - Run metadata (model, collection, timestamp) per profile
    - Latency: mean, p50, p95
    - Throughput: prompt + completion tokens, mean tokens/sec
    - Tool-call success rate on tool + hybrid queries
    - Quality: per-category mean correctness, citation rate,
      refused-correctly rate (OOS only)
    - Retrieval-source overlap: % of queries where both profiles surfaced
      the same top retrieved source

Read-only — never writes files. Safe to re-run as new benchmarks land.
"""

import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "data", "comparison-results")
RUBRIC_CSV = os.path.join(RESULTS_DIR, "rubric-scores.csv")


def latest(profile):
    pattern = os.path.join(RESULTS_DIR, f"results-{profile}-*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        sys.exit(f"No results found for profile={profile} (pattern: {pattern}). "
                 f"Run scripts/benchmark.py first.")
    return matches[-1]


def load(path):
    with open(path) as f:
        return json.load(f)


def percentile(values, pct):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def safe_mean(values):
    return statistics.mean(values) if values else float("nan")


def fmt(x, prec=2):
    if isinstance(x, float) and x != x:  # NaN
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def load_rubric():
    if not os.path.exists(RUBRIC_CSV):
        return {}
    rubric = {}
    with open(RUBRIC_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["profile"], row["id"])
            rubric[key] = {
                "correctness": int(row["correctness"]) if row["correctness"] else None,
                "citation_present": int(row["citation_present"]) if row["citation_present"] else None,
                "refused_correctly": (
                    int(row["refused_correctly"]) if row["refused_correctly"] else None
                ),
                "category": row["category"],
            }
    return rubric


def summarise_profile(run, rubric, profile):
    results = run["results"]
    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    tps = [r["tokens_per_second"] for r in results if r.get("tokens_per_second")]
    prompt_tok = sum(r.get("prompt_tokens") or 0 for r in results)
    comp_tok = sum(r.get("completion_tokens") or 0 for r in results)

    tool_or_hybrid = [r for r in results if r["category"] in ("tool", "hybrid")]
    tool_emitted = sum(1 for r in tool_or_hybrid if r.get("tool_calls_emitted"))

    by_cat = defaultdict(list)
    citation_by_cat = defaultdict(list)
    refused_oos = []
    for r in results:
        key = (profile, r["id"])
        scored = rubric.get(key)
        if scored is None:
            continue
        if scored["correctness"] is not None:
            by_cat[r["category"]].append(scored["correctness"])
        if scored["citation_present"] is not None:
            citation_by_cat[r["category"]].append(scored["citation_present"])
        if r["category"] == "out_of_scope" and scored["refused_correctly"] is not None:
            refused_oos.append(scored["refused_correctly"])

    return {
        "model": run.get("model"),
        "collection": run.get("collection"),
        "timestamp": run.get("timestamp"),
        "n": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "latency_mean": safe_mean(latencies),
        "latency_p50": percentile(latencies, 50),
        "latency_p95": percentile(latencies, 95),
        "tps_mean": safe_mean(tps),
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "tool_emitted": tool_emitted,
        "tool_total": len(tool_or_hybrid),
        "correctness_by_cat": {k: safe_mean(v) for k, v in by_cat.items()},
        "citation_by_cat": {k: safe_mean(v) for k, v in citation_by_cat.items()},
        "refused_oos_rate": safe_mean(refused_oos) if refused_oos else float("nan"),
        "overall_correctness": safe_mean(
            [v for vs in by_cat.values() for v in vs]
        ),
        "overall_citation": safe_mean(
            [v for vs in citation_by_cat.values() for v in vs]
        ),
    }


def overlap(cloud_results, local_results):
    by_id = {r["id"]: r for r in local_results}
    same = 0
    compared = 0
    for r in cloud_results:
        other = by_id.get(r["id"])
        if not other:
            continue
        cs = (r.get("sources") or [])
        ls = (other.get("sources") or [])
        if not cs or not ls:
            continue
        compared += 1
        if cs[0] == ls[0]:
            same += 1
    return same, compared


def line():
    print("-" * 72)


def render(cs, ls, overlap_same, overlap_total):
    line()
    print(f"{'metric':<32} {'cloud':>18} {'local':>18}")
    line()
    print(f"{'model':<32} {cs['model']:>18} {ls['model']:>18}")
    print(f"{'collection':<32} {cs['collection']:>18} {ls['collection']:>18}")
    print(f"{'timestamp (UTC)':<32} {cs['timestamp']:>18} {ls['timestamp']:>18}")
    cs_qe = f"{cs['n']} / {cs['errors']}"
    ls_qe = f"{ls['n']} / {ls['errors']}"
    print(f"{'queries / errors':<32} {cs_qe:>18} {ls_qe:>18}")
    line()
    print(f"{'latency mean (s)':<32} {fmt(cs['latency_mean']):>18} {fmt(ls['latency_mean']):>18}")
    print(f"{'latency p50 (s)':<32} {fmt(cs['latency_p50']):>18} {fmt(ls['latency_p50']):>18}")
    print(f"{'latency p95 (s)':<32} {fmt(cs['latency_p95']):>18} {fmt(ls['latency_p95']):>18}")
    line()
    print(f"{'prompt tokens (sum)':<32} {cs['prompt_tokens']:>18} {ls['prompt_tokens']:>18}")
    print(f"{'completion tokens (sum)':<32} {cs['completion_tokens']:>18} {ls['completion_tokens']:>18}")
    print(f"{'tokens/sec (mean)':<32} {fmt(cs['tps_mean']):>18} {fmt(ls['tps_mean']):>18}")
    line()
    cs_tc = f"{cs['tool_emitted']}/{cs['tool_total']}"
    ls_tc = f"{ls['tool_emitted']}/{ls['tool_total']}"
    print(f"{'tool calls emitted':<32} {cs_tc:>18} {ls_tc:>18}")
    line()
    print(f"{'overall correctness (avg /2)':<32} "
          f"{fmt(cs['overall_correctness']):>18} {fmt(ls['overall_correctness']):>18}")
    for cat in ("rag", "tool", "hybrid", "out_of_scope"):
        c = cs["correctness_by_cat"].get(cat, float("nan"))
        l = ls["correctness_by_cat"].get(cat, float("nan"))
        print(f"  - {cat:<28} {fmt(c):>18} {fmt(l):>18}")
    line()
    print(f"{'citation rate (overall)':<32} "
          f"{fmt(cs['overall_citation']):>18} {fmt(ls['overall_citation']):>18}")
    print(f"{'refused-correctly (OOS)':<32} "
          f"{fmt(cs['refused_oos_rate']):>18} {fmt(ls['refused_oos_rate']):>18}")
    line()
    if overlap_total:
        pct = 100.0 * overlap_same / overlap_total
        print(f"top-source overlap: {overlap_same}/{overlap_total} queries "
              f"({pct:.1f}%) — same first retrieved corpus doc on both profiles")
    else:
        print("top-source overlap: no query had retrieved sources on both profiles")
    line()


def main():
    cloud_path = latest("cloud")
    local_path = latest("local")
    cloud = load(cloud_path)
    local = load(local_path)
    rubric = load_rubric()
    if not rubric:
        print(f"WARNING: {RUBRIC_CSV} not found — quality columns will show n/a")

    cs = summarise_profile(cloud, rubric, "cloud")
    ls = summarise_profile(local, rubric, "local")
    overlap_same, overlap_total = overlap(cloud["results"], local["results"])

    print(f"Cloud results: {cloud_path}")
    print(f"Local results: {local_path}")
    if rubric:
        print(f"Rubric:        {RUBRIC_CSV} ({len(rubric)} rows)")
    render(cs, ls, overlap_same, overlap_total)


if __name__ == "__main__":
    main()
