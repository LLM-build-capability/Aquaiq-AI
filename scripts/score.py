"""Build a rubric-template CSV from the latest benchmark JSONs.

Spec section 3 mandates a rubric and single-rater scoring. This script
reads the most recent results-cloud-*.json and results-local-*.json under
data/comparison-results/, pairs them by query id, and writes a single
CSV (one row per profile per query, 40 rows total) with:

    profile, id, category, query, answer, latency_s, tokens_per_second,
    tool_calls_emitted, sources, correctness, citation_present,
    refused_correctly, notes

The last four columns are blank — fill them in offline. Rubric:

    correctness:        0 (wrong / hallucinated), 1 (partial), 2 (correct)
    citation_present:   0 (no source named) or 1 (corpus or live tool cited)
    refused_correctly:  0/1 — only meaningful for OOS queries; leave blank otherwise

Output is data/comparison-results/rubric-template.csv (gitignored).
Re-running overwrites the file — score on a copy if you've already started.
"""

import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "data", "comparison-results")
OUT_CSV = os.path.join(RESULTS_DIR, "rubric-template.csv")


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


def main():
    cloud_path = latest("cloud")
    local_path = latest("local")
    cloud = load(cloud_path)
    local = load(local_path)
    print(f"Cloud: {cloud_path}  ({len(cloud['results'])} queries)")
    print(f"Local: {local_path}  ({len(local['results'])} queries)")

    rows = []
    for r in cloud["results"]:
        rows.append(("cloud", r))
    for r in local["results"]:
        rows.append(("local", r))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "profile", "id", "category", "query", "answer", "latency_s",
            "tokens_per_second", "tool_calls_emitted", "sources",
            "correctness", "citation_present", "refused_correctly", "notes",
        ])
        for profile, r in rows:
            w.writerow([
                profile, r["id"], r["category"], r["query"],
                (r.get("answer") or r.get("error") or "")[:1500],
                r.get("latency_s"), r.get("tokens_per_second"),
                r.get("tool_calls_emitted"),
                ";".join(r.get("sources") or []),
                "", "", "", "",
            ])

    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")
    print("Fill correctness / citation_present / refused_correctly / notes "
          "manually, then run scripts/aggregate.py.")


if __name__ == "__main__":
    main()
