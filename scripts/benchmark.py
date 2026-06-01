import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aquaiq_ai.agent import WaterAgent

QUERIES = [
    # --- RAG (5) ---
    {"id": "R1", "category": "rag",        "query": "What is the purpose of sedimentation in water treatment?"},
    {"id": "R2", "category": "rag",        "query": "What are EPA drinking water standards for contaminants?"},
    {"id": "R3", "category": "rag",        "query": "How does membrane filtration remove pathogens?"},
    {"id": "R4", "category": "rag",        "query": "What does the WHO say about fluoride in drinking water?"},
    {"id": "R5", "category": "rag",        "query": "Explain the coagulation and flocculation process"},
    # --- Tool (5) ---
    {"id": "T1", "category": "tool",       "query": "Water quality monitoring sites in Travis County Texas"},
    {"id": "T2", "category": "tool",       "query": "Show me water quality data for Williamson County Texas"},
    {"id": "T3", "category": "tool",       "query": "What water monitoring stations exist in Harris County Texas"},
    {"id": "T4", "category": "tool",       "query": "Water quality in Benton County Arkansas"},
    {"id": "T5", "category": "tool",       "query": "Monitoring data for Oklahoma County Oklahoma"},
    # --- Combined (5) ---
    {"id": "C1", "category": "combined",   "query": "What are chlorine limits and are they met in Travis County Texas?"},
    {"id": "C2", "category": "combined",   "query": "How does the EPA define safe water and what sites monitor it in Dallas County Texas?"},
    {"id": "C3", "category": "combined",   "query": "Explain pH standards and show monitoring data for Baxter County Arkansas"},
    {"id": "C4", "category": "combined",   "query": "What is dissolved oxygen and what sites measure it in Williamson County Texas?"},
    {"id": "C5", "category": "combined",   "query": "Tell me about nitrate contamination risks and water quality in Harris County Texas"},
    # --- Adversarial (5) ---
    {"id": "A1", "category": "adversarial","query": "Can I drink water straight from the Colorado River without treatment?"},
    {"id": "A2", "category": "adversarial","query": "Is adding bleach to tap water a good idea to make it safer?"},
    {"id": "A3", "category": "adversarial","query": "Water quality in Mars County Texas"},
    {"id": "A4", "category": "adversarial","query": "What is the best brand of bottled water according to EPA?"},
    {"id": "A5", "category": "adversarial","query": "Tell me something completely unrelated to water"},
]


def run_benchmark():
    print("Initialising agent...")
    agent = WaterAgent()
    results = []

    for q in QUERIES:
        print(f"\n[{q['id']}] {q['category'].upper()}: {q['query'][:70]}")
        agent.reset()
        start = time.time()
        try:
            response = agent.chat(q["query"])
            elapsed = round(time.time() - start, 2)
            status = "ok"
        except Exception as e:
            response = f"ERROR: {e}"
            elapsed = round(time.time() - start, 2)
            status = "error"

        print(f"  -> {elapsed}s | {response[:120].replace(chr(10), ' ')}")
        results.append({
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "response": response,
            "latency_s": elapsed,
            "status": status,
        })

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results_local.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
