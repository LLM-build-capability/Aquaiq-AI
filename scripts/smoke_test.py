"""One-shot smoke test for the active LLM_PROFILE.

Reads .env, prints the resolved profile/model/collection, fires a single
canned query through WaterAgent, and prints the answer. Used as a Phase 3
verification gate (cloud) and Phase 5 verification gate (local).
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from src.aquaiq_ai.config import get_profile, get_llm_model, get_collection_name
from src.aquaiq_ai.agent import WaterAgent

QUERY = "What are the EPA limits for lead in drinking water?"

print("=" * 60)
print(f"Profile:    {get_profile()}")
print(f"Model:      {get_llm_model()}")
print(f"Collection: {get_collection_name()}")
print("=" * 60)

agent = WaterAgent()

print(f"\nQuery: {QUERY}\n")
t0 = time.time()
answer = agent.chat(QUERY)
elapsed = time.time() - t0

print("\n--- Answer ---")
print(answer)
print(f"\n[latency: {elapsed:.2f}s]")
