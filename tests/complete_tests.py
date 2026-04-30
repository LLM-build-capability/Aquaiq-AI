import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.aquaiq_ai.embedding_helper import AzureEmbedder
from src.aquaiq_ai.tools import execute_water_quality_tool, find_county_code
from src.aquaiq_ai.retriever import WaterDocRetriever
from src.aquaiq_ai.agent import WaterAgent
'''wrote this function again instead of importing it from ingest file because
it is giving that message "Collection already exists" and exiting'''
def semantic_chunking(text, chunk_size=800, overlap_sentences=2):
   sentences = re.split(r'(?<=[.!?])\s+', text)
   sentences = [s.strip() for s in sentences if s.strip()]
   if not sentences:
       return []
   chunks = []
   current = []
   current_len = 0
   for sent in sentences:
       sent_len = len(sent)
       if current_len + sent_len > chunk_size and current:
           chunks.append(' '.join(current))
           overlap = min(overlap_sentences, len(current))
           current = current[-overlap:] if overlap > 0 else []
           current_len = sum(len(s) for s in current)
       current.append(sent)
       current_len += sent_len
   if current:
       chunks.append(' '.join(current))
   return chunks
print("TESTING WATER AGENT")
print("-" * 20)
passed = 0
failed = 0
# first test: checking embedding
print("\n1. checking embeddings...")
try:
   e = AzureEmbedder()
   v = e.embed("test")
   if len(v) > 0:
       print(f"  ok - dimension {len(v)}")
       passed += 1
   else:
       print("  fail - empty vector")
       failed += 1
except Exception as e:
   print(f"  fail - {e}")
   failed += 1
# second test - county codes from tools
print("\n2. checking county codes...")
test = find_county_code("travis county texas")
if test == "US:48:453":
   print("  ok - travis county works")
   passed += 1
else:
   print(f"  fail - got {test}")
   failed += 1
test2 = find_county_code("fake county")
if test2 is None:
   print("  ok - fake county fails correctly")
   passed += 1
else:
   print("  fail - fake county should be None")
   failed += 1
# testing api from tools
print("\n3. checking water quality api...")
try:
   result = execute_water_quality_tool("Travis County Texas")
   if "error" not in result:
       sites = result.get('total_sites', 0)
       print(f"  ok - found {sites} sites")
       passed += 1
   else:
       print(f"  fail - {result.get('error')}")
       failed += 1
except Exception as e:
   print(f"  fail - {e}")
   failed += 1
# Checking the DB exists or not
print("\n4. checking chromadb...")
try:
   r = WaterDocRetriever()
   if r.available:
       print(f"there are {r.collection.count()} chunks in db")
       passed += 1
   else:
       print("warning - no data, run ingest.py")
except Exception as e:
   print(f"  fail - {e}")
   failed += 1
# checking semantic chunking
print("\n5. checking chunking...")
sample = "First sentence. Second sentence. Third sentence. Fourth sentence."
chunks = semantic_chunking(sample)
if len(chunks) > 0:
   print(f"  ok - created {len(chunks)} chunks")
   passed += 1
else:
   print("  fail - chunking broken")
   failed += 1
# agent classification (no api call)
print("\n6. checking agent routing...")
try:
   agent = WaterAgent()
   q1 = agent._classify("what is chlorination?")
   q2 = agent._classify("water quality in travis county")
   q3 = agent._classify("according to epa is nitrate in williamson county safe?")
   print(f"  rag query -> {q1}")
   print(f"  tool query -> {q2}")
   print(f"  both query -> {q3}")
   if q1 == "rag" and q2 == "tool":
       print("  ok - routing works")
       passed += 1
   else:
       print("  warn - routing might need tuning")
except Exception as e:
   print(f"  fail - {e}")
   failed += 1
# actual chat test (it is optional)
print("\n7. chat test (uses api credits)...")
answer = input("  run chat test? (y/n): ")
if answer.lower() == 'y':
   try:
       agent = WaterAgent()
       agent.reset()
       resp = agent.chat("say ok")
       print(f"  agent said: {resp[:100]}")
       passed += 1
   except Exception as e:
       print(f"  fail - {e}")
       failed += 1
else:
   print("skipped")
# final summary
print(f"RESULTS: {passed} passed, {failed} failed")
if failed == 0:
   print("all tests are sucessfull")
else:
   print(f"{failed} test(s) failed")