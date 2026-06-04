import os
import sys
import uuid
from datetime import datetime, timezone

# Add the repo root to sys.path so src.aquaiq_ai.* imports work unchanged.
# This file lives at exercise-c/rag-agent/main.py; repo root is two levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAG_DIR = os.path.dirname(os.path.abspath(__file__))
# aquaiq_ai/ sits alongside this script; add rag-agent/ so it is importable.
sys.path.insert(0, RAG_DIR)
sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(REPO_ROOT, ".env"))

# Point Exercise A's retriever at the radar-specific chroma collection that
# lives inside exercise-c/rag-agent/chroma_db/. These env vars are read by
# src/aquaiq_ai/config.py and src/aquaiq_ai/retriever.py — no code changes.
os.environ.setdefault(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(__file__), "chroma_db"),
)
os.environ.setdefault("LOCAL_COLLECTION_NAME", "radar_local")
os.environ.setdefault("CLOUD_COLLECTION_NAME", "radar_local")
os.environ.setdefault("LLM_PROFILE", "local")

import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any
import asyncio

from src.aquaiq_ai.agent import WaterAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PORT = int(os.getenv("RAG_AGENT_PORT", "8081"))
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8083")
AGENT_NAME = "rag-agent"
HEARTBEAT_INTERVAL_S = 20

# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------
class Envelope(BaseModel):
    correlation_id: str
    causation_id: str
    idempotency_key: str
    sender: str
    recipient: str
    capability: str
    payload: dict[str, Any]
    timestamp: str

# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------
def log(level: str, message: str, **fields):
    import json
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "agent": AGENT_NAME,
        "message": message,
        **fields,
    }
    print(json.dumps(entry), flush=True)

# ---------------------------------------------------------------------------
# Idempotency dedup
# ---------------------------------------------------------------------------
seen_keys: set[str] = set()

# ---------------------------------------------------------------------------
# WaterAgent — initialised once at startup; routing embeddings pre-calculated
# ---------------------------------------------------------------------------
agent: WaterAgent | None = None

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------
async def register():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{REGISTRY_URL}/register", json={
                "name": AGENT_NAME,
                "capabilities": ["answer-from-corpus"],
                "endpoint": f"http://localhost:{PORT}",
                "health_url": f"http://localhost:{PORT}/health",
            })
        if resp.status_code == 201:
            log("INFO", "registered with registry")
        else:
            log("WARN", f"registry registration returned {resp.status_code}")
    except Exception as e:
        log("WARN", f"registry registration failed: {e}")

async def deregister():
    try:
        async with httpx.AsyncClient() as client:
            await client.delete(f"{REGISTRY_URL}/deregister/{AGENT_NAME}")
        log("INFO", "deregistered")
    except Exception:
        pass

async def heartbeat_loop():
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{REGISTRY_URL}/heartbeat/{AGENT_NAME}")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    log("INFO", "initialising WaterAgent",
        llm_profile=os.getenv("LLM_PROFILE"),
        chroma_dir=os.getenv("CHROMA_PERSIST_DIR"),
        collection=os.getenv("LOCAL_COLLECTION_NAME"))
    agent = WaterAgent()
    log("INFO", "WaterAgent ready")
    await register()
    asyncio.create_task(heartbeat_loop())
    yield
    await deregister()

app = FastAPI(lifespan=lifespan)

@app.post("/invoke")
async def invoke(request: Request):
    try:
        body = await request.json()
        envelope = Envelope(**body)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "invalid envelope", "detail": str(e)})

    correlation_id = envelope.correlation_id
    log("INFO", "invoke received", correlation_id=correlation_id,
        causation_id=envelope.causation_id, capability=envelope.capability)

    if envelope.capability != "answer-from-corpus":
        return JSONResponse(status_code=400, content={"error": f"unknown capability: {envelope.capability}"})

    # Idempotency dedup
    if envelope.idempotency_key in seen_keys:
        log("INFO", "idempotency deduplicated", correlation_id=correlation_id,
            idempotency_key=envelope.idempotency_key)
        return Envelope(
            correlation_id=correlation_id,
            causation_id=correlation_id,
            idempotency_key=str(uuid.uuid4()),
            sender=AGENT_NAME,
            recipient=envelope.sender,
            capability=envelope.capability,
            payload={"answer": "(deduplicated — same idempotency_key)"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).model_dump()
    seen_keys.add(envelope.idempotency_key)

    query = envelope.payload.get("query")
    if not query:
        return JSONResponse(status_code=400, content={"error": "payload.query is required"})

    try:
        log("INFO", "calling WaterAgent.chat", correlation_id=correlation_id,
            query_preview=query[:100])
        # Reset conversation history so each A2A call is stateless
        agent.reset()
        answer = agent.chat(query)
        log("INFO", "WaterAgent.chat complete", correlation_id=correlation_id,
            answer_length=len(answer or ""))

        response = Envelope(
            correlation_id=correlation_id,
            causation_id=correlation_id,
            idempotency_key=str(uuid.uuid4()),
            sender=AGENT_NAME,
            recipient=envelope.sender,
            capability=envelope.capability,
            payload={"answer": answer or ""},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return response.model_dump()
    except Exception as e:
        log("ERROR", f"WaterAgent.chat failed: {e}", correlation_id=correlation_id)
        return JSONResponse(status_code=500, content={"error": "internal error", "detail": str(e)})

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent": AGENT_NAME,
        "capabilities": ["answer-from-corpus"],
        "llm_profile": os.getenv("LLM_PROFILE", "local"),
        "corpus": os.getenv("LOCAL_COLLECTION_NAME", "radar_local"),
        "agent_ready": agent is not None,
    }

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
