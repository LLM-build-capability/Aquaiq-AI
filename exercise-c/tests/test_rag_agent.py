"""
Tests for exercise-c/rag-agent/main.py

Runs the FastAPI app in-process using httpx.AsyncClient + ASGITransport.
WaterAgent is mocked so tests don't need Ollama or ChromaDB running.
"""
import importlib
import sys
import os
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

# ---------------------------------------------------------------------------
# Path setup — mirror what rag-agent/main.py does so imports resolve
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
RAG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag-agent"))

for p in (RAG_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Force local profile so config doesn't try to build an Azure client
os.environ.setdefault("LLM_PROFILE", "local")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("LOCAL_COLLECTION_NAME", "radar_local")
os.environ.setdefault("CLOUD_COLLECTION_NAME", "radar_local")
os.environ.setdefault(
    "CHROMA_PERSIST_DIR",
    os.path.join(RAG_DIR, "chroma_db"),
)

# ---------------------------------------------------------------------------
# Import the FastAPI app — patch WaterAgent before the module loads
# ---------------------------------------------------------------------------
mock_water_agent_cls = MagicMock()
mock_agent_instance = MagicMock()
mock_agent_instance.chat.return_value = "Mocked answer from RAG corpus."
mock_agent_instance.reset.return_value = None
mock_water_agent_cls.return_value = mock_agent_instance

with patch.dict("sys.modules", {"src.aquaiq_ai.agent": MagicMock(WaterAgent=mock_water_agent_cls)}):
    import main as rag_main  # rag-agent/main.py
    rag_app = rag_main.app
    # Inject the mock agent directly so /invoke works without startup event
    rag_main.agent = mock_agent_instance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_envelope(**overrides: Any) -> dict:
    base: dict = {
        "correlation_id": str(uuid.uuid4()),
        "causation_id": str(uuid.uuid4()),
        "idempotency_key": str(uuid.uuid4()),
        "sender": "orchestrator",
        "recipient": "rag-agent",
        "capability": "answer-from-corpus",
        "payload": {"query": "What is the ADOPT ring on the radar?"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    base.update(overrides)
    return base

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_health_returns_ok():
    async with AsyncClient(transport=ASGITransport(app=rag_app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["agent"] == "rag-agent"
    assert "answer-from-corpus" in body["capabilities"]


@pytest.mark.asyncio
async def test_invoke_returns_envelope():
    env = make_envelope()
    async with AsyncClient(transport=ASGITransport(app=rag_app), base_url="http://test") as client:
        resp = await client.post("/invoke", json=env)
    assert resp.status_code == 200
    body = resp.json()
    assert body["capability"] == "answer-from-corpus"
    assert body["sender"] == "rag-agent"
    assert body["recipient"] == "orchestrator"
    assert body["payload"]["answer"] == "Mocked answer from RAG corpus."
    assert body["correlation_id"] == env["correlation_id"]


@pytest.mark.asyncio
async def test_invoke_bad_capability_returns_400():
    env = make_envelope(capability="no-such-capability")
    async with AsyncClient(transport=ASGITransport(app=rag_app), base_url="http://test") as client:
        resp = await client.post("/invoke", json=env)
    assert resp.status_code == 400
    assert "unknown capability" in resp.json()["error"]


@pytest.mark.asyncio
async def test_invoke_missing_query_returns_400():
    env = make_envelope(payload={})
    async with AsyncClient(transport=ASGITransport(app=rag_app), base_url="http://test") as client:
        resp = await client.post("/invoke", json=env)
    assert resp.status_code == 400
    assert "query" in resp.json()["error"]


@pytest.mark.asyncio
async def test_invoke_invalid_envelope_returns_400():
    async with AsyncClient(transport=ASGITransport(app=rag_app), base_url="http://test") as client:
        resp = await client.post("/invoke", json={"not": "an envelope"})
    assert resp.status_code == 400
    assert "invalid envelope" in resp.json()["error"]


@pytest.mark.asyncio
async def test_idempotency_dedup():
    key = str(uuid.uuid4())
    env = make_envelope(idempotency_key=key)
    # First call — normal
    async with AsyncClient(transport=ASGITransport(app=rag_app), base_url="http://test") as client:
        r1 = await client.post("/invoke", json=env)
        r2 = await client.post("/invoke", json=env)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert "deduplicated" in r2.json()["payload"]["answer"]


@pytest.mark.asyncio
async def test_invoke_agent_error_returns_500():
    mock_agent_instance.chat.side_effect = RuntimeError("Ollama exploded")
    env = make_envelope(idempotency_key=str(uuid.uuid4()))
    try:
        async with AsyncClient(transport=ASGITransport(app=rag_app), base_url="http://test") as client:
            resp = await client.post("/invoke", json=env)
        assert resp.status_code == 500
        assert "internal error" in resp.json()["error"]
    finally:
        mock_agent_instance.chat.side_effect = None
        mock_agent_instance.chat.return_value = "Mocked answer from RAG corpus."


@pytest.mark.asyncio
async def test_response_envelope_has_new_idempotency_key():
    env = make_envelope()
    async with AsyncClient(transport=ASGITransport(app=rag_app), base_url="http://test") as client:
        resp = await client.post("/invoke", json=env)
    body = resp.json()
    # Response must carry a DIFFERENT idempotency_key (fresh UUID)
    assert body["idempotency_key"] != env["idempotency_key"]


@pytest.mark.asyncio
async def test_response_correlation_id_matches_request():
    env = make_envelope()
    async with AsyncClient(transport=ASGITransport(app=rag_app), base_url="http://test") as client:
        resp = await client.post("/invoke", json=env)
    assert resp.json()["correlation_id"] == env["correlation_id"]
