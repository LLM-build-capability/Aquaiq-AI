import asyncio
import json
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI

REGISTRY_URL = "http://localhost:8083"
AGENT_NAME = "mcp-agent"
AGENT_PORT = 8082
AGENT_ENDPOINT = f"http://localhost:{AGENT_PORT}"
HEARTBEAT_INTERVAL = 20

NODE_PATH = "/opt/homebrew/opt/node@22/bin/node"
NODE_CWD = "/Users/ppa1/Aquaiq-AI/exercise-b"
NODE_SCRIPT = "src/index.ts"

node_process: asyncio.subprocess.Process | None = None
heartbeat_task: asyncio.Task | None = None
rpc_lock = asyncio.Lock()
rpc_id = 0
seen_idempotency_keys: dict[str, dict] = {}


def register() -> bool:
    try:
        requests.post(
            f"{REGISTRY_URL}/register",
            json={
                "name": AGENT_NAME,
                "capabilities": ["propose-radar-change"],
                "endpoint": AGENT_ENDPOINT,
                "health_url": f"{AGENT_ENDPOINT}/health",
            },
            timeout=5,
        )
        print(f"[{AGENT_NAME}] registered with registry")
        return True
    except Exception as exc:
        print(f"[{AGENT_NAME}] registry unavailable: {exc}")
        return False


def deregister() -> None:
    try:
        requests.delete(f"{REGISTRY_URL}/deregister/{AGENT_NAME}", timeout=5)
        print(f"[{AGENT_NAME}] deregistered")
    except Exception as exc:
        print(f"[{AGENT_NAME}] deregister failed: {exc}")


async def heartbeat_loop() -> None:
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            requests.post(f"{REGISTRY_URL}/heartbeat/{AGENT_NAME}", timeout=5)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            print(f"[{AGENT_NAME}] heartbeat failed: {exc}")


async def print_node_errors():
    while True:
        if node_process and node_process.stderr:
            err = await node_process.stderr.readline()
            if not err:
                break
            print("NODE ERROR:", err.decode().rstrip())
        else:
            await asyncio.sleep(0.1)


async def start_node_process() -> None:
    global node_process
    node_process = await asyncio.create_subprocess_exec(
        NODE_PATH,
        "--import",
        "tsx/esm",
        NODE_SCRIPT,
        cwd=NODE_CWD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    print(f"[{AGENT_NAME}] Node subprocess started (pid={node_process.pid}) cwd={NODE_CWD}")
    asyncio.create_task(print_node_errors())


async def stop_node_process() -> None:
    global node_process
    if node_process and node_process.returncode is None:
        node_process.terminate()
        try:
            await asyncio.wait_for(node_process.wait(), timeout=5)
        except asyncio.TimeoutError:
            node_process.kill()
            await node_process.wait()


async def call_node(params: dict) -> dict:
    global rpc_id

    if not node_process or not node_process.stdin or not node_process.stdout:
        raise RuntimeError("Node subprocess not running")

    async with rpc_lock:
        rpc_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": "execute",
            "params": params,
            "id": rpc_id,
        }

        node_process.stdin.write((json.dumps(request) + "\n").encode())
        await node_process.stdin.drain()

        line = await node_process.stdout.readline()
        if not line:
            raise RuntimeError("No response from Node")

        return json.loads(line.decode())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_task

    print(f"[{AGENT_NAME}] starting on port {AGENT_PORT}")
    await start_node_process()
    register()

    heartbeat_task = asyncio.create_task(heartbeat_loop())

    try:
        yield
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
        deregister()
        await stop_node_process()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    alive = node_process is not None and node_process.returncode is None
    return {"status": "ok", "node_subprocess_alive": alive}


@app.post("/invoke")
async def invoke(envelope: dict):
    correlation_id = envelope.get("correlation_id")
    idempotency_key = envelope.get("idempotency_key")
    payload = envelope.get("payload", {})

    if idempotency_key and idempotency_key in seen_idempotency_keys:
        return seen_idempotency_keys[idempotency_key]

    try:
        node_response = await call_node(payload)
        response_payload = node_response.get("result", node_response)
    except Exception as e:
        response_payload = {"error": str(e)}

    response = {
        "correlation_id": correlation_id,
        "causation_id": envelope.get("causation_id"),
        "sender": AGENT_NAME,
        "recipient": envelope.get("sender"),
        "capability": envelope.get("capability"),
        "payload": response_payload,
    }

    if idempotency_key:
        seen_idempotency_keys[idempotency_key] = response

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)