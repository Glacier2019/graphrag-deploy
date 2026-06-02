import json
import os
import re
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from urllib.parse import urlencode, parse_qs
import httpx

CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "")

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", CHAT_BASE_URL).rstrip("/")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", CHAT_API_KEY)
EMBED_MODEL = os.getenv("EMBED_MODEL", "")

AZURE_PATH_RE = re.compile(r"^/openai/deployments/([^/]+)/(.+)$")
STRIP_QUERY_PARAMS = {"api-version"}

app = FastAPI(title="LLM Proxy")


def _pick_target(path: str):
    is_embed = "embeddings" in path.lower()
    if is_embed:
        return EMBED_BASE_URL, EMBED_API_KEY, EMBED_MODEL
    return CHAT_BASE_URL, CHAT_API_KEY, CHAT_MODEL


def _strip_azure_path(path: str):
    m = AZURE_PATH_RE.match(path)
    if m:
        return m.group(2)
    return path.lstrip("/")


def _clean_query(query_string: str) -> str:
    if not query_string:
        return ""
    cleaned = {k: v for k, v in parse_qs(query_string, keep_blank_values=True).items() if k not in STRIP_QUERY_PARAMS}
    return urlencode(cleaned, doseq=True) if cleaned else ""


def _forward_headers(headers, api_key: str) -> dict:
    h = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "transfer-encoding"):
            continue
        h[k] = v
    h["authorization"] = f"Bearer {api_key}"
    return h


def _patch_body(body: bytes, override_model: str) -> bytes:
    if not override_model or not body:
        return body
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            data["model"] = override_model
            return json.dumps(data).encode("utf-8")
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return body


async def _proxy(request: Request):
    path = request.url.path
    query = request.url.query
    base_url, api_key, override_model = _pick_target(path)

    target_path = _strip_azure_path(path)
    clean_query = _clean_query(query)
    url = f"{base_url}/{target_path}"
    if clean_query:
        url += f"?{clean_query}"

    headers = _forward_headers(request.headers, api_key)
    raw_body = await request.body()
    body = _patch_body(raw_body, override_model)

    print(f"[proxy] {request.method} {url} model_override={override_model}", flush=True)

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body or None,
        )

    print(f"[proxy] → {resp.status_code}", flush=True)

    resp_headers = {k: v for k, v in resp.headers.items()
                    if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")}

    if "text/event-stream" in resp.headers.get("content-type", ""):
        async def stream():
            async for chunk in resp.aiter_bytes():
                yield chunk
        return StreamingResponse(stream(), status_code=resp.status_code, headers=resp_headers)

    return Response(content=resp.content, status_code=resp.status_code, headers=resp_headers)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(path: str, request: Request):
    return await _proxy(request)


@app.api_route("/openai/deployments/{model}/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def azure_route(model: str, rest: str, request: Request):
    return await _proxy(request)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
