"""
LLM Proxy — 智能路由代理

三层分类策略：
  Layer 1: 敏感词正则 → 强制本地 Ollama
  Layer 2: 简单/复杂正则 → 直接路由
  Layer 3: qwen2.5:0.5b 分类 → 兜底

后端:
  - 本地: Ollama on :11434, model qwen3:14b
  - 远程: 阿里云 DashScope, model qwen3.6-plus, 超时 3s → fallback

提供 create_app() 工厂函数，可由 handler 或独立进程调用。
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Tuple

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger


# ── 加载 .env ──

_ENV_PATH = Path(__file__).parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    logger.info(f"[LLM Proxy] Loaded env from {_ENV_PATH}")


# ── 默认规则文件路径 ──

_RULES_PATH = Path(__file__).parent / "llm_proxy_rules.yaml"


# ── 规则加载 ──

def load_rules(path: Optional[str] = None) -> dict:
    p = Path(path) if path else _RULES_PATH
    with open(p) as f:
        return yaml.safe_load(f.read())


# ── Layer 1: 敏感词 ──

def _check_sensitive(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)


# ── Layer 2: 正则匹配 ──

def _match_patterns(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


# ── Layer 3: 小模型分类 ──

def _classify_with_model(text: str, rules: dict) -> str:
    """调用 qwen2.5:0.5b 分类，返回 'simple' | 'complex' | 'error'"""
    cfg = rules["classifier"]
    prompt = cfg["prompt"].replace("{text}", text)
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "你是一个分类器。只回答一个词。"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 10,
        "temperature": 0,
        "stream": False,
    }
    try:
        resp = httpx.post(
            f"{cfg['api_url']}/chat/completions",
            json=body,
            timeout=cfg["timeout"],
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
        if "complex" in answer:
            return "complex"
        return "simple"
    except Exception as e:
        logger.warning(f"[LLM Proxy] Classifier failed: {e}, defaulting to simple")
        return "simple"


# ── 主分类函数 ──

def classify(messages: List[dict], rules: dict) -> Tuple[str, str, dict]:
    """
    返回 (backend, reason, extra_info)
    backend: 'local' | 'remote'
    reason: 'sensitive' | 'simple_regex' | 'complex_regex' | 'simple_model' | 'complex_model' | 'default'
    """
    # 提取最后一条用户消息
    user_texts = []
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                user_texts.append(content)
            elif isinstance(content, list):
                user_texts.append(" ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                ))
        if len(user_texts) >= 2:
            break
    current_text = user_texts[0] if user_texts else ""
    context_text = " ".join(user_texts)

    info = {"method": "default"}

    # ★ 系统内部消息（观测、摘要、状态）不分类，直接走本地
    if current_text.startswith("<observation") or current_text.startswith("<environment") or \
       current_text.startswith("<dialogue") or current_text.startswith("<rehydrated") or \
       current_text.startswith("<background"):
        info["method"] = "system_message"
        return "local", "system_message", info

    # Layer 1: 敏感词
    if _check_sensitive(context_text, rules.get("sensitive_keywords", [])):
        info["method"] = "sensitive_keyword"
        return "local", "sensitive", info

    # Layer 2: 正则
    if _match_patterns(current_text, rules.get("simple_patterns", [])):
        info["method"] = "simple_regex"
        return "local", "simple_regex", info

    if _match_patterns(current_text, rules.get("complex_patterns", [])):
        info["method"] = "complex_regex"
        return "remote", "complex_regex", info

    # Layer 3: 小模型
    if rules.get("classifier", {}).get("enabled", False):
        start = time.monotonic()
        result = _classify_with_model(current_text, rules)
        elapsed = (time.monotonic() - start) * 1000
        info["method"] = f"model_{result}"
        info["model_ms"] = round(elapsed)
        if result == "complex":
            return "remote", "complex_model", info
        return "local", "simple_model", info

    return "local", "default", info


# ── 流式转发 ──

async def _proxy_stream(backend_url: str, payload: dict, timeout: float, headers: dict = None):
    """转发到后端并流式返回 SSE 事件"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        async with client.stream("POST", backend_url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.error(f"[LLM Proxy] Backend {backend_url} returned {resp.status_code}: {error_body[:200]}")
                yield f"data: {json.dumps({'error': f'Backend error {resp.status_code}'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            async for line in resp.aiter_lines():
                if line:
                    yield line + "\n"
                else:
                    yield "\n"


def _extract_model(payload: dict, rules: dict, backend_key: str) -> Tuple[dict, dict]:
    """注入后端实际模型名，返回 (json_payload, http_headers)"""
    new_payload = dict(payload)
    headers = {}
    backend_cfg = rules["backends"][backend_key]
    if backend_cfg.get("model"):
        new_payload["model"] = backend_cfg["model"]
    if backend_key == "remote":
        api_key = os.getenv(backend_cfg.get("api_key_env", ""), "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return new_payload, headers


# ── FastAPI 应用 ──

def create_app(rules_path: Optional[str] = None) -> FastAPI:
    rules = load_rules(rules_path)
    cfg = rules["proxy"]

    # ★ 预热分类器模型（首次加载耗时 10-30 秒，之后 <100ms）
    _warmup_classifier(rules)

    app = FastAPI(title="LLM Proxy", docs_url=None, redoc_url=None)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        t0 = time.monotonic()
        messages = body.get("messages", [])

        # 分类
        backend, reason, info = classify(messages, rules)
        backend_cfg = rules["backends"].get(backend, rules["backends"]["local"])
        backend_url = f"{backend_cfg['url']}/chat/completions"

        current_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                current_text = (m.get("content", "") or "")[:60]
                break

        logger.info(
            f"[LLM Proxy] route={reason} backend={backend_cfg['label']} "
            f"text='{current_text}' delay={info.get('model_ms', 0)}ms"
        )

        # 敏感词额外标记 + 日志脱敏
        if reason == "sensitive":
            logger.info("[LLM Proxy] SENSITIVE content detected, routing to local only (no remote logging)")

        # 流式转发
        is_stream = body.get("stream", False)
        timeout = backend_cfg.get("timeout", 30)
        payload, http_headers = _extract_model(body, rules, backend)

        if not is_stream:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                resp = await client.post(backend_url, json=payload, headers=http_headers)
                if resp.status_code != 200:
                    if backend == "remote":
                        logger.warning(f"[LLM Proxy] API failed, falling back to local")
                        fallback_url = f"{rules['backends']['local']['url']}/chat/completions"
                        fallback_payload, _ = _extract_model(body, rules, "local")
                        resp2 = await client.post(fallback_url, json=fallback_payload)
                        return JSONResponse(content=resp2.json(), status_code=resp2.status_code)
                    return JSONResponse(content={"error": f"Backend {resp.status_code}"}, status_code=resp.status_code)
                return JSONResponse(content=resp.json(), status_code=resp.status_code)

        # 流式
        async def stream_with_fallback():
            try:
                async for chunk in _proxy_stream(backend_url, payload, timeout, headers=http_headers):
                    yield chunk
            except Exception as e:
                logger.warning(f"[LLM Proxy] Stream error: {e}, falling back to local")
                fallback_url = f"{rules['backends']['local']['url']}/chat/completions"
                fallback_payload, _ = _extract_model(body, rules, "local")
                async for chunk in _proxy_stream(fallback_url, fallback_payload, 60):
                    yield chunk

        elapsed = (time.monotonic() - t0) * 1000
        logger.debug(f"[LLM Proxy] Request completed in {elapsed:.0f}ms route={reason}")

        return StreamingResponse(
            stream_with_fallback(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Proxy-Backend": backend_cfg["label"],
                "X-Proxy-Reason": reason,
            },
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "port": cfg["port"]}

    return app


# ── 预热 ──

def _warmup_classifier(rules: dict):
    """发送一条预热请求，让 Ollama 将模型加载到内存。"""
    if not rules.get("classifier", {}).get("enabled", False):
        return
    cfg = rules["classifier"]
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": "warmup"}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    try:
        logger.info(f"[LLM Proxy] Warming up classifier model {cfg['model']}...")
        resp = httpx.post(
            f"{cfg['api_url']}/chat/completions",
            json=body,
            timeout=60.0,  # 首次加载可能很久
        )
        resp.raise_for_status()
        logger.info(f"[LLM Proxy] Classifier warmup complete")
    except Exception as e:
        logger.warning(f"[LLM Proxy] Classifier warmup failed (will retry on first use): {e}")


# ── 便捷启动 ──

def start_proxy(rules_path: Optional[str] = None, host: str = "127.0.0.1", port: int = 11440):
    """启动代理服务（阻塞调用）"""
    import uvicorn
    app = create_app(rules_path)
    logger.info(f"[LLM Proxy] Starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def start_proxy_in_thread(rules_path: Optional[str] = None, host: str = "127.0.0.1", port: int = 11440):
    """在后台线程启动代理（非阻塞）"""
    import threading
    import uvicorn

    app = create_app(rules_path)

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True, name="llm-proxy")
    t.start()
    logger.info(f"[LLM Proxy] Started in background on {host}:{port}")
    return t


# ── CLI ──

if __name__ == "__main__":
    start_proxy()
