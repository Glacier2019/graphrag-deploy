"""
KgEnrichHandler v2 — 知识图谱上下文注入（完全在 git 仓库外）

位置: /data/cy/graphrag-deploy/oac_plugin/kg_enrich_handler.py

两种输出模式（由 config 选择）：

  output_mode: "text"       → 在用户消息中注入 KG 上下文（简单 pipeline 用）
  output_mode: "perception" → 输出为 PERCEPTION_CONTEXT，ChatAgent 纳入 L3 环境状态（agent 用）

依赖: requests（OpenAvatarChat 已有），无额外 pip 安装。

MCP: JSON-RPC POST http://host:8011/mcp
"""

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from chat_engine.common.handler_base import (
    HandlerBase,
    HandlerBaseInfo,
    HandlerDataInfo,
    HandlerDetail,
)
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import (
    ChatEngineConfigModel,
    HandlerBaseConfigModel,
)
from chat_engine.data_models.runtime_data.data_bundle import DataBundle
from chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition
from chat_engine.data_models.runtime_data.data_bundle import DataBundleEntry


# ── MCP 客户端 ──

class _McpClient:
    """HTTP JSON-RPC 调用 graphrag-deploy 的 MCP Server (容器 :8011)"""

    def __init__(self, base_url: str = "http://localhost:8011", timeout: float = 3.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def _call(self, tool_name: str, arguments: dict) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": 1,
        }
        try:
            resp = self._session.post(
                f"{self._base_url}/mcp",
                json=payload,
                headers={"Accept": "application/json"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            if body.get("error"):
                return {"error": str(body["error"])}
            text = body["result"]["content"][0]["text"]
            return json.loads(text)
        except Exception as e:
            logger.warning(f"[KgEnrich] MCP error: {e}")
            return {"error": str(e)}

    def entity_lookup(self, name: str, limit: int = 3) -> List[dict]:
        r = self._call("age_entity_lookup", {"name": name, "limit": limit})
        if "error" in r:
            return []
        return [_unwrap(row) for row in r.get("rows", [])]

    def entity_relations(self, entity_name: str, max_rel: int = 5) -> List[dict]:
        cypher = (
            f"MATCH (e:Entity {{title: '{entity_name}'}})-[r:RELATED_TO]-(o:Entity) "
            f"RETURN {{"
            f"  source: startNode(r).title, target: endNode(r).title, "
            f"  relation: r.description, weight: r.weight"
            f"}} AS result "
            f"ORDER BY r.weight DESC "
            f"LIMIT {max_rel}"
        )
        r = self._call("age_cypher_query", {"cypher": cypher})
        if "error" in r:
            return []
        return [_unwrap(row) for row in r.get("rows", [])]


def _unwrap(row: Any) -> dict:
    raw = row.get("result", row)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return raw if isinstance(raw, dict) else {}


# ── 实体提取 ──

def _extract_keywords(text: str) -> List[str]:
    candidates = []
    for m in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text):
        candidates.append(m.group())
    seen = set()
    result = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            result.append(c)
    return result[:3]


# ── 通用问题检测 ──

_GENERIC_PATTERNS = [
    (r'(几|多少).*篇.*(文章|文档)', 'doc_count'),
    (r'(几|多少).*(实体|节点|人|组织)', 'entity_count'),
    (r'(几|多少).*条.*(关系|关联)', 'relation_count'),
    (r'(有哪些|列出|所有).*(文章|文档|实体|类型)', 'list'),
    (r'(数据库|知识库|图谱).*(概况|统计|总览|概览)', 'overview'),
    (r'(文章|文档).*(讲了|是什么|关于|内容|介绍|说)', 'doc_content'),
    (r'(how many|count).*(articles?|documents?|entities?|nodes?)', 'doc_count'),
    (r'(overview|summary|stats?).*(database|knowledge)', 'overview'),
    (r'what.*(articles?|documents?).*about', 'doc_content'),
]


def _detect_generic_query(text: str) -> str:
    """检测通用统计/概览类问题，返回查询类型。"""
    lower = text.lower()
    for pattern, qtype in _GENERIC_PATTERNS:
        if re.search(pattern, lower):
            return qtype
    return ""


def _query_generic(mcp, qtype: str) -> str:
    """执行通用查询（计数、统计、列表）。"""
    try:
        if qtype in ('overview', 'list'):
            cypher = (
                "MATCH (e:Entity) "
                "RETURN e.type AS type, count(e) AS count "
                "ORDER BY count DESC"
            )
            rows = mcp._call("age_cypher_query", {"cypher": cypher})
            type_counts = []
            for r in rows.get("rows", []):
                raw = _unwrap(r)
                type_counts.append(f"{raw.get('type', '?')}: {raw.get('count', 0)} 个")
            doc_ct = mcp._call("age_cypher_query", {
                "cypher": "MATCH (d:Document) RETURN { total: count(d), titles: collect(d.title)[..5] } AS result"
            })
            doc_info = ""
            if doc_ct.get("rows"):
                d = _unwrap(doc_ct["rows"][0])
                doc_info = f"文档: {d.get('total', 0)} 篇"
            return f"知识图谱概况:\n  {doc_info}\n  实体分类: {', '.join(type_counts) if type_counts else '无'}"

        if qtype == 'doc_count':
            cypher = "MATCH (d:Document) RETURN { total: count(d) } AS result"
            rows = mcp._call("age_cypher_query", {"cypher": cypher})
            if rows.get("rows"):
                d = _unwrap(rows["rows"][0])
                return f"知识库共有 {d.get('total', 0)} 篇文章/文档"

        if qtype == 'entity_count':
            cypher = "MATCH (e:Entity) RETURN { total: count(e) } AS result"
            rows = mcp._call("age_cypher_query", {"cypher": cypher})
            if rows.get("rows"):
                d = _unwrap(rows["rows"][0])
                return f"知识图谱共有 {d.get('total', 0)} 个实体"

        if qtype == 'relation_count':
            cypher = "MATCH ()-[r:RELATED_TO]->() RETURN { total: count(r) } AS result"
            rows = mcp._call("age_cypher_query", {"cypher": cypher})
            if rows.get("rows"):
                d = _unwrap(rows["rows"][0])
                return f"知识图谱共有 {d.get('total', 0)} 条关系"

        if qtype == 'doc_content':
            cypher = (
                "MATCH (d:Document) "
                "RETURN { titles: collect(d.title), texts: collect(d.text)[..3] } AS result "
                "LIMIT 10"
            )
            rows = mcp._call("age_cypher_query", {"cypher": cypher})
            if rows.get("rows"):
                d = _unwrap(rows["rows"][0])
                titles = d.get("titles", [])
                texts = d.get("texts", [])
                if isinstance(titles, str):
                    try:
                        titles = json.loads(titles)
                    except Exception:
                        titles = [titles]
                parts = []
                parts.append(f"知识库共有 {len(titles)} 篇文章")
                for i, t in enumerate(titles[:5]):
                    preview = ""
                    if texts and i < len(texts):
                        txt = str(texts[i])[:100]
                        preview = f" — {txt}"
                    parts.append(f"  {i+1}. {t}{preview}")
                return "\n".join(parts)
            return ""

    except Exception as e:
        logger.warning(f"[KgEnrich] Generic query failed: {e}")
    return ""


# ── LLM 代理全局状态 ──

_proxy_started: bool = False
_proxy_lock = threading.Lock()


# ── Handler ──

class KgEnrichConfig(HandlerBaseConfigModel):
    enabled: bool = True
    output_mode: str = "text"  # "text" | "perception"
    graphrag_enabled: bool = True  # ★ false=停用知识图谱，纯透传，数字人不受影响
    mcp_url: str = "http://localhost:8011"
    mcp_timeout: float = 3.0
    max_entities: int = 2
    max_relations: int = 3
    context_max_chars: int = 400
    min_interval_seconds: float = 20.0
    # LLM 代理
    llm_proxy_enabled: bool = False
    llm_proxy_rules: str = ""      # rules YAML path
    llm_proxy_port: int = 11440
    # LLM Proxy config（全局仅启动一次）
    llm_proxy_enabled: bool = False
    llm_proxy_rules: str = "/data/cy/graphrag-deploy/oac_plugin/llm_proxy_rules.yaml"
    llm_proxy_port: int = 11440


class KgEnrichContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.last_time: float = 0.0
        self.input_buffer: str = ""


class KgEnrichHandler(HandlerBase):

    def __init__(self):
        super().__init__()
        self._mcp: Optional[_McpClient] = None
        self._cfg: Optional[KgEnrichConfig] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(config_model=KgEnrichConfig)

    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        global _proxy_started
        self._cfg = handler_config if isinstance(handler_config, KgEnrichConfig) else KgEnrichConfig()
        self._mcp = _McpClient(base_url=self._cfg.mcp_url, timeout=self._cfg.mcp_timeout)
        mode = self._cfg.output_mode
        if mode == "perception":
            self._output_type = ChatDataType.PERCEPTION_CONTEXT
            self._display_type = None
        else:
            self._output_type = ChatDataType.AUGMENTED_HUMAN_TEXT  # ChatAgent 消费
            self._display_type = ChatDataType.HUMAN_TEXT           # 前端UI显示
        logger.info(f"[KgEnrich] loaded, mode={mode}, output={self._output_type}, display={self._display_type}, MCP={self._cfg.mcp_url}")

        # ★ LLM 代理：在第一个 handler 实例加载时启动，全局只启动一次
        if self._cfg.llm_proxy_enabled:
            with _proxy_lock:
                if not _proxy_started:
                    try:
                        import uvicorn
                        from llm_proxy import create_app
                        app = create_app(self._cfg.llm_proxy_rules)
                        port = self._cfg.llm_proxy_port
                        t = threading.Thread(
                            target=lambda: uvicorn.run(
                                app, host="127.0.0.1", port=port,
                                log_level="warning",
                            ),
                            daemon=True,
                            name="llm-proxy",
                        )
                        t.start()
                        _proxy_started = True
                        logger.info(
                            f"[KgEnrich] LLM proxy started on "
                            f"http://127.0.0.1:{port}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[KgEnrich] Failed to start LLM proxy: {e}"
                        )

    def create_context(self, session_context, handler_config=None):
        return KgEnrichContext(session_context.session_info.session_id)

    def start_context(self, session_context, handler_context):
        pass

    def get_handler_detail(self, session_context, context):
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry(name="main_data"))
        outputs = {
            self._output_type: HandlerDataInfo(
                type=self._output_type,
                definition=definition,
            ),
        }
        if self._display_type and self._display_type != self._output_type:
            outputs[self._display_type] = HandlerDataInfo(
                type=self._display_type,
                definition=definition,
            )
        return HandlerDetail(
            inputs={
                ChatDataType.HUMAN_TEXT: HandlerDataInfo(type=ChatDataType.HUMAN_TEXT),
            },
            outputs={
                self._output_type: HandlerDataInfo(
                    type=self._output_type,
                    definition=definition,
                ),
            },
        )

    def handle(self, context, inputs, output_definitions):
        ctx = context
        text = inputs.data.get_main_data() if inputs.data else ""

        logger.info(f"[KgEnrich] handle() called: input_type={inputs.type}, is_last={inputs.is_last_data}, text_len={len(text) if isinstance(text, str) else type(text).__name__}")

        if not isinstance(text, str):
            logger.warning(f"[KgEnrich] Non-string data received: type={type(text).__name__}, skipping")
            return

        # 积累流式中间片
        ctx.input_buffer += text
        if not inputs.is_last_data:
            return

        full_text = ctx.input_buffer.strip()
        ctx.input_buffer = ""

        if not full_text:
            return

        if self._cfg.output_mode == "perception":
            # Agent 模式：输出 PERCEPTION_CONTEXT
            kg_text = self._build_kg_context(ctx, full_text)
            if kg_text:
                perception = json.dumps({
                    "scene_summary": kg_text,
                    "source": "knowledge_graph",
                    "timestamp": time.time(),
                }, ensure_ascii=False)
                self._emit(context, perception, output_definitions, finish=True)
            # 无KG数据时不输出，避免覆盖视觉perception
        else:
            # Text 模式：注入 KG 到用户消息
            enriched = self._enrich_text(ctx, full_text)
            # ★ AUGMENTED_HUMAN_TEXT → ChatAgent（单次响应）
            self._emit(context, enriched, output_definitions, finish=True)
            # UI 文字由 SemanticTurnDetector / RtcClient 的 HUMAN_TEXT 信号提供
            # 不需要额外输出

    def _emit(self, context, text: str, output_definitions, finish=False):
        if not text:
            return
        info = output_definitions.get(self._output_type)
        logger.info(f"[KgEnrich] _emit types: output_def_keys={list(output_definitions.keys()) if output_definitions else 'NONE'}, looking_for={self._output_type}")
        if info:
            bundle = DataBundle(info.definition)
            bundle.set_main_data(text)
            streamer = context.data_submitter.get_streamer(self._output_type)
            logger.info(f"[KgEnrich] _emit sending to {self._output_type}: '{text[:60]}'")
            streamer.stream_data(bundle, finish_stream=finish)

    def _emit_display(self, context, text: str, output_definitions, finish=False):
        """输出原始文本到 HUMAN_TEXT → 前端UI显示"""
        if not self._display_type or not text:
            return
        display_info = output_definitions.get(self._display_type)
        if not display_info:
            return
        bundle = DataBundle(display_info.definition)
        bundle.set_main_data(text)
        streamer = context.data_submitter.get_streamer(self._display_type)
        streamer.stream_data(bundle, finish_stream=finish)

    def _build_kg_context(self, ctx: KgEnrichContext, text: str) -> str:
        """构建知识图谱上下文字符串（perception 模式）"""
        kg_info = self._query_kg(ctx, text)
        if not kg_info:
            return ""
        return kg_info

    def _enrich_text(self, ctx: KgEnrichContext, text: str) -> str:
        """将 KG 上下文注入用户消息（text 模式）"""
        kg_info = self._query_kg(ctx, text)
        if not kg_info:
            return text
        return f"{kg_info}\n\n---\n用户问题: {text}"

    def _query_kg(self, ctx: KgEnrichContext, text: str) -> str:
        """查询知识图谱，返回格式化上下文"""
        if not self._cfg or not self._mcp:
            return ""
        # ★ 关闭图谱时不查询，纯透传，数字人不受影响
        if not self._cfg.graphrag_enabled:
            return ""

        now = time.time()
        if now - ctx.last_time < self._cfg.min_interval_seconds:
            return ""
        ctx.last_time = now

        # 先检测通用统计/概览类问题
        qtype = _detect_generic_query(text)
        if qtype:
            result = _query_generic(self._mcp, qtype)
            if result:
                logger.info(f"[KgEnrich] Generic query ({qtype}) → {result[:80]}")
                return result

        # 再检测实体名
        keywords = _extract_keywords(text)
        if not keywords:
            return ""

        parts = []
        total = 0
        limit = self._cfg.context_max_chars

        for kw in keywords[:self._cfg.max_entities]:
            entities = self._mcp.entity_lookup(kw, limit=1)
            if not entities:
                continue

            e = entities[0]
            e_name = e.get("title", kw)
            e_type = e.get("type", "").lower()
            e_desc = (e.get("description") or "")[:200]

            entry = f"[{e_name}] ({e_type}): {e_desc}"
            total += len(entry)

            rels = self._mcp.entity_relations(e_name, self._cfg.max_relations)
            if rels:
                rel_lines = []
                for r in rels[:self._cfg.max_relations]:
                    other = r.get("target") if r.get("source") == e_name else r.get("source")
                    if not other:
                        continue
                    rel_desc = (r.get("relation") or "")[:80]
                    line = f"  - {other}: {rel_desc}"
                    if total + len(line) < limit:
                        rel_lines.append(line)
                        total += len(line)
                if rel_lines:
                    entry += "\n" + "\n".join(rel_lines)

            parts.append(entry)

        if not parts:
            return ""

        header = "知识图谱参考（来自后台知识库）:\n"
        result = header + "\n".join(parts)
        logger.info(f"[KgEnrich] KG query for: {keywords[:2]}, entities: {len(entities)}")
        return result

    def destroy_context(self, context):
        pass
