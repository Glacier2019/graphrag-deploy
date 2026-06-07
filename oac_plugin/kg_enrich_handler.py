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
        t0 = time.monotonic()
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
            elapsed = 1000 * (time.monotonic() - t0)
            logger.info(f"[KgEnrich] MCP call {tool_name} → {resp.status_code} in {elapsed:.0f}ms")
            resp.raise_for_status()
            body = resp.json()
            if body.get("error"):
                return {"error": str(body["error"])}
            text = body["result"]["content"][0]["text"]
            return json.loads(text)
        except Exception as e:
            elapsed = 1000 * (time.monotonic() - t0)
            logger.warning(f"[KgEnrich] MCP error ({tool_name}, {elapsed:.0f}ms): {e}")
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


def _unwrap(row) -> dict:
    """AGE 查询结果转换为字典"""
    raw = row.get("result", row)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return raw if isinstance(raw, dict) else {}


def _extract_keywords(text: str) -> List[str]:
    """从用户输入提取可能的实体名（英文+中文）。兼容中英文混合。"""
    candidates = []
    # 英文专有名词
    for m in re.finditer(r'(?:^|[\s，。！？,.!?])([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text):
        candidates.append(m.group(1))
    for m in re.finditer(r'(?:[\u4e00-\u9fff])([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text):
        candidates.append(m.group(1))
    # ★ 中文实体（工程/科研/教育领域）
    zh_entities = [
        r'(公路|桥梁|隧道|路基|路面|交叉口|立交|涵洞|边坡|挡墙|圆曲线|纵坡|缓和曲线|竖曲线)',
        r'(高速|一级|二级|三级|四级).{0,2}公路',
        r'(设计|施工|养护|管理|检测|评定|加固).{0,3}(标准|规范|规程|指南|手册)',
        r'(高考|中考|分数|录取|志愿|学校|考试|真题|模拟)',
        r'(语文|数学|英语|物理|化学|生物|历史|地理|政治)',
    ]
    for p in zh_entities:
        for m in re.finditer(p, text):
            candidates.append(m.group())
    seen = set()
    result = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            result.append(c)
    return result[:5]  # ★ 3→5，中文实体更多


# ── LightRAG 客户端（直连原生 REST API）──

class _LightRagClient:
    """HTTP 调用 LightRAG 内置 Server。使用 subprocess+curl 绕开 Python GIL 竞争。
    用临时文件代替管道 I/O，消除 subprocess communicate() 的 GIL 等待。"""

    def __init__(self, base_url: str = "http://localhost:9621", timeout: float = 2.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def _call(self, path: str, method: str = "POST", json_data: dict = None):
        """HTTP API 调用，兼容 GraphRAG MCP 和 LightRAG REST"""
        t0 = time.monotonic()
        try:
            if method == "GET":
                resp = self._session.get(
                    f"{self._base_url}{path}",
                    timeout=self._timeout,
                )
            else:
                resp = self._session.post(
                    f"{self._base_url}{path}",
                    json=json_data,
                    timeout=self._timeout,
                )
            resp.raise_for_status()
            text = resp.text
            if not text:
                return {}
            return json.loads(text)
        except Exception as e:
            elapsed = 1000 * (time.monotonic() - t0)
            logger.warning(f"[KgEnrich] LightRAG call error ({path}, {elapsed:.0f}ms): {e}")
            return {}

    def search(self, query: str, mode: str = "mix", only_need_context: bool = False) -> str:
        try:
            payload = {"query": query, "mode": mode, "only_need_context": only_need_context}
            logger.info(f"[KgEnrich] LightRAG HTTP POST {self._base_url}/query mode={mode}")
            t0 = time.monotonic()  # ★ 在 logger 之后计时，避免管道阻塞影响
            body = self._call("/query", "POST", payload)
            t1 = time.monotonic()
            data = body.get("response", "")
            logger.info(f"[KgEnrich] LightRAG HTTP 200, response_len={len(data)}, http={(t1-t0)*1000:.0f}ms json=0ms")
            if not data:
                return ""
            chunks = []
            for m in re.finditer(r'"content"\s*:\s*"([^"]*)"', data):
                chunks.append(m.group(1))
            if chunks:
                result = "LightRAG检索结果:\n" + "\n".join(f"  - {c[:200]}" for c in chunks[:5])
                logger.info(f"[KgEnrich] LightRAG parsed {len(chunks)} chunks")
                return result
            logger.warning(f"[KgEnrich] LightRAG no chunks parsed, raw data len={len(data)}")
            return data[:600]
        except Exception as e:
            logger.warning(f"[KgEnrich] LightRAG query error: {e}")
            return ""

    def stats(self) -> dict:
        try:
            labels = self._call("/graph/label/list", "GET")
            counts_data = self._call("/documents/status_counts", "GET")
            counts = {}
            sc = counts_data.get("status_counts", {})
            total = sc.get("all", 0) - sc.get("pending", 0) - sc.get("parsing", 0)
            counts["documents"] = max(total, 0)
            counts["entities"] = len(labels) if isinstance(labels, list) else 0
            return counts
        except Exception as e:
            logger.warning(f"[KgEnrich] LightRAG stats error: {e}")
            return {}


# ── 知识库路由 ──

def _select_kb(text: str, engine: str, mcp_kb: dict, lightrag_kb: dict, default_mcp):
    """根据问题特征和引擎类型选择知识库。返回 (kb_name, mcp_client, lightrag_client)。
    engine=graphrag → GraphRAG MCP 优先（准确），LightRAG 回落
    engine=lightrag → LightRAG 优先（快），GraphRAG MCP 回落
    """
    def _route_rg(kb_name: str):
        """engine=graphrag: MCP 优先；engine=lightrag: LightRAG 优先"""
        mc = mcp_kb.get(kb_name)
        lr = lightrag_kb.get(f"lightrag_{kb_name}")
        if engine == "graphrag":
            if mc: return (kb_name, mc, None)
            if lr: return (kb_name, None, lr)
        if engine == "lightrag":
            if lr: return (kb_name, None, lr)
            if mc: return (kb_name, mc, None)
        # fallback: 哪个可用用哪个
        if mc: return (kb_name, mc, None)
        if lr: return (kb_name, None, lr)
        return (kb_name, None, None)

    # 科研/工程类 → research
    if _match_patterns(text, [
        r'(公路|桥梁|隧道|标准|规范|工程|技术|施工|设计|交通|路基|路面|交叉|渐变|超高|立交|涵洞|边坡|挡墙|圆曲线|纵坡|缓和曲线|竖曲线)',
        r'(JTG|JTG_B|公路工程)',
    ]):
        return _route_rg("research")
    # 高考/教育类 → gaokao
    if _match_patterns(text, [
        r'(高考|分数|录取|志愿|学校|考试|真题|模拟)',
        r'(语文|数学|英语|物理|化学|生物|历史|地理|政治)',
    ]):
        return _route_rg("gaokao")
    # 默认 → podcast
    result = _route_rg("podcast")
    if result[1] or result[2]:
        return result
    return ("default", default_mcp, None)


# ── RAG 引擎路由 ──

def _detect_rag_engine(text: str, available_kbs: dict = None) -> str:
    """根据问题特征和可用 KB 选择引擎。
    先判断问题适合什么引擎 → 再看目标 KB 有没有 → 没有则直接用可用的。
    available_kbs: {kb_name: {'graphrag': bool, 'lightrag': bool}, ...}
    """
    # ★ 手动指定优先
    if re.search(r'^\s*@graphrag', text, re.I):
        return "graphrag"
    if re.search(r'^\s*@lightrag', text, re.I):
        return "lightrag"

    # 概念/解释类 → LightRAG 优先
    lightrag_patterns = [
        r'(什么是|指什么|的定义|的概念|是什么|什么意思|指的是|值什么|参数|指标|规定|要求|是多少)',
        r'(摘要|总结|概括)',
        r'(为什么.{2,10}(原因|原理|机制))',
        r'(lightrag|light.rag|向量检索|语义搜索)',
    ]
    is_concept = any(re.search(p, text, re.I) for p in lightrag_patterns)

    # GraphRAG 匹配（关系/计数/实体/工程关键词）
    graphrag_patterns = [
        r'(关系|关联|连接|路径|网络|相关|有关|联系)',
        r'(第.*篇|第一篇|第二篇|第三篇)',
        r'(几|多少).*(篇|个|条).*(文章|文档|实体|关系)',
        r'(数据库|知识库|图谱).*(概况|统计|总览)',
        r'(graphrag|graph.rag|图数据库|图索引|图引擎)',
        r'(公路|桥梁|隧道|路基|路面|交叉|渐变|超高|立交|涵洞|边坡|挡墙|圆曲线|纵坡|缓和曲线|竖曲线|交通|设计|施工|标准|规范)',
    ]
    has_entity_word = any(re.search(p, text, re.I) for p in graphrag_patterns)

    # 实体名 → GraphRAG
    en_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    if en_names:
        return "graphrag"

    # ★ 概念/解释类：LightRAG 绝对优先（语义搜索定义/概念最擅长）
    if is_concept:
        return "lightrag"

    # ★ 工程/实体类：GraphRAG 优先（图数据库查关系/实体最擅长）
    if has_entity_word:
        return "graphrag"

    return "graphrag"  # 默认


def _match_patterns(text: str, patterns: list) -> bool:
    """检查文本是否匹配任一正则模式"""
    return any(re.search(p, text) for p in patterns)


# ── 通用问题检测 ──

_GENERIC_PATTERNS = [
    (r'(几|多少).*篇.*(文章|文档)', 'doc_count'),
    (r'(几|多少).*(实体|节点|人|组织)', 'entity_count'),
    (r'(几|多少).*条.*(关系|关联)', 'relation_count'),
    (r'(有哪些|列出|所有).*(文章|文档|实体|类型)', 'list'),
    (r'(数据库|知识库|图谱).*(概况|统计|总览|概览)', 'overview'),
    (r'(文章|文档).*(讲了|是什么|关于|内容|介绍|说)', 'doc_content'),
    (r'how many.*(articles?|documents?|entities?|nodes?)', 'doc_count'),
    (r'overview|summary|stats?.*(database|knowledge)', 'overview'),
    (r'what.*(articles?|documents?).*about', 'doc_content'),
    # ★ 工程/标准类（research KB）
    (r'(标准|规范|规程|指南).*(要求|规定|指标|参数|数值|是多少|是什么)', 'doc_content'),
    (r'(公路|桥梁|隧道|路基|路面).*(设计|施工|标准|规范|要求)', 'doc_content'),
    (r'(定义|术语|分类).{0,10}(标准|规范)', 'doc_content'),
    # ★ 教育/考试类（gaokao KB）
    (r'(考试|试卷|真题).*(内容|答案|解析)', 'doc_content'),
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
    graphrag_enabled: bool = True  # ★ false=停用知识图谱，纯透传
    mcp_url: str = "http://localhost:8011"
    mcp_timeout: float = 3.0
    max_entities: int = 2
    max_relations: int = 3
    context_max_chars: int = 400
    min_interval_seconds: float = 20.0
    # 双 RAG 引擎
    rag_engine: str = "graphrag"          # "graphrag" | "lightrag" | "auto"
    lightrag_timeout: float = 2.0
    # ★ GraphRAG 知识库开关 + MCP 地址
    kb_podcast_enabled: bool = True
    kb_podcast_url: str = "http://localhost:8011"
    kb_research_enabled: bool = False
    kb_research_url: str = "http://localhost:8012"
    kb_gaokao_enabled: bool = False
    kb_gaokao_url: str = "http://localhost:8013"
    # ★ LightRAG 知识库开关 + 实例地址
    kb_lightrag_research_enabled: bool = True
    kb_lightrag_research_url: str = "http://localhost:9621"
    kb_lightrag_podcast_enabled: bool = False
    kb_lightrag_podcast_url: str = "http://localhost:9622"
    kb_lightrag_gaokao_enabled: bool = False
    kb_lightrag_gaokao_url: str = "http://localhost:9623"
    # LLM 代理
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
        self._mcp: Optional[_McpClient] = None  # 主 MCP，保持兼容
        self._mcp_kb: Dict[str, _McpClient] = {}  # ★ 多 GraphRAG KB MCP 客户端
        self._lightrag: Optional[_LightRagClient] = None  # 主 LightRAG，保持兼容
        self._lightrag_kb: Dict[str, _LightRagClient] = {}  # ★ 多 LightRAG KB 客户端
        self._cfg: Optional[KgEnrichConfig] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(config_model=KgEnrichConfig)

    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        global _proxy_started
        self._cfg = handler_config if isinstance(handler_config, KgEnrichConfig) else KgEnrichConfig()
        self._mcp = _McpClient(base_url=self._cfg.mcp_url, timeout=self._cfg.mcp_timeout)
        # ★ 多知识库 MCP 客户端
        self._mcp_kb = {}
        mcp_timeout = self._cfg.mcp_timeout
        if self._cfg.kb_podcast_enabled:
            self._mcp_kb["podcast"] = _McpClient(base_url=self._cfg.kb_podcast_url, timeout=mcp_timeout)
            logger.info(f"[KgEnrich] KB podcast ready: {self._cfg.kb_podcast_url}")
        if self._cfg.kb_research_enabled:
            self._mcp_kb["research"] = _McpClient(base_url=self._cfg.kb_research_url, timeout=mcp_timeout)
            logger.info(f"[KgEnrich] KB research ready: {self._cfg.kb_research_url}")
        if self._cfg.kb_gaokao_enabled:
            self._mcp_kb["gaokao"] = _McpClient(base_url=self._cfg.kb_gaokao_url, timeout=mcp_timeout)
            logger.info(f"[KgEnrich] KB gaokao ready: {self._cfg.kb_gaokao_url}")
        # ★ 多 LightRAG 客户端
        lightrag_timeout = self._cfg.lightrag_timeout
        self._lightrag_kb = {}
        if self._cfg.kb_lightrag_research_enabled:
            self._lightrag_kb["lightrag_research"] = _LightRagClient(
                base_url=self._cfg.kb_lightrag_research_url, timeout=lightrag_timeout)
            logger.info(f"[KgEnrich] LightRAG research ready: {self._cfg.kb_lightrag_research_url}")
        if self._cfg.kb_lightrag_podcast_enabled:
            self._lightrag_kb["lightrag_podcast"] = _LightRagClient(
                base_url=self._cfg.kb_lightrag_podcast_url, timeout=lightrag_timeout)
            logger.info(f"[KgEnrich] LightRAG podcast ready: {self._cfg.kb_lightrag_podcast_url}")
        if self._cfg.kb_lightrag_gaokao_enabled:
            self._lightrag_kb["lightrag_gaokao"] = _LightRagClient(
                base_url=self._cfg.kb_lightrag_gaokao_url, timeout=lightrag_timeout)
            logger.info(f"[KgEnrich] LightRAG gaokao ready: {self._cfg.kb_lightrag_gaokao_url}")
        # 主 LightRAG 指向 research（保持兼容）
        self._lightrag = self._lightrag_kb.get("lightrag_research", None)
        mode = self._cfg.output_mode
        logger.info(f"[KgEnrich] loaded, engine={self._cfg.rag_engine}, mode={mode}, "
                    f"graphRAG_kb={list(self._mcp_kb.keys())}, lightRAG_kb={list(self._lightrag_kb.keys())}")
        if mode == "perception":
            self._output_type = ChatDataType.PERCEPTION_CONTEXT
            self._display_type = None
        else:
            self._output_type = ChatDataType.AUGMENTED_HUMAN_TEXT
            self._display_type = ChatDataType.HUMAN_TEXT

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

        # ★ LightRAG 预热：后台线程发送 ping 查询，避免首次查询 1300ms 冷启动
        def _warmup_lightrag():
            warmup_query = "test"
            for name, client in list(self._lightrag_kb.items()):
                try:
                    t0 = time.monotonic()
                    client.search(warmup_query, mode="naive", only_need_context=True)
                    elapsed = (time.monotonic() - t0) * 1000
                    logger.info(f"[KgEnrich] LightRAG warmup {name}: {elapsed:.0f}ms")
                except Exception as e:
                    logger.warning(f"[KgEnrich] LightRAG warmup {name} failed: {e}")
        threading.Thread(target=_warmup_lightrag, daemon=True, name="lr-warmup").start()

        # ★ GraphRAG MCP 预热：后台线程发送 ping 查询
        def _warmup_mcp():
            for name, client in list(self._mcp_kb.items()):
                try:
                    t0 = time.monotonic()
                    client._call("age_cypher_query", {
                        "cypher": "MATCH (e:Entity) RETURN { total: count(e) } AS result LIMIT 1"
                    })
                    elapsed = (time.monotonic() - t0) * 1000
                    logger.info(f"[KgEnrich] MCP warmup {name}: {elapsed:.0f}ms")
                except Exception as e:
                    logger.warning(f"[KgEnrich] MCP warmup {name} failed: {e}")
        threading.Thread(target=_warmup_mcp, daemon=True, name="mcp-warmup").start()

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
        t0 = time.monotonic()
        ctx = context
        text = inputs.data.get_main_data() if inputs.data else ""

        logger.info(f"[KgEnrich] >>> HANDLE START type={inputs.type} is_last={inputs.is_last_data} text_len={len(text) if isinstance(text, str) else type(text).__name__}")

        if not isinstance(text, str):
            logger.warning(f"[KgEnrich] Non-string data received: type={type(text).__name__}, skipping")
            return

        # 积累流式中间片
        ctx.input_buffer += text
        if not inputs.is_last_data:
            logger.info(f"[KgEnrich] <<< HANDLE END (partial, waiting for last) +{len(text)}ms total_buffer={len(ctx.input_buffer)}")
            return

        full_text = ctx.input_buffer.strip()
        ctx.input_buffer = ""
        t1 = time.monotonic()

        if not full_text:
            logger.info(f"[KgEnrich] <<< HANDLE END (empty text)")
            return

        logger.info(f"[KgEnrich] processing text='{full_text[:60]}'")

        if self._cfg.output_mode == "perception":
            kg_text = self._build_kg_context(ctx, full_text)
            if kg_text:
                perception = json.dumps({
                    "scene_summary": kg_text,
                    "source": "knowledge_graph",
                    "timestamp": time.time(),
                }, ensure_ascii=False)
                self._emit(context, perception, output_definitions, finish=True)
        else:
            t2 = time.monotonic()
            enriched = self._enrich_text(ctx, full_text)
            t3 = time.monotonic()
            self._emit(context, enriched, output_definitions, finish=True)
            t4 = time.monotonic()
            logger.info(f"[KgEnrich] <<< HANDLE END enrich={(t3-t2)*1000:.0f}ms emit={(t4-t3)*1000:.0f}ms total={(t4-t0)*1000:.0f}ms result_len={len(enriched)}")

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
        """查询知识图谱，返回格式化上下文。根据 rag_engine 配置路由到不同后端。"""
        t0 = time.monotonic()
        if not self._cfg:
            logger.warning(f"[KgEnrich] _query_kg: no config")
            return ""
        if not self._cfg.graphrag_enabled:
            logger.warning(f"[KgEnrich] _query_kg: graphrag_enabled=False, SKIPPING!")
            return ""

        now = time.time()
        if now - ctx.last_time < self._cfg.min_interval_seconds:
            logger.info(f"[KgEnrich] _query_kg: throttled ({(now-ctx.last_time):.1f}s < {self._cfg.min_interval_seconds}s)")
            return ""
        ctx.last_time = now

        # ★ 闲聊/视觉/打断类 → 跳过所有 KB
        if _match_patterns(text, [
            r'^(停|暂停|等一下|嗯|哦|啊|好|是的|对|行|喂|听|你听|听见).{,5}$',
            r'(看到|看见|摄像头|镜头|你能|你.*看)',
            r'^(.{1,3})$',  # 1-3字超短句
        ]):
            logger.info(f"[KgEnrich] skip_kb: casual/visual for '{text[:40]}'")
            return ""

        # ★ 双 RAG 引擎路由
        engine = self._cfg.rag_engine
        if engine == "auto":
            engine = _detect_rag_engine(text)
        logger.info(f"[KgEnrich] engine={engine} for text='{text[:50]}'")

        if engine == "lightrag":
            # ★ KB路由（LightRAG优先）
            kb_name, mc, lr_client = _select_kb(text, "lightrag", self._mcp_kb, self._lightrag_kb, self._mcp)
            # 实体/计数类优先 GraphRAG（即使 engine=lightrag）
            if mc and _match_patterns(text, [
                r'(几|多少).*(篇|个|条|文章|文档|实体|关系)',
                r'(第.*篇|第一篇|第二篇)',
                r'[A-Z][a-z].*[A-Z][a-z]',
                r'(关系|关联|连接|有关|相关)',
            ]):
                logger.info(f"[KgEnrich] KB {kb_name}: entity query, using GraphRAG instead")
                return self._query_graphrag(ctx, text, mc)
            logger.info(f"[KgEnrich] KB selected: {kb_name} [LightRAG], routing")
            try:
                result = self._query_lightrag(ctx, text, client=lr_client or self._lightrag)
            except Exception as e:
                logger.error(f"[KgEnrich] _query_lightrag crashed: {e}")
                result = ""
            if not result and mc:
                logger.info(f"[KgEnrich] LightRAG failed, fallback to GraphRAG")
                return self._query_graphrag(ctx, text, mc)
            return result
        elif engine == "graphrag":
            # ★ KB路由（GraphRAG MCP 优先）
            kb_name, mc, lr = _select_kb(text, "graphrag", self._mcp_kb, self._lightrag_kb, self._mcp)
            # 双引擎可用时：实体/计数/关系类优先 GraphRAG
            if mc and lr and _match_patterns(text, [
                r'(几|多少).*(篇|个|条|文章|文档|实体|关系)',
                r'(第.*篇|第一篇|第二篇)',
                r'[A-Z][a-z].*[A-Z][a-z]',  # 英文学体名
                r'(关系|关联|连接|有关|相关)',
            ]):
                logger.info(f"[KgEnrich] KB {kb_name}: both available, using GraphRAG (structured query)")
                return self._query_graphrag(ctx, text, mc)
            # 有 LightRAG 且适合语义搜索 → LightRAG
            if lr:
                logger.info(f"[KgEnrich] KB {kb_name}: using LightRAG (semantic search)")
                return self._query_lightrag(ctx, text, client=lr)
            # 无 LightRAG → GraphRAG
            if mc:
                logger.info(f"[KgEnrich] KB selected: {kb_name} [GraphRAG], routing")
                return self._query_graphrag(ctx, text, mc)
        else:
            logger.warning(f"[KgEnrich] Unknown rag_engine: {engine}")
            return ""

    def _query_graphrag(self, ctx: KgEnrichContext, text: str, mcp: _McpClient = None) -> str:
        """GraphRAG (PostgreSQL AGE) 查询"""
        t0 = time.monotonic()
        client = mcp or self._mcp
        logger.info(f"[KgEnrich] _query_graphrag entry, mcp_ok={client is not None}")
        if not client:
            logger.warning(f"[KgEnrich] MCP client not available")
            return ""

        # 先检测通用统计/概览类问题
        qtype = _detect_generic_query(text)
        logger.info(f"[KgEnrich] _query_graphrag qtype={qtype}, checking entities next")
        if qtype:
            result = _query_generic(client, qtype)
            if result:
                logger.info(f"[KgEnrich] Generic query ({qtype}) → {result[:80]}")
                return result

        # 再检测实体名
        logger.info(f"[KgEnrich] _query_graphrag checking entities for '{text[:30]}'")
        keywords = _extract_keywords(text)
        logger.info(f"[KgEnrich] Entity keywords: {keywords} from text='{text[:50]}'")
        if not keywords:
            return ""

        parts = []
        total = 0
        limit = self._cfg.context_max_chars

        for kw in keywords[:self._cfg.max_entities]:
            entities = client.entity_lookup(kw, limit=1)
            if not entities:
                continue

            e = entities[0]
            e_name = e.get("title", kw)
            e_type = e.get("type", "").lower()
            e_desc = (e.get("description") or "")[:200]

            entry = f"[{e_name}] ({e_type}): {e_desc}"
            total += len(entry)

            rels = client.entity_relations(e_name, self._cfg.max_relations)
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
        elapsed = 1000 * (time.monotonic() - t0)
        logger.info(f"[KgEnrich] KG query for: {keywords[:2]}, entities: {len(entities)}, elapsed={elapsed:.0f}ms")
        return result

    def _query_lightrag(self, ctx: KgEnrichContext, text: str, client=None) -> str:
        """LightRAG 查询（向量语义搜索）。client 可选指定特定 KB 实例。"""
        lr = client or self._lightrag
        logger.info(f"[KgEnrich] _query_lightrag entry, has_client={lr is not None}")
        if not lr:
            return ""

        # 通用查询（统计/计数）映射到 lightrag_stats
        qtype = _detect_generic_query(text)
        if qtype in ("doc_count", "entity_count", "relation_count", "overview", "list"):
            stats = lr.stats()
            if not stats:
                return ""
            parts = []
            if stats.get("documents"):
                parts.append(f"知识库共有 {stats['documents']} 篇文章/文档")
            if stats.get("entities"):
                parts.append(f"知识图谱共有 {stats['entities']} 个实体")
            if stats.get("relations"):
                parts.append(f"知识图谱共有 {stats['relations']} 条关系")
            result = "\n".join(parts) if parts else ""
            if result:
                logger.info(f"[KgEnrich] LightRAG stats → {result[:80]}")
            return result

        # 文档内容查询 → lightrag_search
        # 文档内容查询 → lightrag_search（始终用 naive，mix 在单KB场景太慢）
        mode = "naive"
        result = lr.search(text, mode=mode, only_need_context=True)
        if result:
            result = f"知识图谱参考（LightRAG）:\n{result[:self._cfg.context_max_chars]}"
            logger.info(f"[KgEnrich] LightRAG search({mode}) → {result[:80]}")
            return result
        logger.warning(f"[KgEnrich] LightRAG search({mode}) returned empty for '{text[:40]}'")
        return ""

    def destroy_context(self, context):
        pass
