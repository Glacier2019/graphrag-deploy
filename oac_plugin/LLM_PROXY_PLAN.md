# LLM 代理 — 完整设计文档与分步实现计划

## 架构总览

```
用户输入
    │
    ▼
ChatAgent (Ollama qwen3:14b, api_url → :11440)
    │
    ▼
LLM Proxy (:11440, 内嵌在 KgEnrich handler 中)
    │
    ├── Layer 1: 敏感词（正则, <1ms）──────→ Ollama 本地 (强制)
    ├── Layer 2: 简单/复杂词（正则, <1ms）─→ Ollama 本地 / 阿里云 API
    └── Layer 3: qwen2.5:0.5b 分类（~100ms）─→ 兜底路由
         │
         ├── Ollama 本地 (:11434, qwen3:14b)
         └── 阿里云 DashScope (qwen3.6-plus, 超时3s → fallback)
```

## 数据流（不改现有 KG 管道）

```
语音: Mic → VAD → ASR → SemanticTurnDetector(qwen3:8b) → HUMAN_AUDIO → KgEnrichVoice → AUGMENTED_HUMAN_TEXT → ChatAgent(qwen3:14b)
                                                                                                              │
键盘: RtcClient → HUMAN_TEXT → KgEnrichKeyboard → AUGMENTED_HUMAN_TEXT ──────────────────────────────────────────┘
                                                                                                              │
                                                                                                   POST :11440/v1/chat/completions
                                                                                                              │
                                                                                                        LLM Proxy
                                                                                                       /          \
                                                                                                   Ollama      阿里云API
```

## 文件清单

| 文件 | 位置 | 操作 | 是否改 git |
|------|------|------|-----------|
| `llm_proxy.py` | `/data/cy/graphrag-deploy/oac_plugin/` | 新建 | 否 |
| `llm_proxy_rules.yaml` | `/data/cy/graphrag-deploy/oac_plugin/` | 新建 | 否 |
| `kg_enrich_handler.py` | `/data/cy/graphrag-deploy/oac_plugin/` | 修改 | 否 |
| `chat_duplex_agent_kg_merged.yaml` | `/data/cy/graphrag-deploy/oac_plugin/` | 修改 1 行 | 否 |
| `chat_data_type.py` | `src/chat_engine/data_models/` | 已加 1 行 | 已改 |

## 模型选型

| 用途 | 模型 | 显存 |
|------|------|------|
| 端点检测 | qwen3:8b | 5.5 GB |
| 主对话 | qwen3:14b | 9 GB |
| 分类器 | qwen2.5:0.5b | 0.4 GB |
| FlashHead | SoulX-FlashHead-1.3B (FP8) | 1.5 GB |
| 其他 | SenseVoice + ONNX | ~1 GB |
| **合计** | | **~17.4 GB（32GB 很安全）** |

---

## 分步实现计划

### Step 1 — 创建分类规则配置 `llm_proxy_rules.yaml`

**目标**: 定义正则规则 + 分类模型配置 + 后端配置

**测试**: 用 Python 直接解析 YAML，验证结构正确

---

### Step 2 — 创建代理核心 `llm_proxy.py`

**目标**: 实现 FastAPI 代理，包含三层分类 + 流式转发

**模块**:
- `classify(messages)` — 三层分类
- `route(decision)` — 路由到 Ollama 或 API
- `proxy_stream()` — 流式透传
- `create_app()` — FastAPI 工厂函数

**测试**: 启动代理，用 curl 模拟 ChatAgent 请求，验证分类路由

---

### Step 3 — 代理单元测试

**目标**: 独立启动代理，验证每种分类场景

**测试用例**:
1. "数据库有几篇文章" → simple → Ollama
2. "比较比特币和以太坊的区别" → complex → API
3. "我的身份证号码是" → sensitive → Ollama
4. "你好" → simple(默认) → Ollama
5. API 超时 → fallback Ollama

**验证方法**: 检查代理日志，确认路由正确

---

### Step 4 — 集成到 `kg_enrich_handler.py`

**目标**: 在 `load()` 中启动代理线程，随引擎启停

**修改**: 在 `KgEnrichHandler.load()` 末尾添加 ~10 行启动代码

**测试**: 启动数字人，curl :11440 验证代理存活

---

### Step 5 — 更新配置

**目标**: ChatAgent 的 `api_url` 指向 `:11440`

**修改**: 1 行配置

**测试**: 与 Step 4 合并测试

---

### Step 6 — 端到端测试

**目标**: 完整数字人启动，验证 KG + 分类路由全部正常

**测试**:
1. 语音问"数据库有几篇文章" → 本地 Ollama 回答 10 篇
2. 键盘问"比较下第一和第二篇文章" → 走 API
3. 键盘问含敏感词的"我的密码是" → 走本地 + 通用回复
4. 切回语音正常

---

## 约束

- 不改 git clone 源代码（除已添加的 1 行 AUGMENTED_HUMAN_TEXT）
- 不装新 pip 依赖（FastAPI/uvicorn/httpx 均已存在）
- 代理随引擎自动启停
- 显存预算 < 20GB
