# OpenAvatarChat 双 RAG 数字人部署方案

> **零源码入侵** · 双知识图谱 · 全双工打断 · 本地+云端混合 · 开箱即用

---

## 一、系统概述

基于 OpenAvatarChat 构建的**实时对话数字人**，集成 **GraphRAG (AGE 图数据库)** 和 **LightRAG (向量语义搜索)** 双知识库引擎，支持**语音+文字**输入，**实时打断**，**智能 LLM 路由**（简单题本地、复杂题云端）。

### 核心卖点

| 卖点 | 说明 |
|------|------|
| 🔒 **全本地运行** | 3 个本地模型 + 本地向量/图数据库，隐私零外泄 |
| 🧠 **自带知识库** | 你的文档/资料喂进去，数字人就能答出来 |
| ⚡ **实时打断** | 全双工，说"停"就停，不说"停"不抢话 |
| 🎯 **零源码修改** | OpenAvatarChat 只改了 1 行枚举值，其余全部在独立插件目录 |
| 💰 **按月维护** | 知识库追加、模型更新、系统运维一条龙 |

---

## 二、架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OpenAvatarChat (:8283)                       │
│                                                                     │
│  麦克风 ──→ VAD ──→ ASR ──→ 端点检测 ──→ KgEnrich ──→ ChatAgent    │
│                                  │            │            │         │
│                          ┌───────┘            │            │         │
│                          ▼                    ▼            ▼         │
│                   SemanticTurn     ┌──────────────┐  ┌──────────┐   │
│                   Detector         │ 双RAG富化器  │  │LLM Proxy │   │
│                   (8b+0.5b)        │              │  │ :11440   │   │
│                   ~267ms           │ GraphRAG MCP │  │ 简单→本地 │   │
│                                    │ :8011 :8013  │  │ 复杂→云端 │   │
│                                    │ LightRAG API │  │ 敏感→强制 │   │
│                                    │ :9621 :9622  │  └────┬─────┘   │
│                                    └──┬───┬───┬───┘       │         │
└───────────────────────────────────────│───│───│───│───────│─────────┘
                                        │   │   │   │       │
    ┌───────────────────────────────────┼───┼───┼───┼───────┼───────┐
    │                    底层服务         │   │   │   │       │       │
    │                                    ▼   ▼   ▼   ▼       ▼       │
    │  PostgreSQL:5432   MCP:8011    MCP:8013   LR:9621  Ollama     │
    │  ┌──────────────┐  (播客)     (科研)    (科研)   :11434       │
    │  │ graphRAG     │                                           │
    │  │ 503实体      │  graphrag-    LightRAG                   │
    │  │ 1148关系     │  deploy       Server                     │
    │  │ 10文档       │                                          │
    │  │              │  ─────── 模型层 ───────                   │
    │  ├──────────────┤  qwen3:14b-8k  主对话  14.2G             │
    │  │ graphRAG_    │  qwen3:8b-8k   端点检测  9.8G             │
    │  │ research     │  qwen2.5:0.5b  分类+意图  1.0G           │
    │  │ 1文档        │  FlashHead FP8 渲染     1.5G             │
    │  │ 1204实体     │  ─────────────────────                    │
    │  └──────────────┘  显存总计: ~28.5G / 32G                   │
    └──────────────────────────────────────────────────────────────┘
```

---

## 三、知识库路由

### 开关面板

编辑 `oac_plugin/chat_duplex_agent_kg_merged.yaml` 即可控制：

```yaml
KgEnrich:
    # ── GraphRAG (AGE 图数据库) ──
    kb_podcast_enabled: true       # 播客库 :8011 (10篇)
    kb_research_enabled: true      # 科研库 :8013 (1篇, 1204实体)
    kb_gaokao_enabled: false       # 高考库 :8012 (待建)

    # ── LightRAG (向量语义搜索) ──
    kb_lightrag_research_enabled: true  # :9621 (1204实体)
    kb_lightrag_podcast_enabled: false  # :9622 (待加文档)
    kb_lightrag_gaokao_enabled: false   # :9623 (待加文档)
```

### 自动路由规则

```
"公路渐变率指什么"      → LightRAG (概念/定义, 165ms)
"圆曲线和竖曲线的关系"  → GraphRAG (关系/实体, 468ms)
"数据库有几篇文章"      → GraphRAG (计数, 52ms)
"你能看见什么"          → skip_kb (视觉/闲聊, 3ms)
"比较比特币和以太坊"    → LLM代理 → DeepSeek API (复杂题)
```

### 手动指定引擎

```
@graphrag 纵坡和桥梁的关系    → 强制 GraphRAG
@lightrag 超高渐变率的概念    → 强制 LightRAG
```

---

## 四、LLM 智能路由

ChatAgent 通过内置代理 (`:11440`) 调用 LLM，三层分类：

```
用户消息
    │
    ├─ Layer 1: 敏感词匹配 (<1ms) → 强制本地 Ollama
    ├─ Layer 2: 正则规则 (<1ms) → 简单→Ollama / 复杂→DeepSeek
    └─ Layer 3: qwen2.5:0.5b 分类 (~224ms) → 兜底路由
```

### 延迟对比

| 引擎 | 延迟 | 模型 | 场景 |
|------|------|------|------|
| Ollama 本地 | 437ms | qwen3:14b-8k | 90% 日常对话 |
| DeepSeek API | 600-800ms | deepseek-v4-flash | 复杂分析/创作 |
| 敏感内容 | 本地强制 | — | 隐私保护 |

---

## 五、延迟分析

### 完整链路（说话→数字人开口）

```
VAD(20) → ASR(100) → 端点检测(267) → KG查询 → LLM(437) → TTS(500) → Flash(40)
                                         │
                              ┌──────────┼──────────┐
                              ▼          ▼          ▼
                          闲聊skip    LightRAG   GraphRAG
                           3ms        165ms      468ms
```

| 场景 | KG | LLM | TTS | **总延迟** |
|------|-----|------|-----|----------|
| 闲聊/视觉 | 3ms | 437ms | 500ms | **~1.0s** |
| 概念查询 | 165ms | 437ms | 500ms | **~1.2s** |
| 关系查询 | 468ms | 437ms | 500ms | **~1.5s** |
| 复杂题 | — | 800ms(API) | 500ms | **~1.4s** |

### 端点检测 (267ms)

```
打断检测 qwen3:8b-8k (267ms) ┐
意图判断 qwen2.5:0.5b (224ms) ├─ 并行 (Ollama NUM_PARALLEL=4)
补全检测 已关闭               ┘
```

---

## 六、模型清单

| 模型 | 用途 | 显存 | 延迟 |
|------|------|------|------|
| `qwen3:14b-8k` | ChatAgent 主对话 | 14.2G | 437ms |
| `qwen3:8b-8k` | 端点检测 + LightRAG 提取 | 9.8G | 267ms |
| `qwen2.5:0.5b-4k` | 分类器 + 意图判断 | 1.0G | 224ms |
| `SenseVoiceSmall` | 语音识别 ASR | ~0.2G | 100ms |
| `CosyVoice API` | 语音合成 TTS | API | 500ms |
| `FlashHead 1.3B FP8` | 数字人渲染 | 1.5G | 40ms |

**总显存**: ~28.5G / 32G（RTX PRO 4500），余量 7G

---

## 七、知识库文档放哪

| RAG 引擎 | 输入目录 | 格式 |
|---------|---------|------|
| **GraphRAG 播客** | `/data/cy/postgreSQL-graphRAG-docker/project_folder/data/input/` | `.txt` |
| **GraphRAG 科研** | `/data/cy/postgreSQL-graphRAG-docker/project_folder/kb/research/md/` | `.md` → `.txt` |
| **GraphRAG 高考** | `/data/cy/postgreSQL-graphRAG-docker/project_folder/kb/gaokao/md/` | `.md` → `.txt` |
| **LightRAG 科研** | `/data/cy/LightRAG/data/instances/research/inputs/` | `.txt` `.md` `.pdf` |
| **LightRAG 播客** | `/data/cy/LightRAG/data/instances/podcast/inputs/` | `.txt` `.md` `.pdf` |
| **LightRAG 高考** | `/data/cy/LightRAG/data/instances/gaokao/inputs/` | `.txt` `.md` `.pdf` |

---

## 八、一键启动

```bash
bash /data/cy/graphrag-deploy/oac_plugin/start_all.sh
```

### Docker 容器说明

| 容器 | 端口 | 用途 | 自启 |
|------|------|------|------|
| `postgres` | 5432 | AGE 图数据库 | ✅ |
| `mcp-agent` | 8011 | GraphRAG MCP (播客) | ✅ |
| `mcp-research` | 8013 | GraphRAG MCP (科研) | ✅ |
| `graphrag-proxy` | 4000 | LLM 代理 (GraphRAG用) | ✅ |

### LightRAG 实例

| 实例 | 端口 | Workspace | 状态 |
|------|------|-----------|------|
| research | 9621 | research | ✅ |
| podcast | 9622 | podcast | ⬜ 待加文档 |
| gaokao | 9623 | gaokao | ⬜ 待加文档 |

```bash
# 管理命令
cd /data/cy/LightRAG/data/instances
./manage.sh start research    # 启动
./manage.sh status            # 状态
./manage.sh stop research     # 停止
./manage.sh logs research     # 实时日志
```

---

## 九、项目结构

```
/data/cy/
├── graphrag-deploy/              ← 部署目录（全部可改）
│   ├── oac_plugin/               ← ★ 数字人插件
│   │   ├── kg_enrich_handler.py  ← 双 RAG 富化器 (~870行)
│   │   ├── llm_proxy.py          ← LLM 智能路由代理
│   │   ├── llm_proxy_rules.yaml  ← 分类规则配置
│   │   ├── chat_duplex_agent_kg_merged.yaml ← 主运行配置
│   │   └── start_all.sh          ← 一键启动脚本
│   ├── docker-compose.yaml       ← GraphRAG 容器编排
│   └── .env                      ← 数据库 + LLM API 密钥
│
├── LightRAG/                     ← Git 仓库 (只改 .env)
│   └── data/instances/           ← 多实例配置
│
├── postgreSQL-graphRAG-docker/   ← Git 仓库 (不动)
│   └── project_folder/kb/        ← 知识库文档 + 建图脚本
│
└── OpenAvatarChat/               ← Git 仓库 (仅改 1 行)
    └── src/chat_engine/data_models/chat_data_type.py  (+1行 AUGMENTED_HUMAN_TEXT)
```

**不改源码**：99.9% 代码在独立目录，仅 `chat_data_type.py` 加 1 行枚举值。

---

## 十、部署报价（参考）

### 服务套餐

| 等级 | 价格 | 内容 |
|------|------|------|
| **基础版** | ¥2,000-3,000 | 单知识库 (GraphRAG) + 本地模型 + 无定制 |
| **标准版** | ¥5,000-8,000 | 双 RAG + 智能路由 + 打断 + 1 次 KB 建图 |
| **定制版** | ¥10,000-15,000 | 多知识库 + 私有数据 + 品牌定制 + 远程部署 |
| **维护** | ¥500-1,000/月 | 系统更新、KB 追加、故障响应 |

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | RTX 3090 (24G) | RTX 4090 (24G) / RTX PRO 4500 (32G) |
| RAM | 32GB | 64GB |
| 磁盘 | 50GB | 200GB+ (模型 + KB) |
| 系统 | Ubuntu 22.04 | Ubuntu 22.04 + CUDA 12.8 |

---

## 十一、新增文档流程

### GraphRAG 追加文档

```bash
cd /data/cy/postgreSQL-graphRAG-docker/project_folder/kb

# 1. 放文档
cp 新文档.md research/md/

# 2. Markdown → TXT
bash convert.sh

# 3. 重建图谱（增量）
bash reindex.sh research

# 4. 重启 MCP（重新加载 AGE 图）
docker restart mcp-research
```

### LightRAG 追加文档

```bash
# 直接放文件，服务器自动扫描处理
cp 新文档.txt /data/cy/LightRAG/data/instances/research/inputs/

# 触发扫描
curl -X POST http://localhost:9621/documents/scan
```

---

## 十二、故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| MCP :8013 拒绝连接 | research 容器未启动 | `docker start mcp-research` |
| LightRAG :9621 无响应 | 进程挂了 | `./manage.sh restart research` |
| 数字人回答"我无法处理" | Ollama 模型未拉取 | `ollama pull qwen3:14b-8k` |
| 回答两次 | Config 未设 AUGMENTED_HUMAN_TEXT | 检查 `input_type_override` |
| GraphRAG 查不到实体 | 图未建或 MCP 连错图 | 检查 `AGE_GRAPH_NAME` |
| UI 不显示说话文字 | WebRTC 连接断开 | 刷新页面 (Ctrl+Shift+R) |

---

## 十三、技术亮点

1. **零源码侵入** — 全部插件在 `/data/cy/graphrag-deploy/oac_plugin/`，独立部署
2. **双 RAG 引擎** — GraphRAG (关系/结构) + LightRAG (语义/概念) 互补
3. **全双工打断** — 启发式 (<1ms) + LLM 语义 (267ms) 双层检测
4. **智能路由** — 三层分类 (敏感词/正则/模型) 决定本地/云端
5. **多 KB 开关** — YAML 一行 `true/false` 控制，热重启生效
6. **中文实体提取** — 工程/科研/高考三大领域关键词自动识别
7. **启动预热** — GraphRAG MCP + LightRAG 后台预加载，首次查询零冷启动
8. **显存优化** — 模型 8K context (40K → 8K)，三模型 28.5G 驻留 32G GPU

---

## 十四、抖音口播脚本

> 时长约 55 秒，适合抖音完播率。⚠️ 标注处加画面特效。

```
（镜头：数字人正脸特写，无表情）

你看到的不是我，是我的数字分身。

（⚠️ 特效：数字人面部突然微笑 + 右侧弹出半透明架构图）

我给它装了双引擎——一个是图数据库，一个是向量语义库。
简单说：它能读懂你的文档，然后替你跟人聊天。

（镜头切换到部署画面：终端快速滚动 start_all.sh）

部署一套这样的数字人，我在后台跑了三个本地模型、
两个知识库、一个智能路由。
从头到尾，只改了一行代码。
剩下的全部是外部插件，零侵入。

（⚠️ 特效：屏幕中央出现 "一行代码" 大字闪现）

最绝的是打断——
你跟它说话的时候，随时说"停"，它真的停。
不是那种念完才关的假打断，是真全双工。

（镜头切回数字人，用户说"停"→ 数字人立刻闭嘴）

现在我帮人部署——
把你的文档喂进去，它就是你的专属数字分身。
老师放教学资料，律师放案例库，老板放公司手册。
放什么，它就会什么。

（⚠️ 特效：字幕 "基础版 ¥2000 / 标准版 ¥5000 / 定制版 ¥10000"）

基础版两千起，定制版一万。
每个月我帮你更新数据、维护系统。

想做自己的数字人，私信我"数字人"三个字。

（镜头慢慢拉近数字人脸，定格）

它知道你的数据库里有什么。
而且只说真话。
```

