#!/bin/bash
# OpenAvatarChat 一键启动 + 管理命令速查
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/chat_duplex_agent_kg_merged.yaml"

# ═══════════════════════════════════════════
# 环境变量
# ═══════════════════════════════════════════
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-sk-b6c5995951d3432cbaceafbe584f27d2}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-106b2ce18ece44ffbd0d7d8c349f5f9d}"

# ═══════════════════════════════════════════
# 1. GraphRAG (AGE 图数据库)
# ═══════════════════════════════════════════
echo "=== [1/3] GraphRAG (PostgreSQL + AGE) ==="
docker start postgres graphrag-proxy mcp-agent mcp-research 2>/dev/null || true
sleep 2

for port in 8011 8013; do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${port}/mcp -X POST \
        -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","method":"tools/list","id":1}' 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then echo "  ✅ MCP :${port} ready"; else echo "  ⚠️  MCP :${port} (code=$code)"; fi
done

# ═══════════════════════════════════════════
# 2. LightRAG (向量语义搜索)
# ═══════════════════════════════════════════
echo "=== [2/3] LightRAG (Vector Search) ==="
cd /data/cy/LightRAG/data/instances
bash manage.sh start research 2>&1 | tail -1
sleep 3

code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:9621/health 2>/dev/null || echo "000")
if [ "$code" = "200" ]; then echo "  ✅ LightRAG :9621 ready"; else echo "  ⚠️  LightRAG :9621 (code=$code)"; fi

# ═══════════════════════════════════════════
# 3. OpenAvatarChat
# ═══════════════════════════════════════════
echo "=== [3/3] OpenAvatarChat ==="
echo "  📋 Config: $CONFIG"
echo "  🌐 https://0.0.0.0:8283"
echo ""
cd /mnt/win_data/xx/OpenAvatarChat
uv run --no-sync src/demo.py --config "$CONFIG"
