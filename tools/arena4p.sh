#!/bin/bash
# 四麻 ELO 天梯（网页版）
#
# 阵容：v1.2 四个（标准/防守/速攻/大牌）+ maj-pro4v1.1（真人牌谱 BC）
#      —— maj-normal4v1.3 还没拿到，拿到后加到 --models 里就是 6 个 ✓
#
# 用法:  bash tools/arena4p.sh         然后浏览器打开 http://127.0.0.1:8771
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8771}"
MODELS=(
  maj-normal4v1.2 maj-mountain4v1.2 maj-flash4v1.2 maj-huge4v1.2
  maj-pro4v1.1
)
# 拿到了就自动带上 ✓
[ -f aimodel/weights/maj-normal4v1.3.npz ] && MODELS+=(maj-normal4v1.3)

echo "🏟  四麻天梯：${#MODELS[@]} 个模型 × 3 副本"
printf '     %s\n' "${MODELS[@]}"
echo "     http://127.0.0.1:${PORT}"
exec .venv/bin/python -m aimodel.ui.arena_server \
     --players 4 --port "${PORT}" --copies 3 --models "${MODELS[@]}"
