#!/bin/bash
# 四麻 ELO 天梯（网页版）
#
# 阵容 = **盘上所有四麻模型**（v1 / v1.1 / v1.2 / v1.25(或 v1.3) / pro）
#       自动扫 ✓ 加了新权重重启一下就带上 ✓ 不用改脚本 ✓
#
# ⚠️ 必须兼容 **macOS 自带的 bash 3.2** ✗ —— 它没有 `mapfile`、`sort -V` 也时有时无 ✓
#    （第一版用了 mapfile，在真机上直接 `command not found` ✓
#      —— 是「跑脚本本身」才发现的，光看语法检查过不了这一关 ✓）
#
# 用法:  bash tools/arena4p.sh
#        然后浏览器打开 http://127.0.0.1:8771
#
# 环境变量: PORT（默认 8771）、COPIES（默认 3）
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8771}"
COPIES="${COPIES:-3}"

# 扫所有四麻权重：maj-<风格>4<版本>.npz
#   排除 _bcbase / _prev（中间产物，不是正式版本 ✓）
MODELS=()
while IFS= read -r _m; do
  [ -n "${_m}" ] && MODELS+=("${_m}")
done <<LIST
$(ls aimodel/weights/ 2>/dev/null \
  | grep -E '^maj-[a-z]+4v[0-9.]+\.npz$' \
  | grep -vE '_(bcbase|prev)\.npz$' \
  | sed 's/\.npz$//' \
  | LC_ALL=C sort)
LIST

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "❌ aimodel/weights/ 里没找到四麻模型"
  exit 1
fi

echo "🏟  四麻天梯：${#MODELS[@]} 个模型 × ${COPIES} 副本"
for _m in "${MODELS[@]}"; do echo "     ${_m}"; done
echo "     →  http://127.0.0.1:${PORT}"
# 用户特意点名要这两个：出现时明确说一声 ✓
for _f in maj-normal4v1.25 maj-normal4v1.3; do
  [ -f "aimodel/weights/${_f}.npz" ] && echo "     ✓ 已带上 ${_f}"
done
echo

exec .venv/bin/python -m aimodel.ui.arena_server \
     --players 4 --port "${PORT}" --copies "${COPIES}" --models "${MODELS[@]}"
