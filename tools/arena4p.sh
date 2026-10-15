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

# ⚠️ `--loose` 默认开 ✗ —— 不加的话两端会**静默冻结** ✓
#
# 实测：同一批模型跑两轮，排名**完全洗牌** ✗
#   maj-flash4v1.2   825.5(#1) -> 627.8(#8)
#   maj-pro4v1.1     613.5(#2) -> 970.9(#1)
# 根因是 `--max-gap 200`：评分散开之后，**两端都找不到分差 ≤200 的对手** ✓
# → 他们的分数不再更新 ✓ → 早先一波运气好的被**永久冻在第一** ✓
# 证据：第 1 名只有 264 场、第 15 名 162 场 ✓ 而中间的第 8 名有 567 场 ✓
#
# **这和「选优判据」是同一类毛病** ✗：一个带阈值的规则在阈值外会**静默停止工作** ✓
LOOSE_FLAG=()
[ "${LOOSE:-1}" = "0" ] || LOOSE_FLAG=(--loose)

exec .venv/bin/python -m aimodel.ui.arena_server \
     --players 4 --port "${PORT}" --copies "${COPIES}" \
     "${LOOSE_FLAG[@]}" --models "${MODELS[@]}"
