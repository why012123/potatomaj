#!/bin/bash
# RL 长跑：分块 + 每块之间配对赛（可信判据）+ 不涨就早停
#
# 为什么不用 RL 内部的「最优」：
#   实测它内部评测 +11.3 vs 可信配对赛 +1.89（虚高 6 倍）
#   -> 内部那个「新高」是噪声，不能作为进度依据
#
# 所以：每 N 轮存一个检查点，然后用同一协议（同牌山 + SPRT）配对赛
#       -> 得到可信的进度曲线
#
# 用法: bash tools/rl_ladder.sh <起点权重> <块数> [每块轮数] [连续不涨几次就停]
#
# ═══════════════════════════════════════════════════════════════════════════
# 🔴 v2 —— v1 有三个 bug，白跑 3.4 小时、产出两份无效权重 ✗（2026-10-17 发现）
#
#   v1 写的是：`train rl --model=maj-pro4v1.1 --resume --out chunkNN.npz`
#   实测 chunk01/02：hidden 变成 **[256,256]** ✓ 而起点是 **[512,512]** ✗
#                     两份的 steps **都是 20,118** ✗（= 各自从随机网络重训）
#                     配对赛 -47.892 ✓（半个宽度的随机网络，本该差这么多 ✓）
#
#   三个 bug，都在脚本里：
#     ① **`--model` 只选「规格」，不选「权重」** ✗
#        它决定 dim/kinds/得分条件 ✓ 权重从哪来由 `--out` 决定 ✓
#        我把它当成了「从这里初始化」✗
#     ② **`--hidden` 默认 `256,256`** ✗ 而起点是 512 ✓
#        `build_net` 只在「`--out` 不存在」时才用 `--hidden` 新建 ✓
#     ③ ⭐ **`--out` 每块都是新路径** ✗
#        `build_net(resume=True)`：`--out` 存在 -> `Net.load(--out)` ✓
#                                  `--out` 不存在 -> **静默新建随机网络** ✗✗
#        -> 每一块都是「新随机网络 + 20 轮 RL」✗ **阶梯根本没接上** ✗
#
#   ⭐ 最该记住的是 ③ 的形态 ✗：
#      **「接着上次训」在文件不在时，静默降级成「从头训」** ✗
#      它不报错 ✓ 不警告 ✓ 照常产出文件 ✓ 照常有漂亮的 steps 数字 ✓
#      -> 唯一能看出来的地方是 **hidden 和 steps** ✗（我正是这么发现的 ✓）
#      -> 所以 v2 每次开跑前**先 cp 再断言**✗ 断言失败就**立刻退出** ✓
#         （宁可什么都不做，也不要安静地产出一个错的东西 ✓）
# ═══════════════════════════════════════════════════════════════════════════
#
# 注意: 所有变量加 ${} —— 因为下面有全角标点紧跟变量，bash 会把全角字符
#       当成变量名的一部分（已踩过一次，报 "CHUNKS?: unbound variable"）
set -euo pipefail
cd "$(dirname "$0")/.."

# ── 跨平台：python 解释器路径 ──────────────────────────────────────────────
# Unix 是 `.venv/bin/python` ✓ Windows 是 `.venv/Scripts/python.exe` ✗
# （写死 Unix 路径 -> server 那台直接起不来 ✓ 一个字符的差别 ✓）
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
elif [ -x ".venv/Scripts/python" ]; then
    PY=".venv/Scripts/python"
else
    PY="$(command -v python3 || command -v python)" || {
        echo "❌ 找不到 python（试过 .venv/bin、.venv/Scripts、PATH）"; exit 1; }
fi

# ── 跨平台：nice ───────────────────────────────────────────────────────────
# Windows 的 Git Bash **没有 nice** ✗（直接调会 `command not found` -> 整个脚本挂 ✓）
# 所以探测一下 ✓ 没有就裸跑 ✓
NICE=""
if command -v nice >/dev/null 2>&1; then NICE="nice -n 10"; fi
run_nice() {           # 有 nice 就降优先级，没有就照跑（Windows）
    if [ -n "${NICE}" ]; then ${NICE} "$@"; else "$@"; fi
}

START="${1:?起点权重}"
CHUNKS="${2:-10}"
ITERS="${3:-20}"
PATIENCE="${4:-2}"          # 连续几块「没涨」就早停
MODEL="${MODEL:-maj-pro4v1.1}"
OUTDIR="_data/rl_ladder"
mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/ladder.log"

fail() { echo "❌ $*" | tee -a "${LOG}"; exit 1; }

echo "=== RL ladder v2.1 (跨平台): ${CHUNKS} chunks x ${ITERS} iters, start=${START} ===" | tee "${LOG}"
echo "    python = ${PY}" | tee -a "${LOG}"
echo "    降优先级 = ${NICE:-（本机没有 nice，裸跑）}" | tee -a "${LOG}"

# ── 起点自检：把架构钉死，后面每一块都要和它一致 ──────────────────────────
read_meta() {
    "${PY}" - "$1" <<'PYEOF'
import json, sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
m = json.loads(bytes(z["_meta"]).decode())
print(f"{m['dim']}|{','.join(str(h) for h in m['hidden'])}|{m.get('steps')}")
PYEOF
}
SPEC_START=$(read_meta "${START}") || fail "读不出起点 ${START} 的 _meta"
START_DIM="${SPEC_START%%|*}"
_rest="${SPEC_START#*|}"; START_HID="${_rest%%|*}"; START_STEPS="${_rest##*|}"
echo "    起点身份: dim=${START_DIM}  hidden=[${START_HID}]  steps=${START_STEPS}" | tee -a "${LOG}"
echo "    早停: 连续 ${PATIENCE} 块不涨就停（判据=配对赛 p<0.05 且差值为正）" | tee -a "${LOG}"

BEST="${START}"
BEST_NAME="start"
stale=0
# ⚠️ 用 C 式循环，**不用 `seq 1 ${CHUNKS}`** ✗
#    macOS 的 `seq 1 0` 输出 **"1 0"**（不是空 ✗；GNU seq 才是空 ✓）
#    -> 我拿 CHUNKS=0 做「不训练的冒烟测试」✗ 结果**真的启动了一整块训练** ✗
#    （同类形状：**两个平台对同一输入给出不同结果** ✓
#      而我又是**先用了一个没验过的假设**✗ —— 与⑮同源 ✓）
for (( c = 1; c <= CHUNKS; c++ )); do
    TAG=$(printf '%02d' "${c}")
    CKPT="${OUTDIR}/chunk${TAG}.npz"
    echo "" | tee -a "${LOG}"
    echo "---- chunk ${c} / ${CHUNKS} : RL ${ITERS} iters（从 ${BEST_NAME} 续）----" | tee -a "${LOG}"

    # ⭐ 关键：`--out` 既是「读哪份」也是「写哪份」✗
    #    所以必须**先把起点拷到 out** ✓ 否则 build_net 静默新建随机网络 ✗
    cp "${BEST}" "${CKPT}"
    GOT=$(read_meta "${CKPT}") || fail "拷过来的 ${CKPT} 读不出 _meta"
    GOT_DIM="${GOT%%|*}"; _r="${GOT#*|}"; GOT_HID="${_r%%|*}"
    [ "${GOT_DIM}" = "${START_DIM}" ] || fail "chunk${TAG} dim ${GOT_DIM} != 起点 ${START_DIM}"
    [ "${GOT_HID}" = "${START_HID}" ] || fail "chunk${TAG} hidden [${GOT_HID}] != 起点 [${START_HID}] —— 架构降级了，停"
    echo "    ✅ 架构自检通过: dim=${GOT_DIM} hidden=[${GOT_HID}]" | tee -a "${LOG}"

    START_TS=$(date +%s)
    # run_nice: 有 nice 就降优先级（别和桌面 GUI 抢 CPU ✓）没有就裸跑（Windows ✗）
    run_nice "${PY}" -m aimodel.train rl --model="${MODEL}" --resume \
        --iters "${ITERS}" --matches 300 --eval-matches 80 --ppo-epochs 3 \
        --out "${CKPT}" --log "${OUTDIR}/chunk${TAG}.json" \
        >> "${OUTDIR}/chunk${TAG}.stdout" 2>&1 \
        || fail "chunk ${c} 训练失败（见 ${OUTDIR}/chunk${TAG}.stdout 尾部）"
    echo "  chunk ${c} done, took $(( $(date +%s) - START_TS ))s, weights ${CKPT}" | tee -a "${LOG}"

    # 训练后再断言一次：确认它**没有**把架构换掉（写坏也不许静默 ✓）
    GOT=$(read_meta "${CKPT}") || fail "训完读不出 ${CKPT} 的 _meta"
    GOT_DIM="${GOT%%|*}"; _r="${GOT#*|}"; GOT_HID="${_r%%|*}"; GOT_STEPS="${_r##*|}"
    [ "${GOT_DIM}" = "${START_DIM}" ] || fail "训完 dim 变了: ${GOT_DIM}"
    [ "${GOT_HID}" = "${START_HID}" ] || fail "训完 hidden 变了: [${GOT_HID}] —— 架构降级，停"
    echo "    ✅ 训后自检: dim=${GOT_DIM} hidden=[${GOT_HID}] steps=${GOT_STEPS}" | tee -a "${LOG}"

    # ── 配对赛（可信判据）：本块 vs **当前最好的那一块**（不是只比上一块 ✓）
    echo "  paired test: chunk ${c} vs ${BEST_NAME} ..." | tee -a "${LOG}"
    run_nice "${PY}" -m aimodel.arena_test --a "${CKPT}" --b "${BEST}" \
        --target-delta 2 --jobs 4 --max-games 6000 \
        > "${OUTDIR}/pair${TAG}.txt" 2>&1 || true
    grep -E "配对差值|95% 单侧|SPRT 判决" "${OUTDIR}/pair${TAG}.txt" \
        | sed 's/^/    /' | tee -a "${LOG}"

    DIFF=$(grep -oE "配对差值（A−B）[[:space:]]+-?[0-9]+(\.[0-9]+)?" "${OUTDIR}/pair${TAG}.txt" \
           | grep -oE "\-?[0-9]+(\.[0-9]+)?$" | head -1)
    PVAL=$(grep -oE "单侧 p ≈ [0-9]+(\.[0-9]+)?" "${OUTDIR}/pair${TAG}.txt" \
           | grep -oE "[0-9]+(\.[0-9]+)?$" | head -1)
    # 判据读不出来就**停** ✓ 绝不静默继续（v1 的教训 ✓）
    [ -n "${DIFF}" ] && [ -n "${PVAL}" ] \
        || fail "配对赛结果解析失败（${OUTDIR}/pair${TAG}.txt）—— 判据不明，停"

    if awk -v d="${DIFF}" -v p="${PVAL}" 'BEGIN{exit !(p < 0.05 && d > 0)}'; then
        echo "    🎉 显著更强: 差值 ${DIFF}（p≈${PVAL}）-> 成为新的最好" | tee -a "${LOG}"
        BEST="${CKPT}"; BEST_NAME="chunk${TAG}"; stale=0
    else
        stale=$(( stale + 1 ))
        echo "    ➖ 没涨: 差值 ${DIFF}（p≈${PVAL}）-> 不换最好；连续不涨 ${stale}/${PATIENCE}" \
            | tee -a "${LOG}"
        if [ "${stale}" -ge "${PATIENCE}" ]; then
            echo "" | tee -a "${LOG}"
            echo "=== 早停：连续 ${PATIENCE} 块没有显著变强 -> 继续跑无意义 ===" | tee -a "${LOG}"
            break
        fi
    fi
done
echo "" | tee -a "${LOG}"
echo "=== 结束。最好的一块 = ${BEST_NAME}（${BEST}）===" | tee -a "${LOG}"
echo "    ⚠️ 这份**没有**替换任何已部署的权重 ✗ 要不要换是单独的决定 ✓" | tee -a "${LOG}"
