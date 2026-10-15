#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  全流程：天鳳 4 万局牌谱 → MJAI → BC 数据集 → BC 训练 → RL
#
#  为什么写成「一个脚本跑到底」：
#    · 这台 Mac **不回收长进程** ✓ 所以不需要 shortjob 那套切片 ✓
#    · 但**分阶段**仍必要：抽取那一步崩一次不能全丢 ✓
#      所以抽取按 MJAI 文件分批，每批独立、跳过已完成的 ✓
#
#  预计总耗时：抽取 ~11 小时 + BC 训练 ~2.5 小时 + RL ~3 小时
#
#  用法:
#      bash tools/train_pro4_full.sh            # 全流程
#      bash tools/train_pro4_full.sh extract    # 只抽数据
#      bash tools/train_pro4_full.sh bc         # 只跑 BC
#      bash tools/train_pro4_full.sh rl         # 只跑 RL
# ═══════════════════════════════════════════════════════════
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
XLSM="${HOME}/Downloads/GL1.xlsm"

DATA="${ROOT}/_data/pro4"
MJAI="${DATA}/mjai"
TBC="${DATA}/tbc"
LOG="${DATA}/pipeline.log"
BC_DATA="${DATA}/bc_all.npz"
# ⚠️ 用一个**区别于远端**的名字 ✗ 两边独立跑、之后用配对赛比谁的更好 ✓
WEIGHTS="${ROOT}/aimodel/weights/maj-pro4v1.1_mac.npz"

STAGE="${1:-all}"
mkdir -p "${MJAI}" "${TBC}"
find "${TBC}" -maxdepth 1 -name "batch_*.npz" -type d -exec rm -rf {} + 2>/dev/null   # 上一版建错的目录

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }
stage_done() { touch "${DATA}/.done_$1"; }
stage_ok() { [ -f "${DATA}/.done_$1" ]; }

# ── ① GL1.xlsm → MJAI（约 5 分钟）────────────────────────────
if { [ "${STAGE}" = "all" ] || [ "${STAGE}" = "extract" ]; } && ! stage_ok mjai; then
    log "══ 阶段 1/4：GL1.xlsm → MJAI ══"
    if [ ! -f "${XLSM}" ]; then
        log "❌ 找不到 ${XLSM} —— 原始牌谱不在，无法继续"
        exit 1
    fi
    "${PY}" -m aimodel.mjlog "${XLSM}" --out "${MJAI}" --batch 1500 2>&1 | tail -8 | tee -a "${LOG}"
    n=$(ls "${MJAI}"/*.json 2>/dev/null | wc -l | tr -d ' ')
    log "  产出 ${n} 个 MJAI 文件"
    [ "${n}" -gt 0 ] && stage_done mjai
fi

# ── ② MJAI → BC 数据集（约 11 小时，分批可续）─────────────────
if { [ "${STAGE}" = "all" ] || [ "${STAGE}" = "extract" ]; } && ! stage_ok tbc; then
    log "══ 阶段 2/4：MJAI → BC 数据集 ══"
    total=$(ls "${MJAI}"/*.json 2>/dev/null | wc -l | tr -d ' ')
    i=0; skipped=0; rows=0
    for f in "${MJAI}"/*.json; do
        i=$((i + 1))
        name=$(basename "${f}" .json)
        out="${TBC}/${name}"            # ← 是**目录** ✗ 传文件名会建出同名目录
        # ★ 跳过已完成的批 —— 崩了重跑不会白干 ✓
        if [ -f "${out}/tbc_0000.npz" ]; then
            skipped=$((skipped + 1))
            continue
        fi
        "${PY}" -m aimodel.tenhou_bc --src "${f}" --out "${out}" \
                --players 4 --shard 0 --quiet >>"${LOG}" 2>&1
        if [ $((i % 10)) -eq 0 ]; then
            done_n=$(find "${TBC}" -name 'tbc_*.npz' -type f 2>/dev/null | wc -l | tr -d ' ')
            log "  进度 ${done_n}/${total} 批（本轮跳过 ${skipped}）"
        fi
    done
    done_n=$(find "${TBC}" -name 'tbc_*.npz' -type f 2>/dev/null | wc -l | tr -d ' ')
    log "  完成 ${done_n}/${total} 批"
    [ "${done_n}" -ge "${total}" ] && [ "${total}" -gt 0 ] && stage_done tbc
fi

# ── ③ 合并 + BC 训练（约 2.5 小时）──────────────────────────
if { [ "${STAGE}" = "all" ] || [ "${STAGE}" = "bc" ]; } && ! stage_ok bc; then
    log "══ 阶段 3/4：合并 + BC 训练 ══"
    if [ ! -f "${BC_DATA}" ]; then
        log "  合并 $(find "${TBC}" -name 'tbc_*.npz' -type f | wc -l | tr -d ' ') 个分片 → ${BC_DATA}"
        "${PY}" - <<PYEOF 2>&1 | tail -6 | tee -a "${LOG}"
import sys, pathlib
sys.path.insert(0, "${ROOT}")
import numpy as np
parts = sorted(p for p in pathlib.Path("${TBC}").rglob("tbc_*.npz") if p.is_file())
acc, keys = [], None
for p in parts:
    with np.load(p, allow_pickle=False) as d:
        if keys is None:
            keys = list(d.files)
        acc.append({k: d[k] for k in keys})
out = {}
for k in keys:
    out[k] = np.concatenate([a[k] for a in acc], axis=0)
np.savez_compressed("${BC_DATA}", **out)
print(f"合并 {len(parts)} 片 -> {out[keys[0]].shape[0]:,} 条")
PYEOF
    fi
    log "  备份当前权重（3 epoch 版本）"
    # 从远端的 3-epoch 权重起步（复制一份，别改原件）
    [ -f "${WEIGHTS}" ] || cp "${ROOT}/aimodel/weights/maj-pro4v1.1.npz" "${WEIGHTS}"
    log "  开始 BC 训练 30 epoch（从 ${WEIGHTS##*/} 继续）"
    "${PY}" -m aimodel.train bc --model=maj-pro4v1.1 \
            --bc-data="${BC_DATA}" --resume --epochs 30 \
            --out "${WEIGHTS}" --log "${DATA}/bc_train.log" 2>&1 | tee -a "${LOG}"
    stage_done bc
fi

# ── ④ RL（约 3 小时）────────────────────────────────────────
if { [ "${STAGE}" = "all" ] || [ "${STAGE}" = "rl" ]; } && ! stage_ok rl; then
    log "══ 阶段 4/4：自对弈强化学习（RL）══"
    "${PY}" -m aimodel.train rl --model=maj-pro4v1.1 \
            --iters 20 --matches 300 --eval-matches 80 --ppo-epochs 3 \
            --out "${WEIGHTS}" --log "${DATA}/rl_train.log" 2>&1 | tee -a "${LOG}"
    stage_done rl
fi

log "══ 全部完成 ══"
ls -la "${WEIGHTS}" 2>/dev/null | awk '{print "  最终权重: "$5" B"}'
