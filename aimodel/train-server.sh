#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  土豆麻将 · 服务器训练脚本
#
#  在一台（可能很强的）机器上跑完整一轮训练：
#    行为克隆 → 存纯克隆基线 → 自对弈强化学习
#
#  用法:
#     ./aimodel/train-server.sh                      # 用默认参数跑当前版本
#     ./aimodel/train-server.sh --version v1.3       # 训下一版
#     ./aimodel/train-server.sh --hands 12000 --iters 30 --hidden 512,512
#     ./aimodel/train-server.sh --jobs 8             # 手动指定并行数
#
#  只依赖 numpy，不需要 PyTorch / CUDA。
# ═══════════════════════════════════════════════════════════
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

VERSION=""
HANDS=6000
EPOCHS=20
HIDDEN="512,512"
ITERS=20
MATCHES=350
EVAL=200
POOL=0.25
JOBS=""
PY="${POTATOMAJ_PY:-$ROOT/.venv/bin/python}"

while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --hands)   HANDS="$2";   shift 2 ;;
        --epochs)  EPOCHS="$2";  shift 2 ;;
        --hidden)  HIDDEN="$2";  shift 2 ;;
        --iters)   ITERS="$2";   shift 2 ;;
        --matches) MATCHES="$2"; shift 2 ;;
        --eval)    EVAL="$2";    shift 2 ;;
        --pool)    POOL="$2";    shift 2 ;;
        --jobs)    JOBS="$2";    shift 2 ;;
        --python)  PY="$2";      shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 环境 ────────────────────────────────────────────────────
if [ ! -x "$PY" ]; then
    echo "▸ 没找到 Python（$PY），先建一个虚拟环境"
    python3 -m venv "$ROOT/.venv" || exit 1
    PY="$ROOT/.venv/bin/python"
fi
if ! "$PY" -c "import numpy" 2>/dev/null; then
    echo "▸ 装依赖（只有 numpy）"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q -r "$ROOT/requirements.txt" || exit 1
fi
NUMPY=$("$PY" -c "import numpy;print(numpy.__version__)")

# ── 自动决定并行数 ──────────────────────────────────────────
CORES=$("$PY" - <<'PY'
import os
try: print(os.cpu_count() or 4)
except Exception: print(4)
PY
)
# 内存（GB）
MEM=$(("$PY" - <<'PY'
import os
try:
    print(int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1073741824))
except Exception:
    print(16)
PY
))
if [ -z "$JOBS" ]; then
    # ⚠️ 这里算的是**训练进程**的内存（克隆阶段样本常驻），不是数据生成：
    #    · 训练：四麻 15000 局时单进程约 5GB —— 这才是并行数的上限
    #    · **selfplay 数据生成：实测每 worker 仅 ~186MB** ✗
    #      （曾把 5GB 套到数据生成上，白白限制成 4 并行，浪费了很多轮 —— 见 README）
    # 克隆阶段每个进程要吃 1.5~2GB（5000~6000 局的样本常驻内存），
    # 所以并行数同时受核数和内存限制；留 2GB 给系统。
    BY_MEM=$(( (MEM - 2) / 3 ))
    BY_CPU=$(( CORES / 2 ))
    JOBS=$BY_CPU
    [ "$BY_MEM" -lt "$JOBS" ] && JOBS=$BY_MEM
    [ "$JOBS" -lt 1 ] && JOBS=1
    [ "$JOBS" -gt 8 ] && JOBS=8
fi

cat <<EOF
════════════════════════════════════════════
  土豆麻将 · 服务器训练
════════════════════════════════════════════
  机器        ${CORES} 核 / ${MEM} GB / numpy ${NUMPY}
  并行        ${JOBS} 个模型同时训（每个进程限 1 个 BLAS 线程）
  版本        ${VERSION:-（当前版本）}
  克隆        ${HANDS} 局 × ${EPOCHS} epoch，网络 ${HIDDEN}
  自对弈      ${ITERS} 轮 × ${MATCHES} 场，评测 ${EVAL} 场，对手池 ${POOL}
  日志        aimodel/weights/logs/*.log
════════════════════════════════════════════
EOF
echo "▸ 开始，Ctrl-C 可中断（已训完的模型下次会自动跳过）"
T0=$(date +%s)

VER_ARG=()
[ -n "$VERSION" ] && VER_ARG=(--version "$VERSION")

# ── ① 行为克隆 ──────────────────────────────────────────────
echo
echo "【1/3】行为克隆（跟各自的专属教师学）"
"$PY" -u -m aimodel.train_all --stage bc "${VER_ARG[@]}" \
    --jobs "$JOBS" --hands "$HANDS" --epochs "$EPOCHS" --hidden "$HIDDEN" --threads 1 || exit 1

# ── ② 存纯克隆基线（后面要用它做「RL 到底有没有用」的对照）──
echo
echo "【2/3】保存纯克隆基线 * _bcbase.npz"
"$PY" - "$VERSION" <<'PY'
import sys, shutil
from aimodel import variants
ver = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
for spec in variants.all_specs(ver):
    if not spec.path.exists():
        continue
    base = spec.path.with_name(spec.path.stem + "_bcbase.npz")
    if not base.exists():
        shutil.copy2(spec.path, base)
    print(f"   ✅ {base.name}")
PY

# ── ③ 自对弈强化学习（对手池 = 其它风格）───────────────────
echo
echo "【3/3】自对弈强化学习（对手池里混其它风格的模型）"
"$PY" -u -m aimodel.train_all --stage rl "${VER_ARG[@]}" \
    --jobs "$JOBS" --iters "$ITERS" --matches "$MATCHES" \
    --eval-matches "$EVAL" --pool-ratio "$POOL" --threads 1 || exit 1

DT=$(( $(date +%s) - T0 ))
echo
echo "════════════════════════════════════════════"
echo "  全部完成，用时 $((DT / 60)) 分钟"
"$PY" -m aimodel.train --list-models 2>/dev/null | tail -12
echo
echo "  把权重拷回主力机即可（只传 npz，一个 1~2MB）："
echo "    rsync -av aimodel/weights/*.npz 你的机器:potatomaj/aimodel/weights/"
echo "  或反过来，把那边的 weights 拉回来："
echo "    rsync -av 服务器:$ROOT/aimodel/weights/ ./aimodel/weights/"
echo "════════════════════════════════════════════"
