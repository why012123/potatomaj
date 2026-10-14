#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  土豆麻将 · 三人对战台
#  你 + AI 助手 + 脚本 AI 同桌打三麻
#
#  双击「土豆麻将对战.command」运行，或在终端执行：
#      ./run-battle.sh
#
#  想换配置（改完这些不用动代码）：
#      SEATS=human,ml,teacher ./run-battle.sh   # 让神经网络顶对面那个位子
#      SEATS=ml,ml,ml         ./run-battle.sh   # 纯 AI 自战，看热闹
#      PORT=8800 WAIT=60      ./run-battle.sh   # 换端口 / 缩短托管超时
# ═══════════════════════════════════════════════════════════

cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

SEATS="${SEATS:-human,external,teacher}"
PORT="${PORT:-8765}"
WAIT="${WAIT:-240}"
NO_OPEN="${NO_OPEN:-}"

echo "========================================"
echo "  🀄  土豆麻将 · 三人对战台"
echo "========================================"
echo "  座位： $SEATS"
case "$SEATS" in
    *external*) echo "        （external = AI 助手的位子，由对话这边的命令行接管）" ;;
esac

# ---- 1. 虚拟环境 ----
if [ ! -x "$PY" ]; then
    echo "▸ 首次运行，正在创建虚拟环境…"
    python3 -m venv "$VENV" || { echo "❌ 创建虚拟环境失败，请先安装 python3"; exit 1; }
fi

# ---- 2. 依赖（只要 numpy）----
if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
    echo "▸ 正在安装依赖…"
    "$PY" -m pip install -q numpy || { echo "❌ 依赖安装失败，可以手动执行：$PY -m pip install numpy"; exit 1; }
fi

# ---- 3. 权重（只有 ml 座位才需要）----
if [ ! -f "$ROOT/aimodel/weights/policy.npz" ]; then
    case "$SEATS" in
        *ml*)
            echo "⚠️ 还没有训练好的权重 aimodel/weights/policy.npz，ml 座位会退化成启发式教师"
            echo "   先训练：$PY -m aimodel.train bc --hands 3000"
            ;;
    esac
fi

# ---- 4. 挑端口：已经开着就直接用，被占了就往后挪 ----
port_listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
is_our_ui() { curl -s -m 1 "http://127.0.0.1:$1/api/state?seat=0" >/dev/null 2>&1; }

# 人坐第几号位（默认 0）
HUMAN_SEAT=0
_n=0
IFS=',' read -ra _seats <<< "$SEATS"
for _s in "${_seats[@]}"; do
    if [ "$_s" = "human" ]; then HUMAN_SEAT=$_n; break; fi
    _n=$((_n + 1))
done

if is_our_ui "$PORT"; then
    URL="http://127.0.0.1:$PORT/?seat=$HUMAN_SEAT"
    echo "▸ $PORT 端口上已经开着一桌了，直接给你打开浏览器"
    echo "  👉 $URL"
    [ -z "$NO_OPEN" ] && open "$URL"
    exit 0
fi
while port_listening "$PORT"; do
    echo "▸ $PORT 被别的程序占用，换到 $((PORT + 1))"
    PORT=$((PORT + 1))
done

# ---- 5. 起服务 ----
echo "▸ 启动对战台（Ctrl-C 结束）…"
"$PY" -m aimodel.ui.server --seats "$SEATS" --port "$PORT" --wait "$WAIT" &
SERVER_PID=$!
trap 'echo; echo "▸ 正在关闭对战台…"; kill $SERVER_PID 2>/dev/null; exit 0' INT TERM

for _ in $(seq 1 60); do
    is_our_ui "$PORT" && break
    sleep 0.25
done

URL="http://127.0.0.1:$PORT/?seat=$HUMAN_SEAT"
echo
echo "========================================"
echo "  👉 浏览器打开： $URL"
echo "========================================"
if [ "$HUMAN_SEAT" = "0" ] || [ "$HUMAN_SEAT" -ge 0 ] 2>/dev/null; then
    echo "  你坐 P$HUMAN_SEAT 号位，在网页里点牌出牌。"
fi
case "$SEATS" in
    *external*) echo "  轮到我（AI 助手）时，在对话里说一声「该你了」即可。" ;;
esac
echo "  等 ${WAIT}s 没人动就自动托管，不会卡死。"
echo

[ -z "$NO_OPEN" ] && open "$URL"
wait $SERVER_PID
