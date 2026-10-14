#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  土豆麻将 · ELO 混战池
#  让不同模型、不同版本互相对打，打出一个天梯。
#  规则：初始 500 分 · 桌面分差 ≤200 · 低于 0 分淘汰
# ═══════════════════════════════════════════════════════════
# 这个脚本在 aimodel/ui/ 下，要退两层才到项目根（.venv 在根目录）
cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"

PLAYERS=3
PORT=8770
COPIES=3
while [ $# -gt 0 ]; do
    case "$1" in
        --players) PLAYERS="$2"; shift 2 ;;
        --port)    PORT="$2";    shift 2 ;;
        --copies)  COPIES="$2";  shift 2 ;;
        -h|--help)
            cat <<EOF
用法: ./run-arena.sh [--players 3|4] [--port 8770] [--copies 3]

  --players 3   三麻池（默认）
  --players 4   四麻池
  --port        端口（默认 8770）
  --copies 3    每个模型版本派几个参赛者（默认 3，各记各的分）

打开后：填「场次」→ 点「开始」。页面实时显示按 ELO 排序的天梯，
柱子分别是 ELO / 一位率 / 和了率，右侧是分数走势和最近对局。
EOF
            exit 0 ;;
        *) echo "未知参数: $1（--help 看用法）"; exit 1 ;;
    esac
done

if [ ! -x "$PY" ]; then
    echo "❌ 找不到 Python 环境：$PY"
    echo "   项目根目录: $ROOT"
    echo "   先建虚拟环境：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    echo "   或者看 aimodel/README.md"
    read -r -p "按回车关闭…"; exit 1
fi

# 已经在跑就直接开浏览器
if curl -s -m 2 "http://127.0.0.1:$PORT/api/arena" >/dev/null 2>&1; then
    echo "▸ 端口 $PORT 上已经有一个池子在跑，直接打开页面"
    open "http://127.0.0.1:$PORT/" 2>/dev/null
    exit 0
fi

echo "========================================"
echo "  🏟  土豆麻将 · ELO 混战池（${PLAYERS} 麻）"
echo "========================================"
echo "  每个模型 $COPIES 个参赛者（各记各的分）"
echo "  初始 500 分 · 分差上限 200 · 低于 0 分淘汰"
echo "  页面: http://127.0.0.1:$PORT/"
echo "  停止: 在这个窗口按 Ctrl-C"
echo

"$PY" -m aimodel.ui.arena_server --players "$PLAYERS" --port "$PORT" --copies "$COPIES" &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; exit 0' INT TERM

for _ in $(seq 1 40); do
    sleep 0.4
    if curl -s -m 1 "http://127.0.0.1:$PORT/api/arena" >/dev/null 2>&1; then
        open "http://127.0.0.1:$PORT/" 2>/dev/null
        echo "▸ 浏览器已打开（没自动弹的话手点上面的地址）"
        break
    fi
done

wait $SRV
