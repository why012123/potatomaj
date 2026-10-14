#!/bin/bash
# 一键开桌：编译（必要时）→ 开 Java 窗口。
#
# **后端由 Java 界面自己用 ProcessBuilder 拉起并托管**（拿着进程句柄），
# 关窗口时会把它一起停掉；所以这里不再 nohup 起服务，免得留下关不掉的进程。
# 如果你想自己起后端（比如要在终端看日志），加 --no-server。
#
#   ./run.sh                            默认 8765 端口，窗口里选三麻/四麻再开局
#   ./run.sh --port 8844                换端口（界面会用这个端口起后端）
#   ./run.sh --no-server                后端你自己起，界面只连不管
#   ./run.sh --url http://127.0.0.1:8844 指定地址（非本机地址＝不托管）
#   ./run.sh --build                    先强制重新编译
#   ./run.sh --seat 1 --no-log          其余参数原样传给界面（maj.Main）
#
# 注意：macOS 自带 bash 是 3.2，别用 ${arr[@]+...} 那种写法（会被解析坏）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"   # aimodel/ui/java -> 项目根

PORT=8765
URL=""
FORCE_BUILD=0
UI_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --port)  PORT="${2:?--port 后面要跟端口}"; shift 2 ;;
        --url)   URL="${2:?--url 后面要跟地址}"; shift 2 ;;
        --build) FORCE_BUILD=1; shift ;;
        *)       UI_ARGS[${#UI_ARGS[@]}]="$1"; shift ;;
    esac
done
if [ -z "$URL" ]; then
    URL="http://127.0.0.1:$PORT"
fi

# --- 编译 ---------------------------------------------------------------
NEED_BUILD=$FORCE_BUILD
if [ ! -d "$HERE/out" ] || [ ! -f "$HERE/out/maj/Main.class" ]; then
    NEED_BUILD=1
elif [ -n "$(find "$HERE/src" -name '*.java' -newer "$HERE/out/maj/Main.class" 2>/dev/null)" ]; then
    NEED_BUILD=1
fi
if [ "$NEED_BUILD" = 1 ]; then
    bash "$HERE/build.sh"
fi

# --- 提示 ---------------------------------------------------------------
if command -v curl >/dev/null 2>&1 && curl -s -m 2 -o /dev/null "$URL/api/models"; then
    echo "▸ $URL 上已经有后端在跑，界面会直接连它（外面起的，界面不负责停）"
else
    echo "▸ $URL 上还没有后端：界面会在你点「开始对局」时自己拉起，关窗口一起停"
    echo "  （想自己起：cd \"$ROOT\" && .venv/bin/python -m aimodel.ui.server --seats human,teacher,teacher --port ${PORT}）"
fi

# --- 界面 ---------------------------------------------------------------
cd "$HERE"
if [ "${#UI_ARGS[@]}" -gt 0 ]; then
    exec java -Dfile.encoding=UTF-8 -cp out maj.Main --url "$URL" --root "$ROOT" "${UI_ARGS[@]}"
else
    exec java -Dfile.encoding=UTF-8 -cp out maj.Main --url "$URL" --root "$ROOT"
fi
