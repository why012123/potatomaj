#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  土豆麻将 · Electron 版启动脚本 (macOS)
#  在自带浏览器里开雀魂，并直接读牌局数据（不用装扩展）
#  双击「土豆麻将浏览器.command」等价
# ═══════════════════════════════════════════════════════════

cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
EL="$ROOT/electron"
NODE_BIN="${POTATOMAJ_NODE:-node}"

echo "========================================"
echo "  🀄  土豆麻将 · 雀魂浏览器"
echo "========================================"
echo "  提示直接画在游戏画面上（不开额外窗口）"
echo
echo "  用法: ./run-electron.sh                 默认"
echo "        ./run-electron.sh --no-recapture  不每巡重认手牌（更省，但没有每张牌的收益）"
echo "        ./run-electron.sh --badge         额外开一个独立状态窗口"
echo "        ./run-electron.sh --no-dump       不记录牌局帧"
echo "        ./run-electron.sh --verbose       逐帧打印收到的消息（排查用）"
echo "        ./run-electron.sh --setup-only    只配好环境就退出（不启动游戏）"
echo
echo "  画面上会显示："
echo "    · 每张候选牌上方的「打这张的收益」（金色 = 建议打的那张，红色 = 你已经打的）"
echo "    · 左下角：手牌 / 宝牌 / 各家牌河 / 向听"
echo "  牌局状态写到 build/gamestate.json（现场图 build/shot-*.png）"
echo
echo "  牌局帧默认记录到 build/ws-frames/ —— 用来继续破协议（摸牌那一条已经破一半）。"
echo "  只记牌局消息，登录/账号类不落盘。不想要就加 --no-dump。"
echo

# ---- 1. 依赖 ----
if [ ! -d "$EL/node_modules/electron" ]; then
    echo "▸ 首次运行，正在安装 Electron（约 250MB，走国内镜像）…"
    command -v pnpm >/dev/null 2>&1 || { echo "❌ 需要先安装 pnpm：npm i -g pnpm"; exit 1; }
    ( cd "$EL" && ELECTRON_MIRROR="https://npmmirror.com/mirrors/electron/" pnpm install ) || exit 1
fi

# ---- 2. Electron 二进制 ----
ELECTRON_BIN="$EL/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron"
if [ ! -x "$ELECTRON_BIN" ]; then
    echo "▸ 正在补齐 Electron 二进制…"
    ( cd "$EL" && ELECTRON_MIRROR="https://npmmirror.com/mirrors/electron/" \
        "$NODE_BIN" node_modules/electron/install.js ) || {
        echo "❌ Electron 下载失败。可以手动重试："
        echo "   cd electron && ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ node node_modules/electron/install.js"
        exit 1
    }
fi

# ---- 3. Python 环境 ----
# 牌效和危险度算在 Python 侧（numpy）。缺了的话 Electron 照样能启动，
# 但**一句提示都给不出来** —— 用户看到的是「分析失败」，完全不知道该干什么。
# 所以这里发现缺就自己建，而不是等到运行时再报错。
# 虚拟环境位置可以覆盖（测试要用，也方便用自定义环境）。
# 注意：Electron 侧找的是 $ROOT/.venv/bin/python，要用别的位置得同时设
# POTATOMAJ_PYTHON 指到同一个解释器。
VENV="${POTATOMAJ_VENV:-$ROOT/.venv}"
python_ok() {
    [ -x "$VENV/bin/python" ] || return 1
    "$VENV/bin/python" -c "import numpy, PIL" >/dev/null 2>&1
}
if python_ok; then
    echo "▸ Python 环境就绪（.venv）"
else
    echo "▸ Python 环境还没配好，正在创建并安装依赖…"
    echo "   位置：$VENV"
    command -v python3 >/dev/null 2>&1 || {
        echo "❌ 找不到 python3。装一个再来：https://www.python.org/downloads/"
        read -r -p "按回车关闭…"; exit 1
    }
    if [ ! -x "$VENV/bin/python" ]; then
        python3 -m venv "$VENV" || {
            echo "❌ 创建虚拟环境失败。手动跑："
            echo "   cd \"$ROOT\" && python3 -m venv .venv"
            read -r -p "按回车关闭…"; exit 1
        }
    fi
    echo "   安装依赖（numpy，约 20MB）…"
    "$VENV/bin/python" -m pip install -q --upgrade pip >/dev/null 2>&1
    if ! "$VENV/bin/python" -m pip install -q -r "$ROOT/requirements.txt"; then
        echo "❌ 依赖安装失败（多半是网络）。手动跑："
        echo "   $VENV/bin/pip install -r \"$ROOT/requirements.txt\""
        read -r -p "按回车关闭…"; exit 1
    fi
    python_ok || {
        echo "❌ 依赖装上了但导入还是失败，请把这段发我："
        "$VENV/bin/python" -c "import numpy, PIL" 2>&1 | tail -3
        read -r -p "按回车关闭…"; exit 1
    }
    echo "▸ Python 环境就绪"
fi
echo

if [ "$1" = "--setup-only" ]; then
    echo "▸ --setup-only：环境已备好，不启动游戏。"
    exit 0
fi

# ---- 4. 抓包目录 ----
mkdir -p "$ROOT/build/ws-frames"
echo "▸ 抓到的帧写到： build/ws-frames/"
echo "   （只存牌局消息，登录/账号类不落盘）"
echo
echo "▸ 第一次启动要下载 100MB+ 游戏资源，画面会接近全黑 —— 这是正常的，"
echo "   进度条在屏幕中间偏下。下完之后第二次启动就很快了。"
echo

# ---- 5. 启动 ----
# 输出同时进 build/app.log（终端能实时看，出问题也能把日志发我）。
# 但进程替换（/dev/fd/…）在少数受限环境里会被禁掉，那时就退化成只写文件 ——
# 功能一点不受影响，只是终端里看不到实时输出。
mkdir -p "$ROOT/build"
LOG="$ROOT/build/app.log"
# 日志轮转：挂机打一晚上会写不少（尤其开 --verbose 时是逐帧的），
# 攒到 20MB 就把它挪成 app.log.1，只留一份历史。不清空是因为出问题时
# 上一次的日志往往正好有用。
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG" 2>/dev/null || echo 0)" -gt 20971520 ]; then
    mv -f "$LOG" "$LOG.1" 2>/dev/null || true
    echo "▸ 上次的日志已挪到 build/app.log.1（超过 20MB）"
fi
if ( : > >(cat >/dev/null) ) 2>/dev/null; then
    exec > >(tee -a "$LOG") 2>&1
else
    exec >> "$LOG" 2>&1
    echo "▸ 这个环境不支持进程替换，日志只写文件：$LOG"
    echo "  想实时看就另开一个终端：tail -f \"$LOG\""
fi
echo "──── $(date '+%Y-%m-%d %H:%M:%S') 启动 ────"

# 提示默认画在**游戏页面里**（少一个置顶窗口压在 WebGL 上，流畅得多）。
# --badge 才额外开一个独立的状态窗口；--no-recapture 则不每巡重认手牌。
BADGE_FLAG=""
RECAP_FLAG=""
for a in "$@"; do
    [ "$a" = "--badge" ] && BADGE_FLAG="--badge"
    [ "$a" = "--no-recapture" ] && RECAP_FLAG="--no-recapture"
done

# 有些环境会设 ELECTRON_RUN_AS_NODE=1，那会让 Electron 退化成纯 Node，
# 于是所有窗口参数都会被当成非法选项（报 "bad option: --user-data-dir"）。
# 这里直接清掉，保证它一定以「浏览器」的身份启动。
unset ELECTRON_RUN_AS_NODE

# user-data-dir 放在项目里：游戏资源缓存跟着项目走，不污染系统目录
# POTATOMAJ_ELECTRON_FLAGS 可以追加 Electron 自己的参数
# （比如在受限环境里需要 --no-sandbox）
#
# 默认**不带** --dump：落盘是每帧一次磁盘写 + JSON 序列化，打牌时纯属跟游戏抢时间。
# 现在牌全走协议、协议也早就解完了，只有想留录像研究时才需要 --dump。
#
# 也不带 --tips：牌面气泡默认关（信息框那行「建议打 X」就够），要就加 --tips。
# shellcheck disable=SC2086
exec "$ELECTRON_BIN" \
    --user-data-dir="$ROOT/build/electron-userdata" \
    $POTATOMAJ_ELECTRON_FLAGS \
    "$EL" $BADGE_FLAG $RECAP_FLAG "$@"
