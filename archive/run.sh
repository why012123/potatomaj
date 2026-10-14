#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  土豆麻将 · 启动脚本 (macOS)
#  双击这个文件，或在终端执行 ./run.sh
# ═══════════════════════════════════════════════════════════

cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
VENV="$ROOT/.venv"

echo "========================================"
echo "  🀄  土豆麻将 · 雀魂截图识图"
echo "========================================"

# ---- 1. 准备虚拟环境 ----
if [ ! -x "$VENV/bin/python" ]; then
    echo "▸ 首次运行，正在创建虚拟环境…"
    python3 -m venv "$VENV" || { echo "❌ 创建虚拟环境失败，请先安装 python3"; exit 1; }
fi

PY="$VENV/bin/python"

# ---- 2. 检查依赖 ----
if ! "$PY" -c "import numpy, PIL, PySide6" >/dev/null 2>&1; then
    echo "▸ 正在安装依赖（首次会下载 PySide6，约 100MB，请稍候）…"
    "$PY" -m pip install --upgrade pip --quiet 2>/dev/null
    "$PY" -m pip install -r "$ROOT/requirements.txt" --quiet || {
        echo "❌ 依赖安装失败。可以手动执行："
        echo "   $PY -m pip install -r requirements.txt"
        exit 1
    }
fi

# ---- 3. 检查模板 ----
if [ ! -f "$ROOT/templates/index.json" ]; then
    echo "❌ 找不到牌面模板 templates/，请确认项目完整"
    exit 1
fi

# ---- 4. 屏幕录制权限提示 ----
echo "▸ 提示：用「屏幕」当截图来源时，需要在"
echo "   系统设置 → 隐私与安全性 → 屏幕录制 里勾选你的终端"
echo "   （用「剪贴板」来源则不需要权限）"
echo

# ---- 5. 启动 ----
echo "▸ 启动悬浮窗…"
exec "$PY" -m potatomaj.gui "$@"
