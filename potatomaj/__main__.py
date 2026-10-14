"""``python -m potatomaj`` —— 这里现在没有命令行入口了。

牌全部来自**协议解码**（electron/lib/actions.js），界面是 Electron 那套。
以前这里有个「把截图认成牌局」的命令行工具，那一整条识图的路已经删掉了 ——
原因写在 electron/lib/liqi.js 开头的注释里。

    python -m potatomaj           # 显示这段说明
    python -m potatomaj.bridge    # 给界面用的 stdio 服务（一般不手动跑）
"""

from __future__ import annotations


def main() -> int:
    print(__doc__.strip())
    print()
    print("日常使用请双击：土豆麻将.command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
