#!/bin/bash
# 双击这个文件即可开一桌（macOS 只有 .command 支持双击运行）
# Java 版牌桌：Swing 窗口，人类点牌 / AI 对战 / 配置面板。
cd "$(dirname "$0")" || exit 1
exec ./aimodel/ui/java/run.sh "$@"
