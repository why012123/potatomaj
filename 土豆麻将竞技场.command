#!/bin/bash
# 双击这个文件：打开 ELO 混战池（不同模型 / 不同版本互相对打，实时天梯 + 图表）
#
# 想要四麻池：在终端里跑 ./aimodel/ui/run-arena.sh --players 4
cd "$(dirname "$0")" || exit 1
exec ./aimodel/ui/run-arena.sh "$@"
