#!/bin/bash
# 双击这个文件即可启动（macOS 只有 .command 支持双击运行）
#
# 和「土豆麻将.command」是同一个东西，只是名字更直白 —— 两个都指向
# run-electron.sh，所以只用维护那一处。留着它是因为之前一直在用这个名字。
cd "$(dirname "$0")" || exit 1
exec ./run-electron.sh "$@"
