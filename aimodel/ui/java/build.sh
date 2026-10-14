#!/bin/bash
# 编译 Java 牌桌：javac -> out/，再打一个可运行的 jar。
#
#   bash build.sh          正常编译
#   bash build.sh --clean  先删掉 out/ 再编译
#
# 要求：JDK 21（javac 在 PATH 里），没有任何第三方依赖。
set -euo pipefail

cd "$(dirname "$0")" || exit 1

if [ "${1:-}" = "--clean" ]; then
    rm -rf out maj-ui.jar
    echo "▸ 已清空 out/ 和 maj-ui.jar"
fi

if ! command -v javac >/dev/null 2>&1; then
    echo "❌ 找不到 javac，请先装 JDK 21（brew install openjdk@21）" >&2
    exit 1
fi

JAVAC_VERSION="$(javac -version 2>&1 | awk '{print $2}')"
echo "▸ javac $JAVAC_VERSION  →  out/"

mkdir -p out
# 源文件里有大量中文，必须 -encoding UTF-8；
# 打开全部 lint（serial 对 Swing 组件没意义，this-escape 对 Swing 子类必报，关掉），
# 并且 -Werror：有警告就当编译失败。
# shellcheck disable=SC2046
javac -encoding UTF-8 -Xlint:all,-serial,-this-escape,-missing-explicit-ctor -Werror \
      -d out $(find src -name '*.java' | sort)

# 把 class 文件打成 jar（-C out . 表示不写进 out 目录本身）
jar --create --file maj-ui.jar --main-class maj.Main -C out . 2>/dev/null \
    || jar cf maj-ui.jar -C out .
echo "▸ 编译完成：$(find out -name '*.class' | wc -l | tr -d ' ') 个 class，maj-ui.jar 已生成"
echo "  跑界面：bash run.sh"
echo "  跑自检：java -cp out maj.SelfTest http://127.0.0.1:8765"
