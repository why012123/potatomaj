#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  对齐上游 / 切分支 / 硬重置**之前**跑一次
#
#  为什么需要这个：`git reset --hard origin/main` 会**静默删掉**
#  「你本地有、但上游没有」的文件 —— 不报错、不提示、不问你。
#
#  这不是假设。本项目已经因此丢过两样东西：
#    · `.gitignore` 里的牌谱保护规则（5 轮后才发现）
#    · `aimodel/selfplay.py`（24KB 的完整交付物，只存在于本地支线）
#
#  用法:
#      bash tools/safe_git_sync.sh              # 默认比对 origin/main
#      bash tools/safe_git_sync.sh origin/其它分支
#
#  输出为空 = 可以安全对齐；有任何一条 ⚠️ = 先抢救再对齐。
# ═══════════════════════════════════════════════════════════
set -uo pipefail
UPSTREAM="${1:-origin/main}"
DANGER=0

echo "═══════════════════════════════════════════════════════"
echo "  对齐前检查（目标上游：${UPSTREAM}）"
echo "═══════════════════════════════════════════════════════"

# ── ① 未提交的改动（硬重置会全部抹掉）──────────────────────
echo
echo "【1】工作区未提交的改动 —— 硬重置会**全部抹掉**"
DIRTY=$(git status --porcelain)
if [ -n "${DIRTY}" ]; then
    DANGER=1
    echo "${DIRTY}" | sed 's/^/   ⚠️  /'
    echo "   → 先 commit 或 stash，再对齐"
else
    echo "   ✅ 干净"
fi

# ── ② 已提交、但上游没有的文件（对齐后会消失）──────────────
echo
echo "【2】本地已提交、但上游没有的文件 —— 对齐后会**消失**"
if git rev-parse --verify "${UPSTREAM}" >/dev/null 2>&1; then
    LOST=$(git diff --name-status HEAD "${UPSTREAM}" | awk '$1=="D"{print $2}')
    if [ -n "${LOST}" ]; then
        DANGER=1
        echo "${LOST}" | sed 's/^/   ⚠️  /'
        echo "   → 这些文件一旦对齐就没了。抢救办法："
        echo "       git checkout <你现在的提交> -- <文件>     # 先捞出来"
        echo "       或者 git push origin HEAD:<备份分支名>     # 先推上去"
    else
        echo "   ✅ 没有（本地没有上游缺的文件）"
    fi
else
    echo "   ⚠️  上游分支 ${UPSTREAM} 不存在（先 git fetch）"
    DANGER=1
fi

# ── ③ 未跟踪文件（git clean -fd 会删掉）────────────────────
echo
echo "【3】🔴 未跟踪文件 —— **这一类就是被清掉过的东西**"
UNTRACKED=$(git ls-files --others --exclude-standard)
if [ -n "${UNTRACKED}" ]; then
    DANGER=1
    echo "${UNTRACKED}" | head -20 | sed 's/^/   ⚠️  /'
    N=$(echo "${UNTRACKED}" | wc -l | tr -d ' ')
    [ "${N}" -gt 20 ] && echo "   …（共 ${N} 个）"
    echo "   → **这些不在 git 里、删了就没了** ✗ 已经因此丢过一次："
    echo "     平台侧的 webapp/（约 2000 行）+ run.py + 对 engine.py 的改动，全部归零 ✗"
    echo "   → 先提交：git add <你写的文件> && git commit"
    echo "   → 如果这些是**别人的**：先让本人提交，或者用手动备份"
else
    echo "   ✅ 没有"
fi

# ── ④ stash 里有没有东西（容易被忘）────────────────────────
echo
echo "【4】stash 里的东西（容易被忘在角落）"
if git stash list | grep -q .; then
    git stash list | sed 's/^/   ·  /'
else
    echo "   ✅ 空"
fi

echo
echo "═══════════════════════════════════════════════════════"
if [ "${DANGER}" -eq 1 ]; then
    echo "  ⚠️  有风险 —— **先处理上面标 ⚠️ 的，再对齐**"
    exit 1
fi
echo "  ✅ 可以安全对齐"
echo "═══════════════════════════════════════════════════════"
