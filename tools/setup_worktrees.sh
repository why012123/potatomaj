#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  给多个 AI 会话各开一个独立的 git worktree
#
#  为什么需要这个：这台机器上**三个会话共用一个工作树**（训练侧 / 数据侧 / 平台侧）✗
#  于是任何一个人跑 `git checkout` / `git clean` / 硬重置，
#  都会把**别人未提交的文件**一起清掉 ✓
#
#  已经因此丢过一次：平台侧的 `webapp/`（约 2000 行：服务端 + 验收脚本 + 前端源码）
#  + 根目录 `run.py` + 它对 `engine.py` 的改动 —— **全部从未提交** ✗
#
#  worktree 之后各人一个目录、各自的分支，互不影响 ✓
#  共享的 .git 只有一份，不占额外空间（工作文件是各自的副本）✓
#
#  用法:
#      bash tools/setup_worktrees.sh              # 在主仓库里跑一次
#      bash tools/setup_worktrees.sh --check      # 只看当前状态，不动手
# ═══════════════════════════════════════════════════════════
set -uo pipefail

MAIN="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "${MAIN}" ]; then
    echo "❌ 当前目录不在 git 仓库里"
    exit 1
fi
PARENT="$(dirname "${MAIN}")"
BASE="$(basename "${MAIN}")"

# 每个会话一个 worktree：名字 → 分支
WORKTREES=(
    "wt-train|train-p0-fixes|训练侧：encode/net/agents/train/rewards"
    "wt-data|data-pipeline|数据侧：mjlog/logcheck/logdata/selfplay 数据"
    "wt-platform|webapp|平台侧：webapp/ + run.py"
)

if [ "${1:-}" = "--check" ]; then
    echo "═══════════════════════════════════════════════════"
    echo "  当前 worktree 状态"
    echo "═══════════════════════════════════════════════════"
    git worktree list
    echo
    echo "⚠️ 未提交的改动（谁会因为别人跑 git clean 而丢东西）："
    git status --porcelain | sed 's/^/   /'
    echo
    echo "⚠️ 未跟踪文件（**就是这类文件被清掉过**）："
    git ls-files --others --exclude-standard | head -20 | sed 's/^/   /'
    N=$(git ls-files --others --exclude-standard | wc -l | tr -d ' ')
    [ "${N}" -gt 20 ] && echo "   …（共 ${N} 个）"
    exit 0
fi

echo "═══════════════════════════════════════════════════"
echo "  主仓库：${MAIN}"
echo "  worktree 建在：${PARENT}/"
echo "═══════════════════════════════════════════════════"

# ── 先检查：现在有没有人会因为这次操作丢东西 ──────────────
DIRTY=$(git status --porcelain)
UNTRACKED=$(git ls-files --others --exclude-standard)
if [ -n "${DIRTY}" ] || [ -n "${UNTRACKED}" ]; then
    echo
    echo "⚠️  工作区**不干净** —— 建 worktree 之前先确认这些不属于别人："
    [ -n "${DIRTY}" ] && echo "${DIRTY}" | sed 's/^/   ·  /'
    [ -n "${UNTRACKED}" ] && echo "${UNTRACKED}" | head -10 | sed 's/^/   ?  /'
    echo
    echo "   → 如果是别人的成果：**先让本人提交**，或者手动备份 ✗"
    echo "   → 建 worktree 本身不会动它们，但之后各人分开就安全了 ✓"
    echo
    read -r -p "   继续？[y/N] " ans
    [ "${ans}" = "y" ] || { echo "已取消"; exit 0; }
fi

for entry in "${WORKTREES[@]}"; do
    IFS='|' read -r name branch desc <<< "${entry}"
    dir="${PARENT}/${BASE}-${name}"
    printf "\n── %s ──\n   %s\n" "${name}" "${desc}"
    if [ -d "${dir}" ]; then
        echo "   ⏭  ${dir} 已存在，跳过"
        continue
    fi
    # 分支不存在就从当前 HEAD 建
    if git show-ref --verify --quiet "refs/heads/${branch}"; then
        git worktree add "${dir}" "${branch}" >/dev/null 2>&1 \
            && echo "   ✅ ${dir}（分支 ${branch}）" \
            || echo "   ❌ 失败：${dir}"
    else
        git worktree add -b "${branch}" "${dir}" >/dev/null 2>&1 \
            && echo "   ✅ ${dir}（新分支 ${branch}）" \
            || echo "   ❌ 失败：${dir}"
    fi
done

echo
echo "═══════════════════════════════════════════════════"
echo "  完成。接下来："
echo "    · 每个会话 cd 到自己的目录干活 ✓"
echo "    · 提交只推自己的分支 ✓ 不要直接推 main ✗"
echo "    · 查看全部：git worktree list"
echo "    · 删掉一个：git worktree remove <路径>"
echo "    · 动 git 前仍然要先跑：bash tools/safe_git_sync.sh"
echo "═══════════════════════════════════════════════════"
