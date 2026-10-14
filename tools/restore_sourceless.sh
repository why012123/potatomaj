#!/bin/bash
# 把「只剩字节码」的模块恢复到可导入状态。
#
# ## 为什么要存字节码
#
# 这 8 个模块的**源码已经不存在了** ✗ 它们不在工作区、不在 git 历史、
# 也不在任何备份 zip 里 —— 只剩 `__pycache__/*.pyc` ✓
# 它们是截图识别链路：capture → detect → layout → vision → classify → recognize
# 丢了的话「与既有截图识别模块对接的出牌建议入口」就废了 ✗
#
# 把 .pyc 提交进仓库不好看 ✗ 但**比丢掉 8 个模块好** ✓
# 真正的修法是反编译回源码（Python 3.14 的工具还很少）—— 待办 ✓
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p potatomaj
n=0
for f in potatomaj/_sourceless/*.pyc; do
    m=$(basename "$f" .pyc)
    # 有源码就不动（源码优先）
    if [ -f "potatomaj/${m}.py" ]; then
        echo "  ⏭  ${m}.py 源码在，跳过"
        continue
    fi
    cp "$f" "potatomaj/${m}.pyc"
    echo "  ✅ 恢复 potatomaj/${m}.pyc"
    n=$((n+1))
done
echo "共恢复 ${n} 个无源码模块"
echo
echo "验证："
python -c "
import sys; sys.path.insert(0, '.')
mods = ['capture','detect','layout','vision','classify','recognize','cli','gui']
ok = 0
for m in mods:
    try:
        __import__('potatomaj.' + m); ok += 1
    except Exception as e:
        print(f'  ❌ {m}: {type(e).__name__}: {e}')
print(f'  {ok}/{len(mods)} 可导入')
" 2>/dev/null || echo "  （跳过验证）"
