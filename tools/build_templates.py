#!/usr/bin/env python
"""把 ``webapp/static/tiles/*.svg`` 渲染成 ``templates/<牌id>.png`` —— 截图识别要的模板库。

## 为什么要有这个脚本（而不是把 PNG 提交进仓库）

仓库**有意不存位图** ✓（见 ``tools/extract_tiles.py`` 开头：位图糊、又占地方 ✓
牌面一律由 SVG 生成 ✓）。而截图识别的 ``TemplateLibrary`` 要的偏偏是 **PNG** ✗
—— 它 ``glob('*.png')`` 然后拿文件名当牌 id ✓（``docs/平台需求.md`` 也这么写的 ✓）

-> 位图**本地生成** ✓ 脚本进仓库 ✓ 谁需要谁跑一次 ✓

## 为什么渲染完还要「非均匀缩放」一下

qlmanage 会把 SVG **拉满成正方形** ✗ 实测（2026-10-20）：

    viewBox="0 0 300 400"  --qlmanage -s 400-->  400x400
    alpha 包围盒 = **整整 400x400**（不是居中留白）✓
    -> 说明它把 300x400 的内容**均匀拉伸**填满了正方形 ✓
       宽高比 1.000 ✗ 而牌面应该是 **0.75** ✗

**试过但不管用**：给每张图套一层「正方形画布 + 居中 3:4 内层 ``<svg>``」的外壳 ✗
—— qlmanage 连嵌套 ``<svg>`` 一起拉平 ✓ 依旧 400x400 ✗（37 张全中）

**所以改成**：直接渲染原 SVG，再把 400x400 **非均匀缩放回 300x400** ✓
它的拉伸是**均匀覆盖**（实测包围盒就是满幅 ✓）
-> 这个缩回**恰好是那次拉伸的逆运算** ✓ 牌面恢复正常比例 ✓

## ⚠️ 实测：**合成模板不够用**（2026-10-20，生成完立刻测了）

    samples/sanma5.webp（整窗截图）                  -> 手牌只认出 **2 张**（应 13~14）✗
    samples/real/strip-2026-10-14.png（局间压暗难帧）  -> **0 张** ✗（真值 10 张）

-> **SVG 渲染出来的模板匹配不了真实截图** ✗
   真实牌面有缩放 / 抗锯齿 / 阴影 / 桌面底色 / 压暗 ✓ 合成图一个都没有 ✗
-> 原来那份 `templates/` 大概率是**从真实截图逐张裁出来**的 ✓（那才是对的做法 ✗ 已丢失）
-> 所以本脚本的产物**只能让这条链路跑起来、方便迭代** ✓
   **绝不能当成「截图识别修好了」** ✗
   认出的张数会明显偏少（`advise` 会打「手牌读到 N 张」的警告 ✓ 别忽略它 ✗）
-> 要真修好：拿**真实截图**逐张裁 + 人工标牌 id ✓ 存成 `templates/<id>.png` ✓
   现成素材只有 `samples/real/strip-2026-10-14.png` + `labels.json`
   （10 张真值 ✓ 但只有 10 个牌 id ✗ 覆盖不到 37 张 ✗）-> **还需要更多真实帧** ✗

## 用法

    python tools/build_templates.py                 # 生成到 templates/
    python tools/build_templates.py --out /tmp/tpl  # 换个地方（试）
    python tools/build_templates.py --keep          # 已存在也重渲染

依赖：macOS 的 ``qlmanage``（系统自带 ✓ 负责 SVG -> PNG）和 Pillow。
没有 ``qlmanage`` 就**报错退出** ✓ **不静默降级** ✗
（静默降级会让「截图那条路又坏了」变得很难查 —— 本项目栽过这类坑）
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SVG_DIR = ROOT / "webapp" / "static" / "tiles"
#: 不是牌面的两张（背面 / 空白），前端用得上，模板库不需要
NOT_TILES = {"back", "blank"}
#: 交给 qlmanage 的渲染边长（正方形 —— 这是它的行为，我们事后再纠回来）
CANVAS = 400
#: 牌面真实宽高比（viewBox 300x400）—— 输出一律还原成这个比例
FACE_W, FACE_H = 300, 400


def render_one(src: Path, dst: Path) -> tuple[int, int]:
    """渲染一张牌面，并把它从「qlmanage 的正方形」还原成 3:4。"""
    tmp = Path(tempfile.mkdtemp(prefix="tpl_"))
    try:
        subprocess.run(
            ["qlmanage", "-t", "-s", str(CANVAS), "-o", str(tmp), str(src)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # qlmanage 的输出名是 ``<原文件名>.png``
        out = tmp / (src.name + ".png")
        if not out.is_file():
            raise RuntimeError(f"qlmanage 没产出图：{out}")
        im = Image.open(out).convert("RGBA")
        if im.size[0] != im.size[1]:
            # 「均匀拉伸到正方形」是这里做逆缩放的前提 ✓
            # 一旦 qlmanage 换了行为（比如改成居中留白），这个逆缩放就是错的 ✗
            raise RuntimeError(
                f"qlmanage 这次没给正方形（{im.size}）—— 「均匀拉伸」这个前提不成立，"
                "逆缩放的假设要重验"
            )
        # 逆运算：qlmanage 把 3:4 拉成了 1:1 ✗ 这里非均匀缩回 3:4 ✓
        im = im.resize((FACE_W, FACE_H), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst)
        return im.size
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SVG 牌面 -> 截图识别的 PNG 模板库")
    ap.add_argument("--out", default=str(ROOT / "templates"), help="输出目录")
    ap.add_argument("--svg-dir", default=str(SVG_DIR), help="SVG 牌面目录")
    ap.add_argument("--keep", action="store_true", help="输出目录已有 PNG 时也重渲染")
    args = ap.parse_args(argv)

    if shutil.which("qlmanage") is None:
        raise SystemExit(
            "❌ 没有 qlmanage（macOS 自带）。\n"
            "   Linux/Windows 上请用 rsvg-convert / cairosvg 直接渲染成 300x400 ✓\n"
            "   别照抄这里的「逆缩放」✗ —— 那是为了抵 qlmanage 的拉伸才需要的"
        )
    svgs = sorted(Path(args.svg_dir).glob("*.svg"))
    if not svgs:
        raise SystemExit(f"❌ {args.svg_dir} 里没有 .svg —— 先跑 tools/extract_tiles.py")
    tiles = [s for s in svgs if s.stem not in NOT_TILES]
    outdir = Path(args.out)

    done, skipped, bad = 0, 0, []
    for i, src in enumerate(tiles, 1):
        dst = outdir / (src.stem + ".png")
        if dst.is_file() and not args.keep:
            skipped += 1
            continue
        try:
            w, h = render_one(src, dst)
        except Exception as e:                       # noqa: BLE001
            bad.append(f"{src.stem}: {type(e).__name__}: {e}")
            continue
        print(f"  [{i:2d}/{len(tiles)}] {src.stem:4s} -> {w}x{h}（比例 {w/h:.3f}）")
        done += 1

    print()
    print(f"✅ 生成 {done} 张，跳过 {skipped} 张（已存在），失败 {len(bad)} 张 -> {outdir}")
    if bad:
        for b in bad:
            print("  ❌", b)
        raise SystemExit(1)
    print("   ⚠️ 这批模板是从 **SVG 牌面**渲染的，不是真实截图裁出来的 ✓")
    print("      真实对局里牌面有缩放 / 透视 / 压暗 —— 命中率要拿真截图验 ✗")
    print("      现成的验证素材：samples/real/strip-2026-10-14.png + labels.json（10 张真值）✓")


if __name__ == "__main__":
    main(sys.argv[1:])
