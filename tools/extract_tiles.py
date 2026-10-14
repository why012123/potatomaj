#!/usr/bin/env python
"""从 npm 包 ``riichi-mahjong-tiles`` 抽出立直麻将牌面，生成 SVG（37 张）。

这套牌面是 **FluffyStuff 的 CC0 立直牌**（就是天凤 / 雀魂那个风格），
仓库里不放位图（糊、又占地方），而是生成 SVG：清晰、体积小、任何分辨率都不糊。

    python tools/extract_tiles.py                      # 自动找包
    python tools/extract_tiles.py --pkg <目录>          # 指定解压好的包目录
    python tools/extract_tiles.py --out webapp/static/tiles

包里的每个组件都是 ``<svg viewBox="0 0 300 400">…</svg>`` + React 的
camelCase 属性（``fillRule`` / ``strokeWidth`` …）。这里把它转回标准 SVG
属性，输出：

``1m..9m / 0m / 1p..9p / 0p / 1s..9s / 0s / 1z..7z / back / blank``

如果连包都没有，就直接从 ``riichi-mahjong-tiles`` 的 tgz 里读
（``--tgz``），不需要解压。
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "webapp" / "static" / "tiles"

#: 组件名 -> 牌 id
COMPONENT_TO_ID = {
    **{f"Man{i}": f"{i}m" for i in range(1, 10)},
    **{f"Pin{i}": f"{i}p" for i in range(1, 10)},
    **{f"Sou{i}": f"{i}s" for i in range(1, 10)},
    "Man5Dora": "0m",
    "Pin5Dora": "0p",
    "Sou5Dora": "0s",
    "Ton": "1z",
    "Nan": "2z",
    "Shaa": "3z",
    "Pei": "4z",
    "Haku": "5z",
    "Hatsu": "6z",
    "Chun": "7z",
    "Back": "back",
    "Blank": "blank",
}

#: React 的 camelCase 属性 -> SVG 属性
CAMEL = {
    "fillRule": "fill-rule",
    "fillOpacity": "fill-opacity",
    "strokeWidth": "stroke-width",
    "strokeLinecap": "stroke-linecap",
    "strokeLinejoin": "stroke-linejoin",
    "strokeOpacity": "stroke-opacity",
    "strokeDasharray": "stroke-dasharray",
    "strokeMiterlimit": "stroke-miterlimit",
    "clipPath": "clip-path",
    "clipRule": "clip-rule",
    "fillOpacity2": "fill-opacity",
    "stopColor": "stop-color",
    "stopOpacity": "stop-opacity",
}

#: 只保留这些属性（style 会被展开，其余丢掉）
KEEP = {
    "d", "fill", "fill-rule", "fill-opacity", "stroke", "stroke-width",
    "stroke-linecap", "stroke-linejoin", "stroke-opacity", "stroke-dasharray",
    "stroke-dashoffset", "stroke-miterlimit", "transform", "cx", "cy", "r",
    "rx", "ry", "x", "y", "x1", "y1", "x2", "y2", "width", "height",
    "points", "offset", "stop-color", "stop-opacity", "clip-path", "clip-rule",
    "id", "mask", "opacity", "gradientUnits", "patternUnits", "viewBox",
}


def strip_jsx(src: str) -> str:
    """把 React 组件里的 ``<svg>…</svg>`` 抠出来，转成标准 SVG。

    用**手写扫描器**而不是正则：JSX 里的属性有

    * ``style={{ fill: "#000", fillRule: "evenodd" }}`` —— 要摊平成属性
    * ``{...props}`` / ``transform={x}`` —— 整段丢掉
    * ``<mask id="a_svg__b">`` 这种带下划线的 id —— 正则很容易误伤

    正则版本踩过一次：``xmlns:xlink`` 的值被当成独立属性留下来，生成出
    ``<svg xmlns:xlink="..." xmlns="..." http://www.w3.org/1999/xlink" ...>``
    这种 XML 直接报错的垃圾（37 张里有 24 张渲染不出来）。
    逐字符扫描虽然啰嗦，但不会有这种误伤。
    """
    i = src.find("<svg")
    j = src.find("</svg>")
    if i < 0 or j < 0:
        raise ValueError("这个组件里没有 <svg>")
    body = src[i : j + 6]
    return _scan(body).strip()


def _expand_style(inner: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k, v in re.findall(r"([A-Za-z][\w]*)\s*:\s*(\"[^\"]*\"|'[^']*'|[^,}]+)", inner):
        out.append((CAMEL.get(k, k), v.strip().strip("\"'")))
    return out


def _scan(body: str) -> str:
    """扫一遍，重建出干净的 SVG 文本。"""
    out: list[str] = []
    n = len(body)
    i = 0
    while i < n:
        ch = body[i]
        if ch != "<":
            # 文本节点：丢掉 JSX 表达式
            if ch == "{":
                i = _skip_brace(body, i)
                continue
            if ch == "}":
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        # 标签：找它的结束 '>'，注意属性值里的 '>' 要跳过
        k = i + 1
        while k < n:
            if body[k] in "\"'":
                q = body[k]
                k = body.find(q, k + 1)
                if k < 0:
                    k = n
                else:
                    k += 1
                continue
            if body[k] == ">":
                break
            k += 1
        tag = body[i : k + 1]
        out.append(_clean_tag(tag))
        i = k + 1
    return "".join(out)


def _skip_brace(body: str, i: int) -> int:
    """跳过 ``{...}``（可能嵌套）。"""
    depth = 0
    n = len(body)
    while i < n:
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _clean_tag(tag: str) -> str:
    """``<path style={{...}} fillRule="x" cx={12} {...p} />`` -> 干净的标准 SVG 标签。

    JSX 里属性值可能是表达式：``cx={135.055}``。**不能直接丢掉** —— 丢掉的
    结果是 ``cx`` 变成空串、几何塌缩到原点，牌面上会出现一个红色大方块
    （0m/0p/0s 三张就是这么坏掉的）。这里的办法是：先把 ``style={{...}}``
    里的字面量收成一张表，表达式优先从表里查；查不到才丢。
    """
    if tag.startswith("</"):
        return tag
    self_close = tag.rstrip().endswith("/>")
    inner = tag[1:-2].strip() if self_close else tag[1:-1].strip()
    m = re.match(r"([A-Za-z_][\w:.-]*)", inner)
    if not m:
        return tag
    name = m.group(1)
    rest = inner[m.end() :]
    raw: list[tuple[str, str, bool]] = []      # (名字, 值, 是不是表达式)
    n = len(rest)
    i = 0
    while i < n:
        if rest[i].isspace():
            i += 1
            continue
        if rest[i] == "{":                     # {...props} 之类，整段丢
            i = _skip_brace(rest, i)
            continue
        am = re.match(r"([A-Za-z_][\w:.-]*)\s*=\s*", rest[i:])
        if not am:
            i += 1
            continue
        aname = am.group(1)
        i += am.end()
        expr = False
        if i < n and rest[i] in "\"'":
            q = rest[i]
            end = rest.find(q, i + 1)
            if end < 0:
                break
            aval = rest[i + 1 : end]
            i = end + 1
        elif i < n and rest[i] == "{":
            j = _skip_brace(rest, i)
            aval = rest[i + 1 : j - 1]
            expr = True
            i = j
        else:
            am2 = re.match(r"[\w./#:-]+", rest[i:])
            aval = am2.group(0) if am2 else ""
            i += len(aval)
        raw.append((aname, aval, expr))

    # style 里的字面量收成表，给表达式兜底
    table: dict[str, str] = {}
    for aname, aval, expr in raw:
        if aname == "style" and expr:
            for k, v in _expand_style(aval.strip()):
                table.setdefault(k, v)

    attrs: list[tuple[str, str]] = []
    for aname, aval, expr in raw:
        if aname == "style" and expr:
            attrs.extend(_expand_style(aval.strip()))
            continue
        if expr:
            key = CAMEL.get(aname, aname)
            if key in table:
                aval = table[key]
            elif re.fullmatch(r"-?[\d.]+", aval.strip()):
                aval = aval.strip()
            else:
                continue          # 算不出来就丢（宁缺勿错）
        key = CAMEL.get(aname, aname)
        if key in KEEP or key in ("xmlns", "viewBox", "xmlns:xlink"):
            attrs.append((key, aval))

    seen: set[str] = set()
    keep: list[str] = []
    for k, v in attrs:
        if k in seen or v == "":
            continue
        seen.add(k)
        keep.append(f'{k}="{v}"')
    if "xmlns" not in seen:
        keep.insert(0, 'xmlns="http://www.w3.org/2000/svg"')
    if "xmlns:xlink" not in seen:
        keep.insert(1, 'xmlns:xlink="http://www.w3.org/1999/xlink"')
    return f"<{name} {' '.join(keep)}{' /' if self_close else ''}>"


def read_from_dir(pkg: Path) -> dict[str, str]:
    """从解压好的包里读 ``components/regular/regular/*.tsx``。

    两个特殊项：

    * ``Haku``（白板）：这套素材里它就是一个空白组件（真牌的白板本来也是
      一张空牌面），所以这里用牌面底色 + 细边自己拼一张。
    * ``Back``：牌背不在 regular 目录里，用 ``back/regular/BackBackBlue.tsx``。
    """
    d = pkg / "components" / "regular" / "regular"
    if not d.is_dir():
        raise SystemExit(f"找不到 {d}")
    out = {}
    for f in sorted(d.glob("Regular*.tsx")):
        name = f.stem[len("Regular") :]
        if name in ("Front",):
            continue
        out[name] = f.read_text(encoding="utf-8")
    return out


def read_from_tgz(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with tarfile.open(path) as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            stem = Path(m.name).stem
            if "/regular/regular/Regular" in m.name:
                name = stem[len("Regular") :]
                if name == "Front":
                    continue
            elif stem.startswith("BackBack"):
                # 牌背：BackBackBlue -> Back
                name = "Back"
            else:
                continue
            fh = t.extractfile(m)
            if fh is not None:
                out[name] = fh.read().decode("utf-8")
    return out


def find_sources(pkg: str | None, tgz: str | None) -> dict[str, str]:
    if tgz:
        return read_from_tgz(Path(tgz))
    cands = []
    if pkg:
        cands.append(Path(pkg))
    cands += [
        ROOT / "_tools" / "tilesrc" / "riichitiles" / "package",
        ROOT / "webapp" / "client" / "node_modules" / "riichi-mahjong-tiles",
        ROOT / "node_modules" / "riichi-mahjong-tiles",
    ]
    for c in cands:
        if (c / "components" / "regular" / "regular").is_dir():
            return read_from_dir(c)
    for t in [
        ROOT / "_tools" / "tilesrc" / "riichi-mahjong-tiles-0.1.0.tgz",
    ]:
        if t.exists():
            return read_from_tgz(t)
    raise SystemExit(
        "找不到 riichi-mahjong-tiles。先跑：\n"
        "  npm pack riichi-mahjong-tiles --registry https://registry.npmmirror.com\n"
        "（把 tgz 放到 _tools/tilesrc/，或 --tgz 指定）"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=None, help="解压好的包目录")
    ap.add_argument("--tgz", default=None, help="没解压的 .tgz")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args(argv)

    comps = find_sources(args.pkg, args.tgz)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for f in out.glob("*.svg"):
            f.unlink()

    written: list[str] = []
    missing: list[str] = []
    # 白板：这套素材里 Haku 是个空白组件（真牌的白板本来也是空牌面），
    # 所以直接换成「牌面底色 + 细边」，别去解析那个空文件。
    comps["Haku"] = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 400">'
        '<rect x="0" y="0" width="300" height="400" rx="26" fill="#f3efe9"/>'
        '<rect x="4" y="4" width="292" height="392" rx="24" fill="none" '
        'stroke="#c8c1b4" stroke-width="4"/></svg>'
    )
    for comp, tid in COMPONENT_TO_ID.items():
        src = comps.get(comp)
        if not src:
            missing.append(f"{tid}({comp})")
            continue
        try:
            svg = strip_jsx(src)
        except ValueError as exc:
            missing.append(f"{tid}({comp}: {exc})")
            continue
        (out / f"{tid}.svg").write_text(svg, encoding="utf-8")
        written.append(tid)

    print(f"输出目录：{out}")
    print(f"写出 {len(written)} 张：" + " ".join(sorted(written)))
    if missing:
        print(f"⚠️ 缺 {len(missing)} 张：{missing}")
        return 1
    print("✅ 牌面齐全（1m~9m/0m、1p~9p/0p、1s~9s/0s、1z~7z、back、blank）")
    print("   素材：FluffyStuff 的 CC0 立直牌（经 npm 包 riichi-mahjong-tiles）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
