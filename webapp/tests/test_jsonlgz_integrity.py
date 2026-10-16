"""回归测试：`iter_jsonl_gz` **不得静默截断**（⑯「正常返回」≠「完整返回」）。

用法（不需要 pytest ✓ 与平台其余测试同为独立脚本 ✓）：
    .venv\\Scripts\\python.exe webapp/tests/test_jsonlgz_integrity.py

背景（2026-10-16 实况发现）：
    `iter_jsonl_gz` 遇到**中间**的坏字节时，走的是
        `except zlib.error: out = b""; i = total`   # 跳到文件末尾
        ...
        `if not d.unused_data: return`              # 直接返回，后面全不读
    -> **正常返回 ✓ 但只读了 13.6%** ✗ 调用方无从察觉 ✗
    实测代价：1.44 MB 的分片只读出 196,608 B ✓ 其后约 10 小时的对局全丢 ✓

负向验证（铁律④）：
    把 `recording.py::iter_jsonl_gz` 里的 `_next_member` 重同步删掉
    （恢复 `i = total` + `if not d.unused_data: return`）
    -> 本脚本**必须红** ✓（实测：改成旧写法后 5 项里挂 3 项）
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from webapp.server.recording import iter_jsonl_gz  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'✅' if cond else '🔴'} {label}")
    if not cond:
        FAILED.append(label)


def _member(tag: str, n: int = 3) -> bytes:
    """造一个**完整**的 gzip 成员，内含 n 条 JSONL。"""
    body = b"".join(
        json.dumps({"tag": tag, "i": i}).encode() + b"\n" for i in range(n)
    )
    return gzip.compress(body)


def _corrupt_middle(m: bytes) -> bytes:
    """把成员**中段**的字节改成垃圾，制造 `invalid block type` 级别的损坏。

    ⚠️ 必须改中段而不是头部：改头部的话解压器在**第一行之前**就抛 ✓
       那测的是 `gzip.open` 那条路径 ✗ 不是本测试要覆盖的「中途损坏」✓
    """
    b = bytearray(m)
    lo = len(b) // 3
    hi = max(lo + 1, len(b) * 2 // 3)
    for k in range(lo, hi):
        b[k] = 0x5A if b[k] != 0x5A else 0x00
    return bytes(b)


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="jsonlgz_test_"))
    a, b, c = _member("A"), _member("B"), _member("C")

    print("─" * 74)
    print("① 对照组：干净的多成员分片必须**读全**（证明本测试不是恒真 ✓）")
    print("─" * 74)
    p = tmp / "clean.jsonl.gz"
    p.write_bytes(a + b + c)
    st: dict = {}
    recs = list(iter_jsonl_gz(p, st))
    check(len(recs) == 9, f"读到 9 条（实际 {len(recs)}）")
    check({r['tag'] for r in recs} == {"A", "B", "C"}, "三个成员都读到")
    check(st.get("corrupt") is False, "干净文件不报 corrupt")
    check(st.get("complete") is True, "干净且收尾完整 -> complete=True")

    print()
    print("─" * 74)
    print("② ⭐ 核心：中间成员坏掉，**后面那个好成员仍必须被读到**")
    print("   （修复前的失败点：`i = total` -> 跳到末尾 -> C 一条都读不到 ✗）")
    print("─" * 74)
    p2 = tmp / "broken.jsonl.gz"
    p2.write_bytes(a + _corrupt_middle(b) + c)
    st2: dict = {}
    recs2 = list(iter_jsonl_gz(p2, st2))
    tags = {r["tag"] for r in recs2}
    check("A" in tags, "坏点**之前**的成员保住了")
    check("C" in tags, "🔴 坏点**之后**的成员也读到了（不静默截断）")
    check(len([r for r in recs2 if r["tag"] == "C"]) == 3, "C 的三条一条不少")

    print()
    print("─" * 74)
    print("③ ⭐ ⑯的正解：**「正常返回」必须伴随「不完整」的信号**")
    print("   （否则「读全了」和「读到一半就放弃」在调用方看来一模一样 ✗）")
    print("─" * 74)
    check(st2.get("corrupt") is True, "corrupt=True（中间有坏块）")
    check(st2.get("complete") is False, "complete=False")
    check(st2.get("members_corrupt", 0) >= 1, "损坏成员数报出来了")
    check(st2.get("resyncs", 0) >= 1, "重同步次数报出来了")
    check(st2.get("bytes_total", 0) > 0 and st2.get("records", 0) > 0, "基本量齐全")

    print()
    print("─" * 74)
    print("④ 「正在写」≠「损坏」✗ —— 尾巴没收尾是**正常**形态，不能报 corrupt")
    print("   （实况里的分片正是这个形态，报成 corrupt 会造成假警报 ✗）")
    print("─" * 74)
    p3 = tmp / "tail.jsonl.gz"
    p3.write_bytes(a + b[:-8])            # 砍掉最后一个成员的 CRC/ISIZE
    st3: dict = {}
    recs3 = list(iter_jsonl_gz(p3, st3))
    check(st3.get("tail_open") is True, "tail_open=True")
    check(st3.get("corrupt") is False, "🔴 尾截断不报 corrupt（不产生假警报）")
    check(st3.get("complete") is False, "但也不算 complete")
    check(len(recs3) >= 3, "A 至少读全")

    print()
    print("─" * 74)
    print("⑤ 向后兼容：不传 `stats` 时行为与旧版一致（现有调用方一个都不用改）")
    print("─" * 74)
    p4 = tmp / "opt.jsonl.gz"
    p4.write_bytes(_member("A") + _member("B"))
    check(len(list(iter_jsonl_gz(p4))) == 6, "只传一个参数也读得到 6 条")

    print()
    print("=" * 74)
    if FAILED:
        print(f"🔴 **失败 {len(FAILED)} 项**：")
        for f in FAILED:
            print(f"    · {f}")
        print("=" * 74)
        return 1
    print("✅ 全部通过")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
