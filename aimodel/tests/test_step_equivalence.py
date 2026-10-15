"""数值等价性测试：**重构 `Net.step()` 时的安全网**。

为什么需要它（派单①的纪律）
========================
「把头批量化」是**重构**（行为应当不变）✗ 不是**改算法** ✓
重构的安全网只有一个：**改造前后数值一致** ✓
没有它，你没法知道自己是「快了」还是「快了但算错了」✗ —— 后者是负优化 ✓

做法
====
* 用**固定 seed** 的合成数据（不依赖任何真实牌谱 ✓ 可复现 ✓）
* 跑固定步数，对权重做**确定性校验和** ✗
* 校验和必须与 `GOLDEN` 一致 ✓（`GOLDEN` 是**改造前**实测写入的 ✓）

⚠️ 校验和必须**对「某个头漏乘梯度」敏感** ✗
（那正是最可能的注错方式 —— 逐头循环改批量时漏掉一个头 ✓）
sensitivity 自检在最后 ✓

跑法：``.venv/bin/python aimodel/tests/test_step_equivalence.py``
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel.encode import ACTION_DIM, DIM, HEAD_SIZE  # noqa: E402
from aimodel.net import KINDS, Net  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


#: 用四麻的维度（560），与派单里的对照口径一致
DIM4 = 560
N = 1536          # 足够小以便快速跑，又足够覆盖 9 个头
BATCH = 256
STEPS = 3

#: **改造前实测写入的金值**（见文件末尾 `--record` 说明）
#:
#: ⚠️⚠️ **这个金值是在「KL 分支恒为 False」的状态下录制的** ✗
#:
#: 背景（server 2026-10-16 实测发现）：`Net.step()` 里的 KL 锚定段落
#: 因为 `ref_probs[k]`（全数据集，N_k 行）与 minibatch 的 `p`（batch 行）
#: **形状永不相等**，所以那个 `if` 恒为 False -> 整段（梯度 + 测量）从不执行
#: -> `kl` 恒等于 fallback `0.0`。
#:
#: 后果 ✗：**这个金值只在「KL 不生效」的前提下有效** ✓
#:   · 若**以后修好 KL**（让形状对齐 / 改成现算），`step()` 的数值**必然改变** ✓
#:     -> 那时这个金值**会失效** ✓ 必须**重新录制** ✓ 并且**重测所有依赖它的结论** ✗
#:     （尤其：A 组的 +1.893 是「无锚」数字 ✓ 修好 KL 后必须重跑 ✓）
#:   · 因此：**本次重构（头批量化）必须原样保留 KL 分支的行为** ✗
#:     不修、不"顺手改对"、不改变它是否执行 ✓
#:     —— 否则你分不清「重构引入了偏差」还是「KL 被激活了」✗
#:
#: 判据（mac 采纳为正式验收条款）：重构后若数值不一致，
#:   ① 先看 `kl` 是否从 0.0 变成非 0 ✗
#:      变了 -> KL 分支被激活 ✓ 那是「修好了」不是「改错了」✓
#:      没变 -> 才去查搬运错误 ✓
GOLDEN = "ee6024dc7f6fa27c727312eb13f81a8c5fadc20414e6b683bd4368b1327718be"


def make_data(seed: int = 7) -> dict:
    """固定 seed 的合成数据。**不依赖任何真实牌谱** ✓"""
    rng = np.random.default_rng(seed)
    kind = np.array(KINDS * (N // len(KINDS) + 1))[:N]
    rng.shuffle(kind)
    mask = np.zeros((N, 34), dtype=np.float32)
    action = np.zeros(N, dtype=np.int64)
    for k in KINDS:
        m = kind == k
        sz = HEAD_SIZE[k]
        # 前 sz 列合法（discard 用前 14 张，其余按头宽）✓
        n_ok = 14 if k == "discard" else sz
        mask[m, :n_ok] = 1.0
        action[m] = rng.integers(0, n_ok, size=int(m.sum()))
    return {
        "X": rng.normal(size=(N, DIM4)).astype(np.float32),
        "mask": mask,
        "actfeat": rng.normal(size=(N, 34, ACTION_DIM)).astype(np.float32),
        "action": action,
        "logp": np.zeros(N, dtype=np.float32),
        "kind": kind,
        "adv": rng.normal(size=N).astype(np.float32),
        "ret": rng.normal(size=N).astype(np.float32),
    }


def weight_checksum(net: Net) -> str:
    """确定性校验和：按**参数名排序**后逐数组喂进 sha256 ✓

    ⚠️ 必须对「某个头漏乘梯度」敏感 ✗ —— 见末尾的敏感性自检 ✓
    """
    h = hashlib.sha256()
    for name, arr in sorted(net.params(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


def run_steps(net: Net, data: dict, steps: int = STEPS) -> None:
    """就地跑 `steps` 步（每步一个 batch，按固定顺序切）✓"""
    starts = list(range(0, N, BATCH))[:steps]
    for s in starts:
        sel = slice(s, s + BATCH)
        b = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
        b["mode"] = "pg"
        net.step(
            b, clip=0.2, ent_coef=0.008, vf_coef=0.5,
            ent_target=None, ent_coef_adapt=False,
        )


def checksum_of_current_impl() -> str:
    net = Net(dim=DIM4, hidden=(512, 512), seed=123, lr=3e-4)
    data = make_data()
    run_steps(net, data)
    return weight_checksum(net)


def main(argv: list[str]) -> int:
    if "--record" in argv:
        print("  （录制模式：把下面这行填进 GOLDEN）")
        print(f'GOLDEN = "{checksum_of_current_impl()}"')
        return 0

    print("== 1) 改造前后数值必须一致 ==")
    got = checksum_of_current_impl()
    print(f"  当前实现: {got}")
    print(f"  金值    : {GOLDEN}")
    if GOLDEN == "PENDING":
        print("  ⚠️ GOLDEN 还没录制 —— 先跑一次 --record 填进去 ✗")
        FAILED.append("GOLDEN 未录制")
    else:
        check(got == GOLDEN,
              f"step() 的数值变了！重构必须行为不变 ✗\n"
              f"      期望 {GOLDEN}\n      实得 {got}")

    print("\n== 2) 校验和必须**确定性**（同输入两次跑结果相同）==")
    again = checksum_of_current_impl()
    check(again == got, "同一输入两次跑的校验和不同 -> 校验和本身不确定 ✗（测试无效）")
    print(f"  第二次: {again}  {'一致 ✓' if again == got else '不一致 ✗'}")

    print("\n== 3) ⭐ 敏感性自检：校验和抓不抓得到「某个头漏乘梯度」==")
    print("     （这正是逐头循环改批量时最可能的错法 ✓）")
    net = Net(dim=DIM4, hidden=(512, 512), seed=123, lr=3e-4)
    data = make_data()
    run_steps(net, data)
    base = weight_checksum(net)

    # 模拟「某个头漏乘梯度」：把该头的策略权重整体回退一步（相当于它没被训）
    hit = None
    for name, arr in net.params():
        if name.startswith("Wp_") and name.endswith("ron"):
            arr += 1e-3          # 极小扰动，模拟漏掉一个头
            hit = name
            break
    check(hit is not None, "找不到用于注入的头权重（测试要更新）")
    perturbed = weight_checksum(net)
    check(perturbed != base,
          f"校验和对「{hit} 的 1e-3 扰动」不敏感 -> 它抓不到漏头 ✗（测试无效）")
    print(f"  扰动 {hit} 1e-3 -> 校验和变化: {perturbed != base} ✓")

    if FAILED:
        print(f"\n❌ {len(FAILED)} 条失败")
        return 1
    print("\n✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
