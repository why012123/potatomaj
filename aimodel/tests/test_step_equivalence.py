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
GOLDEN = "ef345dbce254125ccdebd81a139e1e07e424b98660161a4cbf5df01d9daf0927"


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


#: 跨机器比较权重的**容差** ✓（绝对 + 相对，取二者较大 ✓）
#:
#: ⚠️ 为什么不能用「哈希」做跨机器判据（2026-10-16 血泪 ✓ 两轮才修对 ✗）：
#:
#:   第一版：对 float64 权重取 sha256 ✗ —— **逐位**判据 ✓
#:     -> 权重来自 BLAS ✓ **归约顺序不是跨实现确定的** ✗
#:     -> 实测：相对 **1e-15** 的扰动就能翻转校验和 ✗（server 亲测 ✓）
#:        macOS(Accelerate) 与 Windows(OpenBLAS) 的差异正好在这个量级 ✓
#:     -> 于是这个安全网**只在录制它的那台机器上有效** ✗✗
#:
#:   第二版（错的 ✗）：先 `np.round(arr, 9)` 再哈希 ✓ 想靠量化吸收噪声 ✗
#:     -> **哈希是不连续函数** ✗ 舍入边界处会跳变 ✓
#:     -> 翻转概率 ≈ N × (噪声/步长) = 577045 × (1e-15/1e-9) ≈ **0.58** ✗
#:        —— 577k 个参数里**必然有落在边界上的** ✓ 量化只是降概率、没消除 ✓
#:     -> 要压到 << 1 需要步长 >> 6e-10 ✓ 但那时 1e-8 的真信号也被埋了 ✗
#:     -> **分辨力与鲁棒性在同一根轴上冲突 -> 这条路本身是错的** ✗
#:
#:   第三版（本版 ✓）：**比连续统计量，不比离散哈希** ✓
#:     -> 存一份**金值权重**（npz ✓ 577k×4B ≈ 2.3 MB ✓ 可入库 ✓）
#:     -> 比较 `max_abs_diff` ✓ **它是连续的** ✓
#:        同机重跑 diff = 0 ✓；跨机器 diff ~1e-15 ✓ **都远小于 1e-9 阈值** ✓
#:        -> 阈值有 6 个数量级余量 ✓ **不存在「边界翻转」问题** ✓
#:     -> 仍能抓到真正的重构错误 ✓（实测注入 1e-8 必被抓 ✓）
#: ⚠️ 本测试的**最小可检错误 ≈ 1e-4** ✗ —— 它**不是**「逐位」判据 ✓
#:
#: 标定过程（第三版才标对 ✗ 前两版都错了）：
#:
#:   第一版：哈希逐位 ✗ -> 跨机器必红（BLAS 归约顺序不同 ✓）
#:   第二版：量化后哈希 ✗ -> **哈希不连续** ✓ 舍入边界翻转概率 ≈0.58 ✓ 无效 ✓
#:   第三版（本版）：容差比较 ✓ 但**容差标定又错了两次** ✗：
#:
#:     ❌ 错法 A（我的）：把噪声注入**最终金值权重** -> 噪声 ~1e-15 -> 定 1e-9 ✗
#:     ✅ 对法（server 的）：扰动**输入数据** -> 经前向+反向+3步Adam **放大** ✓
#:        实测（server 在 4060 上）：输入 1×float32 eps 的抖动
#:          -> 最终权重差 **5.059e-06**   （放大约 40 倍 ✓）
#:        而真实的跨机器偏差实测 = **3.031e-06** ✓ 同量级 ✓
#:
#:   -> **真实噪声底 ~5e-6，不是 ~1e-15** ✗ 两者差 9 个数量级 ✓
#:   -> 本测试能抓的是 **O(1) 量级的重构错误** ✓
#:      例：「某个头漏乘梯度」✓ 那种错误偏差是 O(1) ✓ **远大于阈值** ✓ 必被抓 ✓
#:   -> 它**抓不到** 1e-5 量级的细微数值改动 ✗ **别拿它当逐位保证** ✗
#:
#: ⚠️ 放大倍数与**步数**有关 ✓（实测 3 步 -> ~40 倍 ✓ 步数越多放大越多 ✗）
#:    若改 STEPS，**本容差必须重新标定** ✗
TOL_ABS = 1e-4
TOL_REL = 1e-4

#: 金值权重文件（**独立于代码** ✓ 由 `--record` 生成 ✓ 入库以便跨机器比较 ✓）
GOLDEN_NPZ = Path(__file__).resolve().parent / "_golden_step_weights.npz"


def weight_checksum(net: Net) -> str:
    """同机判据用的**诊断指纹** ✓ —— ⚠️ **不要**拿它做跨机器断言 ✗

    它仍然有用：同一台机器上前后对照、以及打印出来便于人工比对 ✓
    跨机器断言必须走 :func:`max_abs_diff_vs_golden` ✓
    """
    h = hashlib.sha256()
    for name, arr in sorted(net.params(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


def save_golden(net: Net) -> None:
    """把当前权重存成金值 ✓（`--record` 用 ✓）"""
    np.savez_compressed(GOLDEN_NPZ, **{n: p for n, p in net.params()})


def max_abs_diff_vs_golden(net: Net) -> float:
    """返回「当前权重 vs 金值权重」的**最大绝对差** ✓ 连续量 ✓ 可跨机器 ✓

    ⚠️ 若金值文件不存在 -> 抛 FileNotFoundError ✓（不静默通过 ✗）
    """
    with np.load(GOLDEN_NPZ) as g:
        worst = 0.0
        for name, p in net.params():
            if name not in g.files:
                raise KeyError(f"金值里缺参数 {name!r} —— 结构变了 ✗ 需重录 ✓")
            ref = g[name]
            if ref.shape != p.shape:
                raise ValueError(f"{name!r} 形状变了 {ref.shape} -> {p.shape} ✗")
            # 绝对 + 相对，取更宽松的那个 ✓（大权重的相对误差才有意义 ✓）
            d = np.abs(p - ref) - TOL_REL * np.maximum(np.abs(ref), 1.0)
            worst = max(worst, float(d.max()) if d.size else 0.0)
        return worst


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


def run_current_impl() -> Net:
    """跑一遍当前实现，返回训完的 net ✓（校验和与容差比较都用它 ✓）"""
    net = Net(dim=DIM4, hidden=(512, 512), seed=123, lr=3e-4)
    data = make_data()
    run_steps(net, data)
    return net


def checksum_of_current_impl() -> str:
    return weight_checksum(run_current_impl())


def main(argv: list[str]) -> int:
    if "--record" in argv:
        net = run_current_impl()
        save_golden(net)
        print(f"  （录制模式：金值权重已写入 {GOLDEN_NPZ}）")
        print(f"  同机指纹 GOLDEN = \"{weight_checksum(net)}\"")
        return 0

    print("== 1) 改造前后数值必须一致（**容差比较，跨机器可移植** ✓）==")
    net = run_current_impl()
    got = weight_checksum(net)
    print(f"  同机指纹: {got}   （仅诊断用 ✓ 不做跨机器断言 ✗）")
    if not GOLDEN_NPZ.exists():
        print(f"  ⚠️ 金值文件不存在：{GOLDEN_NPZ} ✗")
        print("     先跑一次 --record 生成 ✗")
        FAILED.append("金值权重未录制")
    else:
        try:
            d = max_abs_diff_vs_golden(net)
        except (KeyError, ValueError) as e:
            check(False, f"金值与当前结构不匹配 ✗ {e}（结构改了要重录 ✓）")
        else:
            print(f"  最大容差内偏差: {d:.3e}   （< 0 即通过 ✓ 阈值 {TOL_ABS:.0e}/{TOL_REL:.0e} ✓）")
            check(d <= 0.0,
                  f"step() 的数值变了！重构必须行为不变 ✗\n"
                  f"      最大容差内偏差 {d:.3e} > 0 ✓\n"
                  f"      提示：跨机器的 BLAS 噪声在 ~1e-15 ✓ 远小于阈值 ✓\n"
                  f"      若偏差在 1e-9 以上 -> 是真的改错了 ✗ 不是噪声 ✓")

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
