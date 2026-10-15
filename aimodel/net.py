"""纯 numpy 神经网络：共享躯干 + 多个策略头 + 一个价值头。

为什么不直接用 PyTorch？因为整个工程（含外层的识图模块）只依赖
``numpy + Pillow``；为了一个 20 万参数的小网络装 200MB+ 的框架不划算。
这里手写前向 / 反向 + Adam，实测一次 minibatch（256）更新只要几毫秒。

结构::

    x(DIM) -> Linear -> ReLU -> Linear -> ReLU -> 特征
                                                   |
        +--- discard 头(34) --- call 头(3) --- ... ---+--- value 头(1, tanh)

训练目标是 PPO 的截断目标；**行为克隆就是「优势恒为 1、ratio 恒为 1」的特例**，
所以模仿学习和强化学习走同一条代码路径。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .encode import ACTION_DIM, DIM, HEAD_SIZE

#: 打牌用的动作塔宽度（每张候选牌先变成一个分数，再加到打牌头的 logit 上）
ACTION_HIDDEN = 32

#: 参与训练的决策种类（顺序固定，存档里会记下来）
KINDS: tuple[str, ...] = (
    "discard", "tsumo", "ron", "riichi", "call", "chi", "kita", "kan", "kyuushu",
)

#: 加了新头之后，旧权重里缺哪些头是可以明确说清的。
#: ``chi`` 是四麻的「吃」：engine.py:877 会向 agent 抛 chi 决策，
#: 但老版本 encode.HEAD_SIZE / 这里的 KINDS 都没有它 —— 于是 chi 一条都
#: 没被记录过（实测四麻 12.6% 的决策丢失）。
_NEW_HEADS: dict[str, str] = {
    "chi": "四麻的「吃」头（老版本完全没记录过 chi，见 P0-2）",
}


class Net:
    """策略 + 价值网络。"""

    def __init__(
        self,
        dim: int = DIM,
        hidden: tuple[int, ...] = (256, 256),
        seed: int = 0,
        lr: float = 1e-3,
        l2: float = 1e-5,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.hidden = tuple(hidden)
        self.lr = lr
        self.l2 = l2
        self.kinds = list(KINDS)
        #: 加载时因「旧权重缺这个头」被零初始化补上的决策种类。
        #: MLAgent 见到它们会走启发式兜底，保证老模型行为不变。
        self.untrained_heads: set[str] = set()
        #: **累计训过多少个 epoch**（跨 `--resume` 累加）。
        #:
        #: 为什么要存这个：`--resume` 只恢复**权重**，`train_bc` 里的
        #: `for ep in range(epochs)` 永远从 0 开始 ✗ 于是
        #: `--resume --epochs 30` 不是「训到第 30 轮」，而是
        #: 「在现有权重上**再训 30 轮**」✓ 每跑一次就多加 30 轮 ✓
        #: 存下来才能在续训时把「已经训了多少」显示出来 ✓
        self.epochs_done: int = 0
        #: ``untrained_heads`` 的别名 —— 平台侧请求里用的名字是这个。
        #: 语义完全相同：**这些决策必须交给兜底策略，不要让零初始化的头参与**。
        self.missing_heads: tuple[str, ...] = ()
        #: 逐头熵系数（P1-1）。默认值在第一次 step(ent_coef_adapt=True) 时播种。
        self._ent_coef: dict[str, float] = {}
        #: KL 锚定的参考策略（P1-2）：**不参与求导**的冻结副本。
        #: 全仓库原来 `grep -i "kl|anchor|reference"` 在 train/net 里零命中。
        self.ref_probs: dict[str, np.ndarray] = {}
        self.kl_coef: float = 0.0

        sizes = [dim, *self.hidden]
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(self.hidden)):
            fan_in = sizes[i]
            self.W.append(
                rng.normal(0.0, np.sqrt(2.0 / fan_in), (sizes[i], sizes[i + 1])).astype(np.float32)
            )
            self.b.append(np.zeros(sizes[i + 1], dtype=np.float32))

        last = self.hidden[-1]
        self.Wp: dict[str, np.ndarray] = {}
        self.bp: dict[str, np.ndarray] = {}
        for k in self.kinds:
            n = HEAD_SIZE[k]
            self.Wp[k] = (rng.normal(0.0, np.sqrt(2.0 / last), (last, n)) * 0.1).astype(np.float32)
            self.bp[k] = np.zeros(n, dtype=np.float32)
        self.Wv = (rng.normal(0.0, np.sqrt(2.0 / last), (last, 1)) * 0.1).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)
        # 动作塔：把「这一张打出去会怎样」直接变成一个 logit 偏置
        self.Wa1 = (rng.normal(0.0, np.sqrt(2.0 / ACTION_DIM), (ACTION_DIM, ACTION_HIDDEN))
                    ).astype(np.float32)
        self.ba1 = np.zeros(ACTION_HIDDEN, dtype=np.float32)
        self.Wa2 = (rng.normal(0.0, np.sqrt(2.0 / ACTION_HIDDEN), (ACTION_HIDDEN, 1)) * 0.1
                    ).astype(np.float32)
        self.ba2 = np.zeros(1, dtype=np.float32)

        self._adam: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        #: `_apply` 里按形状复用的两个临时缓冲（避免每步每参数分配临时数组）✓
        #: ⚠️ 只是**缓冲**，不是状态 ✗ 不进存档、不影响数值 ✓
        self._scratch: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        #: Adam 的步数（决定偏差修正）；**读档时必须从 0 开始**
        self.steps = 0
        #: 这个网络累计被更新过多少次（读档后保留，只用于展示）
        self.total_steps = 0

    # -- 参数遍历 -----------------------------------------------------------
    def params(self):
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            yield f"W{i}", w
            yield f"b{i}", b
        for k in self.kinds:
            yield f"Wp_{k}", self.Wp[k]
            yield f"bp_{k}", self.bp[k]
        yield "Wv", self.Wv
        yield "bv", self.bv
        yield "Wa1", self.Wa1
        yield "ba1", self.ba1
        yield "Wa2", self.Wa2
        yield "ba2", self.ba2

    def copy(self) -> "Net":
        other = Net(self.dim, self.hidden, seed=0, lr=self.lr, l2=self.l2)
        for name, p in self.params():
            dict(other.params())[name][...] = p
        # ⚠️ 这里**必须把「标量状态」全部搬过去** ✗
        #    漏一个的后果是静默的：`cmd_bc` 存的是 `best_net`（copy 出来的副本），
        #    漏了 `epochs_done` 就导致「续训时累计轮数显示不出来」✓
        #    （实测踩过：加了这个字段之后 copy() 没跟上 ✓）
        #    `test_net.py` 里有一条测试会**自动**抓出这类遗漏 ✓ 别再靠人记 ✓
        other.kinds = list(self.kinds)
        other.steps = self.steps
        other.total_steps = self.total_steps
        other.epochs_done = self.epochs_done
        other.untrained_heads = set(self.untrained_heads)
        other.missing_heads = tuple(self.missing_heads)
        return other

    def n_params(self) -> int:
        return sum(p.size for _, p in self.params())

    # -- KL 锚定（P1-2）-----------------------------------------------------
    def set_reference(self, ref: "Net", data: dict, kl_coef: float = 0.1) -> None:
        """冻结一份参考策略，之后 ``step`` 会把策略往它身上拉。

        为什么需要：BC 学出来的牌效是**好权重**，而 RL 训崩了会把它冲掉。
        老代码的「新高才保存」只保护**文件**，不保护**内存里的权重** ——
        崩掉那一轮之后的训练数据就是用崩掉的策略采的，越跑越远。
        加一个 KL 锚点，等于给 RL 系了根安全绳。

        ``ref_probs`` 按决策头整批缓存（不是逐样本），所以内存很小。
        """
        self.ref_probs = {}
        for k in self.kinds:
            idx = np.nonzero(data["kind"] == k)[0]
            if len(idx) == 0:
                continue
            size = HEAD_SIZE[k]
            m = data["mask"][idx][:, :size]
            af = None
            if k == "discard" and data.get("actfeat") is not None:
                af = np.asarray(data["actfeat"][idx], dtype=np.float32)
            z = ref.logits(ref.trunk(data["X"][idx]), k, af)
            z = np.where(m > 0, z, -1e9)
            self.ref_probs[k] = softmax_rows(z).astype(np.float32)
        self.kl_coef = float(kl_coef)

    # -- 前向 ---------------------------------------------------------------
    def _forward(self, X: np.ndarray):
        """返回 (每层输入, 每层线性输出)；最后一层隐层不接激活函数。"""
        acts = [X]
        zs: list[np.ndarray] = []
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ w + b
            zs.append(z)
            acts.append(np.maximum(z, 0.0) if i < len(self.W) - 1 else z)
        return acts, zs

    def trunk(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[0][-1]

    def logits(self, h2: np.ndarray, kind: str, actfeat: np.ndarray | None = None) -> np.ndarray:
        """策略头输出。打牌头会额外加上「动作塔」给出的每张牌分数。"""
        z = h2 @ self.Wp[kind] + self.bp[kind]
        if kind == "discard" and actfeat is not None:
            z = z + self.action_scores(actfeat)
        return z

    def action_scores(self, actfeat: np.ndarray) -> np.ndarray:
        """``(N, 34, ACTION_DIM)`` -> ``(N, 34)``：每张牌打出去值多少分。"""
        a1 = np.maximum(actfeat @ self.Wa1 + self.ba1, 0.0)
        return (a1 @ self.Wa2 + self.ba2)[..., 0]

    def value(self, h2: np.ndarray) -> np.ndarray:
        return np.tanh(h2 @ self.Wv + self.bv)[:, 0]

    # -- 推理 ---------------------------------------------------------------
    def policy(
        self, x: np.ndarray, kind: str, mask: np.ndarray,
        actfeat: np.ndarray | None = None,
    ) -> np.ndarray:
        """单个局面的动作概率（已按掩码归一化）。"""
        h2 = self.trunk(x[None, :])
        af = None if actfeat is None else np.asarray(actfeat, dtype=np.float32)[None, ...]
        return softmax_masked(self.logits(h2, kind, af)[0], mask)

    def act(
        self,
        x: np.ndarray,
        kind: str,
        mask: np.ndarray,
        rng: np.random.Generator | None = None,
        greedy: bool = False,
        actfeat: np.ndarray | None = None,
    ) -> tuple[int, float, np.ndarray]:
        """采样一个动作，返回 ``(动作下标, log 概率, 概率)``。"""
        p = self.policy(x, kind, mask, actfeat).astype(np.float64)
        p = p / max(p.sum(), 1e-12)
        if greedy:
            a = int(np.argmax(p))
        else:
            rng = rng if rng is not None else np.random.default_rng()
            a = int(rng.choice(len(p), p=p))
        return a, float(np.log(max(p[a], 1e-12))), p.astype(np.float32)

    # -- 训练 ---------------------------------------------------------------
    def step(
        self,
        batch: dict,
        clip: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        ent_target: dict[str, float] | None = None,
        ent_coef_adapt: bool = False,
        ent_adapt_rate: float = 0.25,
        label_smoothing: float = 0.0,
        kl_coef: float | None = None,
    ) -> dict:
        """一次 minibatch 更新。``batch`` 由 :mod:`aimodel.train` 组装。

        ``ent_target`` / ``ent_coef_adapt``（P1-1）：**分头**自适应熵系数。
        实测打牌头熵 H=0.291（上限 2.27）、argmax 与采样 87.7% 重合 ——
        探索已经塌了，而全局 ent_coef 从 0.01 降到 0.002 之后梯度贡献 <1%。

        做法：每个决策头有自己的熵系数 ``self._ent_coef[k]``，按
        「实际熵 vs 目标熵」做比例反馈：

            coef_k *= exp(rate * (target_k - H_k) / target_k)

        熵低于目标就**调大** coef（逼它多探索），高于目标就调小。
        二分类头（立直/和了）目标熵天然低，所以**必须分头设目标** ——
        用一个全局系数会把这些头推坏。
        """
        X = batch["X"]
        mask = batch["mask"]
        act = batch["action"]
        adv = batch["adv"]
        ret = batch["ret"]
        old_logp = batch["logp"]
        kinds = batch["kind"]
        mode = batch.get("mode", "pg")
        af_all = batch.get("actfeat")
        n = len(X)

        acts, zs = self._forward(X)
        feat = acts[-1]

        grads: dict[str, np.ndarray] = {}
        g_feat = np.zeros_like(feat)
        pol_loss = 0.0
        ent_sum = 0.0
        clip_frac = 0.0
        n_kind = 0
        kl_sum = 0.0
        kl_n = 0
        kc = self.kl_coef if kl_coef is None else float(kl_coef)
        #: 逐头熵（P1-1）：把 8 个头的熵分开记，否则看不出是哪个头塌了
        ent_by_kind: dict[str, float] = {}

        for k in self.kinds:
            idx = np.nonzero(kinds == k)[0]
            if len(idx) == 0:
                continue
            n_kind += 1
            size = HEAD_SIZE[k]
            m = mask[idx][:, :size]
            z = feat[idx] @ self.Wp[k] + self.bp[k]
            af = None
            z1 = a1 = None
            if k == "discard" and af_all is not None:
                af = np.asarray(af_all[idx], dtype=np.float32)
                z1 = af @ self.Wa1 + self.ba1
                a1 = np.maximum(z1, 0.0)
                z = z + (a1 @ self.Wa2 + self.ba2)[..., 0]
            z = np.where(m > 0, z, -1e9)
            p = softmax_rows(z)
            a = act[idx]
            logp = np.log(np.maximum(p[np.arange(len(idx)), a], 1e-12))
            ent = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)

            if mode == "bc":
                ratio = np.ones_like(logp)  # 行为克隆：优势恒为 1、不截断 = 交叉熵
            else:
                ratio = np.exp(np.clip(logp - old_logp[idx], -20, 20))
            a_i = adv[idx]
            unclipped = ratio * a_i
            clipped = np.clip(ratio, 1.0 - clip, 1.0 + clip) * a_i
            use_un = unclipped <= clipped
            coef = np.where(use_un, -ratio * a_i, 0.0)

            # 注意符号：dL/dz = coef * (onehot - p)，写反了就会「越训越差」
            dz = -coef[:, None] * p
            dz[np.arange(len(idx)), a] += coef
            # 软标签 / label smoothing（P1-1）。硬交叉熵 + 确定性教师
            # = 学出「一个确定性策略的三份镜像」，打牌头熵实测只有 0.291。
            #
            # 目标换成 (1-eps)*onehot + eps*(1/K)（K = 该头合法动作数）之后：
            #   损失 = (1-eps)*CE + eps*mean(log K)
            # 第二项与参数**无关**（常数），所以精确梯度就是原来的
            #   dL/dz = coef*(p - onehot)  乘上 (1-eps)
            #
            # 别画蛇添足：`-(eps/K)*coef*(1 - K*p)` 看着像均匀项，但它对每行
            # 求和恒等于 0（softmax 行和对 z 的导数为 0），加上去是错的。
            keep = 1.0
            if mode == "bc" and label_smoothing > 0.0:
                keep = 1.0 - float(min(max(label_smoothing, 0.0), 0.999))
            # 分头熵系数（P1-1）：每个头有自己的系数，别用一个全局值
            k_coef = self._ent_coef.get(k, ent_coef) if ent_coef_adapt else ent_coef
            if k_coef:
                dz += k_coef * p * (np.log(np.maximum(p, 1e-12)) + ent[:, None])
            # KL 锚定（P1-2）：把策略拉回克隆阶段的参考策略。
            #   KL(pref || p) = sum pref*(log pref - log p)
            #   dKL/dz_j = -sum_k pref_k*(delta_kj - p_j) = p_j - pref_j
            # 也就是 **dKL/dz = p - pref** —— 干净得不像话。
            #
            # 我最初写的版本是 `p * (log(p/pref) + KL_row)`，看着像标准推导，
            # 但它**连「对动作求和必须为 0」都不满足**，数值梯度一验就露了
            # （最大绝对差 1.67）。别自己推，用 p - pref。
            if kc > 0.0 and mode != "bc":
                pref = self.ref_probs.get(k)
                if pref is not None and pref.shape == p.shape:
                    dz += kc * (p - pref)
                    kl_row = np.sum(
                        pref * (np.log(np.maximum(pref, 1e-12))
                                - np.log(np.maximum(p, 1e-12))),
                        axis=1,
                    )
                    kl_sum += float(np.mean(kl_row))
                    kl_n += 1
            dz = dz * keep / n

            h_k = float(np.mean(ent))
            ent_by_kind[k] = h_k
            # 比例反馈：熵低于目标 -> 调大系数（逼它探索），高于目标 -> 调小
            if ent_coef_adapt and ent_target and k in ent_target and ent_target[k] > 0:
                ratio = (ent_target[k] - h_k) / ent_target[k]
                new_c = self._ent_coef.get(k, ent_coef) * float(
                    np.exp(np.clip(ent_adapt_rate * ratio, -0.5, 0.5))
                )
                self._ent_coef[k] = float(np.clip(new_c, 1e-4, 1.0))

            grads[f"Wp_{k}"] = feat[idx].T @ dz + self.l2 * self.Wp[k]
            grads[f"bp_{k}"] = dz.sum(axis=0)
            g_feat[idx] += dz @ self.Wp[k].T
            if af is not None:
                da2 = dz[..., None]
                grads["Wa2"] = np.einsum("nck,ncj->kj", a1, da2) + self.l2 * self.Wa2
                grads["ba2"] = da2.sum(axis=(0, 1))
                dz1 = (da2 @ self.Wa2.T) * (z1 > 0)
                grads["Wa1"] = np.einsum("nck,ncl->kl", af, dz1) + self.l2 * self.Wa1
                grads["ba1"] = dz1.sum(axis=(0, 1))

            pol_loss += float(-np.mean(np.where(use_un, unclipped, clipped)))
            ent_sum += h_k
            clip_frac += float(np.mean(~use_un))

        v = np.tanh(feat @ self.Wv + self.bv)[:, 0]
        value_loss = float(np.mean((v - ret) ** 2))
        # 价值损失 L = vf_coef * mean((v-ret)^2)，所以梯度里带那个 2
        dv = (2.0 * vf_coef * (v - ret) / n)[:, None] * (1.0 - v * v)[:, None]
        grads["Wv"] = feat.T @ dv + self.l2 * self.Wv
        grads["bv"] = dv.sum(axis=0)
        g_feat += dv @ self.Wv.T

        # 躯干反向
        g = g_feat
        for i in reversed(range(len(self.W))):
            grads[f"W{i}"] = acts[i].T @ g + self.l2 * self.W[i]
            grads[f"b{i}"] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (zs[i - 1] > 0)

        self.steps += 1
        self.total_steps += 1
        self._apply(grads)
        nk = max(1, n_kind)
        return {
            "policy_loss": pol_loss / nk,
            "value_loss": value_loss,
            # 整体熵保留（向后兼容），但**逐头熵**才是能看出「哪个头塌了」的东西
            "entropy": ent_sum / nk,
            "entropy_by_kind": ent_by_kind,
            # ⚠️ 这个默认值 0.0 是个**静默耦合**（mac 审 kl-anneal 接线时点出来的）：
            #    `tools/kl_anneal.py` 的 DESCEND 条件是 `measured_kl < kl_lo(0.05)`，
            #    而 **0.0 < 0.05** -> 一旦 `kl_n == 0`（这一轮没算 KL），
            #    退火会**每轮误判「有余量」并降档** ✗
            #    现状安全：`--kl-floor` 默认 0.01 且**永不归零** -> kl_coef >= 0.01
            #    -> kl_n > 0 ✓
            #    **但若以后允许 kl_coef == 0，这条就变成真 bug** ✓
            #    -> 那时应当要么让 kl_anneal 显式区分「没算 KL」与「KL 真的很小」，
            #       要么在这里返回 None 并让调用方处理 ✓
            "kl": (kl_sum / kl_n) if kl_n else 0.0,
            "clip_frac": clip_frac / nk,
        }

    # -- Adam ---------------------------------------------------------------
    def _apply(self, grads: dict, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        """Adam 更新。

        ⚠️ **本函数的数学与运算顺序必须与旧版逐位一致** ✓
        它占了 `step()` 的 **47%**（cProfile 实测，见提交记录）✗
        所以这里只做「少分配临时数组」，**不动公式** ✗
        安全网：`aimodel/tests/test_step_equivalence.py` 的金值校验和 ✓
        （金值是在**旧实现**下录制的 ✓ 改完必须仍然匹配 ✓）

        逐位保持的要点（容易写错的地方都标了）：
          · `m = beta1*m + (1-beta1)*g` -> 用 out= 复用缓冲，**加法交换律**保证等价 ✓
          · `p -= lr*mh / (sqrt(vh)+eps)` -> 原式是 **`(lr*mh)/X`**（左结合）✗
            **不能**写成 `mh/X*lr` ✗ 浮点结合律不成立 ✗
          · 标量乘除一律用 `np.multiply(..., out=)`，dtype 与原来一致 ✓
        """
        t = self.steps
        c1 = 1.0 - beta1 ** t
        c2 = 1.0 - beta2 ** t
        b1, b2 = beta1, beta2
        o1, o2 = 1.0 - beta1, 1.0 - beta2
        for name, p in self.params():
            g = grads.get(name)
            if g is None:
                continue
            # ⚠️ 这里原来写的是：
            #     m, v = self._adam.get(name, (np.zeros_like(p), np.zeros_like(p)))
            #   **`dict.get(k, default)` 会先求值 default** ✗
            #   -> 即使键已存在（稳态下**总是存在**），每步每参数都白分配两个
            #      全尺寸数组 ✓ 28 个参数 × 2 = **56 次分配/步** ✗
            #   profile 实测：`zeros_like` 1425 次 / 25 batch = **57 次/步** ✓ 对得上 ✓
            mv = self._adam.get(name)
            if mv is None:
                m = np.zeros_like(p)
                v = np.zeros_like(p)
                self._adam[name] = (m, v)
            else:
                m, v = mv

            # 两个可复用的临时缓冲（按形状缓存，**跨参数跨步复用**）✓
            s, u = self._scratch_for(p)

            # m = beta1*m + (1-beta1)*g   （原地写回 m 里）
            np.multiply(g, o1, out=u)      # u = (1-beta1)*g
            np.multiply(m, b1, out=s)      # s = beta1*m
            np.add(s, u, out=m)            # m = beta1*m + (1-beta1)*g ✓

            # v = beta2*v + (1-beta2)*(g*g)   （原地写回 v 里）
            np.multiply(g, g, out=u)       # u = g*g
            np.multiply(u, o2, out=u)      # u = (1-beta2)*(g*g)
            np.multiply(v, b2, out=s)      # s = beta2*v
            np.add(s, u, out=v)            # v = beta2*v + (1-beta2)*(g*g) ✓

            # p -= (lr * (m/c1)) / (sqrt(v/c2) + eps)
            np.divide(m, c1, out=s)        # s = mh
            np.divide(v, c2, out=u)        # u = vh
            np.sqrt(u, out=u)              # u = sqrt(vh)
            np.add(u, eps, out=u)          # u = sqrt(vh) + eps
            np.multiply(s, self.lr, out=s)  # s = lr*mh   （**先乘 lr**，保住左结合 ✓）
            np.divide(s, u, out=s)         # s = (lr*mh) / (sqrt(vh)+eps) ✓
            if s.dtype != p.dtype:
                s = s.astype(p.dtype)
            p -= s

    def _scratch_for(self, p: np.ndarray):
        """按形状缓存两个临时缓冲（避免每步每参数都分配）✓"""
        key = (p.shape, p.dtype.str)
        got = self._scratch.get(key)
        if got is None:
            got = (np.empty_like(p), np.empty_like(p))
            self._scratch[key] = got
        return got

    # -- 存档 ---------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: p for name, p in self.params()}
        arrays["_meta"] = np.frombuffer(
            json.dumps(
                {
                    "dim": self.dim,
                    "hidden": list(self.hidden),
                    "kinds": self.kinds,
                    "steps": self.total_steps,
                    "epochs_done": self.epochs_done,
                    "lr": self.lr,
                    # l2 必须存档：BC 阶段用的 L2 比构造函数默认值大 10 倍，
                    # 不存的话 load() 会悄悄回落到 1e-5，RL 阶段的权重衰减
                    # 就变了（P0-4）。
                    "l2": self.l2,
                    "action_dim": int(self.Wa1.shape[0]),
                }
            ).encode(),
            dtype=np.uint8,
        )
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path, *, strict: bool = False) -> "Net":
        """加载权重。

        ``strict=True``：缺决策头时**直接抛 ValueError** ✗
            给「我要确保这个权重和我现在的代码完全配套」的调用方用
            （比如训练脚本加载自己的检查点做续训 ✓）。
        ``strict=False``（默认）：缺的头**零初始化 + 警告 + 记进
            :attr:`untrained_heads`**，相关决策交给兜底策略 ✓

        ⚠️ **为什么默认是 False 而不是 True**：
        全项目有 21 处调用会加载**存量权重**（天梯 / 网页平台 / 对拍 / 权重自检…）✓
        默认 True 的话，「给网络加一个决策头」会让它们**全部当场失效** ✗
        —— 实测踩过一次：44 个权重全部加载失败，天梯、Java 牌桌、
        网页平台一起瘫 ✗

        而且**训练侧其实不需要默认 True**：它加载的是自己刚存的检查点，
        存的时候就带着当前的 `KINDS` ✓ 永远不会有缺头 ✓
        真想要「配套性断言」的调用方，显式传 `strict=True` 即可 ✓
        """

        data = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(data["_meta"]).decode())
        saved_action = int(meta.get("action_dim", -1))
        if saved_action != ACTION_DIM:
            raise ValueError(
                f"{path}: 这是旧结构的权重（动作特征 {saved_action} 维，"
                f"当前代码是 {ACTION_DIM} 维）。改了特征就必须重新训练 —— "
                f"用 `python -m aimodel.train bc` 重跑一遍。"
            )
        # 旧权重缺新头（例如后来加的 chi）。
        #
        # ⚠️ 这里**不能直接抛错** —— 一抛错，加一个头就让**全部** 44 个存量权重
        # 加载失败，天梯 / Java 牌桌 / 网页平台一起瘫（踩过一次）。
        #
        # 正确做法：**零初始化补上，并记进 `untrained_heads`**：
        #   · 零权重的头输出恒为 0，对已有头的输出**没有任何影响**；
        #   · `MLAgent` 见到 untrained 的头会走**启发式兜底** ——
        #     于是老模型的行为和加头之前**逐比特一致**（三麻根本走不到 chi；
        #     四麻继续用兜底规则吃牌，跟以前一样）；
        #   · 重训之后的权重里有这个头，就不再是 untrained，正常使用。
        saved_kinds = list(meta.get("kinds", ()))
        added = [k for k in KINDS if k not in saved_kinds]
        if added and strict:
            why = "；".join(f"{k}: {_NEW_HEADS[k]}" for k in added if k in _NEW_HEADS)
            raise ValueError(
                f"{path}: 旧权重里没有这些决策头：{added}"
                + (f"（{why}）" if why else "")
                + "。strict=True 下不允许缺头 —— 要么用 `train bc` 重训，"
                "要么调用时改用 strict=False（缺的头会零初始化并走兜底策略）。"
            )
        if added:
            why = "；".join(f"{k}: {_NEW_HEADS[k]}" for k in added if k in _NEW_HEADS)
            import warnings
            warnings.warn(
                f"{path}: 旧权重里没有这些决策头：{added}"
                + (f"（{why}）" if why else "")
                + "。已零初始化并标记为「未训练」，这些决策会走启发式兜底；"
                "想让模型自己学这些决策，用 `python -m aimodel.train bc` 重训。",
                stacklevel=2,
            )
        net = cls(
            meta["dim"],
            tuple(meta["hidden"]),
            lr=meta.get("lr", 1e-3),
            # 老权重没有 l2 字段：兜底用构造默认值，别让它们加载失败。
            l2=meta.get("l2", 1e-5),
        )
        net.kinds = list(meta["kinds"])
        # 注意：Adam 的 m/v 不存档，所以偏差修正的步数必须从 0 重新开始。
        # 如果这里沿用原来的步数，前几步会变成 m/(1-0.9^t) ≈ m、v/(1-0.999^t) ≈ v，
        # 相当于把学习率放大 3 倍多 —— 一脚就能把行为克隆学的牌效踹散。
        net.total_steps = int(meta.get("steps", 0))
        net.epochs_done = int(meta.get("epochs_done", 0))
        net.untrained_heads = set(added)
        net.missing_heads = tuple(added)
        net.steps = 0
        missing = []
        for name, p in net.params():
            if name in data:
                p[...] = data[name]
            else:
                missing.append(name)
        # 补上的新头必须**显式清零** —— 留着随机初始化的话，万一哪天被走到，
        # 输出就是噪声（这正是当初写那句报错要防的事）。
        for k in added:
            if k in net.Wp:
                net.Wp[k][...] = 0.0
                net.bp[k][...] = 0.0
        if missing:
            # 多半是「旧结构 + 新代码」：特征维度或动作塔变了。
            # 这时候缺的那部分会是随机权重，打出来的牌没有意义，必须明说。
            print(
                f"⚠️ {path}: 存档里缺 {len(missing)} 组参数（{', '.join(missing[:4])}…），"
                "这部分是随机初始化 —— 特征或网络结构变过了，请重新训练"
            )
        return net


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)


def softmax_masked(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask > 0, z, -1e9)
    z = z - z.max()
    e = np.exp(z) * (mask > 0)
    s = e.sum()
    if s <= 0:
        fallback = (mask > 0).astype(np.float32)
        if fallback.sum() <= 0:  # 掩码全零也要给个合法分布，不能出 NaN
            return np.full(len(mask), 1.0 / len(mask), dtype=np.float32)
        return fallback / fallback.sum()
    return (e / s).astype(np.float32)


__all__ = ["Net", "KINDS", "HEAD_SIZE", "softmax_masked", "softmax_rows"]
