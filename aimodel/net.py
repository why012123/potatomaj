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
        other.kinds = list(self.kinds)
        other.steps = self.steps
        other.total_steps = self.total_steps
        return other

    def n_params(self) -> int:
        return sum(p.size for _, p in self.params())

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
    ) -> dict:
        """一次 minibatch 更新。``batch`` 由 :mod:`aimodel.train` 组装。"""
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
            if ent_coef:
                dz += ent_coef * p * (np.log(np.maximum(p, 1e-12)) + ent[:, None])
            dz /= n

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
            ent_sum += float(np.mean(ent))
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
            "entropy": ent_sum / nk,
            "clip_frac": clip_frac / nk,
        }

    # -- Adam ---------------------------------------------------------------
    def _apply(self, grads: dict, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        t = self.steps
        for name, p in self.params():
            g = grads.get(name)
            if g is None:
                continue
            m, v = self._adam.get(name, (np.zeros_like(p), np.zeros_like(p)))
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g * g)
            self._adam[name] = (m, v)
            mh = m / (1 - beta1 ** t)
            vh = v / (1 - beta2 ** t)
            p -= (self.lr * mh / (np.sqrt(vh) + eps)).astype(p.dtype)

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
    def load(cls, path: str | Path) -> "Net":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(data["_meta"]).decode())
        saved_action = int(meta.get("action_dim", -1))
        if saved_action != ACTION_DIM:
            raise ValueError(
                f"{path}: 这是旧结构的权重（动作特征 {saved_action} 维，"
                f"当前代码是 {ACTION_DIM} 维）。改了特征就必须重新训练 —— "
                f"用 `python -m aimodel.train bc` 重跑一遍。"
            )
        # 旧权重缺新头：必须显式报错并说清是哪个头，否则那些头会是随机权重，
        # 模型会「永远学不会吃」还看不出来。
        saved_kinds = list(meta.get("kinds", ()))
        added = [k for k in KINDS if k not in saved_kinds]
        if added:
            why = "；".join(f"{k}: {_NEW_HEADS[k]}" for k in added if k in _NEW_HEADS)
            raise ValueError(
                f"{path}: 旧权重里没有这些决策头：{added}"
                + (f"（{why}）" if why else "")
                + "。加头会让参数量变化，旧权重不兼容 —— "
                "用 `python -m aimodel.train bc` 重新训练。"
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
        net.steps = 0
        missing = []
        for name, p in net.params():
            if name in data:
                p[...] = data[name]
            else:
                missing.append(name)
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
