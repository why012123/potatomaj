"""两阶段训练：① 行为克隆（学教师）→ ② 自对弈 PPO（超过教师）。

为什么这么分？纯强化学习从随机权重起步，在三麻这种「一局几十个决策、
奖励还稀疏」的环境里非常难收敛；先用教师的高质量出牌做行为克隆，
策略一上来就有牌效，再让 PPO 在自对弈里学教师写不出来的东西
（什么时候弃和、什么时候贪、立直还是默听）。

用法::

    # ① 行为克隆：跟教师打 N 局，把它的每一步都学下来
    python -m aimodel.train bc --hands 2500

    # ② 自对弈强化学习（从 BC 的权重继续）
    python -m aimodel.train rl --matches 600 --iters 30

    # 一键跑完
    python -m aimodel.train all --hands 2500 --matches 400 --iters 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import rewards, sanma as S, variants
from .agents import Agent, HeuristicAgent, MLAgent, RandomAgent, Tracker
from .encode import (
    HEAD_SIZE, action_features, action_index, action_label, action_mask, encode,
    layout_for,
)
from .engine import Decision, Game, View
from .net import Net

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

#: 各决策头的**目标熵**（P1-1）。必须分头设 —— 打牌头有 34 个动作、上限 log34≈3.53，
#: 而二分类头上限只有 log2≈0.693。用一个全局系数会把立直/和了这些头推坏。
#:
#: 取值思路：**二分类头保持低熵**（它们本来就该果断：能和就和、能立就立），
#: 多分类头给足探索空间（打牌头是策略的主体，也是实测塌得最狠的那个：
#: H=0.291 / 上限 2.27，argmax 与采样 87.7% 重合）。
ENT_TARGETS: dict[str, float] = {
    "discard": 1.2,    # 上限 ln34≈3.53 —— 给足探索，别让它变成确定性策略
    "call": 0.55,      # 上限 ln3≈1.10
    "chi": 0.35,       # 上限 ln2≈0.69
    "tsumo": 0.06,     # 上限 ln2 —— 能和就和，几乎不该探索
    "ron": 0.06,
    "riichi": 0.25,    # 立不立直可以有探索
    "kita": 0.15,
    "kan": 0.10,
    "kyuushu": 0.05,
}

#: 默认模型（不指定 --model 时用它）
DEFAULT_MODEL = "maj-normal"


def spec_of(args) -> variants.ModelSpec:
    return variants.get(getattr(args, "model", DEFAULT_MODEL))


# ---------------------------------------------------------------------------
# 记录用的包装 agent
# ---------------------------------------------------------------------------


class RecordingAgent(Agent):
    """把内层 agent 的每一步决策记进 :class:`Tracker`（行为克隆用）。"""

    def __init__(self, inner: Agent, tracker: Tracker, logp: float = 0.0) -> None:
        self.inner = inner
        self.tracker = tracker
        self.logp = logp
        self.name = getattr(inner, "name", "agent")

    def decide(self, decision: Decision, view: View) -> int:
        choice = self.inner.decide(decision, view)
        kind = decision.kind
        if kind in HEAD_SIZE:
            x = encode(view)
            mask = action_mask(decision)[: HEAD_SIZE[kind]]
            head = action_index(decision, choice)
            self.tracker.record(
                kind, x, mask, head, self.logp, view.player, action_label(decision, choice),
                action_features(view) if kind == "discard" else None,
            )
        return choice


# ---------------------------------------------------------------------------
# ① 行为克隆
# ---------------------------------------------------------------------------


def collect_bc(
    n_hands: int, seed: int = 0, verbose: bool = True, spec: variants.ModelSpec | None = None
) -> dict:
    """让这个模型的专属教师打若干局，逐局记录。

    价值目标用的是**这个模型自己的得分条件**（不是原始得失点）——
    追大牌的模型就该按番数估价值，追防守的模型就该按放铳风险估价值。
    """
    spec = spec or variants.get(DEFAULT_MODEL)
    n = spec.players
    rule = S.rule_for(n)
    tracker = Tracker()
    rets: list[float] = []
    hands_done = 0
    game_seed = seed
    t0 = time.perf_counter()
    while hands_done < n_hands:
        agents = [
            RecordingAgent(HeuristicAgent(game_seed * n + i, style=spec.style), tracker)
            for i in range(n)
        ]
        game_start = len(tracker)
        r = Game(agents, seed=game_seed, rule=rule).play()
        for i in range(game_start, len(tracker)):
            rets.append(rewards.norm_reward(spec.reward, r, tracker.players[i]))
        hands_done += len(r.hands)
        game_seed += 1
    data = tracker.arrays()
    data["ret"] = np.asarray(rets, dtype=np.float32)
    if verbose:
        dt = time.perf_counter() - t0
        print(
            f"  行为克隆数据：{len(data['X'])} 条决策 / {hands_done} 局 / {dt:.1f}s"
            f"（{game_seed - seed} 场对局，{spec.cn}教师）"
        )
    return data


def train_bc(
    net: Net,
    data: dict,
    epochs: int = 6,
    batch_size: int = 256,
    seed: int = 0,
    verbose: bool = True,
    holdout: dict | None = None,
    patience: int = 3,
    label_smoothing: float = 0.05,
    ref_holdout: dict | None = None,
    ref_name: str = "参照",
) -> tuple[list[dict], Net, dict]:
    """行为克隆。返回 ``(hist, best_net, best_info)``。

    **为什么返回 best_net 而不是就用传进来的 net**：留出集一致率通常在
    中途见顶然后回落（权重日志里实测 18.5 个百分点：训练 87.6%->93.4%，
    留出 75.9%->74.9%）。老代码只打印留出率、**不做决策**，
    ``cmd_bc`` 又无条件 ``net.save()`` —— 存下来的是**过拟合最重的那一轮**。

    现在每轮记一次留出率，**只在变好时留一份网络快照**，跑完把最好的那份
    返回给调用方去存。``patience`` 轮没有改善就提前停（0 = 不早停）。
    """
    rng = np.random.default_rng(seed)
    n = len(data["X"])
    adv = np.ones(n, dtype=np.float32)
    hist: list[dict] = []
    # ⚠️⚠️ `best_hold` **不能初始化为 -1** ✗ 那是个死值 ✓
    #
    # 后果（远端实测发现的，很严重）：`improved = hold > best_hold` 在**第一轮永远为真** ✗
    # 于是 `--resume` 分块续训时，**第一块结束就把这一块内的「最好」存盘覆盖掉了** ✗
    # —— 跟「进来时的权重本来是多少」毫无关系 ✓
    #
    # 实测证据：进来的权重留出 90.0%（3 轮），训 1 轮后存下来的是 **84.2%** ✗
    # **越训越差，还存盘了** ✓ 而分块续训正是「杀窗口短」环境下唯一的活路 ✗
    #
    # 正确做法：**先量一次「进来时」的 holdout，拿它当初始最优** ✓
    # 这样只有**真的比进来时更好**才会覆盖 ✓ 一轮都没进步就把原权重原样留着 ✓
    base_hold = agreement(net, holdout) if holdout else agreement(net, data)
    best_hold = base_hold
    best_epoch = 0
    # ⚠️ 必须是 **copy()**，不能是 `net` 本身 ✗✗
    #    `net` 是**引用** —— 训练过程会**原地改它** ✗ 所以「保留原权重」根本保留不住 ✓
    #    （我的测试当场抓到了这条：`steps 0 -> 24`，说明 best_net 跟着一起被训了 ✓）
    best_net = net.copy()
    kept_original = True    # ← 直到有某一轮真的超过 base_hold 才变 False
    stale = 0
    print(f"    进来时的留出率 {base_hold:.1%}"
          "（只有超过它才会覆盖权重 ✓ 分块续训靠这个才不会越训越差 ✗）")
    # ⚠️ `--resume` 只恢复权重，**不恢复 epoch 计数** ✗
    #    所以 `--resume --epochs 30` = 「在现有权重上再训 30 轮」，
    #    不是「训到第 30 轮」✓ 每跑一次就多加 30 轮 ✓
    #    （远端新会话读代码时发现的 —— 交接书里我正是这么写错的 ✗）
    if net.epochs_done > 0:
        print(f"    ⚠️ 这个权重**已经训过 {net.epochs_done} 轮**；"
              f"本次再训 {epochs} 轮 → 累计 {net.epochs_done + epochs} 轮")
        print("       （--resume 不恢复 epoch 计数，它是在现有权重上继续训 ✗）")

    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        nb = 0
        for s in range(0, n, batch_size):
            sel = idx[s : s + batch_size]
            if len(sel) < 8:
                continue
            st = net.step(
                {
                    "X": data["X"][sel],
                    "mask": data["mask"][sel],
                    "actfeat": data["actfeat"][sel],
                    "action": data["action"][sel],
                    "logp": data["logp"][sel],
                    "kind": data["kind"][sel],
                    "adv": adv[sel],
                    "ret": data["ret"][sel],
                    "mode": "bc",
                },
                clip=0.0,
                ent_coef=0.001,
                vf_coef=0.5,
                label_smoothing=label_smoothing,
            )
            for k in tot:
                tot[k] += st[k]
            nb += 1
        acc = agreement(net, data)
        hold = agreement(net, holdout) if holdout else acc
        # 累计计数在**打分之后、存最优之前**加 —— 这样存下来的最优权重
        # 带的是「它实际训过多少轮」✓
        net.epochs_done += 1
        # 参照指标：只打印、**不参与选择** ✗
        # 为什么要它：用户换标签（人类数据）之后，「教师留出率」会掉 ✗
        # 但那不代表变差 ✓ 所以选择用**同源留出集**，教师那个只当对照 ✓
        if ref_holdout is not None:
            row_ref = agreement(net, ref_holdout)
        else:
            row_ref = None
        improved = hold > best_hold
        if improved:
            # net.copy() 是深拷贝（含 steps / total_steps），所以拿回来能直接存
            best_hold, best_epoch, best_net = hold, ep + 1, net.copy()
            kept_original = False
            stale = 0
        else:
            stale += 1
        row = {
            "epoch": ep + 1,
            "policy_loss": tot["policy_loss"] / max(1, nb),
            "value_loss": tot["value_loss"] / max(1, nb),
            "entropy": tot["entropy"] / max(1, nb),
            "与教师一致率": acc,
            "留出集一致率": hold,
            "最优留出轮次": best_epoch,
        }
        hist.append(row)
        if verbose:
            print(
                f"    epoch {ep + 1}/{epochs}  一致率={acc:.1%}"
                f"（留出 {hold:.1%}）  value_loss={row['value_loss']:.4f}"
                + ("  ★最优" if improved else f"  （已 {stale} 轮没进步）")
            )
        if patience > 0 and stale >= patience:
            if verbose:
                print(f"    ⏹ 留出集连续 {patience} 轮没进步，提前停在 epoch {ep + 1}")
            break
    info = {
        "kept_original": kept_original,   # ← True = 一轮都没超过进来时的留出率 ✓
        "base_holdout": base_hold,
        "best_epoch": best_epoch,
        "best_holdout": best_hold,
        "last_epoch": len(hist),
    }
    if verbose:
        if kept_original:
            print(
                f"    ⚠️ {len(hist)} 轮里**没有一轮超过进来时的 {base_hold:.1%}** —— "
                "原权重原样保留 ✓（不是「第 0 轮最优」，是**这一轮训练白跑了** ✓）"
            )
        else:
            if ref_holdout is not None:
                print(f"      （对照：{ref_name} 一致率 {agreement(best_net, ref_holdout):.1%}"
                      " —— 换标签后会掉，属正常 ✓ 不参与选择 ✗）")
            print(
                f"    ★ 最优留出率 {best_hold:.1%} 出现在 **第 {best_epoch} 轮**"
                f"（共跑 {len(hist)} 轮）—— 存的是这一轮，不是最后一轮"
            )
    return hist, best_net, info


def agreement(net: Net, data: dict, limit: int = 4000) -> float:
    """网络出牌与教师标签的一致率（只看打牌决策）。"""
    mask_kind = data["kind"] == "discard"
    idx = np.nonzero(mask_kind)[0]
    if len(idx) == 0:
        return 0.0
    if len(idx) > limit:
        idx = idx[:limit]
    h2 = net.trunk(data["X"][idx])
    logits = net.logits(
        h2, "discard", np.asarray(data["actfeat"][idx], dtype=np.float32)
    )
    m = data["mask"][idx][:, :34]
    logits = np.where(m > 0, logits, -1e9)
    pred = logits.argmax(axis=1)
    return float(np.mean(pred == data["action"][idx]))


# ---------------------------------------------------------------------------
# ② 自对弈 PPO
# ---------------------------------------------------------------------------


def collect_selfplay(
    net: Net,
    n_matches: int,
    seed: int = 0,
    teacher_ratio: float = 0.25,
    snapshot: Net | None = None,
    snapshot_ratio: float = 0.1,
    greedy_teacher: bool = False,
    spec: variants.ModelSpec | None = None,
    pool: list[Net] | None = None,
    pool_ratio: float = 0.0,
) -> tuple[dict, dict]:
    """自对弈收集轨迹；只有「用当前网络」的座位上的决策会被记录。

    奖励用这个模型自己的得分条件 —— 这就是四个模型打法分岔的地方。
    """
    spec = spec or variants.get(DEFAULT_MODEL)
    n = spec.players
    rule = S.rule_for(n)
    tracker = Tracker()
    rets: list[float] = []
    rng = np.random.default_rng(seed)
    stats = {"hands": 0, "wins": 0, "deal_in": 0, "scores": [0] * n, "n": 0}
    for m in range(n_matches):
        game_seed = seed * 100003 + m
        agents: list[Agent] = []
        recorded: list[bool] = []
        for seat in range(n):
            u = float(rng.random())
            if u < teacher_ratio:
                agents.append(HeuristicAgent(game_seed * n + seat, style=spec.style))
                recorded.append(False)
            elif u < teacher_ratio + snapshot_ratio and snapshot is not None:
                agents.append(
                    MLAgent(snapshot, seed=game_seed * n + seat, greedy=False,
                            style=spec.style)
                )
                recorded.append(False)
            elif u < teacher_ratio + snapshot_ratio + pool_ratio and pool:
                # **对手池**：坐一个别的风格的模型（v1.1 的另外三个）。
                # 只跟自己+教师下，学不到「面对速攻该怎么防」这类东西。
                opp = pool[int(rng.integers(len(pool)))]
                agents.append(
                    MLAgent(opp, seed=game_seed * n + seat, greedy=False,
                            style=spec.style)
                )
                recorded.append(False)
            else:
                agents.append(
                    MLAgent(net, seed=game_seed * n + seat, tracker=tracker,
                            greedy=False, style=spec.style)
                )
                recorded.append(True)
        game_start = len(tracker)
        r = Game(agents, seed=game_seed, rule=rule).play()
        deltas = r.deltas
        for i in range(game_start, len(tracker)):
            rets.append(rewards.norm_reward(spec.reward, r, tracker.players[i]))
        stats["hands"] += len(r.hands)
        stats["n"] += 1
        stats["scores"] = [stats["scores"][i] + deltas[i] for i in range(n)]
        for p in range(n):
            if recorded[p]:
                st = r.stats[p]
                stats["wins"] += st.tsumo + st.ron
                stats["deal_in"] += st.deal_in
    data = tracker.arrays()
    data["ret"] = np.asarray(rets, dtype=np.float32)
    return data, stats


def compute_advantages(net: Net, data: dict) -> np.ndarray:
    v = net.value(net.trunk(data["X"]))
    adv = data["ret"] - v
    std = float(adv.std())
    return ((adv - adv.mean()) / (std + 1e-6)).astype(np.float32)


def train_rl(
    net: Net,
    data: dict,
    epochs: int = 3,
    batch_size: int = 256,
    seed: int = 0,
    clip: float = 0.2,
    ent_coef: float = 0.008,
    vf_coef: float = 0.5,
    verbose: bool = True,
    ent_coef_adapt: bool = False,
    kl_coef: float = 0.0,
    ref: Net | None = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    data = dict(data)
    data["adv"] = compute_advantages(net, data)
    # KL 锚定（P1-2）：先按这一批轨迹把参考策略的概率冻结下来。
    # 参考就是**克隆阶段的权重**，所以 RL 不会被推着走远。
    if kl_coef > 0.0 and ref is not None:
        net.set_reference(ref, data, kl_coef=kl_coef)
    elif kl_coef <= 0.0:
        net.kl_coef = 0.0
    n = len(data["X"])
    hist = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
               "clip_frac": 0.0, "kl": 0.0}
        ent_by_kind: dict[str, float] = {}
        ent_n: dict[str, int] = {}
        nb = 0
        for s in range(0, n, batch_size):
            sel = idx[s : s + batch_size]
            if len(sel) < 8:
                continue
            st = net.step(
                {
                    "X": data["X"][sel],
                    "mask": data["mask"][sel],
                    "actfeat": data["actfeat"][sel],
                    "action": data["action"][sel],
                    "logp": data["logp"][sel],
                    "kind": data["kind"][sel],
                    "adv": data["adv"][sel],
                    "ret": data["ret"][sel],
                    "mode": "pg",
                },
                clip=clip,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                ent_target=ENT_TARGETS,
                ent_coef_adapt=ent_coef_adapt,
            )
            for k in tot:
                tot[k] += st[k]
            for k, hv in st.get("entropy_by_kind", {}).items():
                ent_by_kind[k] = ent_by_kind.get(k, 0.0) + hv
                ent_n[k] = ent_n.get(k, 0) + 1
            nb += 1
        row = {k: v / max(1, nb) for k, v in tot.items()}
        row["epoch"] = ep + 1
        row["entropy_by_kind"] = {
            k: ent_by_kind[k] / max(1, ent_n[k]) for k in ent_by_kind
        }
        row["ent_coef_by_kind"] = dict(net._ent_coef)
        hist.append(row)
    if verbose and hist:
        h = hist[-1]
        print(
            f"    PPO  policy_loss={h['policy_loss']:+.4f}  entropy={h['entropy']:.3f}"
            f"  clip={h['clip_frac']:.1%}  value_loss={h['value_loss']:.4f}"
        )
        print_entropy_by_kind(h)
    return hist


def print_entropy_by_kind(row: dict) -> None:
    """**逐头**打印熵，并标出离目标最远的头（P1-1）。

    老代码把 8 个头的熵**平均成一个数**（``net.py`` 原 :284），
    所以「打牌头塌到 H=0.291」这种信息会被其他头摊平、根本看不出来。
    这里逐头列出：实际熵 / 目标熵 / 该头的自适应系数。
    """
    by_kind = row.get("entropy_by_kind") or {}
    if not by_kind:
        return
    coefs = row.get("ent_coef_by_kind") or {}
    parts = []
    worst, worst_gap = None, -1.0
    for k in sorted(by_kind):
        h = by_kind[k]
        tgt = ENT_TARGETS.get(k)
        if tgt:
            gap = (tgt - h) / tgt      # >0 表示熵偏低（探索不足）
            if gap > worst_gap:
                worst, worst_gap = k, gap
            parts.append(f"{k}={h:.2f}/{tgt:.2f}")
        else:
            parts.append(f"{k}={h:.2f}")
    print("    逐头熵（实际/目标）: " + "  ".join(parts))
    if coefs:
        print("    逐头熵系数: " + "  ".join(
            f"{k}={coefs[k]:.4f}" for k in sorted(coefs)
        ))
    if worst is not None and worst_gap > 0.3:
        print(f"    ⚠️ 探索最不足的是 **{worst}**"
              f"（{by_kind[worst]:.2f} vs 目标 {ENT_TARGETS[worst]:.2f}"
              f"，差 {worst_gap:.0%}）—— 这个头在向确定性塌缩")


# ---------------------------------------------------------------------------
# 快速评测（训练过程中用）
# ---------------------------------------------------------------------------


#: 对外报告用的种子 —— **必须和选择用的 999 不同**。
#: 同一个测试集既选又报，再多样本也救不了（实测选择偏差 +9.46 段位分）。
REPORT_SEED = 31337


def quick_eval(
    net: Net,
    n_matches: int = 60,
    #: ⚠️ **这是「选择集」的种子，不是「报告集」**。
    #:
    #: 检查点逻辑是「评测新高就保存」——如果报告也用同一份牌局，
    #: 就是在有限测试集上反复挑最大值（赢家诅咒）。
    #: 实测：同一份权重、只换 16 个 150 场测试集，读数跨度 **16.7 段位分**；
    #: 从 16 个同强度候选里挑最高 → **虚高 +9.46 段位分** ✗
    #:
    #: 所以：**这个种子只用来选检查点；对外报告必须换种子**（见 REPORT_SEED）。
    seed: int = 999,
    opponent: str = "teacher",
    greedy: bool = True,
    spec: variants.ModelSpec | None = None,
) -> dict:
    """固定 0 号位是待评测网络，其余座位是教师（或随机）。"""
    spec = spec or variants.get(DEFAULT_MODEL)
    n = spec.players
    rule = S.rule_for(n)
    delta_sum = 0.0
    rp_sum = 0.0
    rank_sum = 0.0
    wins = 0
    deal_in = 0
    hands = 0
    for m in range(n_matches):
        game_seed = seed * 7919 + m
        agents: list[Agent] = [
            MLAgent(net, seed=game_seed, greedy=greedy, style=spec.style)
        ]
        for i in range(1, n):
            if opponent == "teacher":
                agents.append(HeuristicAgent(game_seed * n + i, style=spec.style))
            else:
                agents.append(RandomAgent(game_seed * n + i))
        r = Game(agents, seed=game_seed, rule=rule).play()
        delta_sum += r.deltas[0]
        rp_sum += r.rank_points[0]
        rank_sum += r.ranks[0]
        wins += r.stats[0].tsumo + r.stats[0].ron
        deal_in += r.stats[0].deal_in
        hands += len(r.hands)
    return {
        "平均得点变化": delta_sum / n_matches,
        "平均段位分": rp_sum / n_matches,
        "平均顺位": rank_sum / n_matches + 1,
        "和了率": wins / max(1, hands),
        "放铳率": deal_in / max(1, hands),
        "对局数": n_matches,
    }


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def build_net(args, resume: bool = True) -> Net:
    spec = spec_of(args)
    # 特征维度跟人数走：三麻 486、四麻 560。搞错了就是 matmul 直接炸。
    dim = layout_for(spec.players)[1]
    if resume and Path(args.out).exists():
        net = Net.load(args.out)
        if net.dim != dim:
            raise SystemExit(
                f"{args.out} 是 {net.dim} 维的，但 {spec.name}（{spec.cn}）需要 {dim} 维 —— "
                f"换人数就得重训，删掉这个权重再来。"
            )
        print(f"  从 {args.out} 继续训练（累计已更新 {net.total_steps} 次）")
        return net
    hidden = tuple(int(x) for x in str(args.hidden).split(","))
    return Net(dim=dim, hidden=hidden, seed=args.seed, lr=args.lr, l2=args.l2)


def acquire_out_lock(out: Path) -> Path | None:
    """**单写者锁**：两个训练进程写同一个权重文件时，第二个直接拒绝启动。

    为什么需要：`--out` 指向同一个文件的两个进程**不会互相报错** ✗
    —— 它们只是各训各的，**最后写完的赢** ✓ 中间没有任何提示 ✓

    实测踩过（远端）：用户手动起了一次 6×5 轮，同时另一个会话起了 27 轮，
    两个进程训同一个模型、写同一个文件，**每轮从 14 分钟拖到 28 分钟** ✗
    而且看不出来是谁在写 ✓

    实现：`<out>.lock` 里写 pid ✓ `O_CREAT|O_EXCL` 保证原子 ✓
    **陈旧锁会自动接管**（pid 已经不在了 ✓）—— 否则一次崩溃就永久锁死 ✓

    返回锁文件路径（成功）或 `None`（**别人正在写** ✓）
    """
    import os as _os

    lock = out.with_suffix(out.suffix + ".lock")

    def _owner_alive() -> bool:
        try:
            pid = int(lock.read_text().strip() or 0)
        except (OSError, ValueError):
            return False
        if pid <= 0:
            return False
        try:                        # 信号 0 = 只探测存活，不真发信号 ✓
            _os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:     # 存在但不是我的进程 ✓ 也算活着 ✓
            return True

    try:
        fd = _os.open(lock, _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY)
        _os.write(fd, str(_os.getpid()).encode())
        _os.close(fd)
        return lock
    except FileExistsError:
        if _owner_alive():
            return None
        # 陈旧锁（进程没了）→ 接管 ✓
        print(f"  ⚠️ 发现陈旧锁 {lock}（pid 已不在）—— 接管它 ✓")
        try:
            lock.unlink()
        except OSError:
            return None
        return acquire_out_lock(out)


def guard_out_path(args, resume: bool) -> Path:
    """**止损门禁**：不让人一裸跑就把已训好的权重覆盖掉。

    老行为：直接 `python -m aimodel.train bc` 会用**默认的 256×256 新建网络**
    去覆盖 `weights/maj-normalv1.1.npz` —— 无备份、无确认。
    一次手滑就丢掉一个训好的模型。

    规则（``cmd_bc`` / ``cmd_rl`` 都走这里）：
    * 目标不存在            -> 正常写
    * 目标存在 + ``--force`` -> 正常写（明确授权覆盖）
    * 目标存在 + ``--resume``-> 正常写（本来就是接着它训）
    * 目标存在 + 都没有      -> **改写 ``<name>.new.npz``**，并打印提示
    """
    out = Path(args.out)
    if not out.exists():
        return out
    if getattr(args, "force", False) or resume:
        return out
    alt = out.with_name(out.stem + ".new.npz")
    print(
        f"  ⚠️ {out} 已存在 —— 裸跑会用**新建的网络**把它覆盖掉。\n"
        f"     已改为写 {alt}\n"
        f"     （要覆盖原文件就加 --force；要接着它训就加 --resume）"
    )
    return alt


def cmd_bc(args) -> Net:
    spec = spec_of(args)
    print(f"① 行为克隆：{spec.cn}（{spec.name}）")
    print(f"   {spec.describe()}")
    net = build_net(args, resume=args.resume)
    args.out = str(guard_out_path(args, resume=args.resume))
    # ★ 单写者锁：两个进程写同一个权重时，第二个直接拒绝 ✓
    _lock = acquire_out_lock(Path(args.out))
    if _lock is None:
        raise SystemExit(
            f"❌ {args.out} 正被**另一个训练进程**写（锁文件 {args.out}.lock）\n"
            "   两个进程写同一个文件不会报错，只会「最后写完的赢」✗\n"
            "   要么等它跑完，要么换一个 --out ✓\n"
            "   （确认对方已经不在了就删掉那个 .lock ✓）"
        )

    # --bc-data: consume a pre-extracted dataset instead of generating one from
    # the heuristic teacher.
    #
    # WHY: `collect_bc` clones `HeuristicAgent`, a hand-written rule set, so the
    # cloned policy can at best match those rules -- the teacher is the ceiling.
    # `aimodel/tenhou_bc.py` extracts the SAME schema from 40,000 real Tenhou
    # hands (Houou-level humans: 20.98% per-seat win rate), and
    # `aimodel/shortjob.py` builds it in ~9s slices so that a process kill --
    # which on this box happens every 15s-3.5min -- cannot lose more than one
    # slice. Together they let BC imitate humans rather than our heuristics.
    #
    # The HOLDOUT still comes from the teacher, deliberately: that is the fixed
    # yardstick the existing maj-* models were measured against, so keeping it
    # makes the new model's holdout number directly comparable to theirs.
    src = getattr(args, "bc_data", None)
    if src:
        import numpy as _np

        with _np.load(src, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
        print(f"  行为克隆数据：{len(data['X'])} 条决策 <- {src}（预抽，不是教师生成的）")
        if data["X"].shape[1] != net.dim:
            raise SystemExit(
                f"维度不符：数据集 {data['X'].shape[1]} vs 网络 {net.dim}"
                "（四麻 560 / 三麻 486 —— 拿错数据集了）"
            )
    else:
        data = collect_bc(args.hands, seed=args.seed, spec=spec)

    # ★ 留出集必须和训练数据**同源** ✗
    #
    # 踩过的坑：原来不管数据从哪来，留出集一律 `collect_bc`（**教师现打**）✓
    # 那在「克隆教师」时没问题 ✓ 但**换标签之后就是致命的** ✗：
    #
    #   实测（人类牌谱 BC）：人类数据和教师的出牌只一致约 63% ✓
    #     · 未训练模型：教师留出 90.0%，同源一致率 **45.3%**
    #     · 训 1 轮后：教师留出 **61.3%**（掉了 ✗），同源一致率 **63.6%**（涨了 ✓）
    #   → 模型其实**在进步** ✓ 但教师留出集一路下降 ✗
    #   → 早停/选优会因此判定「越训越差」并拒绝保存 ✗✗
    #
    # 所以：**选择用同源留出集** ✓ 教师留出集降级成**只打印的对照** ✓
    # （对照仍有价值：它让新模型和老模型的数字**可比** ✓）
    if src:
        _n = len(data["X"])
        _rng = np.random.default_rng(args.seed + 555_555)
        _idx = _rng.permutation(_n)
        _nh = max(2000, _n // 20)
        holdout = {k: data[k][_idx[:_nh]] for k in data}
        data = {k: data[k][_idx[_nh:]] for k in data}
        print(f"  留出集：{len(holdout['X'])} 条（**从同一份数据里切** ✓ 同源 ✓）")
        print(f"  训练集：{len(data['X'])} 条")
        ref_holdout = collect_bc(
            max(60, args.hands // 20), seed=args.seed + 555_555,
            verbose=False, spec=spec,
        )
        ref_name = "教师"
        print(f"  对照留出集（教师生成 {len(ref_holdout['X'])} 条）："
              "**只打印 ✓ 不参与选择 ✗**（换标签后它会掉，属正常 ✓）")
    else:
        holdout = collect_bc(
            max(60, args.hands // 20), seed=args.seed + 555_555, verbose=False, spec=spec
        )
        ref_holdout = None
        ref_name = "参照"
        print(f"  留出集：{len(holdout['X'])} 条决策（种子不同，用来判断有没有过拟合）")
    hist, best_net, bc_info = train_bc(
        net, data, epochs=args.epochs, batch_size=args.batch, seed=args.seed,
        holdout=holdout, patience=args.patience,
        ref_holdout=ref_holdout, ref_name=ref_name,
    )
    # 存**留出最优**那一轮，不是最后一轮。老代码无条件 net.save()，
    # 存下来的是过拟合最重的那份（实测训练 87.6%->93.4% 时留出 75.9%->74.9%）。
    saved_net = best_net
    saved_net.lr = getattr(args, "lr", saved_net.lr)
    if _lock is not None:
        try:
            _lock.unlink()
        except OSError:
            pass
    saved_net.save(args.out)
    if bc_info.get("kept_original"):
        print(
            f"  已保存 -> {args.out}（{saved_net.n_params()} 参数）"
            f"  [**一轮都没超过进来时的 {bc_info['base_holdout']:.1%}，"
            f"原权重原样保留** ✓]"
        )
    else:
        print(
            f"  已保存 -> {args.out}（{saved_net.n_params()} 参数）"
            f"  [最优留出 {bc_info['best_holdout']:.1%} @ 第 {bc_info['best_epoch']} 轮"
            f" / 共 {bc_info['last_epoch']} 轮]"
        )
    save_log(args, {"bc": {
        "hist": hist, "samples": int(len(data["X"])),
        "best_epoch": bc_info["best_epoch"],
        "best_holdout": bc_info["best_holdout"],
        "last_epoch": bc_info["last_epoch"],
    }})
    return saved_net


def cmd_rl(args) -> Net:
    spec = spec_of(args)
    print(f"② 自对弈强化学习（PPO）：{spec.cn}（{spec.name}）")
    print(f"   得分条件 = {rewards.MODE_CN.get(spec.reward, spec.reward)}")
    net = build_net(args, resume=True)
    net.lr = args.rl_lr  # 强化学习用更小的步长，别把克隆来的牌效打散
    args.out = str(guard_out_path(args, resume=True))
    _lock = acquire_out_lock(Path(args.out))
    if _lock is None:
        raise SystemExit(
            f"❌ {args.out} 正被**另一个训练进程**写（锁文件 {args.out}.lock）\n"
            "   两个进程写同一个文件不会报错，只会「最后写完的赢」✗"
        )
    # 对手池：同人数的**其它风格**模型（比如训「防守」时让「速攻」「大牌」上桌）
    pool: list[Net] = []
    if args.pool_ratio > 0:
        from . import variants as _v

        for other in _v.same_players(spec.players):
            if other.style == spec.style:
                continue
            if other.path.exists():
                try:
                    pool.append(Net.load(other.path))
                except Exception:
                    pass
        if pool:
            print(f"   对手池：{len(pool)} 个其它风格的模型（占比 {args.pool_ratio:.0%}）")
    log: dict = {"rl": []}
    best = quick_eval(
        net, n_matches=args.eval_matches, seed=args.seed + 777, spec=spec
    )
    print(f"  起始：{fmt_eval(best)}")
    best_delta = best["平均段位分"]
    # KL 锚定的参考 = **进入 RL 之前**的那份权重（BC 克隆出来的牌效）。
    ref_net = net.copy()
    snapshot = net.copy()
    # 「新高才保存」原来只保护**文件**，不保护**内存里的权重**：
    # 崩掉那一轮之后的训练数据是用崩掉的策略采的，会越跑越远。
    # 所以同时留一份内存里的最优备份，跑完回滚回去。
    best_net = net.copy()
    for it in range(1, args.iters + 1):
        t0 = time.perf_counter()
        data, st = collect_selfplay(
            net,
            args.matches,
            seed=args.seed * 1000 + it,
            teacher_ratio=args.teacher_ratio,
            snapshot=snapshot,
            spec=spec,
            pool=pool,
            pool_ratio=args.pool_ratio,
        )
        hist = train_rl(
            net,
            data,
            epochs=args.ppo_epochs,
            batch_size=args.batch,
            seed=args.seed + it,
            ent_coef=max(0.002, 0.01 * (1 - it / (args.iters + 1))),
            kl_coef=args.kl_coef,
            ref=ref_net,
        )
        ev = quick_eval(
            net, n_matches=args.eval_matches, seed=args.seed + 777, spec=spec
        )
        dt = time.perf_counter() - t0
        print(
            f"  [{it}/{args.iters}] {len(data['X'])} 决策  {dt:.0f}s  {fmt_eval(ev)}"
        )
        log["rl"].append({"iter": it, "samples": int(len(data["X"])), "eval": ev,
                          "selfplay": {k: v for k, v in st.items() if k != "scores"}})
        if ev["平均段位分"] > best_delta:
            best_delta = ev["平均段位分"]
            net.save(args.out)
            # ★ 同时把最优权重**留一份在内存里**（P1-2）：
            # 只存文件的话，后面训崩了内存里已经变坏，下一轮自对弈就是
            # 用坏策略采的数据，等于自己把自己越推越远。
            best_net = net.copy()
            print(f"    ↑ 新高，已保存 {args.out}（内存里也留了一份最优）")
        if it % 5 == 0:
            snapshot = net.copy()
    # 跑完把最优那份**回滚进内存**，并且以它为准落盘
    if best_net is not net:
        net = best_net
        net.save(args.out)
        print(f"  ↩ 已回滚到最优权重（段位分 {best_delta:+.1f}）并保存 -> {args.out}")
    elif not Path(args.out).exists():
        net.save(args.out)
    save_log(args, log)
    return net


def fmt_eval(ev: dict) -> str:
    return (
        f"对教师 段位分 {ev['平均段位分']:+.1f}  平均顺位 {ev['平均顺位']:.2f}"
        f"  和了率 {ev['和了率']:.1%}  放铳率 {ev['放铳率']:.1%}"
    )


def save_log(args, extra: dict) -> None:
    path = Path(args.log)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except json.JSONDecodeError:
            old = {}
    old.update(extra)
    old["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(old, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="三麻 AI 训练")
    ap.add_argument("stage", nargs="?", default="all", choices=["bc", "rl", "all"],
                    help="训练阶段（默认 all）")
    ap.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="训哪个模型（maj-normal / maj-mountain / maj-flash / maj-huge，四麻加 4 后缀，"
             "别名 flash4 也认）",
    )
    ap.add_argument("--list-models", action="store_true", help="列出八个模型和状态")
    ap.add_argument("--hands", type=int, default=2500, help="行为克隆的对局数")
    ap.add_argument(
        "--bc-data", default=None, metavar="NPZ",
        help="用预先抽好的 BC 数据集代替教师自打（例如 "
             "aimodel/tenhou_bc.py 从天鳳牌谱抽出的那批）。此时 --hands 只影响留出集大小。",
    )
    ap.add_argument("--matches", type=int, default=300, help="每轮自对弈的场数")
    ap.add_argument("--iters", type=int, default=20, help="自对弈轮数")
    ap.add_argument("--epochs", type=int, default=6, help="行为克隆的 epoch 数")
    ap.add_argument("--patience", type=int, default=3,
                    help="留出集连续多少轮没进步就早停（0 = 不早停，始终跑满 --epochs）")
    ap.add_argument("--ppo-epochs", type=int, default=3, help="每轮 PPO 的 epoch 数")
    ap.add_argument("--batch", type=int, default=256, help="minibatch 大小")
    ap.add_argument("--lr", type=float, default=1e-3, help="行为克隆的学习率")
    ap.add_argument("--rl-lr", type=float, default=3e-4, help="自对弈 PPO 的学习率")
    ap.add_argument("--hidden", type=str, default="256,256", help="隐层宽度")
    ap.add_argument("--l2", type=float, default=1e-4, help="L2 正则（抑制过拟合）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-ratio", type=float, default=0.25, help="自对弈里教师的占比")
    ap.add_argument("--pool-ratio", type=float, default=0.0,
                    help="自对弈时让别的风格的模型上桌的比例（0 = 只用自己和教师）")
    ap.add_argument("--eval-matches", type=int, default=80, help="每轮评测的场数")
    ap.add_argument("--out", type=str, default=str(WEIGHTS_DIR / "policy.npz"))
    ap.add_argument("--log", type=str, default=str(WEIGHTS_DIR / "train_log.json"))
    ap.add_argument("--resume", action="store_true", help="从已有权重继续")
    ap.add_argument("--force", action="store_true",
                    help="目标权重已存在时允许直接覆盖（不加就改写成 <name>.new.npz）")
    ap.add_argument("--kl-coef", type=float, default=0.1,
                    help="KL 锚定强度（0 = 关）。把 RL 策略拉回克隆阶段的参考策略，"
                         "防止把 BC 学的牌效训崩；太紧 RL 动不了，太松等于没锚")
    args = ap.parse_args(argv)

    if args.list_models:
        print(f"{'模型':<16}{'人数':<6}{'得分条件':<34}{'权重'}")
        for spec in variants.all_specs():
            mark = "✅ 已训练" if spec.path.exists() else "—"
            print(f"{spec.name:<16}{spec.players:<6}"
                  f"{rewards.MODE_CN.get(spec.reward, spec.reward):<34}{mark}")
        return

    spec = variants.get(args.model)
    if args.out == str(WEIGHTS_DIR / "policy.npz"):
        # 没显式指定输出 -> 按模型名存
        args.out = str(spec.path)

    t0 = time.perf_counter()
    if args.stage in ("bc", "all"):
        cmd_bc(args)
    if args.stage in ("rl", "all"):
        cmd_rl(args)
    print(f"总耗时 {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
