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
) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(data["X"])
    adv = np.ones(n, dtype=np.float32)
    hist: list[dict] = []
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
            )
            for k in tot:
                tot[k] += st[k]
            nb += 1
        acc = agreement(net, data)
        hold = agreement(net, holdout) if holdout else acc
        row = {
            "epoch": ep + 1,
            "policy_loss": tot["policy_loss"] / max(1, nb),
            "value_loss": tot["value_loss"] / max(1, nb),
            "entropy": tot["entropy"] / max(1, nb),
            "与教师一致率": acc,
            "留出集一致率": hold,
        }
        hist.append(row)
        if verbose:
            print(
                f"    epoch {ep + 1}/{epochs}  一致率={acc:.1%}"
                f"（留出 {hold:.1%}）  value_loss={row['value_loss']:.4f}"
            )
    return hist


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
                agents.append(MLAgent(snapshot, seed=game_seed * n + seat, greedy=False))
                recorded.append(False)
            elif u < teacher_ratio + snapshot_ratio + pool_ratio and pool:
                # **对手池**：坐一个别的风格的模型（v1.1 的另外三个）。
                # 只跟自己+教师下，学不到「面对速攻该怎么防」这类东西。
                opp = pool[int(rng.integers(len(pool)))]
                agents.append(MLAgent(opp, seed=game_seed * n + seat, greedy=False))
                recorded.append(False)
            else:
                agents.append(
                    MLAgent(net, seed=game_seed * n + seat, tracker=tracker, greedy=False)
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
) -> list[dict]:
    rng = np.random.default_rng(seed)
    data = dict(data)
    data["adv"] = compute_advantages(net, data)
    n = len(data["X"])
    hist = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
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
            )
            for k in tot:
                tot[k] += st[k]
            nb += 1
        row = {k: v / max(1, nb) for k, v in tot.items()}
        row["epoch"] = ep + 1
        hist.append(row)
    if verbose and hist:
        h = hist[-1]
        print(
            f"    PPO  policy_loss={h['policy_loss']:+.4f}  entropy={h['entropy']:.3f}"
            f"  clip={h['clip_frac']:.1%}  value_loss={h['value_loss']:.4f}"
        )
    return hist


# ---------------------------------------------------------------------------
# 快速评测（训练过程中用）
# ---------------------------------------------------------------------------


def quick_eval(
    net: Net,
    n_matches: int = 60,
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
        agents: list[Agent] = [MLAgent(net, seed=game_seed, greedy=greedy)]
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


def cmd_bc(args) -> Net:
    spec = spec_of(args)
    print(f"① 行为克隆：{spec.cn}（{spec.name}）")
    print(f"   {spec.describe()}")
    net = build_net(args, resume=args.resume)
    data = collect_bc(args.hands, seed=args.seed, spec=spec)
    holdout = collect_bc(
        max(60, args.hands // 20), seed=args.seed + 555_555, verbose=False, spec=spec
    )
    print(f"  留出集：{len(holdout['X'])} 条决策（种子不同，用来判断有没有过拟合）")
    hist = train_bc(
        net, data, epochs=args.epochs, batch_size=args.batch, seed=args.seed,
        holdout=holdout,
    )
    net.save(args.out)
    print(f"  已保存 -> {args.out}（{net.n_params()} 参数）")
    save_log(args, {"bc": {"hist": hist, "samples": int(len(data["X"]))}})
    return net


def cmd_rl(args) -> Net:
    spec = spec_of(args)
    print(f"② 自对弈强化学习（PPO）：{spec.cn}（{spec.name}）")
    print(f"   得分条件 = {rewards.MODE_CN.get(spec.reward, spec.reward)}")
    net = build_net(args, resume=True)
    net.lr = args.rl_lr  # 强化学习用更小的步长，别把克隆来的牌效打散
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
    snapshot = net.copy()
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
            print(f"    ↑ 新高，已保存 {args.out}")
        if it % 5 == 0:
            snapshot = net.copy()
    if not Path(args.out).exists():
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
    ap.add_argument("--matches", type=int, default=300, help="每轮自对弈的场数")
    ap.add_argument("--iters", type=int, default=20, help="自对弈轮数")
    ap.add_argument("--epochs", type=int, default=6, help="行为克隆的 epoch 数")
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
