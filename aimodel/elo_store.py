"""ELO 期次存档 + 逐手回放。

## 为什么每局只存「种子 + 参战者」

`aimodel/arena.py` 的每一局是**完全确定**的：牌山与每个 AI 的种子都从那一个
「场次种子」派生（``seed_of(index)`` / ``_agent(p, seed*players+i)``）。
**实测**（2026-10-20，一局四麻 633 个动作）：同 seed 跑两次，动作**逐个相同**、
分数名次全同。所以存档只需要

    {seed, players, 参战者列表}

就能用**同一个引擎**逐手重放出完整对局（重放一局约 0.5 秒）。一局约 200 字节：

    4000 局/期 ≈ 800 KB   ｜   5 期/天 ≈ 4 MB/天   ｜   3 天 ≈ 12 MB

—— 100 GB 的配额在这个量级下根本用不到（见 :func:`stats`）。

## 为什么不存「平台那份 23 KB/局的决策日志」

和网页对战平台同格式的那份记录（决策级 + 完整 ``view``）也能回放，但一局 23 KB，
是种子方案的 **100 倍**，而收益只是「不用重跑引擎」。重跑一局 0.5 秒，不值得。
-> 所以默认**只存种子**；若确实需要那份富记录，调用方另存，本模块不管。

## 目录结构（全部在 ``_data/elo/``，已被 gitignore）

    _data/elo/
      seasons/
        2026-10-20T08/            <- 一「期」（官方每天 8/11/14/17/20 各一期）
          meta.json               <- 期元数据（开始/结束时间、参战者、场次、冠军）
          matches.jsonl           <- 一行一局：{seed, players, group, ranks, scores}
          favorite.json           <- 收藏标记（只有管理员能写）
        2026-10-20T11/
        ...
      scores/
        <期号>.json               <- 该期的最终 ELO 表（不随回放数据一起删）
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent
#: 全部 ELO 数据的根。**已被 .gitignore 的 ``_data/`` 覆盖** ✓
#: ``$POTATOMAJ_ELO_DIR`` 可以把它指到独立目录（小样验证绝不碰生产数据 ✓）
DATA_DIR = Path(os.environ.get("POTATOMAJ_ELO_DIR") or (ROOT / "_data" / "elo"))
SEASONS_DIR = DATA_DIR / "seasons"
SCORES_DIR = DATA_DIR / "scores"

#: 保留天数：超过这个天数、且没被收藏的期，只留「冠军名字」
KEEP_DAYS = 3

#: 每一局记录的字段（顺序固定，写 JSONL 时用 separators 省空间）
MATCH_FIELDS = ("seed", "players", "group", "ranks", "scores")
#: 含外部 AI 的局才有的**可选**标记字段（老局没有这些键，老行为不变 ✓）
#:
#: 为什么必须存：Mortal 不是我们引擎里的确定性模型，**回放不能靠重跑**。
#: 标记之后 ``replay`` 会明确说「这一局不可重放」，而不是跑出一个错的对局。
OPTIONAL_MATCH_FIELDS = (
    "external",          #: True = 这一局有外部 AI 参战
    "external_failed",   #: 外部进程在本局崩过/超时/非法动作 -> 该座位降级教师
    "external_specs",    #: 外部 AI 的 spec key，如 ["mortal-akagi"]
    "external_seats",    #: 外部位座位下标，如 [2]
)


class ReplayUnsupported(RuntimeError):
    """这一局不能靠「同种子重跑」复现（含外部 AI）。"""


class ExternalMatchNotReplayable(ReplayUnsupported):
    """含外部参战者的局：明确拒绝回放（不静默跑错）。"""


# ---------------------------------------------------------------------------
# 期（Season）
# ---------------------------------------------------------------------------


@dataclass
class Season:
    """一期的元数据。"""

    sid: str  #: 期号，例如 "2026-10-20T08"
    players: int = 4
    copies: int = 3
    seed_base: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    target: int = 0  #: 计划场次
    done: int = 0  #: 实际完成场次
    champion: str = ""  #: 冠军**模型名**（按模型均分 ✓ 不是某个实例 ✗）
    champion_key: str = ""
    champion_avg: float = 0.0       #: 冠军模型的**均分**
    champion_inst: str = ""         #: 单实例里分最高的那个（参考 ✓ 噪声大 ✗）
    champion_inst_score: float = 0.0
    pruned: bool = False  #: True = 回放数据已删，只剩冠军
    favorite: bool = False  #: 被管理员收藏 -> 永不清理
    note: str = ""

    @property
    def dir(self) -> Path:
        return SEASONS_DIR / self.sid

    def as_dict(self) -> dict:
        return {
            "sid": self.sid, "players": self.players, "copies": self.copies,
            "seed_base": self.seed_base, "started_at": self.started_at,
            "finished_at": self.finished_at, "target": self.target, "done": self.done,
            "champion": self.champion, "champion_key": self.champion_key,
            "champion_avg": self.champion_avg,
            "champion_inst": self.champion_inst,
            "champion_inst_score": self.champion_inst_score,
            "pruned": self.pruned, "favorite": self.favorite, "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Season":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def seed_from_sid(sid: str) -> int:
    """从**期号**派生牌山种子。

    ⚠️ 每期必须换种子 ✗ —— 否则每期都打**同一副牌**（用户 2026-10-20 指出）
    ⚠️ 但不能用纯随机 ✗ —— 补跑 / 续跑同一期要能**复现**
    -> 用期号的哈希：同一期恒定 ✓ 不同期不同 ✓
    """
    import hashlib

    h = hashlib.sha256(sid.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 2_000_000_000


def model_of(key: str) -> str:
    """参战者 key -> **模型名**（去掉 ``#N`` 实例后缀）。"""
    return key.split("#", 1)[0]


def champion_by_model(table: list[dict]) -> tuple[str, float, str, float]:
    """从 ``Arena.table()`` 的行算出冠军 —— **按模型均分**。

    为什么不能只看单个实例：每个模型派 ``copies`` 个参赛者（默认 3 个），
    各自记分。单个实例的第一名**噪声很大**（三份同权重，谁运气好谁高）。
    用户要求「**模型的均分第一也要显示**」，所以这里以均分为准。

    返回 ``(模型名, 均分, 单实例第一名, 该实例分)``。
    """
    if not table:
        return "", 0.0, "", 0.0
    agg: dict[str, list[float]] = {}
    inst_name, inst_score = "", float("-inf")
    for r in table:
        model = str(r.get("模型") or "")
        score = float(r.get("分数") or 0.0)
        if not model:
            continue
        agg.setdefault(model, []).append(score)
        if score > inst_score:
            inst_name, inst_score = str(r.get("名") or model), score
    if not agg:
        return "", 0.0, inst_name, (0.0 if inst_score == float("-inf") else inst_score)
    # 均分最高者；并列时取「份数更少也照样高」的那个（更可信）
    top = max(agg, key=lambda m: (sum(agg[m]) / len(agg[m]), -len(agg[m])))
    avg = sum(agg[top]) / len(agg[top])
    return top, round(avg, 1), inst_name, round(0.0 if inst_score == float("-inf") else inst_score, 1)


def _num(v: Any, default: float = 0.0) -> float:
    """把 JSON 里的数字/数字字符串转成 float（坏值不抛异常）。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def model_ranking(table: list[dict]) -> list[dict]:
    """把实例表聚合成**模型榜** —— ELO 按**场数加权**，不是简单平均。

    为什么必须加权：同一个模型派 ``copies`` 个实例（默认 3），各自记分。
    简单平均会让「只打了几场、运气好分高」的实例和「打了几百场」的实例
    同权 —— 少打的实例把模型均分带偏（用户 2026-10-21 明确要求按局数加权）。

    口径（与页面上写的完全一致，历史/实时共用这一个函数）：
      · ``ELO = Σ(实例分数 × 实例场数) / Σ实例场数``
      · ``一位率 = Σ一位 / Σ场数``
      · ``和了率 = Σ(和了率 × 局数) / Σ局数``（``局数`` 是手数分母）
      · ``放铳率 = Σ(放铳率 × 局数) / Σ局数``
      · ``实例数`` 含没上过场的实例；``已赛实例`` 只算场数 > 0 的
      · 一个模型一场都没打 -> 不进榜（与 ``Arena`` 的 ``summary`` 一致）

    入参就是 ``Arena.table()`` 的行（**中文键**：名/模型/分数/场数/局数/…），
    历史 JSON 和实时快照是同一套字段，所以两处数字必然一致。
    """
    groups: dict[str, list[dict]] = {}
    for row in table or []:
        if not isinstance(row, dict):
            continue
        model = str(row.get("模型") or model_of(str(row.get("名") or "")) or "").strip()
        if not model:
            continue
        groups.setdefault(model, []).append(row)

    out: list[dict] = []
    for model, rows in groups.items():
        rated = [r for r in rows if _num(r.get("场数")) > 0]
        if not rated:
            continue
        counts = [(r, max(0.0, _num(r.get("场数")))) for r in rated]
        games = sum(c for _, c in counts)
        hands = sum(max(0.0, _num(r.get("局数"))) for r, _ in counts)
        # 按场数加权：ELO 的样本量是「打过多少场」，不是「打过多少手」。
        elo = sum(_num(r.get("分数")) * c for r, c in counts) / games if games else 0.0
        best = max(_num(r.get("分数")) for r in rated)
        worst = min(_num(r.get("分数")) for r in rated)
        wins = sum(max(0.0, _num(r.get("一位"))) for r, _ in counts)
        first_rate = wins / games if games else 0.0
        # 和了/放铳的分母是手数：先还原成次数再相加，避免「先四舍五入成比率再加权」。
        agari = sum(_num(r.get("和了率")) * max(0.0, _num(r.get("局数")))
                    for r, _ in counts)
        deal_in = sum(_num(r.get("放铳率")) * max(0.0, _num(r.get("局数")))
                      for r, _ in counts)
        agari_rate = agari / hands if hands else 0.0
        deal_rate = deal_in / hands if hands else 0.0

        def _uniq(key: str) -> str:
            vals: list[str] = []
            for r in rows:
                v = str(r.get(key) or "").strip()
                if v and v not in vals:
                    vals.append(v)
            return " / ".join(vals)

        out.append({
            "模型": model,
            "实例数": len(rows),
            "已赛实例": len(rated),
            "场数": int(games),
            "局数": int(hands),
            "分数": round(elo, 1),
            "ELO": round(elo, 1),
            "均分": round(sum(_num(r.get("分数")) for r in rated) / len(rated), 1),
            "最好": round(best, 1),
            "最差": round(worst, 1),
            "极差": round(best - worst, 1),
            "一位率": round(first_rate, 4),
            "和了率": round(agari_rate, 4),
            "放铳率": round(deal_rate, 4),
            "版本": _uniq("版本"),
            "风格": _uniq("风格"),
            "保底": sum(1 for r in rows
                        if _num(r.get("触底")) > 0
                        or (str(r.get("状态") or "") not in ("", "存活")
                            and _num(r.get("场数")) > 0)),
            "加权": "场数",
        })
    out.sort(key=lambda r: (-r["分数"], r["模型"]))
    return out


def season_start_ts(s: Season) -> float:
    """一期的开始时间；没有时间戳就从期号 ``YYYY-MM-DDTHH`` 推。"""
    if s.started_at:
        return float(s.started_at)
    if s.finished_at:
        return float(s.finished_at)
    try:
        return time.mktime(time.strptime(s.sid[:13], "%Y-%m-%dT%H"))
    except (TypeError, ValueError):
        return 0.0


def build_trend(season_tables: list[tuple[str, list[dict]]]) -> dict:
    """跨期趋势：``[(期号, 模型榜)]`` -> 每个模型一条带缺口的折线。

    **缺期返回 ``None`` 点**（不是补 0、也不是沿用上一期 ✗）。前端遇到
    ``None`` 就断开折线 —— 这是用户明确要求的「缺期要断开，不许连线」。
    一期有存档但没 ELO 表（还在跑 / 被精简）时，该期所有模型都是 ``None``。
    """
    items = sorted(((str(sid), rows or []) for sid, rows in season_tables),
                   key=lambda x: x[0])
    sids = [sid for sid, _ in items]
    by_model: dict[str, dict] = {}
    for sid, rows in items:
        for r in rows:
            model = str(r.get("模型") or "").strip()
            if not model:
                continue
            slot = by_model.setdefault(model, {
                "模型": model,
                "版本": str(r.get("版本") or ""),
                "风格": str(r.get("风格") or ""),
                "points": {},
            })
            slot["points"][sid] = {
                "sid": sid,
                "ELO": r.get("ELO", r.get("分数")),
                "场数": r.get("场数", 0),
                "一位率": r.get("一位率", 0.0),
            }
    series: list[dict] = []
    for d in by_model.values():
        points = [d["points"].get(sid) for sid in sids]
        last = next((p for p in reversed(points) if p), None)
        series.append({
            "模型": d["模型"],
            "版本": d["版本"],
            "风格": d["风格"],
            "points": points,
            "last": last,
            "n": sum(1 for p in points if p),
        })
    series.sort(key=lambda r: (-(r["last"]["ELO"] if r["last"] else -1e9), r["模型"]))
    missing = [sid for sid, rows in items if not rows]
    return {"sids": sids, "series": series, "missing": missing}


def season_id(when: float | None = None, slot_hour: int | None = None) -> str:
    """期号：``YYYY-MM-DDTHH``。``slot_hour`` 给了就用它（官方固定档位）。"""
    t = time.localtime(when if when is not None else time.time())
    hour = slot_hour if slot_hour is not None else t.tm_hour
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}T{hour:02d}"


def open_season(sid: str, *, players: int = 4, copies: int = 3,
                seed_base: int = 0, target: int = 0) -> Season:
    """开一期（若已存在就返回已有的 ✓ 幂等）。"""
    s = Season(sid=sid, players=players, copies=copies,
               seed_base=seed_base, target=target)
    s.dir.mkdir(parents=True, exist_ok=True)
    meta = s.dir / "meta.json"
    if meta.exists():
        try:
            old = Season.from_dict(json.loads(meta.read_text(encoding="utf-8")))
            # 保留已记录的进度/收藏，只更新计划
            old.target = target or old.target
            old.copies = copies
            old.players = players
            return old
        except (json.JSONDecodeError, TypeError):
            pass
    s.started_at = time.time()
    save_season(s)
    return s


def save_season(s: Season) -> None:
    s.dir.mkdir(parents=True, exist_ok=True)
    (s.dir / "meta.json").write_text(
        json.dumps(s.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    fav = s.dir / "favorite.json"
    if s.favorite:
        fav.write_text(json.dumps({"favorite": True, "at": time.time()}), encoding="utf-8")
    elif fav.exists():
        fav.unlink()


def load_season(sid: str) -> Season | None:
    meta = SEASONS_DIR / sid / "meta.json"
    if not meta.exists():
        return None
    try:
        s = Season.from_dict(json.loads(meta.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return None
    s.favorite = (SEASONS_DIR / sid / "favorite.json").exists()
    return s


def list_seasons() -> list[Season]:
    """全部期，**新的在前**。"""
    if not SEASONS_DIR.exists():
        return []
    out = [s for s in (load_season(p.name) for p in SEASONS_DIR.iterdir()
                       if p.is_dir()) if s is not None]
    out.sort(key=lambda s: s.sid, reverse=True)
    return out


# ---------------------------------------------------------------------------
# 逐局记录
# ---------------------------------------------------------------------------


def append_match(s: Season, match: dict[str, Any]) -> None:
    """追加一局。老字段固定；含外部 AI 时额外带 :data:`OPTIONAL_MATCH_FIELDS`。

    ⚠️ 可选字段只在**调用方给了**才写 —— 不含外部 AI 的局仍然只有 5 个键，
    老的 ``matches.jsonl`` / 老测试全部不受影响 ✓
    """
    rec = {k: match[k] for k in MATCH_FIELDS if k in match}
    for k in OPTIONAL_MATCH_FIELDS:
        if k in match:
            rec[k] = match[k]
    if "seed" not in rec or "group" not in rec:
        raise ValueError("记录一局至少要有 seed 和 group")
    s.dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
    with (s.dir / "matches.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    s.done += 1


def iter_matches(s: Season) -> Iterable[dict]:
    """逐局读回来（**流式** ✓ 一期 4000 行也不占内存）。"""
    p = s.dir / "matches.jsonl"
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def get_match(s: Season, index: int) -> dict | None:
    """取第 ``index`` 局（0 基）。"""
    for i, m in enumerate(iter_matches(s)):
        if i == index:
            return m
    return None


# ---------------------------------------------------------------------------
# 回放：用**同一个引擎**重跑
# ---------------------------------------------------------------------------


def replay(match: dict, *, on_action: Callable[[dict], None] | None = None):
    """按 ``{seed, players, group}`` 用同一个引擎逐手重跑一局。

    ``group`` 的每一项是 :class:`aimodel.arena.Participant` 的 key
    （例如 ``"maj-flashv1.2#2"``）。**顺序必须保持原样** ——
    因为每个 AI 的种子是 ``seed*players + i``，换顺序就换牌。

    ⚠️ **含外部 AI 的局不能重放** ✗：Mortal 是外部进程，不是我们引擎里的
    确定性模型，同一串事件不保证同一个动作（还可能崩）。这种局在
    ``matches.jsonl`` 里带 ``external=true``；这里**明确抛**
    :class:`ExternalMatchNotReplayable`，让调用方提示「不可重放」，
    **绝不**拿重跑出来的假对局冒充存档 ✓

    返回 ``(actions, result)``：``actions`` 是动作列表（每项含 kind/actor/…），
    ``result`` 是引擎的 ``MatchResult``。
    """
    if match.get("external"):
        specs = match.get("external_specs") or []
        seats = match.get("external_seats") or []
        raise ExternalMatchNotReplayable(
            "这一局含外部 AI"
            + (f"（{', '.join(map(str, specs))}）" if specs else "")
            + (f"，座位 {seats}" if seats else "")
            + "，外部进程不是确定性模型，无法用「同种子重跑」回放；"
            "请在期列表里跳过这一局（matches.jsonl 已标记 external=true）"
        )
    from aimodel.agents import HeuristicAgent, MLAgent, RandomAgent
    from aimodel.engine import Game
    from aimodel.net import Net
    from aimodel import variants

    seed = int(match["seed"])
    players = int(match.get("players", 4))
    group = list(match["group"])

    nets: dict[str, Net] = {}
    agents = []
    for i, key in enumerate(group):
        name, _, inst = key.partition("#")
        aseed = seed * players + i
        kind, src = _kind_and_source(name)
        if kind == "model":
            if src not in nets:
                spec = variants.get(src)
                nets[src] = Net.load(spec.path)
            agents.append(MLAgent(nets[src], seed=aseed, greedy=True))
        elif kind == "random":
            agents.append(RandomAgent(aseed))
        else:
            agents.append(HeuristicAgent(aseed))

    actions: list[dict] = []
    g = Game(agents, seed=seed, rounds=players)

    def _hook(a: dict) -> None:
        actions.append(_plain(a))
        if on_action is not None:
            on_action(a)

    g.on_action = _hook
    result = g.play()
    return actions, result


def _kind_and_source(name: str) -> tuple[str, str]:
    """把参战者名解析成 ``(kind, 权重来源)``。

    ⚠️ 基线参战者的 key 是 ``teacher4#1`` / ``random4#1`` ✗ —— **带人数后缀** ✗
       （由 ``arena.loose_participants`` 生成 ✓）
       所以必须按**前缀**判 ✓ 只匹配精确的 ``teacher`` / ``random`` 会漏掉全部基线 ✗
       —— 这个 bug 是 ``test_elo_store.py`` 抓出来的 ✓ 代价是含基线的对局一律回放不了 ✗
    """
    low = name.lower()
    if low.startswith("external:") or low.startswith("external_") \
            or low.startswith("ext:"):
        raise ExternalMatchNotReplayable(
            f"参战者 {name!r} 是外部 AI，无法用种子重放")
    if low.startswith("teacher"):
        return "teacher", ""
    if low.startswith("random"):
        return "random", ""
    return "model", name


def _plain(a: dict, _depth: int = 0) -> dict:
    """把动作 dict 转成可 JSON 序列化的值。

    ⚠️ **要递归** ✗ —— 第一版只处理一层，于是 `start_hand` 里的
       ``hands``（4×34 的整数数组）被整个 ``str()`` 成字符串了 ✗
       前端拿到就只是一坨文本 ✓ 现在递归保留结构 ✓
    """
    def cv(v: Any) -> Any:
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
        if isinstance(v, dict):
            return {str(k): cv(x) for k, x in v.items()}
        return str(v)

    return {str(k): cv(v) for k, v in a.items()}


# ---------------------------------------------------------------------------
# 清理 / 配额
# ---------------------------------------------------------------------------


def prune(*, keep_days: int = KEEP_DAYS, now: float | None = None,
          favorites: set[str] | None = None) -> list[str]:
    """超过 ``keep_days`` 且**没被收藏**的期 -> 删回放数据，只留冠军名。

    返回被清理的期号列表。

    ⚠️ 它**不删** ``scores/<期号>.json``（最终 ELO 表）—— 那是「历史数据」，
       用户要求能查看历史 ✓ 只有逐局回放数据才受 3 天限制 ✓
    """
    now = now if now is not None else time.time()
    cutoff = now - keep_days * 86400
    favorites = favorites or set()
    pruned: list[str] = []
    for s in list_seasons():
        fav = s.favorite or s.sid in favorites
        if s.pruned or fav:
            continue
        ref = s.finished_at or s.started_at
        if ref and ref < cutoff:
            # 记下冠军（从最终 ELO 表取），然后删逐局数据
            if not s.champion:
                champ = _champion_from_scores(s.sid)
                if champ:
                    (s.champion, s.champion_avg,
                     s.champion_inst, s.champion_inst_score) = champ
                    s.champion_key = s.champion
            mf = s.dir / "matches.jsonl"
            if mf.exists():
                mf.unlink()
            s.pruned = True
            s.done = 0
            save_season(s)
            pruned.append(s.sid)
    return pruned


def _champion_from_scores(sid: str) -> tuple[str, float, str, float] | None:
    """从 ``scores/<sid>.json`` 里算冠军（**按模型均分**，与 ``prune`` 之外的口径一致）。

    返回 ``(模型名, 均分, 单实例第一名, 该实例分)``，没有表就返回 ``None``。
    """
    p = SCORES_DIR / f"{sid}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    rows = d.get("table") or []
    if not rows:
        return None
    # `Arena.table()` 的字段名是**中文**（"名"/"模型"/"分数"），不是 label/key/rating。
    return champion_by_model(rows)


def save_scores(sid: str, table: list[dict], summary: dict | None = None) -> None:
    """存该期的最终 ELO 表（「历史数据」不受 3 天限制 ✓）。"""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    (SCORES_DIR / f"{sid}.json").write_text(
        json.dumps({"sid": sid, "at": time.time(), "table": table,
                    "summary": summary or {}}, ensure_ascii=False, indent=1),
        encoding="utf-8")


def load_scores(sid: str) -> dict | None:
    p = SCORES_DIR / f"{sid}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def stats() -> dict:
    """用了多少空间、还剩多少（对照 100 GB 配额）。

    ⚠️ ``per_match_bytes`` 只除 **``matches.jsonl``** 的字节 ✗ 不是全部文件 ✓
       因为 ``scores/*.json`` 是**每期一张** ELO 表（约 24 KB），
       把它除进「每局」会把数字虚高 10 倍（第一版就是 1768 B/局 ✗ 真实 150 B ✓）。
    """
    total = 0
    replay_bytes = 0
    scores_bytes = 0
    n_match = 0
    n_season = 0
    if DATA_DIR.exists():
        for p in DATA_DIR.rglob("*"):
            if not p.is_file():
                continue
            sz = p.stat().st_size
            total += sz
            if p.name == "matches.jsonl":
                n_season += 1
                replay_bytes += sz
                with p.open("rb") as fh:
                    n_match += sum(1 for _ in fh)
            elif p.parent == SCORES_DIR:
                scores_bytes += sz
    quota = 100 * 1024 ** 3
    return {
        "bytes": total, "human": _human(total),
        "replay_bytes": replay_bytes, "replay_human": _human(replay_bytes),
        "scores_bytes": scores_bytes, "scores_human": _human(scores_bytes),
        "quota_bytes": quota, "quota_human": _human(quota),
        "used_pct": round(total / quota * 100, 6),
        "seasons_with_replay": n_season, "matches": n_match,
        "per_match_bytes": round(replay_bytes / n_match, 1) if n_match else 0,
        #: 按当前每局大小，100 GB 还能装多少局
        "matches_capacity": (int(quota / (replay_bytes / n_match))
                             if n_match and replay_bytes else 0),
    }


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
