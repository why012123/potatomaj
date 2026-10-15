"""八个模型的注册表。

命名规则（用户定的）：三麻叫 ``maj-<风格>``，四麻在原名后面加 ``4``。
别名也认：``flash`` / ``flash4`` / ``maj-flash`` / ``maj-flash4`` 都能用。

===============  ======  ==========  ========  ==========================
模型              人数    得分条件     教师风格   一句话
===============  ======  ==========  ========  ==========================
``maj-normal``    3     normal      normal    标准段位分
``maj-mountain``  3     mountain    mountain  防守专精（重罚放铳）
``maj-flash``     3     flash       flash     速攻（按和牌次数给钱）
``maj-huge``      3     huge        huge      大牌（按番数给钱）
``maj-normal4``   4     normal      normal    四麻标准
``maj-mountain4`` 4     mountain    mountain  四麻防守
``maj-flash4``    4     flash       flash     四麻速攻
``maj-huge4``     4     huge        huge      四麻大牌
===============  ======  ==========  ========  ==========================

**得分条件**（奖励函数）在 :mod:`aimodel.rewards`；
**教师风格**（拿来行为克隆的启发式口味）在 :mod:`aimodel.agents`。
两套要配套：追速度的模型就该跟「爱碰爱快的教师」学，不然一上来方向就是歪的。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

#: 当前版本。改名规则跟着用户定的来：**原名称直接接版本号**
#: （三麻 ``maj-flashv1.1``、四麻 ``maj-flash4v1.1``）。
#: 不带版本号的写法一律当别名，指向最新版。
VERSION = "v1.1"

#: 四种风格的说明（中文，报告和日志里用）
STYLE_CN = {
    "normal": "标准",
    "mountain": "防守",
    "flash": "速攻",
    "huge": "大牌",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str  #: 正名（带版本），比如 maj-flash4v1.1
    players: int  #: 3 或 4
    reward: str  #: rewards.MODES 之一
    style: str  #: 教师风格（agents.TeacherStyle 的名字）
    version: str = VERSION

    @property
    def path(self) -> Path:
        return WEIGHTS_DIR / f"{self.name}.npz"

    @property
    def cn(self) -> str:
        return f"{'四麻' if self.players == 4 else '三麻'}·{STYLE_CN.get(self.style, self.style)}"

    def describe(self) -> str:
        from .rewards import MODE_CN

        return f"{self.name:<15} {self.cn:<8} 得分条件: {MODE_CN.get(self.reward, self.reward)}"


def _build(version: str = VERSION) -> dict[str, ModelSpec]:
    out: dict[str, ModelSpec] = {}
    for players, suffix in ((3, ""), (4, "4")):
        for style in ("normal", "mountain", "flash", "huge"):
            name = f"maj-{style}{suffix}{version}"
            out[name] = ModelSpec(name, players, style, style, version)
    return out


#: 正名 -> 规格
MODELS: dict[str, ModelSpec] = _build()


def _register_pro(version: str = VERSION) -> None:
    """注册 ``maj-pro`` —— 用**真实天鳳人类牌谱**做 BC 的那一系。

    为什么单独注册、而不是加进 ``_build`` 的 style 列表：那个列表里的每个
    style 都同时是**教师风格**（``HeuristicAgent(style=...)``），而 pro 不是
    一种教师口味 —— 它的训练数据来自人类，不来自教师。

    但 ``spec.style`` 会被用来生成**留出集**（见 ``train.cmd_bc``），
    而留出集必须是个固定标尺才能和已有的 maj-* 模型比。所以这里
    ``reward`` 沿用用户条件，``style`` 固定成 ``"normal"`` —— 表示
    「留出集用标准教师」，而不是「这个模型是标准风格」。

    训练方式：``python -m aimodel.train bc --model=maj-pro4v1.3
    --bc-data=_data/bc_tv13.npz``（数据集由 ``aimodel/tenhou_bc.py`` +
    ``aimodel/shortjob.py`` 从天鳳牌谱抽出）。
    """
    for players, suffix in ((3, ""), (4, "4")):
        name = f"maj-pro{suffix}{version}"
        reward = "normal" if players == 4 else "normal"
        MODELS[name] = ModelSpec(name, players, reward, "normal", version)


_register_pro()

# 注册之后要重建别名表（下面那段是按 MODELS 生成的）

#: 别名（不带版本、不带 maj- 前缀、下划线等写法都认，一律指向最新版）
_ALIASES: dict[str, str] = {}
for _name, _spec in MODELS.items():
    _bare = _name[: -len(_spec.version)]          # maj-flash4
    _forms = {
        _name,                                     # maj-flash4v1.1
        _bare,                                     # maj-flash4  -> 最新版
        _name.replace("-", "_"),
        _bare.replace("-", "_"),
        _name.replace("maj-", ""),
        _bare.replace("maj-", ""),                 # flash4
        _bare.replace("maj-", "").replace("-", "_"),
    }
    for _f in _forms:
        _ALIASES[_f.lower()] = _name
del _name, _spec, _bare, _forms, _f


def versions(style: str, players: int) -> list[str]:
    """某个风格/人数下，历史上存在过的版本（新→旧）。"""
    out = []
    for f in sorted(WEIGHTS_DIR.glob(f"maj-{style}{'4' if players == 4 else ''}v*.npz")):
        out.append(f.stem)
    return out


def resolve(name: str) -> str:
    """把别名归一成正名。"""
    key = name.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # 宽松匹配：majflash4 / maj_flash_4 / flash4_v1.1 这种写法都认。
    # 注意要把版本号也归一化掉，否则 "majflash4" 对不上 "majflash4v1.1"。
    squash = key.replace("_", "").replace("-", "").replace(".", "")
    for cand, spec in MODELS.items():
        bare = cand[: -len(spec.version)]
        for form in (cand, bare):
            if squash == form.replace("-", "").replace(".", "").lower():
                return spec.name
    raise KeyError(f"没有这个模型: {name}（可选: {', '.join(sorted(MODELS))}）")


_VERSIONED = re.compile(r"^(maj-[a-z]+4?)(v[\d.]+)$")


def get(name: str) -> ModelSpec:
    """按名字取规格。**显式带版本号的写法一定认**（即使那不是当前版本），
    这样训练 v1.2 的时候还能引用 v1.1 的模型当对手。"""
    key = name.strip().lower()
    m = _VERSIONED.match(key)
    if m and key not in MODELS:
        built = _build(m.group(2))
        if key in built:
            return built[key]
    return MODELS[resolve(name)]


def all_specs(version: str | None = None) -> list[ModelSpec]:
    """不传版本就是当前版本；传了就按那个版本生成（训练下一版时用）。"""
    if version is None or version == VERSION:
        return [MODELS[n] for n in sorted(MODELS)]
    built = _build(version)
    return [built[n] for n in sorted(built)]


def same_players(players: int, version: str | None = None) -> list[ModelSpec]:
    return [s for s in all_specs(version) if s.players == players]


def existing() -> list[ModelSpec]:
    """已经训练出来的（有权重文件的）。"""
    return [s for s in all_specs() if s.path.exists()]


def audit_weights(verbose: bool = True) -> dict:
    """**启动自检**：把所有权重试加载一遍，失败的明确列出来。

    为什么要有这个：给网络加一个决策头（比如 chi）会让**全部存量权重**一次性
    加载失败。如果没有自检，症状是「玩家点了『选 AI 对战』才发现某个模型瘫了」
    —— 而那时候已经很难定位了（踩过一次：44 个权重全挂）。

    用法::

        python -m aimodel.variants --audit        # 命令行
        from aimodel.variants import audit_weights
        audit_weights()                           # 服务启动时调用

    返回 ``{"ok": [...], "failed": {名字: 错误}, "untrained": {名字: [缺的头]}}``
    """
    from .net import Net

    ok: list[str] = []
    failed: dict[str, str] = {}
    untrained: dict[str, list[str]] = {}
    import warnings as _w

    missing_files: list[str] = []
    for spec in all_specs():
        if not spec.path.exists():
            # ⚠️ 注册了但权重文件不存在 —— **必须报出来，不能静默跳过** ✗
            # 自检的意义就是发现「模型在列表里但选不了」✓
            # （maj-pro 刚注册时正是这个状态：天梯/平台选到它会失败）
            missing_files.append(spec.name)
            continue
        try:
            # 缺头警告由本函数自己汇总，别让 44 条一样的警告淹掉输出
            with _w.catch_warnings():
                _w.simplefilter("ignore", UserWarning)
                net = Net.load(spec.path)
        except Exception as e:                      # noqa: BLE001
            failed[spec.name] = f"{type(e).__name__}: {e}"
            continue
        ok.append(spec.name)
        miss = sorted(getattr(net, "untrained_heads", ()) or ())
        if miss:
            untrained[spec.name] = miss
    if verbose:
        print(f"权重自检：{len(ok)} 个正常"
              + (f"，{len(failed)} 个**加载失败**" if failed else "，0 个失败"))
        for name, err in failed.items():
            print(f"  ❌ {name}: {err}")
        if untrained:
            heads = sorted({h for v in untrained.values() for h in v})
            print(f"  ⚠️ {len(untrained)} 个权重缺决策头 {heads}"
                  "（已零初始化，这些决策走启发式兜底；想让它自己学就重训）")
            for name in list(untrained)[:4]:
                print(f"      {name}: {untrained[name]}")
    if verbose and missing_files:
        print(f"  ⚠️ {len(missing_files)} 个模型**注册了但权重文件不存在**"
              "（选到它们会失败，先补文件或先别注册）：")
        for name in missing_files:
            print(f"      {name}")
    return {"ok": ok, "failed": failed, "untrained": untrained,
            "missing_files": missing_files}


__all__ = [
    "audit_weights","ModelSpec", "MODELS", "STYLE_CN", "get", "resolve", "all_specs", "existing"]


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="模型注册表 / 权重自检")
    ap.add_argument("--audit", action="store_true", help="把所有权重试加载一遍")
    ap.add_argument("--list", action="store_true", help="列出全部模型")
    args = ap.parse_args(argv)
    if args.audit:
        r = audit_weights()
        bad = bool(r["failed"]) or bool(r.get("missing_files"))
        raise SystemExit(1 if bad else 0)
    for spec in all_specs():
        mark = "✅" if spec.path.exists() else "—"
        print(f"  {mark} {spec.name:<22} {spec.players}麻  {spec.cn}")


if __name__ == "__main__":
    main()
