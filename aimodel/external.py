"""把市面上的开源 mjai AI 当成对手接进本项目。

## 为什么是「协议桥」而不是「把模型搬进来」

本项目的 :class:`~aimodel.agents.Agent` 只有一个同步接口::

    Agent.decide(decision, view) -> int

而开源麻将 AI（Mortal、mjai-manue、Kanachan、akochan …）几乎都活在
**mjai 协议**里：一行一个 JSON 事件，进程从 stdin 读事件、往 stdout 吐动作。
两边对「一局」的记录方式不一样，所以这里只做一层**翻译**：

* 引擎的结构化动作流（``Game.on_action``）-> mjai 事件；
* 外部 AI 的 mjai 动作 -> ``decision.options`` 的下标。

外部 AI 的源码/权重**不进来**，只通过环境变量或 JSON 配置指向它。
没装时构造 :class:`ExternalMJaiAgent` 会抛 :class:`ExternalAINotInstalled`，
错误信息里带安装文档路径 —— **绝不静默降级成 ``return 0``**。

## 怎么用

**配对赛 / 天梯**::

    # 需要先按 docs/接入外部AI.md 装好，然后：
    python -m aimodel.arena_test --a aimodel/weights/maj-pro4v1.1_mac_rl.npz \\
        --b aimodel/weights/maj-normalv1.1.npz --third manue --max-games 200

    # 也可以让外部 AI 当被测方（A 或 B）：
    python -m aimodel.arena_test --a manue --b aimodel/weights/maj-pro4v1.1_mac_rl.npz

**代码里**::

    from aimodel import external
    ag = external.make("manue", seed=1, player_id=2, players=4)
    game = Game([..., ag, ...], seed=1, rule=S.rule_for(4))
    external.attach(game)          # 把 on_action 转发给外部 AI
    game.play()
    ag.close()

``attach`` 只包一层 ``game.on_action``，不碰 engine.py；平台（webapp）只需要在
``Room._on_action`` 里把事件转给外部队列，见 docs/接入外部AI.md。

## 协议细节（和 `aimodel/mjai.py` 的不同）

1. **赤 5 的写法**：``mjai.py`` 用 ``0m/0p/0s``；Mortal / mjai-manue-go 用
   ``5mr/5pr/5sr``。本模块内部统一发后者，读回来两种都认。
2. **明杠**：本项目 ``call=="kan"``，Mortal 的事件名是 ``daiminkan``
   （``kan`` 在原版 mjai 里也常见）；这里发 ``daiminkan``，读的时候
   ``kan``/``daiminkan`` 都认。
3. **荣和**：``hora`` 的 ``target`` 对自摸也必须有；自摸时 ``target=actor``。
4. **可达信息**：只把手牌/自摸摸牌发给「自己」；别家摸牌发 ``"?"``，
   ``start_kyoku`` 的别家 ``tehais`` 也全是 ``"?"``。**不能把 View 里的
   别家手牌泄漏给外部 AI** —— 那会把它变成作弊 AI，评测数字就废了。

## 超时 / 崩溃 / 确定性

* 起进程后有一个读 stdout 的线程，主线程用 ``queue.Queue.get(timeout=...)``
  等动作；进程卡死会抛 :class:`ExternalAITimeout`，**不会**拖死整局。
* 崩溃/超时后会尝试**重启并重放本局已发生的事件**（最多 2 次），
  因为外部 AI 是确定性的，重放同一串事件应该得到同一个动作。
* 仍然失败时：构造函数给了 ``fallback`` 就降级到教师并记 ``degraded``；
  没给就抛异常（评测默认走这条，宁可中止也不要污染数据）。
* 动作映射找不到对应选项时抛 :class:`ExternalAIProtocolError`，
  **不会** ``return 0`` —— 静默 0 在麻将里等于「随便打一张」，会污染
  整个天梯的强弱结论。
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import mjai, sanma as S
from .agents import Agent, HeuristicAgent
from .engine import Decision, View

log = logging.getLogger("potatomaj.external")

#: 还活着的 agent —— 进程退出 / Ctrl-C 时由 atexit 统一 kill
_LIVE: "set[ExternalMJaiAgent]" = set()

ROOT = Path(__file__).resolve().parents[1]
#: 外部 AI 默认放这里（`.gitignore` 里的 `_data/`，不会进 git）
EXTERNAL_DIR = ROOT / "_data" / "external"
DOC_PATH = ROOT / "docs" / "接入外部AI.md"
DOC_HINT = "docs/接入外部AI.md"

#: 赤 5 的牌型下标 -> mjai 外部 AI 的写法（不是 0m！）
_AKA_NAME = {4: "5mr", 13: "5pr", 22: "5sr"}
_NAME_AKA = {v: k for k, v in _AKA_NAME.items()}

#: 这些动作表示「和了」
_HORA = {"hora", "tsumo", "ron"}


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class ExternalAIError(RuntimeError):
    """外部 AI 相关的所有错误的基类。"""


class ExternalAINotInstalled(ExternalAIError):
    """外部 AI 没装 / 找不到可执行文件 / 缺模型。"""


class ExternalAIUnsupported(ExternalAIError):
    """这个外部 AI 的协议不是「stdio 一行一个 JSON」（比如只支持 TCP）。"""


class ExternalAIProtocolError(ExternalAIError):
    """外部 AI 返回了当前决策里不存在的动作（不是超时、不是崩溃）。"""


class ExternalAITimeout(ExternalAIError):
    """等外部 AI 的动作超时（进程卡死）。"""


class ExternalAICrashed(ExternalAIError):
    """外部进程退出 / 管道断裂。"""


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExternalSpec:
    """一个外部 AI 的「怎么起它」以及「它是什么」。"""

    key: str
    title: str
    repo: str
    license: str
    language: str
    #: 只支持几家：mjai 的 start_kyoku 是 4 人格式
    players: int = 4
    #: ``mjai-stdio`` = 一行一个 mjai JSON；``mjai-tcp`` = 只能连 mjai server
    protocol: str = "mjai-stdio"
    #: 安装提示（docs 里也有）
    install: str = ""
    #: 默认命令（``{player_id}`` / ``{seed}`` / ``{manue_bin}`` 会被替换）
    command: tuple[str, ...] = ()
    #: 默认工作目录
    cwd: Path | None = None
    #: 需要检查存在性的文件（相对 cwd）
    needs_files: tuple[str, ...] = ()
    #: 是否按「一批事件一个 JSON 数组」喂（Akagi 的 Mortal bot 是这种变体）
    batch_events: bool = False
    #: 每一局开始都重启进程。用于不认 ``can_act`` 的 bot（manue）：上一局
    #: 迟到的动作会污染下一局的第一手（实测：它想打上一局摸到的 8s）。
    restart_each_hand: bool = False
    #: 给前端/文档的一句话
    note: str = ""


SPECS: dict[str, ExternalSpec] = {
    "mortal": ExternalSpec(
        key="mortal",
        title="Mortal",
        repo="Equim-chan/Mortal",
        license="AGPL-3.0",
        language="Rust + Python",
        protocol="mjai-stdio",
        command=("python3", "mortal.py", "{player_id}"),
        cwd=EXTERNAL_DIR / "Mortal" / "mortal",
        needs_files=("mortal.py",),
        install=(
            "git clone https://github.com/Equim-chan/Mortal && "
            "cargo build -p libriichi --lib --release && "
            "cp target/release/libriichi.so mortal/ && 准备 mortal.pth"
        ),
        note="开源最强立直麻将 AI 之一；mjai stdio，需要 Rust + torch + 权重",
    ),
    "manue": ExternalSpec(
        key="manue",
        title="mjai-manue-go",
        repo="Apricot-S/mjai-manue-go",
        license="BSD-3-Clause",
        language="Go",
        protocol="mjai-stdio",
        command=("{manue_bin}", "--name", "PotatoManue", "--id", "{player_id}",
                 "--seed", "{seed}"),
        needs_files=(),
        restart_each_hand=True,
        install=(
            "go install github.com/Apricot-S/mjai-manue-go/cmd/mjai-manue@latest "
            "（或 go build -o _data/external/bin/mjai-manue ./cmd/mjai-manue）"
        ),
        note="Manue 的 Go 移植；纯 stdio、无需权重，四麻",
    ),
    "kanachan": ExternalSpec(
        key="kanachan",
        title="Kanachan",
        repo="Cryolite/kanachan",
        license="未声明（仓库无 LICENSE 文件）",
        language="C++ + Python",
        protocol="mjai-stdio",
        command=("python3", "bot.py"),
        cwd=EXTERNAL_DIR / "kanachan" / "mjai.app",
        needs_files=("bot.py", "model.kanachan"),
        install=(
            "按 kanachan/mjai.app/build.sh 打 zip；需要自己训练的 "
            "model.kanachan（仓库不提供权重）+ C++ 依赖"
        ),
        note="mjai.app 的 python 包装；仓库不提供训练好的模型",
    ),
    "akochan": ExternalSpec(
        key="akochan",
        title="akochan",
        repo="critter-mj/akochan",
        license="自定义（非 OSI；禁止用 AI 权重发布衍生作品）",
        language="C++",
        protocol="mjai-tcp",
        install="make -f Makefile_MacOS；用 mjai server 连接 system.exe mjai_client",
        note="只支持 TCP mjai 客户端，不能直接接 stdio 桥",
    ),
    "manue-original": ExternalSpec(
        key="manue-original",
        title="mjai-manue（原版）",
        repo="gimite/mjai-manue",
        license="New BSD",
        language="CoffeeScript + Ruby",
        protocol="mjai-tcp",
        install="bundle install；bin/mjai-manue <mjai-server-url>",
        note="只支持 TCP mjai 客户端；stdio 请用 manue（Go 移植）",
    ),
    # Akagi 的 Mortal bot：stdin 一行是一个 JSON **数组**，不是单个事件。
    # 它自带 libriichi + mortal.pth，适合在 macOS 上快速验证 Mortal。
    "mortal-akagi": ExternalSpec(
        key="mortal-akagi",
        title="Mortal（Akagi 打包版）",
        repo="shinkuan/Akagi-MjaiBot-Mortal",
        license="AGPL-3.0",
        language="Python + 预编译 libriichi",
        protocol="mjai-stdio",
        command=("python3", "bot.py"),
        cwd=EXTERNAL_DIR / "Akagi-MjaiBot-Mortal",
        needs_files=("bot.py", "mortal.pth"),
        batch_events=True,
        install=(
            "下载 Akagi-MjaiBot-Mortal 的 release zip，解压到 "
            "_data/external/Akagi-MjaiBot-Mortal；需要 Python 3.12 + torch"
        ),
        note="自带权重和 macOS/arm64 libriichi；按批接收事件",
    ),
}

#: 别名（--third 里可以用）
ALIASES = {
    "mjai-manue": "manue",
    "manue-go": "manue",
    "akagi-mortal": "mortal-akagi",
    "mortal-bot": "mortal-akagi",
}


def spec_names() -> list[str]:
    """所有可用的 ``--third`` 名字（含别名）。"""
    return sorted(set(SPECS) | set(ALIASES))


def get_spec(spec: str) -> ExternalSpec:
    """把 ``mortal`` / ``external:mortal`` / 别名解析成 :class:`ExternalSpec`。"""
    key = normalize_key(spec)
    if key is None:
        raise ExternalAIError(
            f"不认识的外部 AI：{spec!r}；可用：{', '.join(spec_names())}"
        )
    return SPECS[key]


def normalize_key(spec: str) -> str | None:
    """``external:mortal`` / ``ext:mortal`` / ``mortal`` -> ``mortal``。

    不是外部 AI 的名字（比如 ``teacher`` / ``style:speed`` / ``*.npz``）返回
    ``None``，调用方可以继续走原来的分支。
    """
    if not spec:
        return None
    s = spec.strip()
    for prefix in ("external:", "ext:"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    else:
        # 没有前缀时，只有**完全等于**注册名/别名才算外部 AI。
        # 这样 `policy.npz`、`style:speed` 不会被误判。
        if s not in SPECS and s not in ALIASES:
            return None
    s = ALIASES.get(s, s)
    return s if s in SPECS else None


def is_external_spec(spec: str) -> bool:
    return normalize_key(spec) is not None


def spec_help() -> str:
    """给 ``--help`` 用的一行摘要。"""
    return " / ".join(f"{k}（{SPECS[k].title}）" for k in sorted(SPECS))


# ---------------------------------------------------------------------------
# 配置 / 安装检查
# ---------------------------------------------------------------------------
def _env_key(key: str) -> str:
    return "POTATOMAJ_EXTERNAL_" + "".join(
        c if c.isalnum() else "_" for c in key
    ).upper()


def _load_config_file() -> dict:
    """读 ``$POTATOMAJ_EXTERNAL_CONFIG`` 指向的 JSON（可选）。

    格式::

        {"mortal": {"cmd": ["/path/python", "mortal.py", "{player_id}"],
                    "cwd": "/path/Mortal/mortal",
                    "env": {"MORTAL_REVIEW_MODE": "0"}}}
    """
    path = os.environ.get("POTATOMAJ_EXTERNAL_CONFIG", "").strip()
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        log.warning("POTATOMAJ_EXTERNAL_CONFIG 指向的文件不存在：%s", p)
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("外部 AI 配置文件读不了（%s）：%s", p, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _find_manue_binary() -> str | None:
    """找 manue-go 可执行文件：env -> PATH -> _data/external。"""
    for name in ("POTATOMAJ_EXTERNAL_MANUE_BIN", "POTATOMAJ_MANUE_BIN"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    found = shutil.which("mjai-manue")
    if found:
        return found
    for p in (
        EXTERNAL_DIR / "bin" / "mjai-manue",
        EXTERNAL_DIR / "mjai-manue-go" / "mjai-manue",
        EXTERNAL_DIR / "mjai-manue-go" / "mjai-manue.exe",
        ROOT / "tools" / "external" / "bin" / "mjai-manue",
    ):
        if p.exists():
            return str(p)
    return None


def _search_manue() -> str | None:
    return _find_manue_binary()


def resolve_settings(
    spec: str | ExternalSpec,
    *,
    player_id: int = 0,
    seed: int = 0,
    command: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict:
    """算出这个外部 AI 的 ``(cmd, cwd, env)``；不启动进程。

    优先级（后者覆盖前者）：

    1. :data:`SPECS` 里的默认值
    2. ``$POTATOMAJ_EXTERNAL_CONFIG`` JSON
    3. ``$POTATOMAJ_EXTERNAL_<KEY>_CMD`` / ``_DIR`` / ``_BIN``
    4. 函数参数（代码里显式指定）
    """
    sp = spec if isinstance(spec, ExternalSpec) else get_spec(spec)
    cfg = _load_config_file().get(sp.key, {}) or {}

    cmd: list[str] = list(sp.command)
    workdir: Path | None = sp.cwd
    extra_env: dict[str, str] = {}
    batch = sp.batch_events
    restart_each_hand = sp.restart_each_hand

    if isinstance(cfg, dict):
        if isinstance(cfg.get("cmd"), list):
            cmd = [str(x) for x in cfg["cmd"]]
        elif isinstance(cfg.get("command"), list):
            cmd = [str(x) for x in cfg["command"]]
        if cfg.get("cwd"):
            workdir = Path(str(cfg["cwd"])).expanduser()
        if isinstance(cfg.get("env"), dict):
            extra_env.update({str(k): str(v) for k, v in cfg["env"].items()})
        if cfg.get("batch_events") is not None:
            batch = bool(cfg["batch_events"])
        if cfg.get("restart_each_hand") is not None:
            restart_each_hand = bool(cfg["restart_each_hand"])

    ek = _env_key(sp.key)
    env_cmd = os.environ.get(ek + "_CMD", "").strip()
    if env_cmd:
        # 路径里有空格时 shlex 会切错；允许直接写 JSON 数组：
        #   POTATOMAJ_EXTERNAL_MANUE_CMD='["/a b/python", "/a b/bot.py"]'
        if env_cmd.startswith("["):
            try:
                parsed = json.loads(env_cmd)
                cmd = [str(x) for x in parsed]
            except Exception as exc:  # noqa: BLE001
                raise ExternalAINotInstalled(
                    f"{ek}_CMD 不是合法的 JSON 数组：{exc}"
                ) from exc
        else:
            cmd = shlex.split(env_cmd)
    env_dir = os.environ.get(ek + "_DIR", "").strip()
    if env_dir:
        workdir = Path(env_dir).expanduser()
    env_bin = os.environ.get(ek + "_BIN", "").strip()
    if env_bin and cmd:
        # 默认命令第一段是程序名（或 {manue_bin} 占位）时，用 _BIN 覆盖它
        if not os.path.isabs(cmd[0]) or cmd[0] == "{manue_bin}":
            cmd[0] = env_bin
    env_batch = os.environ.get(ek + "_BATCH", "").strip()
    if env_batch:
        batch = env_batch not in ("0", "false", "False", "no")
    env_restart = os.environ.get(ek + "_RESTART_EACH_HAND", "").strip()
    if env_restart:
        restart_each_hand = env_restart not in ("0", "false", "False", "no")

    if command is not None:
        cmd = [str(x) for x in command]
    if cwd is not None:
        workdir = Path(cwd).expanduser()

    values = {
        "player_id": str(player_id),
        "seed": str(seed),
        "key": sp.key,
        "root": str(ROOT),
        "external_dir": str(EXTERNAL_DIR),
        "python": sys.executable or "python3",
        "manue_bin": _search_manue() or "{manue_bin}",
    }
    expanded: list[str] = []
    for token in cmd:
        try:
            expanded.append(token.format_map(values))
        except (KeyError, ValueError) as exc:
            raise ExternalAINotInstalled(
                f"{sp.key} 的命令模板 {token!r} 展开失败（{exc}）；"
                f"请设置 {ek}_CMD"
            ) from exc
    if workdir is not None and not workdir.is_absolute():
        workdir = ROOT / workdir
    proc_env = dict(os.environ)
    proc_env.update(extra_env)
    if env:
        proc_env.update({str(k): str(v) for k, v in env.items()})
    # Python 外部 AI 不加这两条会在 pipe 里缓冲到退出才吐动作
    if expanded and any(Path(x).name.startswith("python") for x in expanded[:2]):
        proc_env.setdefault("PYTHONUNBUFFERED", "1")
        proc_env.setdefault("PYTHONIOENCODING", "utf-8")
    return {"cmd": expanded, "cwd": workdir, "env": proc_env,
            "batch_events": batch, "restart_each_hand": restart_each_hand}


def _command_exists(cmd: Sequence[str], cwd: Path | None) -> tuple[bool, str]:
    if not cmd:
        return False, "没有配置启动命令"
    first = cmd[0]
    if "{manue_bin}" in first:
        return False, "找不到 mjai-manue 可执行文件"
    if os.path.sep in first or first.startswith("."):
        p = Path(first)
        if not p.is_absolute() and cwd is not None:
            p = cwd / p
        if not p.exists():
            return False, f"可执行文件不存在：{p}"
        if not os.access(p, os.X_OK):
            return False, f"文件不可执行：{p}"
        return True, ""
    if shutil.which(first) is None:
        return False, f"PATH 里找不到命令：{first}"
    return True, ""


def check_spec(
    spec: str | ExternalSpec,
    *,
    player_id: int = 0,
    seed: int = 0,
    command: Sequence[str] | None = None,
    cwd: str | Path | None = None,
) -> tuple[bool, str]:
    """只检查「装没装」，**不启动进程**。返回 ``(能不能用, 原因)``。"""
    try:
        sp = spec if isinstance(spec, ExternalSpec) else get_spec(spec)
    except ExternalAIError as exc:
        return False, str(exc)
    if sp.protocol != "mjai-stdio":
        return False, (
            f"{sp.title} 只支持 TCP mjai 客户端，本适配层只接 stdio；"
            f"请用 mjai server 转接，见 {DOC_HINT}"
        )
    try:
        settings = resolve_settings(sp, player_id=player_id, seed=seed,
                                     command=command, cwd=cwd)
    except ExternalAINotInstalled as exc:
        return False, str(exc)
    workdir: Path | None = settings["cwd"]
    ok, why = _command_exists(settings["cmd"], workdir)
    if not ok:
        return False, why
    if workdir is not None and not workdir.exists():
        return False, f"工作目录不存在：{workdir}"
    for name in sp.needs_files:
        p = (workdir or Path.cwd()) / name
        if not p.exists():
            return False, f"缺文件：{p}"
    if sp.key == "mortal" and workdir is not None:
        # Mortal 的权重不在仓库里；给了环境变量就不查
        model = os.environ.get("POTATOMAJ_EXTERNAL_MORTAL_MODEL", "").strip()
        if not model and not any(workdir.glob("*.pth")) and not any(
            workdir.glob("*.pt")
        ):
            return False, (
                f"缺 Mortal 权重（*.pth / *.pt）；把 mortal.pth 放到 {workdir}，"
                f"或设置 POTATOMAJ_EXTERNAL_MORTAL_MODEL"
            )
    return True, ""


def require(spec: str | ExternalSpec, **kwargs: Any) -> None:
    """没装就抛 ``SystemExit``，给命令行一个清晰错误（不是 traceback）。"""
    ok, why = check_spec(spec, **kwargs)
    if not ok:
        key = spec if isinstance(spec, str) else spec.key
        raise SystemExit(
            f"外部 AI {key!r} 没装好：{why}\n"
            f"  → 安装步骤见 {DOC_HINT}\n"
            f"  → 也可以设置 POTATOMAJ_EXTERNAL_{key.upper().replace('-', '_')}_CMD "
            f"指向一个 stdio mjai bot"
        )


# ---------------------------------------------------------------------------
# 牌名转换
# ---------------------------------------------------------------------------
def _tile_name(tile: int, red: bool = False) -> str:
    if red and tile in _AKA_NAME:
        return _AKA_NAME[tile]
    return mjai.tile_to_mjai(tile)


def _parse_tile(name: Any) -> tuple[int, bool]:
    """外部 AI 的牌名 -> ``(牌型下标, 是否赤)``；``?`` 会抛协议错误。"""
    if not isinstance(name, str) or not name:
        raise ExternalAIProtocolError(f"缺少牌名：{name!r}")
    s = name.strip()
    if s in _NAME_AKA:
        return _NAME_AKA[s], True
    if s == "0m" or s == "0p" or s == "0s":
        return mjai.tile_from_mjai(s)
    try:
        return mjai.tile_from_mjai(s)
    except ValueError as exc:
        raise ExternalAIProtocolError(f"认不出的牌名：{name!r}") from exc


def _wind_name(wind: Any) -> str:
    return mjai.WIND_MJAI.get(wind, "E")


def _masked_hand(counts: Iterable[int], reds: Iterable[int] | None = None) -> list[str]:
    out: list[str] = []
    counts = list(counts)
    reds = list(reds) if reds is not None else [0] * 34
    for t, n in enumerate(counts):
        for i in range(n):
            out.append(_tile_name(t, red=i < reds[t]))
    return out


# ---------------------------------------------------------------------------
# 外部 Agent
# ---------------------------------------------------------------------------
class ExternalMJaiAgent(Agent):
    """把一个支持 **mjai 协议**的外部 AI（子进程）包装成本项目的 Agent。

    参数
    ----
    spec : str
        :data:`SPECS` 里的名字（``mortal`` / ``manue`` / …），也可以带
        ``external:`` 前缀。别名见 :data:`ALIASES`。
    seed : int
        传给 ``{seed}`` 占位符。**同一个输入必须同一输出**，所以不要在这里
        用随机数；需要随机性的外部 AI 由它自己的 ``--seed`` 决定。
    player_id : int
        这个 agent 坐在哪一席（0..players-1）。mjai 的 ``start_kyoku`` /
        ``hora`` 等事件都按绝对座位编号。
    players : int
        4（mjai 标准）。外部 AI 目前都只支持四麻；传 3 会抛
        :class:`ExternalAIUnsupported`。
    timeout : float
        等一个动作的秒数（默认 20）。超时会先重启重放一次，再失败。
    fallback : Agent | None
        失败时降级用的 agent。默认 ``None`` = 直接抛异常（评测用）。
        平台/娱乐场景可以传 ``HeuristicAgent``，这样桌子不会崩。
    strict : bool
        保留参数：``fallback`` 给了就降级，没给就抛；不再用 strict 区分。
    batch_events : bool | None
        ``None`` = 用 spec 的默认值。Akagi 的 Mortal bot 要 ``True``。
    """

    name = "外部AI"

    def __init__(
        self,
        spec: str,
        *,
        seed: int = 0,
        player_id: int = 0,
        players: int = 4,
        timeout: float = 2.0,
        start_timeout: float = 60.0,
        fallback: Agent | None = None,
        strict: bool = True,
        command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        log_path: str | Path | None = None,
        on_error: Any = None,
        allow_missing: bool = False,
        batch_events: bool | None = None,
        max_restarts: int = 2,
    ) -> None:
        self.spec = get_spec(spec)
        if players != 4 or self.spec.players != 4:
            raise ExternalAIUnsupported(
                f"{self.spec.title} 只支持四麻（mjai 的 start_kyoku 是 4 人格式）；"
                f"本适配层没有三麻的外部 AI"
            )
        if not (0 <= int(player_id) < int(players)):
            raise ValueError(f"player_id={player_id} 不在 0..{players - 1}")
        self.player_id = int(player_id)
        self.players = int(players)
        self.seed = int(seed)
        self.timeout = float(timeout)
        self.start_timeout = float(start_timeout)
        self.fallback = fallback
        self.strict = bool(strict)
        self.on_error = on_error
        self.log_path = Path(log_path) if log_path else None
        self._settings = resolve_settings(
            self.spec, player_id=self.player_id, seed=self.seed,
            command=command, cwd=cwd, env=env,
        )
        if batch_events is not None:
            self._settings["batch_events"] = bool(batch_events)
        self.batch_events = bool(self._settings["batch_events"])
        self.restart_each_hand = bool(self._settings.get("restart_each_hand"))
        self.max_restarts = max(0, int(max_restarts))
        self.name = f"外部AI:{self.spec.title}"

        #: 外部 AI 用的文件名 / 目录，给报告和平台展示
        self.available, self.available_reason = check_spec(
            self.spec, player_id=self.player_id, seed=self.seed,
            command=self._settings["cmd"], cwd=self._settings["cwd"],
        )
        if not self.available and not allow_missing:
            raise ExternalAINotInstalled(
                f"外部 AI {self.spec.key!r} 没装好：{self.available_reason}\n"
                f"  → 安装步骤见 {DOC_HINT}\n"
                f"  → 或设置 {_env_key(self.spec.key)}_CMD 指向一个 stdio mjai bot"
            )

        # 进程 / 读线程
        self._proc: subprocess.Popen | None = None
        self._gen = 0
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._stderr: deque[str] = deque(maxlen=80)
        self._log_fh = None
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_fh = self.log_path.open("a", encoding="utf-8")
            except OSError:
                self._log_fh = None

        # 协议状态
        self._started = False
        self._closed = False
        self._dead = False
        self._restarts = 0
        self._hand_open = False
        self._hand_events: list[dict] = []
        self._outbox: list[dict] = []
        self._hand_no = 0
        self._kyoku = 1
        self._honba = 0
        self._prev_dealer: int | None = None
        self._prev_wind: Any = None
        self._dora_sent = 1
        self._riichi_pending = [False] * 4
        self._pending_action: dict | None = None
        self._discard_key: tuple | None = None
        self._discard_seq = 0
        #: 每个座位最近摸到的牌（只有自己知道真牌；别家记 "?"）
        self._last_draw: list[str | None] = [None] * 4
        self._discard_response: dict | None = None
        self._discard_response_key: tuple | None = None
        self._last_raw_action: dict | None = None

        # 统计
        self.decisions = 0
        self.errors = 0
        self.fallback_steps = 0
        self._decide_total = 0.0
        self._decide_max = 0.0
        self.degraded = False
        self.degrade_reason = ""
        self.crash_count = 0
        self.timeout_count = 0
        self._bad_action_count = 0

    # -- 生命周期 -----------------------------------------------------------
    def __enter__(self) -> "ExternalMJaiAgent":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - 解释器退出时兜底
        try:
            self.close()
        except Exception:
            pass

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._started and self._proc is not None:
            self._started = False
        self._start()

    def _start(self) -> None:
        """起进程 + 发 ``start_game``。"""
        if self._closed:
            raise ExternalAIError(f"{self.name} 已经 close()")
        if not self.available and not self._settings["cmd"]:
            raise ExternalAINotInstalled(
                f"外部 AI {self.spec.key!r} 没装好：{self.available_reason}"
            )
        cmd = list(self._settings["cmd"])
        cwd = self._settings["cwd"]
        env = dict(self._settings["env"])
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ExternalAINotInstalled(
                f"起不了 {self.spec.key!r}：{exc}；见 {DOC_HINT}"
            ) from exc
        except OSError as exc:
            raise ExternalAICrashed(
                f"起不了 {self.spec.key!r}：{exc}；见 {DOC_HINT}"
            ) from exc
        self._gen += 1
        gen = self._gen
        threading.Thread(target=self._read_stdout, args=(gen,),
                         name=f"ext-{self.spec.key}-out", daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(gen,),
                         name=f"ext-{self.spec.key}-err", daemon=True).start()
        self._started = True
        _LIVE.add(self)
        start = {"type": "start_game", "id": self.player_id,
                 "names": [f"P{i}" for i in range(self.players)]}
        if self.batch_events:
            self._write_batch([start])
            # Akagi 的 bot 对 start_game 会回一行 none；等它，保证后面的读一 一对应
            self._read_action(self.start_timeout, allow_none=True)
        else:
            self._write_one(start)

    def _read_stdout(self, gen: int) -> None:
        proc = self._proc
        try:
            if proc is None or proc.stdout is None:
                return
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    action = json.loads(line)
                except json.JSONDecodeError:
                    if self._log_fh:
                        self._log_fh.write(f"[bad-json] {line[:500]}\n")
                        self._log_fh.flush()
                    self._q.put((gen, "bad", line[:500]))
                    continue
                if not isinstance(action, dict):
                    self._q.put((gen, "bad", str(action)[:500]))
                    continue
                self._q.put((gen, "action", action))
        except Exception as exc:  # noqa: BLE001
            self._q.put((gen, "error", exc))
        finally:
            self._q.put((gen, "eof", None))

    def _read_stderr(self, gen: int) -> None:
        proc = self._proc
        try:
            if proc is None or proc.stderr is None:
                return
            for line in proc.stderr:
                line = line.rstrip("\n")
                if not line:
                    continue
                if gen == self._gen:
                    self._stderr.append(line)
                if self._log_fh:
                    self._log_fh.write(f"[stderr] {line}\n")
                    self._log_fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        """收工：发 ``end_game``、关 stdin、必要时 kill。幂等。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            _LIVE.discard(self)
            proc = self._proc
            if proc is not None and proc.poll() is None:
                try:
                    end = {"type": "end_game"}
                    line = json.dumps(
                        [end] if self.batch_events else end,
                        ensure_ascii=False, separators=(",", ":"),
                    )
                    if proc.stdin is not None:
                        proc.stdin.write(line + "\n")
                        proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        proc.terminate()
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        try:
                            proc.kill()
                        except OSError:
                            pass
            if self._log_fh is not None:
                try:
                    self._log_fh.close()
                except OSError:
                    pass
                self._log_fh = None

    # -- 管道 ---------------------------------------------------------------
    def _write_one(self, event: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise ExternalAICrashed(f"{self.name} 的 stdin 不可用")
        event = {k: v for k, v in event.items() if not k.startswith("_")}
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise ExternalAICrashed(
                f"{self.name} 的 stdin 断了：{exc}；stderr 末尾：{self._stderr_tail()}"
            ) from exc

    def _write_batch(self, events: list[dict]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise ExternalAICrashed(f"{self.name} 的 stdin 不可用")
        clean = [{k: v for k, v in ev.items() if not k.startswith("_")}
                 for ev in events]
        line = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise ExternalAICrashed(
                f"{self.name} 的 stdin 断了：{exc}；stderr 末尾：{self._stderr_tail()}"
            ) from exc

    def _stderr_tail(self, n: int = 6) -> str:
        if not self._stderr:
            return "（空）"
        return " ⏎ ".join(list(self._stderr)[-n:])[-800:]

    def _read_action(self, timeout: float, *, allow_none: bool = False,
                     count_timeout: bool = True) -> dict | None:
        """等一行动作。超时/崩溃抛异常。

        ``count_timeout=False`` 用于「读干净」时的小等待：那是正常的
        「没有更多反应」，不该记成外部 AI 卡死。
        """
        deadline = time.monotonic() + float(timeout)
        while True:
            remain = deadline - time.monotonic()
            if remain <= 0:
                if count_timeout:
                    self.timeout_count += 1
                raise ExternalAITimeout(
                    f"{self.name} 在 {timeout:.1f}s 内没有给动作；"
                    f"stderr 末尾：{self._stderr_tail()}"
                )
            try:
                gen, kind, payload = self._q.get(timeout=remain)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            if kind == "eof":
                raise ExternalAICrashed(
                    f"{self.name} 进程退出了（returncode="
                    f"{self._proc.poll() if self._proc else '?'}）；"
                    f"stderr 末尾：{self._stderr_tail()}"
                )
            if kind == "error":
                raise ExternalAICrashed(
                    f"{self.name} 读 stdout 失败：{payload!r}；"
                    f"stderr 末尾：{self._stderr_tail()}"
                )
            if kind == "bad":
                raise ExternalAIProtocolError(
                    f"{self.name} 输出的不是 JSON 对象：{payload!r}"
                )
            return payload

    # -- 事件翻译 -----------------------------------------------------------
    def feed(self, action: Mapping[str, Any]) -> None:
        """引擎 ``Game.on_action`` 的结构化事件 -> mjai -> 外部进程。"""
        if self._dead or self._closed:
            return
        try:
            events = self._translate(action)
            if not events:
                return
            self._ensure_started()
            for ev in events:
                if ev.get("type") == "start_kyoku":
                    # 新一局：清掉上一局的日志，只保留这一局（恢复时重放用）
                    self._hand_events = [ev]
                elif ev.get("type") != "start_game":
                    self._hand_events.append(ev)
            # 所有事件先缓冲，等真正要决策时再按顺序发送。
            # 否则外部 AI 会对「引擎根本不会问我们」的事件提前吐动作，那些
            # 动作会变成下一条决策的脏数据（实测：manue 的 none 被当成打牌）。
            self._outbox.extend(events)
            self._after_feed(events)
        except ExternalAIError as exc:
            self._handle_fail(exc)

    def _after_feed(self, events: list[dict]) -> None:
        for ev in events:
            t = ev.get("type")
            if t == "tsumo":
                actor = ev.get("actor")
                if isinstance(actor, int) and 0 <= actor < 4:
                    self._last_draw[actor] = ev.get("pai")
            elif t == "dahai":
                actor = ev.get("actor")
                if isinstance(actor, int) and 0 <= actor < 4:
                    self._last_draw[actor] = None
            if t == "start_kyoku":
                self._pending_action = None
                self._discard_key = None
                self._discard_response = None
                self._discard_response_key = None
                self._riichi_pending = [False] * 4
                self._last_draw = [None] * 4
            elif t == "tsumo" and ev.get("actor") == self.player_id:
                # 自己摸牌 = 新的一手；之前的 pending 不该留
                self._pending_action = None
                self._discard_key = None
                self._discard_response = None
                self._discard_response_key = None
            elif t in ("chi", "pon", "daiminkan", "ankan", "kakan") \
                    and ev.get("actor") == self.player_id:
                self._pending_action = None
                self._discard_key = None
                self._discard_response = None
                self._discard_response_key = None
            elif t == "dahai" and ev.get("actor") != self.player_id:
                self._discard_seq += 1
                self._discard_key = (
                    ev.get("actor"), ev.get("pai"), self._discard_seq,
                )
                self._discard_response = None
                self._discard_response_key = None

    def _translate(self, a: Mapping[str, Any]) -> list[dict]:
        k = a.get("kind")
        n = self.players
        if k == "start_hand":
            if self.restart_each_hand and self._proc is not None:
                self._restart_process()
                self._hand_open = False
            out: list[dict] = []
            if self._hand_open:
                out.append({"type": "end_kyoku"})
            self._hand_open = True
            self._hand_no = int(a.get("hand_no") or self._hand_no + 1)
            dealer = int(a.get("dealer", 0))
            wind = a.get("round_wind")
            if self._prev_wind is not None and wind != self._prev_wind:
                self._kyoku = 1
                self._honba = 0
            elif self._prev_dealer is not None and dealer == self._prev_dealer:
                self._honba += 1
            elif self._prev_dealer is not None:
                self._kyoku += 1
                self._honba = 0
            else:
                self._kyoku = 1
                self._honba = 0
            self._prev_dealer = dealer
            self._prev_wind = wind
            hands = a.get("hands") or [[0] * 34 for _ in range(n)]
            reds = a.get("reds") or [[0] * 34 for _ in range(n)]
            tehais: list[list[str]] = []
            for p in range(n):
                if p == self.player_id:
                    # MVP：赤 5 一律按普通 5 发（详见模块文档「赤 5」一节）。
                    # 引擎的 chi 不维护 red 计数，保留赤反而会让���部 AI 的
                    # 手牌模型和引擎对不上；先保证 legality，再谈赤 dora。
                    tehais.append(_masked_hand(hands[p], None))
                else:
                    tehais.append(["?"] * int(S.rule_for(n).hand_size))
            dora = list(a.get("dora") or [])
            if not dora:
                dora = [0]
            out.append({
                "type": "start_kyoku",
                "bakaze": _wind_name(wind),
                "kyoku": max(1, self._kyoku),
                "honba": max(0, self._honba),
                "kyotaku": int(a.get("sticks") or 0),
                "oya": dealer,
                "scores": list(a.get("scores") or [0] * n),
                "dora_marker": _tile_name(int(dora[0])),
                "tehais": tehais,
            })
            for d in dora[1:]:
                out.append({"type": "dora", "dora_marker": _tile_name(int(d))})
            self._dora_sent = max(1, len(dora))
            return out

        if k == "draw":
            p = int(a.get("player", 0))
            if p == self.player_id:
                pai = _tile_name(int(a.get("tile", 0)), False)
            else:
                pai = "?"
            ev = {"type": "tsumo", "actor": p, "pai": pai}
            if a.get("rinshan"):
                # 标记给 _sync_dora 用：新宝牌必须插在岭上摸牌**之前**，
                # 否则 manue 的「等 dora 再进入可打牌状态」会卡住。
                ev["_rinshan"] = True
            return [ev]

        if k == "discard":
            p = int(a.get("player", 0))
            tile = int(a.get("tile", 0))
            # ⚠️ engine 的 discard 事件里 tsumogiri 不可靠：_finish_discard 先把
            # pl.drawn 清成 None 再 _act，于是永远 True。自己摸的牌我们记着，
            # 用牌型比较；别家摸牌是 "?"，只能保守发 False（比谎报摸切好）。
            tsumogiri = False
            if p == self.player_id:
                if self._last_draw[p]:
                    try:
                        tsumogiri = _parse_tile(self._last_draw[p])[0] == tile
                    except ExternalAIProtocolError:
                        tsumogiri = False
            else:
                # 别家打牌：我们本来就不该知道他是摸切还是手切。发 True 有两个好处：
                # 1) 不泄漏信息（View 里也没有别家的摸牌）；
                # 2) manue-go 对「非摸切」会做喰い替え校验，而别家的手牌它只能靠
                #    事件重建，规则一有差异就把整局打崩（实测过）。
                tsumogiri = True
            ev = {
                "type": "dahai",
                "actor": p,
                "pai": _tile_name(tile),
                "tsumogiri": tsumogiri,
            }
            out = [ev]
            # 立直宣言后的第一张打牌 -> reach_accepted
            if bool(a.get("riichi")) and self._riichi_pending[p]:
                out.append({"type": "reach_accepted", "actor": p})
                self._riichi_pending[p] = False
            return out

        if k == "riichi":
            p = int(a.get("player", 0))
            self._riichi_pending[p] = True
            return [{"type": "reach", "actor": p}]

        if k == "call":
            p = int(a.get("player", 0))
            kind = a.get("call")
            tile = int(a.get("tile", 0))
            frm = int(a.get("from_player", p))
            called = int(a.get("called", tile))
            consumed = [_tile_name(int(t)) for t in (a.get("consumed") or [])]
            if kind == "chi":
                return [{
                    "type": "chi", "actor": p, "target": frm,
                    "pai": _tile_name(called), "consumed": consumed,
                }]
            if kind == "pon":
                return [{
                    "type": "pon", "actor": p, "target": frm,
                    "pai": _tile_name(tile), "consumed": consumed,
                }]
            if kind == "kan":
                return [{
                    "type": "daiminkan", "actor": p, "target": frm,
                    "pai": _tile_name(tile), "consumed": consumed,
                }]
            if kind == "ankan":
                return [{
                    "type": "ankan", "actor": p,
                    "consumed": consumed or [_tile_name(tile)] * 4,
                }]
            if kind == "kakan":
                return [{
                    "type": "kakan", "actor": p, "pai": _tile_name(tile),
                    "consumed": consumed or [_tile_name(tile)] * 3,
                }]
            raise ExternalAIProtocolError(f"未知的引擎鸣牌类型：{kind!r}")

        if k == "kita":
            raise ExternalAIUnsupported(
                "外部 AI 只支持四麻，没有拔北；三麻不能用 ExternalMJaiAgent"
            )

        if k == "win":
            winner = int(a.get("winner", 0))
            tile = int(a.get("tile", 0))
            tsumo = bool(a.get("tsumo"))
            loss = a.get("loser")
            ev = {
                "type": "hora",
                "actor": winner,
                "target": winner if tsumo or loss is None else int(loss),
            }
            if tile >= 0:
                ev["pai"] = _tile_name(tile)
            self._hand_open = False
            return [ev, {"type": "end_kyoku"}]

        if k == "ryukyoku":
            self._hand_open = False
            return [{
                "type": "ryukyoku",
                "deltas": list(a.get("deltas") or []),
            }, {"type": "end_kyoku"}]

        return []

    @staticmethod
    def _insert_before_rinshan(events: list[dict], dora: list[dict]) -> None:
        """把 dora 事件插到最后一个岭上 tsumo 之前；没有就追加。"""
        for i in range(len(events) - 1, -1, -1):
            if events[i].get("type") == "tsumo" and events[i].get("_rinshan"):
                events[i:i] = dora
                return
        events.extend(dora)

    def _sync_dora(self, view: View) -> None:
        """杠后宝牌指示牌会变多；引擎没有单独的 dora 事件，用 View 补。"""
        inds = list(getattr(view, "dora_indicators", []) or [])
        if len(inds) <= self._dora_sent:
            return
        new = inds[self._dora_sent:]
        self._dora_sent = len(inds)
        events = [{"type": "dora", "dora_marker": _tile_name(int(d))} for d in new]
        self._insert_before_rinshan(self._hand_events, events)
        self._insert_before_rinshan(self._outbox, events)

    # -- 决策 ---------------------------------------------------------------
    def decide(self, decision: Decision, view: View) -> int:
        t0 = time.perf_counter()
        try:
            return self._decide(decision, view)
        finally:
            dt = time.perf_counter() - t0
            self._decide_total += dt
            self._decide_max = max(self._decide_max, dt)

    def _decide(self, decision: Decision, view: View) -> int:
        self.decisions += 1
        if self.degraded:
            assert self.fallback is not None
            return int(self.fallback.decide(decision, view))
        try:
            players = int(getattr(getattr(view, "rule", None), "players", 4) or 4)
            if players != 4:
                raise ExternalAIUnsupported(
                    f"{self.name} 只支持四麻，当前是 {players} 人局"
                )
            self._sync_dora(view)
            self._prepare_decision(decision)
            return self._choose(decision, view)
        except (ExternalAITimeout, ExternalAICrashed) as exc:
            # 卡死/崩溃：重启并重放本局事件一次；还不行再失败
            if self._restarts < self.max_restarts:
                try:
                    self._recover()
                    self._prepare_decision(decision)
                    return self._choose(decision, view)
                except ExternalAIError as exc2:
                    return self._fail(exc2, decision, view)
            return self._fail(exc, decision, view)
        except ExternalAIError as exc:
            return self._fail(exc, decision, view)

    def _prepare_decision(self, decision: Decision) -> None:
        """把缓冲的事件发出去，并在需要时读回一个动作。

        - 自己的回合（tsumo/kyuushu/kan/riichi/discard）：读到的动作放进
          ``_pending_action``，同一回合后面的决策复用（不会重复读）。
        - 别家打牌后的 ron/call/chi：读到的动作放进 ``_discard_response``，
          同一张打牌的三个决策复用。
        """
        k = decision.kind
        own = k in ("tsumo", "kyuushu", "kan", "riichi", "discard")
        if own:
            if self._pending_action is not None:
                # 立直：tsumo 给的是 reach，后面 discard 还要再读宣言牌
                if not (k == "discard" and self._pending_action.get("type") == "reach"):
                    return
            want = bool(self._outbox) or self._pending_action is None
        else:
            if self._discard_response is not None:
                return
            want = bool(self._outbox)
        if not want:
            return
        action = self._flush_outbox(expect_action=True)
        if action is None:
            return
        if own:
            # 自己的回合不可能是 none；这是上一条事件迟到的脏数据。
            # 继续读几行，拿到真正的动作（最多等 3×0.5s）。
            for _ in range(3):
                if action.get("type") != "none":
                    break
                try:
                    nxt = self._read_action(0.5, allow_none=True,
                                            count_timeout=False)
                except ExternalAITimeout:
                    break
                if nxt is not None:
                    action = nxt
            self._pending_action = action
        else:
            self._discard_response = action
            self._discard_response_key = self._discard_key

    def _flush_outbox(self, *, expect_action: bool) -> dict | None:
        if not self._outbox:
            return None
        events = self._outbox
        self._outbox = []
        self._ensure_started()
        # 只有最后一个「能触发决策」的事件该让外部 AI 动手；它后面的
        # dora/end_kyoku 之类不能算触发（否则 can_act=false 会把真正的
        # tsumo 也压掉，外部 AI 一行都不吐 -> 超时）。
        trigger = _last_trigger_index(events)
        if self.batch_events:
            # Akagi 的 Mortal bot：一行一个 JSON 数组，整批只回一行。
            payload = []
            for i, ev in enumerate(events):
                e = dict(ev)
                if i != trigger:
                    e["can_act"] = False
                payload.append(e)
            self._write_batch(payload)
            if not expect_action:
                return None
            return self._read_action(self.timeout, allow_none=True)
        # 标准 stdio：一行一个事件。非最后事件加 can_act=false（Mortal 会遵守；
        # manue 忽略它，所以后面还要 drain 到安静、取最后一条）。
        for i, ev in enumerate(events):
            e = dict(ev)
            if i != trigger:
                e["can_act"] = False
            self._write_one(e)
        if not expect_action:
            return None
        first = self._read_action(self.timeout, allow_none=True)
        if first is None:
            return None
        last = first
        while True:
            try:
                nxt = self._read_action(0.05, allow_none=True,
                                        count_timeout=False)
            except ExternalAITimeout:
                break
            if nxt is not None:
                last = nxt
        return last

    def _restart_process(self) -> None:
        """重启进程（用于不认 can_act 的 bot，每局开头清掉上一局的迟到动作）。"""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        self._proc = None
        self._started = False
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._outbox = []
        self._hand_events = []
        self._pending_action = None
        self._discard_key = None
        self._discard_response = None
        self._discard_response_key = None
        self._last_draw = [None] * 4
        self._start()

    def _recover(self) -> None:
        """杀掉旧进程，起新进程，把本局已发生的事件原样重放。"""
        self._restarts += 1
        log.warning("%s 超时/崩溃，重启并重放本局（第 %d 次）",
                    self.name, self._restarts)
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        self._proc = None
        self._started = False
        # 清掉旧队列
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._outbox = []
        log.info("%s 重放 %d 个事件（首事件 %r）", self.name,
                 len(self._hand_events),
                 self._hand_events[0].get("type") if self._hand_events else None)
        self._start()
        if self._hand_events:
            if self.batch_events:
                self._write_batch(self._hand_events)
            else:
                for ev in self._hand_events:
                    self._write_one(ev)
        # ⚠️ 重放整局时外部 AI 会对**每一个**能动作的事件各吐一行；
        # 要一直读到安静下来，取最后一行（对应最后一个事件），不能只读一行。
        if self.batch_events:
            action = self._read_action(self.timeout, allow_none=True)
        else:
            action = None
            while True:
                try:
                    a = self._read_action(min(self.timeout, 2.0),
                                          allow_none=True, count_timeout=False)
                except ExternalAITimeout:
                    break
                if a is not None:
                    action = a
        self._pending_action = action

    def _handle_fail(self, exc: ExternalAIError) -> None:
        """记错误；有 fallback 就降级，没有就继续抛。"""
        self.errors += 1
        if isinstance(exc, ExternalAICrashed):
            self.crash_count += 1
        if isinstance(exc, ExternalAIProtocolError):
            self._bad_action_count += 1
        reason = f"{type(exc).__name__}: {exc}"
        self.degrade_reason = reason
        log.error("%s 失败：%s", self.name, reason)
        if self.on_error is not None:
            try:
                self.on_error(self, exc)
            except Exception:  # noqa: BLE001
                log.exception("on_error 回调自己炸了")
        if self.fallback is None:
            raise exc
        self.degraded = True
        self._dead = True
        log.warning("%s 降级为内置教师（原因：%s）", self.name, reason)

    def _fail(self, exc: ExternalAIError, decision: Decision, view: View) -> int:
        self._handle_fail(exc)
        assert self.fallback is not None
        return int(self.fallback.decide(decision, view))

    def _next_action(self, decision: Decision) -> dict:
        if self._pending_action is None:
            raise ExternalAIProtocolError(
                f"{self.name} 没有对 {decision.kind} 决策给出动作"
                f"（事件流可能没接上；确认 external.attach(game) 已调用）"
            )
        return self._pending_action

    def _choose(self, decision: Decision, view: View) -> int:
        k = decision.kind
        if k in ("discard", "riichi", "kan", "tsumo", "kyuushu"):
            return self._choose_own(decision, view)
        if k in ("ron", "call", "chi"):
            return self._choose_call(decision, view)
        if k == "kita":
            raise ExternalAIUnsupported("外部 AI 只支持四麻，没有拔北")
        raise ExternalAIProtocolError(f"不认识的决策类型：{k!r}")

    def _choose_own(self, decision: Decision, view: View) -> int:
        action = self._next_action(decision)
        typ = action.get("type")
        self._last_raw_action = action
        k = decision.kind

        if k == "tsumo":
            # 能自摸 -> yes；否则 no，动作留给后面的 discard/riichi/kan
            return _index_of(decision, "yes") if typ in _HORA else _index_of(decision, "no")

        if k == "kyuushu":
            return _index_of(decision, "yes") if typ == "ryukyoku" else _index_of(decision, "no")

        if k == "kan":
            if typ in ("ankan", "kakan", "daiminkan"):
                tile = _action_kan_tile(action)
                if decision.tile is not None and tile is not None and tile != int(decision.tile):
                    raise ExternalAIProtocolError(
                        f"{self.name} 想杠 {_tile_name(tile)}，但引擎这一手只提供 "
                        f"{_tile_name(int(decision.tile))}；拒绝静默改动作"
                    )
                return _index_of(decision, "kan")
            return _index_of(decision, "pass")

        if k == "riichi":
            return _index_of(decision, "yes") if typ == "reach" else _index_of(decision, "no")

        # discard
        if typ == "reach":
            # reach 事件在 feed 里缓冲，_prepare_decision 已经把它发出去并把
            # 新的 dahai 读进 _pending_action；走到这里说明没读到宣言牌。
            raise ExternalAIProtocolError(
                f"{self.name} 在立直后没有给出宣言牌（仍是 reach）"
            )
        if typ != "dahai":
            raise ExternalAIProtocolError(
                f"{self.name} 在打牌决策给了 {typ!r}（原始动作 {action}）；"
                f"决策种类 discard，选项 {decision.options}"
            )
        return self._map_dahai(action, decision)

    def _choose_call(self, decision: Decision, view: View) -> int:
        key = self._discard_key
        if key is None:
            # 没有收到「别家打牌」事件就要求叫牌，说明事件流没接上
            raise ExternalAIProtocolError(
                f"{self.name} 收到 {decision.kind} 决策，但事件流里没有对应的 dahai；"
                f"请确认用 external.attach(game) 把 Game.on_action 接上了"
            )
        if self._discard_response is None or self._discard_response_key != key:
            self._discard_response = self._read_action(self.timeout)
            self._discard_response_key = key
            self._last_raw_action = self._discard_response
        action = self._discard_response
        typ = action.get("type")
        k = decision.kind

        if k == "ron":
            if typ == "hora":
                return _index_of(decision, "yes")
            if typ in ("none", "pass"):
                return _index_of(decision, "no")
            if typ in ("pon", "chi", "daiminkan", "kan"):
                # 不荣和、改成鸣牌：这个反应留给后面的 call/chi 决策
                return _index_of(decision, "no")
            raise ExternalAIProtocolError(
                f"{self.name} 对打牌的反应 {typ!r} 不可能是荣和决策的答案；"
                f"原始动作 {action}"
            )

        if k == "call":
            opts = list(decision.options)
            if typ in ("none", "pass"):
                return _index_of(decision, "pass")
            if typ == "pon":
                return _index_of(decision, "pon")
            if typ in ("daiminkan", "kan"):
                return _index_of(decision, "kan")
            if typ == "chi":
                if "chi" in opts:
                    return _index_of(decision, "chi")
                # 引擎先问碰/杠，再单独问吃；把 chi 留给后面的 chi 决策
                return _index_of(decision, "pass")
            if typ == "hora":
                raise ExternalAIProtocolError(
                    f"{self.name} 想荣和，但引擎没有给 ron 决策（可能振听判定不同）；"
                    f"拒绝静默跳过和牌"
                )
            raise ExternalAIProtocolError(
                f"{self.name} 对鸣牌决策给了 {typ!r}；选项 {opts}"
            )

        # chi
        opts = list(decision.options)
        if typ in ("none", "pass"):
            return _index_of(decision, "pass")
        if typ == "chi":
            self._validate_chi(action, decision)
            return _index_of(decision, "chi")
        if typ == "hora":
            raise ExternalAIProtocolError(
                f"{self.name} 想吃/荣和，但引擎只给了 chi 决策；选项 {opts}"
            )
        raise ExternalAIProtocolError(
            f"{self.name} 对吃决策给了 {typ!r}；选项 {opts}，原始动作 {action}"
        )

    def _validate_chi(self, action: Mapping[str, Any], decision: Decision) -> None:
        """核对吃的是不是引擎给的那张，以及 consumed 两张是不是合法顺子。"""
        tile = decision.tile
        if tile is None:
            return
        pai = action.get("pai")
        if pai is not None:
            called, _ = _parse_tile(pai)
            if called != int(tile):
                raise ExternalAIProtocolError(
                    f"{self.name} 说吃 {pai}，但引擎给的是 {_tile_name(int(tile))}"
                )
        consumed = action.get("consumed") or []
        if len(consumed) != 2:
            raise ExternalAIProtocolError(
                f"{self.name} 的 chi 动作缺 consumed（{action}）"
            )
        a, _ = _parse_tile(consumed[0])
        b, _ = _parse_tile(consumed[1])
        seq = sorted([int(tile), a, b])
        if seq[1] != seq[0] + 1 or seq[2] != seq[0] + 2 or seq[0] // 9 != seq[2] // 9:
            raise ExternalAIProtocolError(
                f"{self.name} 的 chi consumed {consumed} 和 {_tile_name(int(tile))} "
                f"凑不成顺子"
            )

    def _map_dahai(self, action: Mapping[str, Any], decision: Decision) -> int:
        pai = action.get("pai")
        tile, _red = _parse_tile(pai)
        opts = list(decision.options)
        if tile not in opts:
            raise ExternalAIProtocolError(
                f"{self.name} 想打 {_tile_name(tile)}（{pai}），但引擎的合法打牌是 "
                f"{[_tile_name(int(t)) for t in opts]}；不静默改成第一项"
            )
        return opts.index(tile)

    # -- 统计 ---------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "spec": self.spec.key,
            "title": self.spec.title,
            "license": self.spec.license,
            "player_id": self.player_id,
            "decisions": self.decisions,
            "errors": self.errors,
            "restarts": self._restarts,
            "timeouts": self.timeout_count,
            "crashes": self.crash_count,
            "bad_actions": self._bad_action_count,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "fallback_steps": self.fallback_steps,
            "avg_decide_ms": round(1000 * self._decide_total / max(1, self.decisions), 2),
            "max_decide_ms": round(1000 * self._decide_max, 2),
            "available": self.available,
            "available_reason": self.available_reason,
        }


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
#: 这些事件可能让外部 AI 产生一个动作
_TRIGGER_TYPES = {
    "tsumo", "dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "reach",
}


def _last_trigger_index(events: list[dict]) -> int:
    """批里最后一个可能触发外部 AI 动作的下标；没有就返回 -1。"""
    for i in range(len(events) - 1, -1, -1):
        if events[i].get("type") in _TRIGGER_TYPES:
            return i
    return -1


def _index_of(decision: Decision, value: str) -> int:
    """在 ``decision.options`` 里找字符串选项；找不到**明确报错**。"""
    opts = list(decision.options)
    if value not in opts:
        raise ExternalAIProtocolError(
            f"引擎的 {decision.kind} 选项里没有 {value!r}：{opts}"
        )
    return opts.index(value)


def _action_kan_tile(action: Mapping[str, Any]) -> int | None:
    pai = action.get("pai")
    if isinstance(pai, str):
        try:
            return _parse_tile(pai)[0]
        except ExternalAIProtocolError:
            return None
    consumed = action.get("consumed")
    if isinstance(consumed, list) and consumed:
        try:
            return _parse_tile(consumed[0])[0]
        except ExternalAIProtocolError:
            return None
    return None


def make(spec: str, *, seed: int = 0, player_id: int = 0, players: int = 4,
         **kwargs: Any) -> ExternalMJaiAgent:
    """构造一个外部 agent；没装会抛 :class:`ExternalAINotInstalled`。"""
    return ExternalMJaiAgent(spec, seed=seed, player_id=player_id,
                             players=players, **kwargs)


def attach(game: Any, agents: Iterable[Any] | None = None) -> list[ExternalMJaiAgent]:
    """把 ``game.on_action`` 接到桌上的所有 :class:`ExternalMJaiAgent`。

    只包一层，保留原来的 ``on_action``（比如平台的广播）。返回接上的 agent 列表。
    """
    pool = list(agents if agents is not None else getattr(game, "agents", []) or [])
    ext = [a for a in pool if isinstance(a, ExternalMJaiAgent)]
    if not ext:
        return []
    prev = getattr(game, "on_action", None)

    def dispatch(ev: dict) -> None:
        if prev is not None:
            prev(ev)
        for ag in ext:
            ag.feed(ev)

    game.on_action = dispatch
    return ext


def close_external(agents: Iterable[Any]) -> None:
    """关掉一批 agent 里的外部进程（其它 agent 忽略）。

    也认 ``webapp.server.agents.AISeat`` 这种包装：它把外部 agent 放在
    ``.agent`` / ``._external`` 里。
    """
    for a in agents:
        inner = getattr(a, "agent", a)
        ext = getattr(a, "_external", None)
        if ext is None and isinstance(inner, ExternalMJaiAgent):
            ext = inner
        if ext is not None:
            try:
                ext.close()
            except Exception:  # noqa: BLE001
                log.exception("关闭外部 AI 失败")


def external_models(players: int = 4) -> list[dict]:
    """给平台模型列表用的行；**不启动进程**。

    平台可以把这些行并进 ``available_models()``；``usable=False`` 的置灰显示。
    """
    rows: list[dict] = []
    for key in sorted(SPECS):
        sp = SPECS[key]
        if sp.players != players:
            continue
        ok, why = check_spec(sp)
        rows.append({
            "name": f"external:{key}",
            "external": True,
            "spec": key,
            "title": sp.title,
            "players": sp.players,
            "license": sp.license,
            "language": sp.language,
            "repo": sp.repo,
            "style": "external",
            "style_cn": "外部 AI",
            "cn": sp.title,
            "version": "",
            "usable": bool(ok),
            "missing_heads": [],
            "note": sp.note if ok else why,
        })
    return rows


def close_all() -> None:
    """进程退出 / Ctrl-C 时把所有外部子进程 kill 干净。"""
    for ag in list(_LIVE):
        try:
            ag.close()
        except Exception:  # noqa: BLE001
            pass


atexit.register(close_all)


__all__ = [
    "ExternalAIError", "ExternalAINotInstalled", "ExternalAIUnsupported",
    "ExternalAIProtocolError", "ExternalAITimeout", "ExternalAICrashed",
    "ExternalSpec", "ExternalMJaiAgent", "SPECS", "ALIASES",
    "spec_names", "get_spec", "normalize_key", "is_external_spec", "spec_help",
    "resolve_settings", "check_spec", "require", "make", "attach",
    "close_external", "external_models", "EXTERNAL_DIR", "DOC_PATH",
]
