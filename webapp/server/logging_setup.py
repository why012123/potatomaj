"""日志：同时写 stdout 和**带轮转的文件**。

部署要求里写死了「不要只打 stdout」，所以这里用
:class:`logging.handlers.RotatingFileHandler` 落到 ``logs/`` 下：

* ``maj-server.log``   服务端 + 房间生命周期
* ``maj-game.log``     每个房间的牌局流水（引擎原生日志 + ELO 变化）
* ``maj-error.log``    异常（WARNING 以上独立一份，方便盯）
* ``maj-access.log``   HTTP 访问日志

轮转参数来自 :class:`webapp.server.config.Config`（默认 8MB × 5 份）。
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_CONFIGURED = False

FMT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


class _MaxLevel(logging.Filter):
    """只放行低于某个级别的记录（让 WARNING 以上走 stderr / 独立文件）。"""

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.level


def setup(log_dir: Path, level: str = "INFO", max_bytes: int = 8 << 20,
          backups: int = 5) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(lvl)

    fmt = logging.Formatter(FMT, DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.addFilter(_MaxLevel(logging.WARNING))
    root.addHandler(console)

    err_console = logging.StreamHandler(sys.stderr)
    err_console.setFormatter(fmt)
    err_console.setLevel(logging.WARNING)
    root.addHandler(err_console)

    def rot(name: str, lvl_: int) -> logging.Handler:
        h = logging.handlers.RotatingFileHandler(
            log_dir / name, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
        h.setFormatter(fmt)
        h.setLevel(lvl_)
        return h

    root.addHandler(rot("maj-server.log", lvl))
    root.addHandler(rot("maj-error.log", logging.WARNING))

    # 牌局流水单独一份（压到 INFO 以下也能留住，方便以后查账）
    game = logging.getLogger("maj.game")
    game.propagate = False
    game.setLevel(logging.INFO)
    gh = logging.handlers.RotatingFileHandler(
        log_dir / "maj-game.log", maxBytes=max_bytes, backupCount=backups,
        encoding="utf-8",
    )
    gh.setFormatter(logging.Formatter("%(asctime)s %(message)s", DATEFMT))
    game.addHandler(gh)

    # aiohttp 的访问日志单独走文件，别把终端刷满
    access = logging.getLogger("aiohttp.access")
    access.propagate = False
    access.setLevel(logging.INFO)
    access.addHandler(rot("maj-access.log", logging.INFO))
    logging.getLogger("aiohttp.server").setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger("maj.boot").info("日志已就绪 -> %s", log_dir.resolve())
