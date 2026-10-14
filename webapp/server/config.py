"""运行参数。

全部可以用环境变量覆盖，命令行再覆盖环境变量（见 :func:`parse_args`）。
默认值是按「本机 i9-14900HX + 同时在跑模型训练」这个前提定的 ——
AI 线程池故意开得很小，宁可让观战慢一点，也不要把核吃满。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # potatomaj/
STATIC = Path(__file__).resolve().parents[1] / "static"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    #: 监听地址。公网部署时建议 127.0.0.1 + nginx/frp 反代
    host: str = "127.0.0.1"
    port: int = 8790
    #: 对外 URL（邀请链接用），留空则按请求头推断
    public_url: str = ""

    #: AI 线程池大小。**故意很小**：这台机器同时在跑训练。
    ai_workers: int = 4
    #: 每个房间最多几路并发推理（预留，目前按全局池算）
    ai_workers_per_room: int = 2

    #: 真人思考超时（秒）：超时交给教师托管，绝不卡死整桌
    human_timeout: float = 30.0
    #: 局间等待「下一局」的超时（秒）
    between_timeout: float = 60.0
    #: AI 每手节拍（秒），0 = 全速
    ai_delay: float = 1.2

    #: 房间数上限 / 单房间连接数上限
    max_rooms: int = 200
    max_conns_per_room: int = 40
    max_conns: int = 1200
    #: 空房间保留多久（秒）后回收
    empty_room_ttl: float = 1800.0
    #: 对局结束的房间保留多久
    finished_room_ttl: float = 600.0
    #: 断线后座位保留多久（秒）—— 期间重连能回到原座位
    seat_grace: float = 300.0

    #: 日志
    log_dir: Path = field(default_factory=lambda: Path(_env("MAJ_LOG_DIR", str(ROOT / "logs"))))
    log_level: str = field(default_factory=lambda: _env("MAJ_LOG_LEVEL", "INFO"))
    log_max_bytes: int = 8 * 1024 * 1024
    log_backups: int = 5
    access_log: bool = False

    #: 决策级对局记录（训练数据）。关掉就一条都不写。
    record: bool = field(default_factory=lambda: _env("MAJ_RECORD", "1") != "0")
    data_dir: Path = field(
        default_factory=lambda: Path(_env("MAJ_DATA_DIR", str(ROOT / "data" / "games")))
    )
    #: 单个分片多大就换下一个
    record_part_bytes: int = field(
        default_factory=lambda: _env_int("MAJ_RECORD_PART_MB", 256) << 20
    )
    #: 整个记录目录的容量上限（超了删最老的分片）
    record_total_bytes: int = field(
        default_factory=lambda: _env_int("MAJ_RECORD_TOTAL_MB", 8192) << 20
    )

    #: 默认对局长度：0 = 按人数（三麻 3 局 / 四麻 4 局）
    default_rounds: int = 0
    #: 聊天
    allow_chat: bool = True
    max_chat_len: int = 200
    #: 进房口令（留空 = 只要昵称就够，不裸奔）
    room_password: str = ""

    @property
    def models_dir(self) -> Path:
        return ROOT / "aimodel" / "weights"

    @property
    def tiles_dir(self) -> Path:
        return STATIC / "tiles"

    @property
    def client_dir(self) -> Path:
        return STATIC / "app"

    @property
    def profiles_path(self) -> Path:
        return self.data_dir.parent / "profiles.json"

    def as_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "ai_workers": self.ai_workers,
            "human_timeout": self.human_timeout,
            "ai_delay": self.ai_delay,
            "max_rooms": self.max_rooms,
            "record": self.record,
            "data_dir": str(self.data_dir),
            "need_password": bool(self.room_password),
        }


DEFAULT = Config()


def parse_args(argv: list[str] | None = None) -> Config:
    ap = argparse.ArgumentParser(description="土豆麻将 · 网页对战平台")
    ap.add_argument("--host", default=_env("MAJ_HOST", DEFAULT.host))
    ap.add_argument("--port", type=int, default=_env_int("MAJ_PORT", DEFAULT.port))
    ap.add_argument("--public-url", default=_env("MAJ_PUBLIC_URL", DEFAULT.public_url))
    ap.add_argument("--ai-workers", type=int,
                    default=_env_int("MAJ_AI_WORKERS", DEFAULT.ai_workers))
    ap.add_argument(
        "--ai-delay", type=float, default=_env_float("MAJ_AI_DELAY", DEFAULT.ai_delay),
        help="AI 每手节拍（秒），0 = 全速",
    )
    ap.add_argument(
        "--human-timeout", type=float,
        default=_env_float("MAJ_HUMAN_TIMEOUT", DEFAULT.human_timeout),
        help="真人出牌超时（秒），超时由内置教师托管",
    )
    ap.add_argument("--max-rooms", type=int, default=_env_int("MAJ_MAX_ROOMS", DEFAULT.max_rooms))
    ap.add_argument("--log-level", default=_env("MAJ_LOG_LEVEL", DEFAULT.log_level))
    ap.add_argument("--log-dir", default=str(DEFAULT.log_dir))
    ap.add_argument("--access-log", action="store_true",
                    default=_env("MAJ_ACCESS_LOG", "0") == "1")
    ap.add_argument("--no-chat", action="store_true",
                    default=_env("MAJ_ALLOW_CHAT", "1") == "0")
    ap.add_argument("--no-record", action="store_true",
                    default=_env("MAJ_RECORD", "1") == "0",
                    help="关闭决策级对局记录")
    ap.add_argument("--data-dir", default=_env("MAJ_DATA_DIR", str(DEFAULT.data_dir)))
    ap.add_argument("--room-password", default=_env("MAJ_ROOM_PASSWORD", ""),
                    help="进房口令（留空 = 只要昵称）")
    args = ap.parse_args(argv)

    return Config(
        host=args.host,
        port=args.port,
        public_url=args.public_url.strip(),
        ai_workers=max(1, args.ai_workers),
        ai_delay=max(0.0, args.ai_delay),
        human_timeout=max(3.0, args.human_timeout),
        max_rooms=max(1, args.max_rooms),
        log_dir=Path(args.log_dir),
        log_level=args.log_level,
        access_log=bool(args.access_log),
        allow_chat=not args.no_chat,
        record=not args.no_record,
        data_dir=Path(args.data_dir),
        room_password=args.room_password.strip(),
    )
