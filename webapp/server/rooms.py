"""房间注册表。

**隔离是硬要求**：这张表里存的是一个个独立的 :class:`Room`，
每个房间有自己的引擎、座位表、事件序号和连接集合。
除了这张表本身（一个 dict）和只读的权重缓存 / 推理线程池，
房间之间没有任何共享可变状态。

:meth:`RoomManager.sweep` 负责回收：纯 AI 房自动开打、空房间、
打完很久的房间、掉线太久没回来的座位。
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from typing import Any, Iterable

from .config import Config
from .table import PHASE_FINISHED, PHASE_LOBBY, PHASE_PLAYING, Room, RoomError

log = logging.getLogger("maj.rooms")

#: 房间号字符集：去掉 0/O/1/I 这些看错的
_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


class RoomManager:
    def __init__(
        self,
        cfg: Config,
        loop: asyncio.AbstractEventLoop,
        *,
        writer: Any = None,
        profiles: Any = None,
    ) -> None:
        self.cfg = cfg
        self.loop = loop
        self.lock = threading.RLock()
        self.rooms: dict[str, Room] = {}
        self._sweeper: asyncio.Task | None = None
        #: 决策级记录的写盘器 + 玩家画像表（全局唯一，但只被**追加**）
        self.writer = writer
        self.profiles = profiles

    # ------------------------------------------------------------------
    def new_id(self, length: int = 5) -> str:
        with self.lock:
            for _ in range(200):
                rid = "".join(random.choice(_ALPHABET) for _ in range(length))
                if rid not in self.rooms:
                    return rid
        raise RoomError("房间号用完了，稍后再试")

    def create(
        self,
        *,
        name: str = "",
        players: int = 4,
        host_uid: str,
        host_name: str,
        rounds: int = 0,
        hanchan: bool = False,
        team: dict | None = None,
        ai_delay: float | None = None,
        seats: Iterable[tuple[str, str]] | None = None,
        spectator_reveal: bool = True,
    ) -> Room:
        with self.lock:
            if len(self.rooms) >= self.cfg.max_rooms:
                raise RoomError(f"房间数已达上限（{self.cfg.max_rooms}）")
            rid = self.new_id()
            room = Room(
                rid, self.cfg, loop=self.loop, name=name or f"{host_name} 的牌桌",
                players=players, host_uid=host_uid, host_name=host_name,
                rounds=rounds, hanchan=hanchan, team=team, ai_delay=ai_delay,
                seats=seats,
                spectator_reveal=spectator_reveal,
                writer=self.writer, profiles=self.profiles,
            )
            self.rooms[rid] = room
            log.info("创建房间 %s（%s，%d 人，%s）", rid, room.name, players,
                     "纯AI" if room.human_count == 0 else "有人")
            return room

    def get(self, room_id: str) -> Room | None:
        with self.lock:
            return self.rooms.get((room_id or "").strip().upper())

    def drop(self, room_id: str) -> None:
        with self.lock:
            room = self.rooms.pop(room_id, None)
        if room is not None:
            room.close("房间被回收")
            log.info("回收房间 %s", room_id)

    # ------------------------------------------------------------------
    def list_rooms(self) -> list[dict]:
        with self.lock:
            rooms = list(self.rooms.values())
        out = []
        for r in rooms:
            try:
                s = r.summary()
            except Exception:  # noqa: BLE001
                log.exception("房间 %s 摘要生成失败", r.id)
                continue
            # ★ **打完的房间不再出现在大厅列表里**（用户要求）：
            #   牌局结束（phase == "finished"）之后还挂着，列表只会越来越长；
            #   房内的人不受影响 —— 他们手里有房间号，`/room/{id}` 照样能看结算。
            if s.get("phase") == "finished":
                continue
            out.append(s)
        out.sort(key=lambda x: (-x["humans"], -x["conns"], x["created_at"]))
        return out

    def stats(self) -> dict:
        with self.lock:
            rooms = list(self.rooms.values())
        conns = humans = ais = playing = 0
        for r in rooms:
            s = r.summary()
            conns += s["conns"]
            humans += s["humans"]
            ais += s["ais"]
            playing += 1 if s["phase"] == PHASE_PLAYING else 0
        return {
            "rooms": len(rooms), "conns": conns, "humans": humans, "ais": ais,
            "playing": playing, "max_rooms": self.cfg.max_rooms,
        }

    # ------------------------------------------------------------------
    def find_seat(self, uid: str) -> tuple[str, int] | None:
        """某个 uid 现在坐在哪个房间（防止一个人开一堆座位）。"""
        with self.lock:
            rooms = list(self.rooms.items())
        for rid, room in rooms:
            for s in room.seats:
                if s and s.kind == "human" and s.uid == uid:
                    return rid, s.seat
        return None

    # ------------------------------------------------------------------
    def start_sweeper(self) -> None:
        if self._sweeper is None:
            self._sweeper = self.loop.create_task(self._sweep_loop())

    async def _sweep_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(10)
                self.sweep()
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001
                log.exception("房间清理出错")

    def sweep(self) -> None:
        now = time.time()
        dead: list[str] = []
        with self.lock:
            rooms = list(self.rooms.items())
        for rid, room in rooms:
            try:
                if room.closed:
                    dead.append(rid)
                    continue
                # 纯 AI 房：建好还没开打就自动开 —— 邀请链接点进来是观战，
                # 没人会去点「开始」，AI 房不该卡在大厅。
                if (room.phase == PHASE_LOBBY and room.human_count == 0
                        and now - room.created_at > 3.0):
                    try:
                        room.start()
                        log.info("纯 AI 房 %s 自动开打", rid)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("纯 AI 房 %s 自动开打失败：%s", rid, exc)
                        dead.append(rid)
                    continue
                if not room.conns:
                    ttl = (self.cfg.finished_room_ttl if room.phase == PHASE_FINISHED
                           else self.cfg.empty_room_ttl)
                    ref = room.finished_at or room.last_active or room.created_at
                    if now - ref > ttl:
                        dead.append(rid)
                        continue
                # 掉线太久没回来的真人座位 -> 交回给 AI（只对还没打完的桌子做）
                if room.phase != PHASE_FINISHED:
                    for s in room.seats:
                        if (s and s.kind == "human" and not s.online
                                and s.disconnected_at
                                and now - s.disconnected_at > self.cfg.seat_grace):
                            log.info("房间 %s 座位 %d（%s）掉线太久，交回给 AI",
                                     rid, s.seat, s.name)
                            s.kind = "ai"
                            s.ai = s.ai or "teacher"
                            s.name = "AI"
                            s.token = ""
                else:
                    # 打完了又把连接都断了的桌子：座位立刻释放，只保留观战
                    for s in room.seats:
                        if s and s.kind == "human" and not s.online:
                            s.kind = "ai"
                            s.ai = s.ai or "teacher"
                            s.name = "AI"
                            s.token = ""
                            s.uid = f"done:{rid}:{s.seat}"
            except Exception:  # noqa: BLE001
                log.exception("清理房间 %s 时出错", rid)
        for rid in dead:
            self.drop(rid)

    def shutdown(self) -> None:
        with self.lock:
            rooms = list(self.rooms.values())
            self.rooms.clear()
        for r in rooms:
            r.close("服务端关闭")
        if self._sweeper is not None:
            self._sweeper.cancel()
            self._sweeper = None
        log.info("所有房间已关闭（%d 个）", len(rooms))
