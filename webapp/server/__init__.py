"""土豆麻将 · 网页对战平台服务端。

模块划分
--------

* :mod:`webapp.server.config`   —— 运行参数（环境变量 / 命令行）
* :mod:`webapp.server.tiles`    —— 牌张（下标 <-> id）与牌面资源
* :mod:`webapp.server.protocol` —— 服务端 -> 客户端的 JSON 协议（**座位视角**）
* :mod:`webapp.server.agents`   —— 人 / AI 座位适配（AI 走线程池，不阻塞事件循环）
* :mod:`webapp.server.table`    —— 一个房间 = 一个引擎线程 + 异步事件队列
* :mod:`webapp.server.rooms`    —— 房间注册表（房间之间零共享可变状态）
* :mod:`webapp.server.app`      —— aiohttp：静态资源 + 登录/列表 HTTP + WebSocket
* :mod:`webapp.server.recording`—— 决策级日志 + 玩家画像 + ELO
* :mod:`webapp.server.logging_setup` —— 文件日志 + 轮转

设计上只有一条硬规矩：**服务端权威**。客户端只能发「我想干什么」这种意图，
所有合法性判断、所有牌局状态都在服务端；别人的手牌在协议层就不下发。
"""

from __future__ import annotations

__version__ = "1.0.0"
