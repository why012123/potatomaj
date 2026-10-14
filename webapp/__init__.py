"""``webapp`` —— 土豆麻将网页对战平台。

分两层：

* :mod:`webapp.server` —— aiohttp 服务端（HTTP 静态资源 + WebSocket 实时推送）
* :mod:`webapp.client` —— Vite + TypeScript 前端源码（构建产物落在
  ``webapp/static/app/``，服务端当静态资源发）

版本号在这里写一份，服务端自己那份在 :data:`webapp.server.__version__`。
"""

from __future__ import annotations

VERSION = "1.0.0"

__all__ = ["VERSION"]
