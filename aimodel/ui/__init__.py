"""简易 Web UI：三个人一起打三麻（详见 :mod:`aimodel.ui.server`）。"""

from .server import Session, RemoteAgent, serve, main  # noqa: F401

__all__ = ["Session", "RemoteAgent", "serve", "main"]
