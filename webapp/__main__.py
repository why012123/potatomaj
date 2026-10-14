"""让 ``python -m webapp`` 直接起服务。"""

from __future__ import annotations

from .server.app import main

if __name__ == "__main__":
    main()
