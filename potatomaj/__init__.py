"""potatomaj —— 雀魂（Mahjong Soul）三麻对局的牌效与危险度分析。

牌全部来自协议解码（不在这个包里），这里只负责「拿到牌之后怎么打」。

第一阶段目标：从 Safari 对局截图中精准识别
  * 自己的手牌 (hand)
  * 四家牌河 (discards / pond)
  * 左上角宝牌指示牌 (dora indicators)
"""

__version__ = "0.1.0"

from .tiles import Tile, tile_id, parse_tile, dora_from_indicator, TILE_IDS  # noqa: F401

__all__ = ["Tile", "tile_id", "parse_tile", "dora_from_indicator", "TILE_IDS", "__version__"]
