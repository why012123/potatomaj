"""土豆麻将 GUI：一个置顶小悬浮窗，点「识别」后在下方弹出结果窗。

用 PySide6（Qt）实现：
  * 悬浮窗无边框、置顶、可拖动，只有「识别」按钮和来源选择
  * 结果窗在悬浮窗正下方新建，用真实牌面图显示手牌 / 三家牌河 / 宝牌指示牌，
    并给出向听数与进张
  * 识别跑在后台线程，界面不会卡
"""

from __future__ import annotations

import io
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import __version__, capture as capture_mod, vision as V
from .classify import TemplateLibrary
from .detect import Region
from .recognize import recognize, render_debug
from .shanten import analyze as analyze_hand
from .tiles import parse_tile, sort_tiles, tiles_to_mpsz

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "templates"

# 牌面显示尺寸（宽，高按牌面比例算）
HAND_TILE_W = 46
POND_TILE_W = 30
DORA_TILE_W = 34
FACE_ASPECT = 0.614


# ---------------------------------------------------------------------------
# 牌面图片
# ---------------------------------------------------------------------------


class TileImages:
    """把 templates/*.png 渲染成 Qt 图片，按 (牌, 宽度) 缓存。"""

    def __init__(self, template_dir: Path = TEMPLATE_DIR):
        self.dir = template_dir
        self._cache: dict[tuple[str, int], QPixmap] = {}
        self._raw: dict[str, Image.Image] = {}

    def _load(self, tid: str) -> Image.Image | None:
        if tid in self._raw:
            return self._raw[tid]
        p = self.dir / f"{tid}.png"
        img = None
        if p.exists():
            img = Image.open(p).convert("RGBA")
        self._raw[tid] = img  # type: ignore[assignment]
        return img

    def pixmap(self, tid: str | None, width: int) -> QPixmap:
        key = (tid or "?", width)
        if key in self._cache:
            return self._cache[key]
        height = int(round(width / FACE_ASPECT))
        img = self._load(tid) if tid else None
        if img is None:
            pm = self._placeholder(width, height, tid)
        else:
            resized = img.resize((width, height), Image.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            pm = QPixmap()
            pm.loadFromData(buf.getvalue(), "PNG")
        self._cache[key] = pm
        return pm

    @staticmethod
    def _placeholder(width: int, height: int, label: str | None) -> QPixmap:
        pm = QPixmap(width, height)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(245, 243, 235, 210))
        painter.setPen(QPen(QColor(150, 145, 135), 1))
        painter.drawRoundedRect(0, 0, width - 1, height - 1, 4, 4)
        painter.setPen(QColor(120, 115, 105))
        painter.setFont(QFont("Helvetica", max(9, width // 3)))
        painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, label or "?")
        painter.end()
        return pm


# ---------------------------------------------------------------------------
# 识别线程
# ---------------------------------------------------------------------------


class RecognizeWorker(QObject):
    finished = Signal(object)   # dict
    failed = Signal(str)

    def __init__(self, source: str, file_path: str | None, library: TemplateLibrary):
        super().__init__()
        self.source = source
        self.file_path = file_path
        self.library = library

    def run(self) -> None:
        try:
            cap = capture_mod.capture(self.source, self.file_path)
            res = recognize(cap.image, self.library)
            debug_path = None
            if res.anchor is not None:
                debug_path = ROOT / "build" / "last_recognize.png"
                try:
                    render_debug(cap.image, res, debug_path)
                except Exception:
                    debug_path = None
            self.finished.emit({"capture": cap, "result": res, "debug": debug_path})
        except capture_mod.CaptureError as e:
            self.failed.emit(str(e))
        except Exception:
            self.failed.emit("识别出错：\n" + traceback.format_exc(limit=3))


# ---------------------------------------------------------------------------
# 悬浮窗
# ---------------------------------------------------------------------------


class FloatingWindow(QWidget):
    """置顶小悬浮窗：一个「识别」按钮 + 来源选择。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("土豆麻将")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.library = TemplateLibrary(TEMPLATE_DIR)
        self.images = TileImages()
        self.result_window: ResultWindow | None = None
        self._drag_offset = None
        self._thread: QThread | None = None
        self._worker: RecognizeWorker | None = None
        self._file_path: str | None = None
        self._build_ui()

    # -- 界面 ---------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        panel = QFrame()
        panel.setObjectName("panel")
        panel.setStyleSheet(
            """
            QFrame#panel {
                background: rgba(28, 32, 44, 235);
                border: 1px solid rgba(120, 200, 255, 90);
                border-radius: 12px;
            }
            QLabel { color: #d8e4f0; }
            QLabel#title { color: #8fd3ff; font-weight: 600; }
            QLabel#status { color: #9aa7b8; font-size: 10px; }
            QPushButton#go {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #3d8bfd, stop:1 #2a6ed4);
                color: white; border: none; border-radius: 8px;
                font-size: 15px; font-weight: 600; padding: 10px 0;
            }
            QPushButton#go:hover { background: #4b98ff; }
            QPushButton#go:disabled { background: #3a4356; color: #8b95a5; }
            QPushButton#icon {
                background: transparent; color: #9aa7b8; border: none;
                font-size: 13px; padding: 0 4px;
            }
            QPushButton#icon:hover { color: white; }
            QComboBox {
                background: rgba(255,255,255,18); color: #cfd9e6;
                border: 1px solid rgba(255,255,255,40); border-radius: 6px;
                padding: 3px 6px; font-size: 11px;
            }
            QComboBox QAbstractItemView {
                background: #232936; color: #cfd9e6; selection-background-color: #3d8bfd;
            }
            """
        )
        outer.addWidget(panel)

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 9, 12, 11)
        lay.setSpacing(6)

        # 标题栏
        bar = QHBoxLayout()
        bar.setSpacing(4)
        title = QLabel("🀄 土豆麻将")
        title.setObjectName("title")
        bar.addWidget(title)
        bar.addStretch(1)
        btn_close = QPushButton("✕")
        btn_close.setObjectName("icon")
        btn_close.setFixedSize(20, 20)
        btn_close.clicked.connect(self.close)
        bar.addWidget(btn_close)
        lay.addLayout(bar)

        # 识别按钮
        self.btn_go = QPushButton("识  别")
        self.btn_go.setObjectName("go")
        self.btn_go.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_go.clicked.connect(self.on_recognize)
        lay.addWidget(self.btn_go)

        # 来源
        src = QHBoxLayout()
        src.setSpacing(6)
        lab = QLabel("来源")
        lab.setObjectName("status")
        src.addWidget(lab)
        self.combo = QComboBox()
        for key, text in capture_mod.SOURCE_LABELS.items():
            self.combo.addItem(text, key)
        self.combo.currentIndexChanged.connect(self._on_source_changed)
        src.addWidget(self.combo, 1)
        lay.addLayout(src)

        self.status = QLabel("就绪")
        self.status.setObjectName("status")
        lay.addWidget(self.status)

        self.setFixedWidth(184)
        self.adjustSize()

    # -- 拖动 ---------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self._place_result_window()
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_offset = None

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        act_kb = QAction("打开模板文件夹", self)
        act_kb.triggered.connect(lambda: subprocess.run(["open", str(TEMPLATE_DIR)]))
        menu.addAction(act_kb)
        act_ver = QAction(f"potatomaj {__version__}", self)
        act_ver.setEnabled(False)
        menu.addAction(act_ver)
        menu.exec(event.globalPos())

    # -- 来源 ---------------------------------------------------------------

    def set_source(self, key: str, file_path: str | None = None) -> None:
        """以编程方式切换来源（不弹文件对话框），测试和自动化用。"""
        keys = list(capture_mod.SOURCE_LABELS)
        if key not in keys:
            raise ValueError(f"未知来源 {key!r}")
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(keys.index(key))
        self.combo.blockSignals(False)
        self._file_path = file_path
        self.status.setText(f"文件：{Path(file_path).name}" if file_path else "就绪")

    def _on_source_changed(self, _idx: int) -> None:
        if self.combo.currentData() == "file":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择对局截图", str(Path.home() / "Desktop"),
                "图片 (*.png *.jpg *.jpeg *.webp)",
            )
            if path:
                self._file_path = path
                self.status.setText(f"文件：{Path(path).name}")
            else:
                self.combo.setCurrentIndex(0)
                self._file_path = None
        else:
            self._file_path = None
            self.status.setText("就绪")

    # -- 识别 ---------------------------------------------------------------

    def on_recognize(self) -> None:
        if self._thread is not None:
            return
        source = self.combo.currentData()
        if source == "file" and not self._file_path:
            self._on_source_changed(0)
            if not self._file_path:
                return
        self.btn_go.setEnabled(False)
        self.status.setText("识别中…")

        self._thread = QThread(self)
        self._worker = RecognizeWorker(source, self._file_path, self.library)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _stop_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
            self._worker = None
        self.btn_go.setEnabled(True)

    def _on_done(self, payload: dict) -> None:
        self._stop_thread()
        res = payload["result"]
        cap = payload["capture"]
        if res.anchor is None:
            self.status.setText("没找到牌桌（截图是否包含手牌行？）")
        else:
            self.status.setText(f"{cap.source} · {len(res.hand)} 张")
        if self.result_window is None:
            self.result_window = ResultWindow(self.images, self)
        self.result_window.update_result(res, cap, payload.get("debug"))
        # 先 show 再定位：show 之前窗口还没「可见」，定位逻辑会被跳过，
        # 结果窗就会停在 (0,0)
        self.result_window.show()
        self._place_result_window()
        self.result_window.raise_()

    def _on_failed(self, msg: str) -> None:
        self._stop_thread()
        self.status.setText("失败")
        QMessageBox.warning(self, "识别失败", msg)

    def _place_result_window(self) -> None:
        """结果窗贴在悬浮窗正下方。"""
        if self.result_window is None or not self.result_window.isVisible():
            return
        geo = self.frameGeometry()
        self.result_window.move(int(geo.left()), int(geo.bottom()) + 8)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_thread()
        if self.result_window is not None:
            self.result_window.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# 结果窗
# ---------------------------------------------------------------------------


class ResultWindow(QWidget):
    """显示识别 + 判断结果。"""

    def __init__(self, images: TileImages, parent: FloatingWindow | None = None):
        super().__init__()
        self.images = images
        self.parent_window = parent
        self.setWindowTitle("判断结果")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(
            """
            QWidget { background: #1b2029; color: #dfe7f1; }
            QLabel#section { color: #7fc4ff; font-weight: 600; }
            QLabel#dim { color: #93a1b3; }
            QLabel#verdict { color: #ffd479; font-weight: 600; font-size: 14px; }
            QLabel#warn { color: #ffb36b; }
            QPushButton {
                background: rgba(255,255,255,18); color: #dfe7f1;
                border: 1px solid rgba(255,255,255,40); border-radius: 6px;
                padding: 5px 12px;
            }
            QPushButton:hover { background: rgba(255,255,255,35); }
            """
        )
        self._debug_path: Path | None = None
        self._summary = ""
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        self.verdict = QLabel("—")
        self.verdict.setObjectName("verdict")
        root.addWidget(self.verdict)

        self.hand_row = QHBoxLayout()
        self.hand_row.setSpacing(4)
        self.hand_row.addStretch(1)
        root.addLayout(self.hand_row)

        self.ponds_box = QVBoxLayout()
        self.ponds_box.setSpacing(6)
        root.addLayout(self.ponds_box)

        self.dora_row = QHBoxLayout()
        self.dora_row.setSpacing(4)
        root.addLayout(self.dora_row)

        self.warn = QLabel("")
        self.warn.setObjectName("warn")
        self.warn.setWordWrap(True)
        root.addWidget(self.warn)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_copy = QPushButton("复制文本")
        self.btn_copy.clicked.connect(self._copy)
        btns.addWidget(self.btn_copy)
        self.btn_debug = QPushButton("看标注图")
        self.btn_debug.clicked.connect(self._open_debug)
        btns.addWidget(self.btn_debug)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        btns.addWidget(self.btn_close)
        root.addLayout(btns)

    # -- 更新 ---------------------------------------------------------------

    @staticmethod
    def _clear(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout() is not None:
                ResultWindow._clear(item.layout())

    def _tile_labels(self, ids: list[str | None], width: int) -> list[QLabel]:
        out = []
        for tid in ids:
            lab = QLabel()
            lab.setPixmap(self.images.pixmap(tid, width))
            out.append(lab)
        return out

    def update_result(self, res, cap, debug_path) -> None:
        self._debug_path = Path(debug_path) if debug_path else None
        self.btn_debug.setEnabled(self._debug_path is not None)

        # ---- 手牌 + 向听 ----
        # 注意：这里按「识别到的每一格」显示，认不出来的那格留 ? 占位 ——
        # 直接丢掉的话张数会少，向听数就基于一副残缺的牌算，看着对其实错
        self._clear(self.hand_row)
        hand_tiles = sorted(
            (t for t in res.tiles if t.region is Region.HAND), key=lambda t: t.index
        )
        hand = res.hand
        unknown_hand = [t for t in hand_tiles if t.tile_id is None]
        tiles_text = " ".join(sort_tiles(hand))
        if hand_tiles:
            size_hint = "14 张（含刚摸的牌）" if len(hand_tiles) == 14 else f"{len(hand_tiles)} 张"
            self.hand_row.addWidget(self._title(f"手牌 {size_hint}"))
            for lab in self._tile_labels([t.tile_id for t in hand_tiles], HAND_TILE_W):
                self.hand_row.addWidget(lab)
            self.hand_row.addStretch(1)
            if unknown_hand:
                self.verdict.setText(
                    f"手牌有 {len(unknown_hand)} 张没认出来，向听判断不可靠"
                )
            else:
                # 13 张给进张/听牌，14 张（刚摸到）给「建议打哪张」
                self.verdict.setText(analyze_hand(hand, sanma=True)["text"])
        else:
            self.hand_row.addWidget(self._title("手牌 未识别到"))
            self.verdict.setText("—")

        # ---- 牌河 ----
        self._clear(self.ponds_box)
        seat_names = {
            Region.POND_SELF: "自己",
            Region.POND_RIGHT: "下家",
            Region.POND_TOP: "对家",
            Region.POND_LEFT: "上家",
        }
        any_pond = False
        for region, name in seat_names.items():
            tiles = [t for t in res.tiles if t.region is region]
            if not tiles:
                continue
            any_pond = True
            row = QHBoxLayout()
            row.setSpacing(4)
            unknown = sum(1 for t in tiles if t.tile_id is None)
            label = f"牌河·{name} {len(tiles)} 张"
            if unknown:
                label += f"（{unknown} 张没认出来）"
            row.addWidget(self._title(label))
            for t in tiles:
                lab = QLabel()
                lab.setPixmap(self.images.pixmap(t.tile_id, POND_TILE_W))
                lab.setToolTip(t.tile_id or "没认出来")
                row.addWidget(lab)
            row.addStretch(1)
            self.ponds_box.addLayout(row)
        if not any_pond:
            self.ponds_box.addWidget(self._dim("牌河 未识别到"))

        # ---- 宝牌 ----
        self._clear(self.dora_row)
        drawn = [t.tile_id for t in res.tiles if t.region is Region.DORA and not t.is_back]
        if drawn:
            self.dora_row.addWidget(self._title("宝牌指示牌"))
            for lab in self._tile_labels(list(drawn), DORA_TILE_W):
                self.dora_row.addWidget(lab)
            dora = res.dora
            if dora:
                arrow = QLabel("→ 宝牌")
                arrow.setObjectName("dim")
                self.dora_row.addWidget(arrow)
                for lab in self._tile_labels(list(dora), DORA_TILE_W):
                    self.dora_row.addWidget(lab)
        else:
            self.dora_row.addWidget(self._dim("宝牌指示牌 未识别到"))
        self.dora_row.addStretch(1)

        # ---- 警告 ----
        notes = [f"来源：{cap.source}"]
        if res.warnings:
            notes.append("；".join(res.warnings[:3]))
        self.warn.setText("　|　".join(notes))

        # ---- 文本摘要（复制用）----
        self._summary = (
            f"手牌({len(hand)}): {tiles_text}   [{tiles_to_mpsz(hand)}]\n"
            + self.verdict.text() + "\n"
            + "\n".join(
                f"牌河·{seat_names[r]} " + " ".join(t.tile_id or "?" for t in res.tiles if t.region is r)
                for r in seat_names
                if any(t.region is r for t in res.tiles)
            )
            + f"\n宝牌指示牌: {' '.join(drawn) or '—'}  宝牌: {' '.join(res.dora) or '—'}"
        )
        self.adjustSize()

    def _title(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("section")
        lab.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return lab

    def _dim(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("dim")
        return lab

    # -- 按钮 ---------------------------------------------------------------

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._summary)
        self.btn_copy.setText("已复制 ✓")
        QApplication.processEvents()

    def _open_debug(self) -> None:
        if self._debug_path and self._debug_path.exists():
            subprocess.run(["open", str(self._debug_path)])


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName("土豆麻将")
    app.setApplicationDisplayName("土豆麻将")
    app.setQuitOnLastWindowClosed(True)

    win = FloatingWindow()
    screen = app.primaryScreen().availableGeometry()
    win.move(screen.right() - win.width() - 60, screen.top() + 120)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
