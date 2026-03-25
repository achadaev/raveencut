import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class WaveformWidget(QWidget):
    seek_requested = pyqtSignal(float)

    SPEECH_COLOR   = QColor("#3a7a3a")
    SILENCE_COLOR  = QColor("#7a2a2a")
    RESTORED_COLOR = QColor("#3a5a7a")
    BG_COLOR       = QColor("#111111")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pcm: np.ndarray | None = None
        self._duration = 1.0
        self._speech_segs = []
        self._silence_segs = []
        self._restored: set = set()
        self._zoom = 1.0
        self._scroll_frac = 0.0
        self._playhead_sec = 0.0
        self.setFixedHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def update_audio(self, pcm: np.ndarray, duration: float):
        self._pcm = pcm
        self._duration = max(duration, 0.001)
        self.update()

    def update_segments(self, speech, silence, restored: set):
        self._speech_segs = speech
        self._silence_segs = silence
        self._restored = restored
        self.update()

    def set_playhead(self, sec: float):
        self._playhead_sec = sec
        self._auto_scroll(sec)
        self.update()

    def set_scroll(self, frac: float):
        self._scroll_frac = max(0.0, min(1.0, frac))
        self.update()

    def _visible_duration(self):
        return self._duration / self._zoom

    def _scroll_offset_sec(self):
        return self._scroll_frac * max(0.0, self._duration - self._visible_duration())

    def _time_to_x(self, t: float) -> int:
        offset = self._scroll_offset_sec()
        vis = self._visible_duration()
        frac = (t - offset) / vis if vis > 0 else 0
        return int(frac * self.width())

    def _x_to_time(self, x: int) -> float:
        return self._scroll_offset_sec() + (x / self.width()) * self._visible_duration()

    def _auto_scroll(self, sec: float):
        offset = self._scroll_offset_sec()
        vis = self._visible_duration()
        if sec < offset or sec > offset + vis:
            max_offset = max(0.0, self._duration - vis)
            new_offset = max(0.0, min(max_offset, sec - vis * 0.1))
            self._scroll_frac = (new_offset / max_offset) if max_offset > 0 else 0.0

    def _color_at(self, t: float) -> QColor:
        for i, seg in enumerate(self._silence_segs):
            if seg["start"] <= t < seg["end"]:
                return self.RESTORED_COLOR if i in self._restored else self.SILENCE_COLOR
        return self.SPEECH_COLOR

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            if not painter.isActive():
                return
        except RuntimeError:
            return
        painter.fillRect(self.rect(), self.BG_COLOR)
        if self._pcm is None:
            return
        w, h, mid = self.width(), self.height(), self.height() // 2
        n = len(self._pcm)
        offset, vis = self._scroll_offset_sec(), self._visible_duration()
        for bar_i, amp in enumerate(self._pcm):
            t = offset + (bar_i / n) * vis if n else 0
            x = self._time_to_x(t)
            if 0 <= x < w:
                bar_h = int(amp * mid * 0.9)
                painter.setPen(QPen(self._color_at(t), 1))
                painter.drawLine(x, mid - bar_h, x, mid + bar_h)
        px = self._time_to_x(self._playhead_sec)
        if 0 <= px < w:
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawLine(px, 0, px, h)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            t = max(0.0, min(self._duration, self._x_to_time(int(event.position().x()))))
            self.seek_requested.emit(t)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        cursor_x = int(event.position().x())
        t_under = self._x_to_time(cursor_x)
        self._zoom = max(1.0, min(100.0, self._zoom * factor))
        vis = self._visible_duration()
        max_off = max(0.0, self._duration - vis)
        new_off = t_under - (cursor_x / self.width()) * vis
        self._scroll_frac = max(0.0, min(1.0, new_off / max_off)) if max_off > 0 else 0.0
        self.update()
