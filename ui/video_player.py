from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.utils import fmt_time


class VideoPlayerWidget(QWidget):
    position_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._player = QMediaPlayer()
        self._audio_out = QAudioOutput()
        self._player.setAudioOutput(self._audio_out)
        self._video_widget = QVideoWidget()
        self._player.setVideoOutput(self._video_widget)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.clicked.connect(self._toggle_play)
        self._time_label = QLabel("0:00 / 0:00")

        controls = QHBoxLayout()
        controls.addWidget(self._play_btn)
        controls.addWidget(self._time_label)
        controls.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._video_widget, stretch=1)
        layout.addLayout(controls)

        self._player.positionChanged.connect(self._on_position)
        self._player.playbackStateChanged.connect(self._on_state)

    def load(self, path: str):
        self._player.setSource(QUrl.fromLocalFile(path))

    def seek(self, sec: float):
        self._player.setPosition(int(sec * 1000))

    def _toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_position(self, ms: int):
        sec = ms / 1000.0
        dur = self._player.duration() / 1000.0
        self._time_label.setText(f"{fmt_time(sec)} / {fmt_time(dur)}")
        self.position_changed.emit(sec)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return width * 9 // 16 + 36  # 16:9 video + controls bar

    def _on_state(self, state):
        self._play_btn.setText(
            "⏸" if state == QMediaPlayer.PlaybackState.PlayingState else "▶"
        )
