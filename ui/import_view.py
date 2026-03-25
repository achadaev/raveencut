import os

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from core.constants import SUPPORTED_EXTS


class ImportView(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self._drop_label = QLabel("")
        self._drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_label.setStyleSheet(
            "border: 2px dashed #555; border-radius: 8px;"
            "font-size: 18px; color: #aaa; padding: 60px;"
        )
        self._browse_btn = QPushButton("")
        self._browse_btn.clicked.connect(self._browse)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #e06060;")
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error_label.setVisible(False)

        self._lang_btn = QPushButton("EN")
        self._lang_btn.setFixedWidth(40)

        top_bar = QHBoxLayout()
        top_bar.addStretch()
        top_bar.addWidget(self._lang_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addStretch()
        layout.addWidget(self._drop_label)
        layout.addSpacing(12)
        layout.addWidget(self._browse_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._error_label)
        layout.addStretch()

        self.retranslate()

    def _browse(self):
        path, __ = QFileDialog.getOpenFileName(
            self, _("Open Video"),
            filter=_("Video files (*.mp4 *.mov *.mkv *.avi *.webm)"),
        )
        if path:
            self._handle_path(path)

    def _handle_path(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTS:
            self._error_label.setText(
                _("'{name}' is not supported. Use: mp4, mov, mkv, avi, webm.").format(
                    name=ext or path
                )
            )
            self._error_label.setVisible(True)
            return
        self._error_label.setVisible(False)
        self.file_selected.emit(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            self._handle_path(url.toLocalFile())
            break

    def retranslate(self):
        self._drop_label.setText(_("Drop a video file here"))
        self._browse_btn.setText(_("Browse\u2026"))
