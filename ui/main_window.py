import os
import subprocess

import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QScrollBar, QSizePolicy, QSlider,
    QStackedWidget, QVBoxLayout, QWidget,
)

from core.constants import DEFAULT_MIN_SILENCE, DEFAULT_PADDING, DEFAULT_THRESHOLD
from core.exporter import nvenc_available
from core.i18n import set_language
from core.segments import (
    export_segments_fn, merge_segments, pad_segments,
    probs_to_segments, silence_regions,
)
from core.utils import fmt_time, resolve_output_path
from ui.import_view import ImportView
from ui.video_player import VideoPlayerWidget
from ui.waveform import WaveformWidget
from ui.workers import AnalysisWorker, ExportWorker

class MainView(QWidget):
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path = ""
        self._speech_segs = []
        self._silence_segs = []
        self._restored_indices: set = set()
        self._cached_probs = []
        self._analysis_worker = None
        self._export_worker = None
        self._pending_chip = None
        self._pending_chip_idx = -1
        self._chips = []
        # Format strings — set by retranslate(), defaults prevent AttributeError on init
        self._thr_fmt = "Threshold: {v:.2f}"
        self._sil_fmt = "Min silence: {v:.1f}s"
        self._pad_fmt = "Padding: {v:.2f}s"
        self._stats_fmt = "Kept: {kept} / {total} ({pct:.0f}%)\nSegments: {n}"
        self._restore_fmt = "restore {t}"

        # --- Header ---
        self._title_label = QLabel("RaveenCut")
        self._title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self._back_btn = QPushButton("")
        self._back_btn.clicked.connect(self._on_back)
        header = QHBoxLayout()
        header.addWidget(self._back_btn)
        header.addWidget(self._title_label)
        header.addStretch()
        self._lang_btn = QPushButton("EN")
        self._lang_btn.setFixedWidth(40)
        header.addWidget(self._lang_btn)

        # --- Video player ---
        self._video_player = VideoPlayerWidget()

        # --- Controls panel (fixed 240px) ---
        self._thr_slider = QSlider(Qt.Orientation.Horizontal)
        self._thr_slider.setRange(10, 90); self._thr_slider.setValue(50)
        self._thr_label = QLabel("")

        self._sil_slider = QSlider(Qt.Orientation.Horizontal)
        self._sil_slider.setRange(1, 20); self._sil_slider.setValue(6)
        self._sil_label = QLabel("")

        self._pad_slider = QSlider(Qt.Orientation.Horizontal)
        self._pad_slider.setRange(0, 20); self._pad_slider.setValue(7)
        self._pad_label = QLabel("")

        self._stats_label = QLabel("")
        self._stats_label.setWordWrap(True)

        for sl in (self._thr_slider, self._sil_slider, self._pad_slider):
            sl.valueChanged.connect(self._update_slider_labels)
            sl.sliderReleased.connect(self._on_slider_released)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self._thr_label)
        controls_layout.addWidget(self._thr_slider)
        controls_layout.addWidget(self._sil_label)
        controls_layout.addWidget(self._sil_slider)
        controls_layout.addWidget(self._pad_label)
        controls_layout.addWidget(self._pad_slider)
        controls_layout.addWidget(self._stats_label)
        controls_layout.addStretch()
        controls_widget = QWidget(); controls_widget.setFixedWidth(240)
        controls_widget.setLayout(controls_layout)

        # --- Top row ---
        top_row = QHBoxLayout()
        top_row.addWidget(self._video_player, stretch=1)
        top_row.addWidget(controls_widget)

        # --- Waveform stack ---
        self._waveform = WaveformWidget()
        self._waveform.seek_requested.connect(self._video_player.seek)
        self._video_player.position_changed.connect(self._waveform.set_playhead)

        self._scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self._scrollbar.setRange(0, 1000)
        self._scrollbar.valueChanged.connect(lambda v: self._waveform.set_scroll(v / 1000))

        waveform_layout = QVBoxLayout()
        waveform_layout.setContentsMargins(0, 0, 0, 0)
        waveform_layout.addWidget(self._waveform)
        waveform_layout.addWidget(self._scrollbar)
        waveform_page = QWidget()
        waveform_page.setLayout(waveform_layout)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout = QVBoxLayout()
        progress_layout.addStretch()
        progress_layout.addWidget(self._status_label)
        progress_layout.addWidget(self._progress_bar)
        progress_layout.addStretch()
        progress_page = QWidget()
        progress_page.setLayout(progress_layout)

        self._waveform_stack = QStackedWidget()
        self._waveform_stack.addWidget(waveform_page)   # index 0
        self._waveform_stack.addWidget(progress_page)   # index 1

        # --- Warning label ---
        self._warning_label = QLabel("")
        self._warning_label.setStyleSheet("background: #7a6a00; color: #ffee88; padding: 4px;")
        self._warning_label.setVisible(False)

        # --- Chips strip ---
        self._chips_header = QLabel("")
        self._chips_header.setStyleSheet("color: #888; font-size: 11px;")
        self._chips_layout = QHBoxLayout()
        self._chips_layout.setContentsMargins(4, 2, 4, 2)
        self._chips_layout.addStretch()
        chips_inner = QWidget()
        chips_inner.setLayout(self._chips_layout)
        chips_scroll = QScrollArea()
        chips_scroll.setWidget(chips_inner)
        chips_scroll.setWidgetResizable(True)
        chips_scroll.setFixedHeight(46)
        chips_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        chips_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # --- Export row ---
        self._export_btn = QPushButton("")
        self._export_btn.setEnabled(False)
        self._export_progress = QProgressBar()
        self._export_progress.setRange(0, 100)
        self._export_progress.setVisible(False)
        self._export_label = QLabel("")
        export_row = QHBoxLayout()
        export_row.addWidget(self._export_btn)
        export_row.addWidget(self._export_progress)
        export_row.addWidget(self._export_label)
        export_row.addStretch()
        self._export_btn.clicked.connect(self._on_export)

        # --- Main layout ---
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(header)
        main_layout.addLayout(top_row, stretch=1)
        main_layout.addWidget(self._waveform_stack)
        main_layout.addWidget(self._warning_label)
        main_layout.addWidget(self._chips_header)
        main_layout.addWidget(chips_scroll)
        main_layout.addLayout(export_row)

        self.retranslate()

    # -- Public API --

    def start_analysis(self, path: str):
        self._video_path = path
        self._title_label.setText(os.path.basename(path))
        self._video_player.load(path)
        self._waveform_stack.setCurrentIndex(1)
        self._export_btn.setEnabled(False)
        for sl in (self._thr_slider, self._sil_slider, self._pad_slider):
            sl.setEnabled(False)
        self._analysis_worker = AnalysisWorker(path)
        self._analysis_worker.progress.connect(self._on_analysis_progress)
        self._analysis_worker.analysis_complete.connect(self._on_analysis_complete)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.start()

    def load_analysis(self, speech_segs, pcm, duration, cached_probs):
        self._speech_segs = speech_segs
        self._silence_segs = silence_regions(speech_segs, duration)
        self._restored_indices = set()
        self._cached_probs = cached_probs
        self._pcm = pcm
        self._duration = duration
        self._rebuild_waveform()
        self._rebuild_chips()
        self._waveform_stack.setCurrentIndex(0)
        self._export_btn.setEnabled(bool(speech_segs))
        for sl in (self._thr_slider, self._sil_slider, self._pad_slider):
            sl.setEnabled(True)
        self._update_stats()

    # -- Slots --

    def _on_analysis_progress(self, pct, msg):
        self._progress_bar.setValue(pct)
        self._status_label.setText(msg)

    def _on_analysis_complete(self, segs, pcm, duration):
        worker = self._analysis_worker
        self._analysis_worker = None
        cached_probs = worker.cached_probs if worker else []
        self.load_analysis(segs, pcm, duration, cached_probs)

    def _on_analysis_error(self, msg):
        self._analysis_worker = None
        self._waveform_stack.setCurrentIndex(0)
        self._warning_label.setText(_("Analysis failed: ") + msg)
        self._warning_label.setVisible(True)

    def _on_slider_released(self):
        if self._analysis_worker is not None:
            return
        thr = self._thr_slider.value() / 100
        min_sil = self._sil_slider.value() / 10
        pad = self._pad_slider.value() / 20
        duration = getattr(self, "_duration", 0.0)
        segs = probs_to_segments(self._cached_probs, thr)
        segs = merge_segments(segs, min_gap=min_sil)
        segs = pad_segments(segs, pad=pad, max_duration=duration)
        self._speech_segs = segs
        self._silence_segs = silence_regions(segs, duration)
        self._restored_indices = set()
        self._rebuild_waveform()
        self._rebuild_chips()
        self._update_stats()
        self._export_btn.setEnabled(bool(segs))

    def _update_slider_labels(self):
        self._thr_label.setText(self._thr_fmt.format(v=self._thr_slider.value() / 100))
        self._sil_label.setText(self._sil_fmt.format(v=self._sil_slider.value() / 10))
        self._pad_label.setText(self._pad_fmt.format(v=self._pad_slider.value() / 20))

    def _rebuild_waveform(self):
        pcm = getattr(self, "_pcm", None)
        if pcm is None:
            pcm = np.zeros(1, dtype=np.float32)
        self._waveform.update_audio(pcm, getattr(self, "_duration", 1.0))
        self._waveform.update_segments(self._speech_segs, self._silence_segs, self._restored_indices)

    def _rebuild_chips(self):
        # Clear existing chips
        while self._chips_layout.count() > 1:  # keep the trailing stretch
            item = self._chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._chips = []
        self._pending_chip = None
        self._pending_chip_idx = -1
        for i, seg in enumerate(self._silence_segs):
            label = f"{fmt_time(seg['start'])}-{fmt_time(seg['end'])}"
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            idx = i
            btn.clicked.connect(lambda checked, i=idx, b=btn: self._on_chip_click(i, b))
            self._chips_layout.insertWidget(self._chips_layout.count() - 1, btn)
            self._chips.append(btn)

    def _on_chip_click(self, idx, chip):
        if self._pending_chip_idx == idx:
            # Second click: restore
            self._restored_indices.add(idx)
            if chip in self._chips:
                self._chips.remove(chip)
            chip.setParent(None)
            chip.deleteLater()
            self._pending_chip = None
            self._pending_chip_idx = -1
            self._rebuild_waveform()
        else:
            # First click: seek and mark pending
            if self._pending_chip is not None:
                self._pending_chip.setChecked(False)
                orig_label = f"{fmt_time(self._silence_segs[self._pending_chip_idx]['start'])}-{fmt_time(self._silence_segs[self._pending_chip_idx]['end'])}"
                self._pending_chip.setText(orig_label)
            self._pending_chip = chip
            self._pending_chip_idx = idx
            seg = self._silence_segs[idx]
            self._video_player.seek(seg["start"])
            chip.setChecked(True)
            chip.setText(
                self._restore_fmt.format(t=f"{fmt_time(seg['start'])}-{fmt_time(seg['end'])}")
            )

    def _update_stats(self):
        total = getattr(self, "_duration", 0.0)
        kept = sum(s["end"] - s["start"] for s in self._speech_segs)
        kept += sum(self._silence_segs[i]["end"] - self._silence_segs[i]["start"]
                    for i in self._restored_indices
                    if i < len(self._silence_segs))
        pct = (kept / total * 100) if total > 0 else 0
        self._stats_label.setText(
            self._stats_fmt.format(
                kept=fmt_time(kept),
                total=fmt_time(total),
                pct=pct,
                n=len(self._speech_segs),
            )
        )

    def _on_export(self):
        if not os.path.isdir(os.path.dirname(os.path.abspath(self._video_path))):
            QMessageBox.warning(self, _("Error"), _("Cannot determine output directory."))
            return
        output = resolve_output_path(self._video_path)
        use_gpu = nvenc_available()
        self._export_worker = ExportWorker(
            self._video_path,
            export_segments_fn(self._speech_segs, self._silence_segs, self._restored_indices),
            output, use_gpu,
        )
        self._export_worker.progress.connect(lambda p, m: (
            self._export_progress.setValue(p),
            self._export_label.setText(m),
        ))
        self._export_worker.export_complete.connect(self._on_export_complete)
        self._export_worker.error.connect(self._on_export_error)
        self._export_btn.setVisible(False)
        self._export_progress.setVisible(True)
        self._back_btn.setEnabled(False)
        self._export_worker.start()

    def _on_export_complete(self, path):
        self._export_worker = None
        self._export_progress.setVisible(False)
        self._export_btn.setVisible(True)
        self._back_btn.setEnabled(True)
        dir_ = os.path.dirname(os.path.abspath(path))
        msg = QMessageBox(self)
        msg.setWindowTitle(_("Export complete"))
        msg.setText(_("Saved to:\n{path}").format(path=path))
        open_file_btn = msg.addButton(_("Open File"), QMessageBox.ButtonRole.ActionRole)
        open_folder_btn = msg.addButton(_("Open Folder"), QMessageBox.ButtonRole.ActionRole)
        msg.addButton(_("Close"), QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        if msg.clickedButton() == open_file_btn:
            subprocess.Popen(["explorer", "/select,", path.replace("/", "\\")])
        elif msg.clickedButton() == open_folder_btn:
            subprocess.Popen(["explorer", dir_.replace("/", "\\")])

    def _on_export_error(self, err_msg):
        self._export_worker = None
        self._export_progress.setVisible(False)
        self._export_btn.setVisible(True)
        self._back_btn.setEnabled(True)
        msg = QMessageBox(self)
        msg.setWindowTitle(_("Export failed"))
        msg.setText(err_msg)
        retry_btn = msg.addButton(_("Retry"), QMessageBox.ButtonRole.AcceptRole)
        msg.addButton(_("Cancel"), QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        if msg.clickedButton() == retry_btn:
            self._on_export()

    def closeEvent(self, event):
        self._waveform.hide()
        super().closeEvent(event)

    def _on_back(self):
        if self._analysis_worker is not None:
            self._analysis_worker.terminate()
            self._analysis_worker.wait()
            self._analysis_worker = None
        self.back_requested.emit()

    def retranslate(self):
        self._back_btn.setText(_("<- Back"))
        self._chips_header.setText(
            _("Silence regions (click to preview, click again to restore):")
        )
        self._status_label.setText(_("Analyzing..."))
        self._export_btn.setText(_("Export"))
        self._thr_fmt = _("Threshold: {v:.2f}")
        self._sil_fmt = _("Min silence: {v:.1f}s")
        self._pad_fmt = _("Padding: {v:.2f}s")
        self._stats_fmt = _("Kept: {kept} / {total} ({pct:.0f}%)\nSegments: {n}")
        self._restore_fmt = _("restore {t}")
        self._update_slider_labels()
        self._update_stats()
        if hasattr(self, "_silence_segs"):
            self._rebuild_chips()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RaveenCut")
        self.setMinimumSize(1100, 720)
        self._import_view = ImportView()
        self._main_view   = MainView()
        self._stack = QStackedWidget()
        self._stack.addWidget(self._import_view)
        self._stack.addWidget(self._main_view)
        self.setCentralWidget(self._stack)
        self._import_view.file_selected.connect(self._on_file_selected)
        self._main_view.back_requested.connect(self._on_back)
        self._import_view._lang_btn.clicked.connect(self._on_lang_toggle)
        self._main_view._lang_btn.clicked.connect(self._on_lang_toggle)

    def _on_lang_toggle(self):
        import builtins as _b

        current_ru = _b._("Export") == "Экспорт"
        new_lang = "en" if current_ru else "ru"
        set_language(new_lang)
        btn_label = "RU" if new_lang == "en" else "EN"
        self._import_view._lang_btn.setText(btn_label)
        self._main_view._lang_btn.setText(btn_label)
        self._import_view.retranslate()
        self._main_view.retranslate()

    def _on_file_selected(self, path: str):
        self._stack.setCurrentIndex(1)
        self._main_view.start_analysis(path)

    def _on_back(self):
        self._stack.setCurrentIndex(0)

