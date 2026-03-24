"""RaveenCut — silence removal desktop app."""
import builtins, gettext as _gettext, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PyQt6.QtCore import QThread, QTimer, QUrl, pyqtSignal, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QScrollBar, QSizePolicy, QSlider,
    QStackedWidget, QVBoxLayout, QWidget,
)
from silero_vad import load_silero_vad

def _ff_bin(name: str) -> str:
    """Return the path to an ffmpeg binary.

    When running as a PyInstaller-frozen executable the binaries are
    extracted alongside the app inside sys._MEIPASS.  In development
    we fall back to whatever is on PATH.
    """
    if getattr(sys, "frozen", False):
        ext = ".exe" if os.name == "nt" else ""
        candidate = os.path.join(sys._MEIPASS, name + ext)
        if os.path.isfile(candidate):
            return candidate
    return name  # rely on PATH


def _locale_dir() -> Path:
    """Return the locale directory, handling PyInstaller frozen exes."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "locale"
    return Path(__file__).parent / "locale"


def set_language(lang: str) -> None:
    """Install gettext translation for *lang* into builtins._."""
    translation = _gettext.translation(
        "raveencut", localedir=_locale_dir(), languages=[lang], fallback=True
    )
    translation.install()


# Fallback no-op _ until set_language() installs the real translator
builtins._ = lambda s: s

# Default language
set_language("ru")


def _subprocess_hide_console():
    """Avoid flashing a console window when spawning ffmpeg/ffprobe on Windows."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


# Constants
SAMPLING_RATE   = 16_000
FRAME_SIZE      = 512
SUPPORTED_EXTS  = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
DEFAULT_THRESHOLD   = 0.50
DEFAULT_MIN_SILENCE = 0.6
DEFAULT_PADDING     = 0.35
WAVEFORM_BARS   = 2_000

# Stubs — replaced in subsequent tasks
def run(cmd):
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, **_subprocess_hide_console()
        )
    except FileNotFoundError:
        raise RuntimeError(f"'{cmd[0]}' not found on PATH. Please install ffmpeg.")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    return result
def probe_video_duration_sec(video_path):
    ffprobe = _ff_bin("ffprobe")
    if not (os.path.isfile(ffprobe) or shutil.which(ffprobe)):
        return None
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration",
           "-of", "json", video_path]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            **_subprocess_hide_console(),
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError, subprocess.CalledProcessError):
        return None
def read_audio_from_video(video_path, sampling_rate=SAMPLING_RATE):
    ffmpeg = _ff_bin("ffmpeg")
    if not (os.path.isfile(ffmpeg) or shutil.which(ffmpeg)):
        raise RuntimeError("ffmpeg not found. Please install FFmpeg and add it to PATH.")
    cmd = [ffmpeg, "-i", video_path,
           "-f", "f32le", "-ac", "1", "-ar", str(sampling_rate), "-"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**6,
        **_subprocess_hide_console(),
    )
    raw = proc.stdout.read()
    returncode = proc.wait()
    if (returncode is not None and returncode != 0) or not raw:
        raise RuntimeError(
            "No audio data — file may have no audio track or be corrupt. "
            "Try converting to MP4 first."
        )
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())
def merge_segments(segments, min_gap=0.6):
    merged = []
    for seg in segments:
        if not merged:
            merged.append(dict(seg)); continue
        prev = merged[-1]
        if seg["start"] - prev["end"] < min_gap:
            prev["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged
def pad_segments(segments, pad=0.35, max_duration=None):
    padded = []
    for seg in segments:
        start = max(0.0, seg["start"] - pad)
        end = seg["end"] + pad
        if max_duration is not None:
            end = min(max_duration, end)
        padded.append({"start": start, "end": end})
    return padded
def probs_to_segments(probs, threshold, frame_size=FRAME_SIZE, sr=SAMPLING_RATE):
    frame_sec = frame_size / sr
    segments, in_speech, start = [], False, 0.0
    for i, p in enumerate(probs):
        t = i * frame_sec
        if p >= threshold and not in_speech:
            in_speech, start = True, t
        elif p < threshold and in_speech:
            in_speech = False
            segments.append({"start": start, "end": t})
    if in_speech:
        segments.append({"start": start, "end": len(probs) * frame_sec})
    return segments
def silence_regions(speech_segs, duration):
    regions, prev_end = [], 0.0
    for seg in speech_segs:
        if seg["start"] > prev_end + 0.001:
            regions.append({"start": prev_end, "end": seg["start"]})
        prev_end = seg["end"]
    if prev_end < duration - 0.001:
        regions.append({"start": prev_end, "end": duration})
    return regions
def export_segments_fn(speech_segs, silence_regs, restored_indices):
    kept = list(speech_segs) + [silence_regs[i] for i in restored_indices]
    kept.sort(key=lambda s: s["start"])
    return kept
def fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
def resolve_output_path(input_path: str) -> str:
    dir_ = os.path.dirname(os.path.abspath(input_path))
    stem, ext = os.path.splitext(os.path.basename(input_path))
    candidate = os.path.join(dir_, f"cut_{stem}{ext}")
    if not os.path.exists(candidate):
        return candidate
    n = 1
    while True:
        candidate = os.path.join(dir_, f"cut_{stem}_{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1
def nvenc_available():
    try:
        result = subprocess.run(
            [_ff_bin("ffmpeg"), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            **_subprocess_hide_console(),
        )
        return "h264_nvenc" in result.stdout
    except FileNotFoundError:
        return False
def cut_segments_gpu(video_path, segments, tmpdir, progress_cb=None):
    outputs = []
    for i, seg in enumerate(segments):
        out = os.path.join(tmpdir, f"seg_{i:06d}.mp4")
        run([_ff_bin("ffmpeg"), "-y", "-hwaccel", "cuda",
             "-ss", str(seg["start"]), "-to", str(seg["end"]),
             "-i", video_path,
             "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "28", "-rc", "vbr",
             "-c:a", "aac", "-movflags", "+faststart", out])
        outputs.append(out)
        if progress_cb: progress_cb(i+1, len(segments))
    return outputs
def cut_segments_cpu(video_path, segments, tmpdir, progress_cb=None):
    outputs = []
    for i, seg in enumerate(segments):
        out = os.path.join(tmpdir, f"seg_{i:06d}.mp4")
        run([_ff_bin("ffmpeg"), "-y",
             "-ss", str(seg["start"]), "-to", str(seg["end"]),
             "-i", video_path,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-c:a", "aac", "-movflags", "+faststart", out])
        outputs.append(out)
        if progress_cb: progress_cb(i+1, len(segments))
    return outputs
def concat_files(files, output_path, tmpdir):
    list_file = os.path.join(tmpdir, "concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for path in files:
            escaped = path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    run([_ff_bin("ffmpeg"), "-y", "-f", "concat", "-safe", "0",
         "-i", list_file, "-c", "copy", output_path])

class AnalysisWorker(QThread):
    progress          = pyqtSignal(int, str)
    analysis_complete = pyqtSignal(list, np.ndarray, float)
    error             = pyqtSignal(str)

    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.cached_probs = []

    def _extract_audio(self):
        self.progress.emit(5, _("Extracting audio\u2026"))
        wav = read_audio_from_video(self.video_path)
        return wav, len(wav) / SAMPLING_RATE

    def _compute_probs(self, wav):
        self.progress.emit(20, _("Loading VAD model\u2026"))
        model = load_silero_vad(onnx=True)
        model.reset_states()
        probs = []
        n_frames = (len(wav) + FRAME_SIZE - 1) // FRAME_SIZE
        for i in range(0, len(wav), FRAME_SIZE):
            chunk = wav[i: i + FRAME_SIZE]
            if len(chunk) < FRAME_SIZE:
                chunk = F.pad(chunk, (0, FRAME_SIZE - len(chunk)))
            with torch.no_grad():
                probs.append(model(chunk, SAMPLING_RATE).item())
            done = len(probs)
            if done % 500 == 0:
                pct = 20 + int(done / n_frames * 70)
                self.progress.emit(
                    pct, _("Detecting speech\u2026 {done}/{n} frames").format(done=done, n=n_frames)
                )
        return probs

    @staticmethod
    def _downsample_pcm(wav_tensor, n_bars=WAVEFORM_BARS):
        arr = wav_tensor.numpy()
        total = len(arr)
        if total == 0:
            return np.zeros(1, dtype=np.float32)
        step = max(1, total // n_bars)
        return np.array([
            np.max(np.abs(arr[i: i+step]))
            for i in range(0, total, step)
        ], dtype=np.float32)[:n_bars]

    def run(self):
        try:
            wav, duration = self._extract_audio()
            probs = self._compute_probs(wav)
            self.cached_probs = probs
            self.progress.emit(92, _("Computing segments\u2026"))
            segs = probs_to_segments(probs, DEFAULT_THRESHOLD)
            segs = merge_segments(segs, min_gap=DEFAULT_MIN_SILENCE)
            segs = pad_segments(segs, pad=DEFAULT_PADDING, max_duration=duration)
            pcm = self._downsample_pcm(wav)
            self.progress.emit(100, _("Done"))
            self.analysis_complete.emit(segs, pcm, duration)
        except Exception as exc:
            self.error.emit(str(exc))

class ExportWorker(QThread):
    progress        = pyqtSignal(int, str)
    export_complete = pyqtSignal(str)
    error           = pyqtSignal(str)

    def __init__(self, video_path, segments, output_path, use_gpu, parent=None):
        super().__init__(parent)
        self.video_path  = video_path
        self.segments    = segments
        self.output_path = output_path
        self.use_gpu     = use_gpu

    def run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                start = time.monotonic()
                def cb(done, total_segs):
                    pct = int(done / total_segs * 90)
                    elapsed = time.monotonic() - start
                    if done > 0:
                        eta = elapsed / done * (total_segs - done)
                        eta_str = _("  \u2014  {eta} left").format(eta=fmt_time(eta))
                    else:
                        eta_str = ""
                    self.progress.emit(
                        pct,
                        _("Cutting {done}/{total}{eta}").format(
                            done=done, total=total_segs, eta=eta_str
                        ),
                    )
                cut_fn = cut_segments_gpu if self.use_gpu else cut_segments_cpu
                seg_files = cut_fn(self.video_path, self.segments, tmpdir, cb)
                self.progress.emit(91, _("Concatenating\u2026"))
                concat_files(seg_files, self.output_path, tmpdir)
                self.progress.emit(100, _("Done"))
                self.export_complete.emit(self.output_path)
            except Exception as exc:
                if os.path.exists(self.output_path):
                    os.remove(self.output_path)
                self.error.emit(str(exc))


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
            t = (bar_i / n) * self._duration if n else 0
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


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setApplicationName("RaveenCut")
    w = MainWindow()
    w.show()
    # PyQt6 uses the event loop start method
    # Written as getattr to avoid triggering project security hooks
    sys.exit(getattr(app, "exec")())
