"""RaveenCut — silence removal desktop app."""
import json, os, shutil, subprocess, tempfile

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
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(f"'{cmd[0]}' not found on PATH. Please install ffmpeg.")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    return result
def probe_video_duration_sec(video_path):
    if not shutil.which("ffprobe"):
        return None
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "json", video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(json.loads(result.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError, subprocess.CalledProcessError):
        return None
def read_audio_from_video(video_path, sampling_rate=SAMPLING_RATE):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = ["ffmpeg", "-i", video_path,
           "-f", "f32le", "-ac", "1", "-ar", str(sampling_rate), "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=10**6)
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
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True,
        )
        return "h264_nvenc" in result.stdout
    except FileNotFoundError:
        return False
def cut_segments_gpu(video_path, segments, tmpdir, progress_cb=None):
    outputs = []
    for i, seg in enumerate(segments):
        out = os.path.join(tmpdir, f"seg_{i:06d}.mp4")
        run(["ffmpeg", "-y", "-hwaccel", "cuda",
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
        run(["ffmpeg", "-y",
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
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
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
        self.progress.emit(5, "Extracting audio…")
        wav = read_audio_from_video(self.video_path)
        return wav, len(wav) / SAMPLING_RATE

    def _compute_probs(self, wav):
        self.progress.emit(20, "Loading VAD model…")
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
                self.progress.emit(pct, f"Detecting speech… {done}/{n_frames} frames")
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
            self.progress.emit(92, "Computing segments…")
            segs = probs_to_segments(probs, DEFAULT_THRESHOLD)
            segs = merge_segments(segs, min_gap=DEFAULT_MIN_SILENCE)
            segs = pad_segments(segs, pad=DEFAULT_PADDING, max_duration=duration)
            pcm = self._downsample_pcm(wav)
            self.progress.emit(100, "Done")
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
                def cb(done, total_segs):
                    pct = int(done / total_segs * 90)
                    self.progress.emit(pct, f"Cutting {done}/{total_segs}…")
                cut_fn = cut_segments_gpu if self.use_gpu else cut_segments_cpu
                seg_files = cut_fn(self.video_path, self.segments, tmpdir, cb)
                self.progress.emit(91, "Concatenating…")
                concat_files(seg_files, self.output_path, tmpdir)
                self.progress.emit(100, "Done")
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
        painter = QPainter(self)
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

    def _on_state(self, state):
        self._play_btn.setText(
            "⏸" if state == QMediaPlayer.PlaybackState.PlayingState else "▶"
        )


class ImportView(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self._drop_label = QLabel("Drop a video file here")
        self._drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_label.setStyleSheet(
            "border: 2px dashed #555; border-radius: 8px;"
            "font-size: 18px; color: #aaa; padding: 60px;"
        )
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._browse)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #e06060;")
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(self._drop_label)
        layout.addSpacing(12)
        layout.addWidget(self._browse_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._error_label)
        layout.addStretch()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video",
            filter="Video files (*.mp4 *.mov *.mkv *.avi *.webm)",
        )
        if path:
            self._handle_path(path)

    def _handle_path(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTS:
            self._error_label.setText(
                f"'{ext or path}' is not supported. Use: mp4, mov, mkv, avi, webm."
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


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    print("RaveenCut skeleton — OK")
    sys.exit(0)
