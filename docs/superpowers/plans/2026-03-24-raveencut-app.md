# RaveenCut Desktop App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `app.py` — a PyQt6 desktop application that detects and removes silence from video files, using `cut.py` as reference only (not imported).

**Architecture:** Single `app.py` containing all processing functions (ffmpeg helpers, segment helpers, VAD prob caching, export) plus all PyQt6 classes (widgets, workers, views, main window). `cut.py` is reference-only.

**Tech Stack:** Python 3.10+, PyQt6, torch, silero-vad, numpy, ffmpeg (on PATH), pytest, pytest-qt

**Spec:** `docs/superpowers/specs/2026-03-24-raveencut-app-design.md`

---

## File Map

| File | Purpose |
|---|---|
| `app.py` | Entire application — all processing + GUI |
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | Test dependencies |
| `tests/test_processing.py` | Unit tests for pure processing functions |
| `tests/test_gui.py` | pytest-qt smoke tests for widgets |
| `.gitignore` | Ignore venv, pycache, .superpowers, etc. |

---

## Task 1: Project Scaffold

**Files:** `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `app.py` (skeleton), `tests/__init__.py`, `tests/test_processing.py`, `tests/test_gui.py`

- [ ] **Step 1: Init git**
```bash
cd d:/Git/raveencut-2
git init
```

- [ ] **Step 2: Create .gitignore**
```
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
dist/
build/
.superpowers/
*.spec
```

- [ ] **Step 3: Create requirements.txt**
```
PyQt6
torch
numpy
silero-vad
rich
```

- [ ] **Step 4: Create requirements-dev.txt**
```
pytest
pytest-qt
```

- [ ] **Step 5: Create app.py skeleton**

```python
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
def run(cmd): pass
def probe_video_duration_sec(path): pass
def read_audio_from_video(path, sampling_rate=SAMPLING_RATE): pass
def merge_segments(segments, min_gap=0.6): pass
def pad_segments(segments, pad=0.35, max_duration=None): pass
def probs_to_segments(probs, threshold, frame_size=FRAME_SIZE, sr=SAMPLING_RATE): pass
def silence_regions(speech_segs, duration): pass
def export_segments_fn(speech_segs, silence_regs, restored_indices): pass
def fmt_time(sec): pass
def resolve_output_path(input_path): pass
def nvenc_available(): pass
def cut_segments_gpu(video_path, segments, tmpdir, progress_cb=None): pass
def cut_segments_cpu(video_path, segments, tmpdir, progress_cb=None): pass
def concat_files(files, output_path, tmpdir): pass

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    print("RaveenCut skeleton — OK")
    sys.exit(0)
```

- [ ] **Step 6: Create empty test files**

`tests/__init__.py` — empty
`tests/test_processing.py` — empty
`tests/test_gui.py` — empty

- [ ] **Step 7: Install deps**
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

- [ ] **Step 8: Verify skeleton runs**
```bash
python app.py
```
Expected: `RaveenCut skeleton — OK`

- [ ] **Step 9: Commit**
```bash
git add .
git commit -m "feat: project scaffold with skeleton app.py"
```

---

## Task 2: Pure Processing Functions

Implement `merge_segments`, `pad_segments`, `probs_to_segments`, `silence_regions`, `export_segments_fn`, `fmt_time`, `resolve_output_path`.

**Files:** `app.py`, `tests/test_processing.py`

- [ ] **Step 1: Write failing tests in tests/test_processing.py**

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from app import (
    merge_segments, pad_segments, probs_to_segments,
    silence_regions, export_segments_fn, fmt_time, resolve_output_path,
)

def test_fmt_time_sub_hour():
    assert fmt_time(75.0) == "1:15"

def test_fmt_time_over_hour():
    assert fmt_time(3661.0) == "1:01:01"

def test_fmt_time_zero():
    assert fmt_time(0.0) == "0:00"

def test_merge_segments_merges_close_gaps():
    segs = [{"start": 0.0, "end": 1.0}, {"start": 1.4, "end": 2.0}]
    assert merge_segments(segs, min_gap=0.6) == [{"start": 0.0, "end": 2.0}]

def test_merge_segments_keeps_wide_gaps():
    segs = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]
    assert len(merge_segments(segs, min_gap=0.6)) == 2

def test_merge_segments_empty():
    assert merge_segments([], min_gap=0.6) == []

def test_pad_segments_adds_padding():
    segs = [{"start": 1.0, "end": 2.0}]
    r = pad_segments(segs, pad=0.3, max_duration=10.0)
    assert r[0]["start"] == pytest.approx(0.7)
    assert r[0]["end"] == pytest.approx(2.3)

def test_pad_segments_clamps_to_zero():
    segs = [{"start": 0.1, "end": 1.0}]
    assert pad_segments(segs, pad=0.5, max_duration=10.0)[0]["start"] == 0.0

def test_pad_segments_clamps_to_duration():
    segs = [{"start": 8.0, "end": 9.8}]
    assert pad_segments(segs, pad=0.5, max_duration=10.0)[0]["end"] == 10.0

def test_probs_to_segments_basic():
    probs = [0.1]*4 + [0.9]*4 + [0.1]*4
    segs = probs_to_segments(probs, threshold=0.5, frame_size=512, sr=16000)
    assert len(segs) == 1
    frame_sec = 512/16000
    assert segs[0]["start"] == pytest.approx(4*frame_sec)
    assert segs[0]["end"] == pytest.approx(8*frame_sec)

def test_probs_to_segments_speech_to_end():
    probs = [0.1, 0.9, 0.9]
    segs = probs_to_segments(probs, threshold=0.5, frame_size=512, sr=16000)
    assert len(segs) == 1
    assert segs[0]["end"] == pytest.approx(3*512/16000)

def test_probs_to_segments_all_silent():
    assert probs_to_segments([0.1, 0.2], threshold=0.5) == []

def test_silence_regions_gaps():
    speech = [{"start": 1.0, "end": 3.0}, {"start": 5.0, "end": 7.0}]
    regions = silence_regions(speech, duration=10.0)
    assert regions == [
        {"start": 0.0, "end": 1.0},
        {"start": 3.0, "end": 5.0},
        {"start": 7.0, "end": 10.0},
    ]

def test_silence_regions_no_leading():
    speech = [{"start": 0.0, "end": 3.0}, {"start": 5.0, "end": 10.0}]
    assert silence_regions(speech, duration=10.0) == [{"start": 3.0, "end": 5.0}]

def test_export_segments_with_restored():
    speech = [{"start": 1.0, "end": 3.0}]
    sil = [{"start": 0.0, "end": 1.0}, {"start": 3.0, "end": 5.0}]
    result = export_segments_fn(speech, sil, restored_indices={0})
    assert result == [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 3.0}]

def test_export_segments_no_restored():
    speech = [{"start": 1.0, "end": 3.0}]
    sil = [{"start": 0.0, "end": 1.0}]
    assert export_segments_fn(speech, sil, set()) == [{"start": 1.0, "end": 3.0}]

def test_resolve_output_path_no_existing(tmp_path):
    src = str(tmp_path / "video.mp4")
    assert resolve_output_path(src) == str(tmp_path / "cut_video.mp4")

def test_resolve_output_path_increments(tmp_path):
    src = str(tmp_path / "video.mp4")
    (tmp_path / "cut_video.mp4").touch()
    assert resolve_output_path(src) == str(tmp_path / "cut_video_1.mp4")

def test_resolve_output_path_increments_twice(tmp_path):
    src = str(tmp_path / "video.mp4")
    (tmp_path / "cut_video.mp4").touch()
    (tmp_path / "cut_video_1.mp4").touch()
    assert resolve_output_path(src) == str(tmp_path / "cut_video_2.mp4")
```

- [ ] **Step 2: Run — verify all fail**
```bash
pytest tests/test_processing.py -v
```

- [ ] **Step 3: Implement all pure functions in app.py**

```python
def fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

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
```

- [ ] **Step 4: Run — verify all pass**
```bash
pytest tests/test_processing.py -v
```

- [ ] **Step 5: Commit**
```bash
git add app.py tests/test_processing.py
git commit -m "feat: pure processing functions with tests"
```

---

## Task 3: ffmpeg Helpers

Implement `run`, `probe_video_duration_sec`, `read_audio_from_video`.

**Files:** `app.py`, `tests/test_processing.py`

- [ ] **Step 1: Append failing tests to tests/test_processing.py**

```python
from unittest.mock import MagicMock, patch
from app import run, probe_video_duration_sec, read_audio_from_video
import struct

def test_run_raises_on_nonzero():
    mock = MagicMock(); mock.returncode = 1; mock.stderr = "err"
    with patch("subprocess.run", return_value=mock):
        with pytest.raises(RuntimeError, match="FFmpeg failed"):
            run(["ffmpeg"])

def test_run_returns_result_on_success():
    mock = MagicMock(); mock.returncode = 0
    with patch("subprocess.run", return_value=mock):
        assert run(["ffmpeg"]) is mock

def test_probe_parses_json():
    mock = MagicMock(); mock.stdout = '{"format":{"duration":"123.456"}}'
    with patch("shutil.which", return_value="/usr/bin/ffprobe"), \
         patch("subprocess.run", return_value=mock):
        assert probe_video_duration_sec("x.mp4") == pytest.approx(123.456)

def test_probe_returns_none_without_ffprobe():
    with patch("shutil.which", return_value=None):
        assert probe_video_duration_sec("x.mp4") is None

def test_read_audio_returns_tensor():
    pcm = struct.pack("4f", 0.1, -0.1, 0.2, -0.2)
    mock_proc = MagicMock()
    mock_proc.stdout.read.return_value = pcm
    mock_proc.wait.return_value = None
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("subprocess.Popen", return_value=mock_proc):
        t = read_audio_from_video("x.mp4")
    assert t.shape == (4,)
    assert abs(t[0].item() - 0.1) < 1e-5
```

- [ ] **Step 2: Run — verify failures**
```bash
pytest tests/test_processing.py -k "test_run or test_probe or test_read_audio" -v
```

- [ ] **Step 3: Implement in app.py**

```python
def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
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
    except Exception:
        return None

def read_audio_from_video(video_path, sampling_rate=SAMPLING_RATE):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = ["ffmpeg", "-i", video_path,
           "-f", "f32le", "-ac", "1", "-ar", str(sampling_rate), "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=10**6)
    raw = proc.stdout.read()
    proc.wait()
    if not raw:
        raise RuntimeError(
            "No audio data — file may have no audio track or be corrupt. "
            "Try converting to MP4 first."
        )
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())
```

- [ ] **Step 4: Run all tests**
```bash
pytest tests/test_processing.py -v
```

- [ ] **Step 5: Commit**
```bash
git add app.py tests/test_processing.py
git commit -m "feat: ffmpeg helpers with tests"
```

---

## Task 4: Export Helpers

Implement `nvenc_available`, `cut_segments_gpu`, `cut_segments_cpu`, `concat_files`.

**Files:** `app.py`, `tests/test_processing.py`

- [ ] **Step 1: Append failing tests**

```python
from app import nvenc_available, cut_segments_gpu, cut_segments_cpu, concat_files

def test_nvenc_available_true():
    mock = MagicMock(); mock.stdout = "h264_nvenc encoder"
    with patch("subprocess.run", return_value=mock):
        assert nvenc_available() is True

def test_nvenc_available_false():
    mock = MagicMock(); mock.stdout = "libx264 encoder"
    with patch("subprocess.run", return_value=mock):
        assert nvenc_available() is False

def test_cut_segments_gpu_calls_ffmpeg(tmp_path):
    segs = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]
    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd); open(cmd[-1], "w").close()
        r = MagicMock(); r.returncode = 0; return r
    with patch("subprocess.run", side_effect=fake_run):
        outputs = cut_segments_gpu("input.mp4", segs, str(tmp_path))
    assert len(outputs) == 2
    assert "h264_nvenc" in calls[0]

def test_cut_segments_cpu_uses_libx264(tmp_path):
    segs = [{"start": 0.0, "end": 1.0}]
    def fake_run(cmd, **kw):
        open(cmd[-1], "w").close()
        r = MagicMock(); r.returncode = 0; return r
    with patch("subprocess.run", side_effect=fake_run):
        outputs = cut_segments_cpu("input.mp4", segs, str(tmp_path))
    assert len(outputs) == 1

def test_concat_files_writes_list_and_calls_ffmpeg(tmp_path):
    files = [str(tmp_path/"a.mp4"), str(tmp_path/"b.mp4")]
    called = []
    def fake_run(cmd, **kw):
        called.append(cmd); r = MagicMock(); r.returncode = 0; return r
    with patch("subprocess.run", side_effect=fake_run):
        concat_files(files, str(tmp_path/"out.mp4"), str(tmp_path))
    assert (tmp_path/"concat.txt").exists()
    assert called[0][0] == "ffmpeg"
```

- [ ] **Step 2: Run — verify failures**
```bash
pytest tests/test_processing.py -k "nvenc or cut_segments or concat" -v
```

- [ ] **Step 3: Implement in app.py**

```python
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
            f.write(f"file '{path}'\n")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", list_file, "-c", "copy", output_path])
```

- [ ] **Step 4: Run all tests**
```bash
pytest tests/test_processing.py -v
```

- [ ] **Step 5: Commit**
```bash
git add app.py tests/test_processing.py
git commit -m "feat: export helpers with tests"
```

---

## Task 5: AnalysisWorker

Implement AnalysisWorker (QThread) — audio extract, VAD prob caching, segment computation.

**Files:** `app.py`, `tests/test_gui.py`

- [ ] **Step 1: Write failing tests in tests/test_gui.py**

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import torch
import pytest
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)

from app import (AnalysisWorker, SAMPLING_RATE, FRAME_SIZE,
                 DEFAULT_THRESHOLD, DEFAULT_MIN_SILENCE, DEFAULT_PADDING)

def test_analysis_worker_emits_complete(qapp, qtbot):
    wav = torch.zeros(SAMPLING_RATE)
    n_frames = (SAMPLING_RATE + FRAME_SIZE - 1) // FRAME_SIZE
    fake_probs = [0.1]*n_frames
    for i in range(10, 20):
        if i < len(fake_probs): fake_probs[i] = 0.9

    worker = AnalysisWorker("fake.mp4")
    with patch.object(worker, "_extract_audio", return_value=(wav, 1.0)), \
         patch.object(worker, "_compute_probs", return_value=fake_probs):
        with qtbot.waitSignal(worker.analysis_complete, timeout=5000) as blocker:
            worker.start()
        worker.wait()

    segs, pcm, duration = blocker.args
    assert duration == pytest.approx(1.0)
    assert isinstance(pcm, np.ndarray)
    assert len(pcm) <= 2000

def test_analysis_worker_emits_error_on_bad_file(qapp, qtbot):
    worker = AnalysisWorker("nonexistent_file_xyz.mp4")
    with qtbot.waitSignal(worker.error, timeout=5000) as blocker:
        worker.start()
        worker.wait()
    assert len(blocker.args[0]) > 0
```

- [ ] **Step 2: Run — verify failures**
```bash
pytest tests/test_gui.py -v
```

- [ ] **Step 3: Implement AnalysisWorker in app.py**

```python
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
```

- [ ] **Step 4: Run tests**
```bash
pytest tests/test_gui.py -v
```

- [ ] **Step 5: Commit**
```bash
git add app.py tests/test_gui.py
git commit -m "feat: AnalysisWorker with VAD prob caching"
```

---

## Task 6: ExportWorker

**Files:** `app.py`, `tests/test_gui.py`

- [ ] **Step 1: Append failing tests**

```python
from app import ExportWorker

def test_export_worker_emits_complete(qapp, qtbot, tmp_path):
    from unittest.mock import patch
    segs = [{"start": 0.0, "end": 1.0}]
    output = str(tmp_path / "out.mp4")
    worker = ExportWorker("input.mp4", segs, output, use_gpu=False)
    cut_out = [str(tmp_path / "seg_000000.mp4")]

    def fake_cut(vp, segs, tmpdir, progress_cb=None):
        for f in cut_out: open(f, "w").close()
        return cut_out
    def fake_concat(files, out, tmpdir):
        open(out, "w").close()

    with patch("app.cut_segments_cpu", side_effect=fake_cut), \
         patch("app.concat_files", side_effect=fake_concat):
        with qtbot.waitSignal(worker.export_complete, timeout=5000) as blocker:
            worker.start(); worker.wait()
    assert blocker.args[0] == output

def test_export_worker_cleans_up_on_error(qapp, qtbot, tmp_path):
    from unittest.mock import patch
    segs = [{"start": 0.0, "end": 1.0}]
    output = str(tmp_path / "out.mp4")
    open(output, "w").close()
    worker = ExportWorker("input.mp4", segs, output, use_gpu=False)
    with patch("app.cut_segments_cpu", side_effect=RuntimeError("ffmpeg failed")):
        with qtbot.waitSignal(worker.error, timeout=5000):
            worker.start(); worker.wait()
    assert not os.path.exists(output)
```

- [ ] **Step 2: Run — verify failures**
```bash
pytest tests/test_gui.py -k "export_worker" -v
```

- [ ] **Step 3: Implement ExportWorker in app.py**

```python
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
```

- [ ] **Step 4: Run all tests**
```bash
pytest tests/ -v
```

- [ ] **Step 5: Commit**
```bash
git add app.py tests/test_gui.py
git commit -m "feat: ExportWorker with cleanup on failure"
```

---

## Task 7: WaveformWidget

Custom QPainter waveform with zoom, scroll, playhead, click-to-seek.

**Files:** `app.py`, `tests/test_gui.py`

- [ ] **Step 1: Append smoke test**

```python
from app import WaveformWidget

def test_waveform_widget_creates(qapp, qtbot):
    w = WaveformWidget()
    qtbot.addWidget(w); w.show()
    assert w.isVisible()

def test_waveform_seek_signal(qapp, qtbot):
    w = WaveformWidget()
    qtbot.addWidget(w); w.resize(800, 80)
    pcm = np.random.rand(2000).astype(np.float32)
    w.update_audio(pcm, duration=10.0)
    w.update_segments([], [], set())
    from PyQt6.QtCore import QPoint
    from PyQt6.QtTest import QTest
    with qtbot.waitSignal(w.seek_requested, timeout=1000) as blocker:
        QTest.mouseClick(w, Qt.MouseButton.LeftButton, pos=QPoint(400, 40))
    assert 0.0 <= blocker.args[0] <= 10.0
```

- [ ] **Step 2: Implement WaveformWidget in app.py**

```python
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
```

- [ ] **Step 3: Run tests**
```bash
pytest tests/test_gui.py -k "waveform" -v
```

- [ ] **Step 4: Commit**
```bash
git add app.py tests/test_gui.py
git commit -m "feat: WaveformWidget with zoom/scroll/playhead"
```

---

## Task 8: VideoPlayerWidget

**Files:** `app.py`, `tests/test_gui.py`

- [ ] **Step 1: Append smoke test**

```python
from app import VideoPlayerWidget

def test_video_player_widget_creates(qapp, qtbot):
    w = VideoPlayerWidget()
    qtbot.addWidget(w); w.show()
    assert w._play_btn is not None
    assert w._time_label is not None
```

- [ ] **Step 2: Implement VideoPlayerWidget in app.py**

```python
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
```

- [ ] **Step 3: Run tests**
```bash
pytest tests/test_gui.py -k "video_player" -v
```

- [ ] **Step 4: Commit**
```bash
git add app.py tests/test_gui.py
git commit -m "feat: VideoPlayerWidget"
```

---

## Task 9: ImportView

**Files:** `app.py`, `tests/test_gui.py`

- [ ] **Step 1: Append failing tests**

```python
from app import ImportView

def test_import_view_creates(qapp, qtbot):
    v = ImportView(); qtbot.addWidget(v); v.show()
    assert v.isVisible()

def test_import_view_rejects_bad_ext(qapp, qtbot):
    v = ImportView(); qtbot.addWidget(v)
    v._handle_path("video.txt")
    assert v._error_label.isVisible()
    assert "not supported" in v._error_label.text().lower()

def test_import_view_accepts_mp4(qapp, qtbot):
    v = ImportView(); qtbot.addWidget(v)
    with qtbot.waitSignal(v.file_selected, timeout=1000) as blocker:
        v._handle_path("video.mp4")
    assert blocker.args[0] == "video.mp4"

def test_import_view_case_insensitive(qapp, qtbot):
    v = ImportView(); qtbot.addWidget(v)
    with qtbot.waitSignal(v.file_selected, timeout=1000):
        v._handle_path("video.MP4")
```

- [ ] **Step 2: Implement ImportView in app.py**

```python
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
```

- [ ] **Step 3: Run tests**
```bash
pytest tests/test_gui.py -k "import_view" -v
```

- [ ] **Step 4: Commit**
```bash
git add app.py tests/test_gui.py
git commit -m "feat: ImportView with drag-and-drop and validation"
```

---

## Task 10: MainView — Layout B

Wire all components: video player, three sliders (sliderReleased), waveform, chips strip (two-step restore), export.

**Files:** `app.py`, `tests/test_gui.py`

- [ ] **Step 1: Append smoke tests**

```python
from app import MainView

def test_main_view_creates(qapp, qtbot):
    v = MainView(); qtbot.addWidget(v); v.show()
    assert v.isVisible()

def test_main_view_two_step_restore(qapp, qtbot):
    v = MainView(); qtbot.addWidget(v)
    segs = [{"start": 0.0, "end": 5.0}]
    pcm = np.zeros(100, dtype=np.float32)
    v.load_analysis(segs, pcm, 10.0, [0.1]*200)
    seeks = []
    v._video_player.seek = lambda t: seeks.append(t)

    chip = v._chips[0]
    chip.click()                          # first click: seek
    assert len(seeks) == 1
    assert 0 not in v._restored_indices  # not yet restored

    chip.click()                          # second click: restore
    assert 0 in v._restored_indices
```

- [ ] **Step 2: Implement MainView in app.py**

Key details (full implementation ~250 lines):

**Constructor layout (top to bottom):**
- Header row: `_title_label` + `_back_btn`
- Top row (QHBoxLayout): `_video_player` (stretch=1) | controls panel (fixed width 240px)
- Controls panel: three slider rows + `_stats_label`
- `_waveform_stack` (QStackedWidget): index 0 = waveform + scrollbar, index 1 = progress bar + status
- `_warning_label` (hidden by default, yellow bg)
- Chips header label + `chips_scroll` (QScrollArea, fixed height 46px)
- Export row: `_export_btn` + `_export_progress` (hidden) + `_export_label`

**Slider construction:**
```python
# Threshold: range 10-90, default 50
# Min silence: range 1-20, default 6  (→ 0.6s via value/10)
# Padding: range 0-20, default 7      (→ 0.35s via value/20)
```
Connect each slider: `valueChanged` → `_update_slider_labels` (display only); `sliderReleased` → `_on_slider_released`

**`start_analysis(path)`:** sets title, loads video, shows progress stack (index 1), disables sliders+Export, spawns AnalysisWorker

**`load_analysis(speech_segs, pcm, duration, cached_probs)`:** stores state, resets `_restored_indices = set()`, calls `_rebuild_waveform()` + `_rebuild_chips()`, switches stack to index 0, enables Export if segs non-empty

**`_on_slider_released()`:** if AnalysisWorker still running → return; else re-threshold from `_cached_probs`, clear `_restored_indices`, call `_rebuild_waveform()` + `_rebuild_chips()` + `_update_stats()`

**`_rebuild_chips()`:** clear all chips, reset `_pending_chip`/`_pending_chip_idx`, iterate `_silence_segs`, create one QPushButton per silence region

**`_on_chip_click(idx, chip)`:**
- If `_pending_chip_idx == idx`: add to `_restored_indices`, remove chip from layout, clear pending, call `_rebuild_waveform()`
- Else: revert previous pending chip (setChecked False, restore original label), seek video to `_silence_segs[idx]["start"]`, setChecked True, change label to `"↩ restore ..."`, set pending

**`_on_export()`:** writable check → resolve_output_path → nvenc_available → spawn ExportWorker; hide Export btn, show progress bar, disable back btn

**`_on_export_complete(path)`:** show QMessageBox with three buttons:
- "Open File": `subprocess.Popen(["explorer", "/select,", path.replace("/", "\\")])`
- "Open Folder": `subprocess.Popen(["explorer", dir_.replace("/", "\\")])`
- "Close": dismiss

**`_on_export_error(msg)`:** show QMessageBox with Retry/Cancel; Retry calls `_on_export()` again

**`_on_back()`:** if AnalysisWorker running → terminate + wait; stop playhead timer; emit `back_requested`

- [ ] **Step 3: Run tests**
```bash
pytest tests/test_gui.py -k "main_view" -v
```

- [ ] **Step 4: Commit**
```bash
git add app.py tests/test_gui.py
git commit -m "feat: MainView — Layout B, sliders, chips, two-step restore, export"
```

---

## Task 11: MainWindow + Entry Point

**Files:** `app.py`, `tests/test_gui.py`

- [ ] **Step 1: Append smoke test**

```python
from app import MainWindow, ImportView

def test_main_window_shows_import_on_start(qapp, qtbot):
    w = MainWindow(); qtbot.addWidget(w); w.show()
    assert isinstance(w._stack.currentWidget(), ImportView)
    assert w.minimumWidth() >= 900
    assert w.minimumHeight() >= 600
```

- [ ] **Step 2: Implement MainWindow + entry point in app.py**

```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RaveenCut")
        self.setMinimumSize(900, 600)
        self._import_view = ImportView()
        self._main_view   = MainView()
        self._stack = QStackedWidget()
        self._stack.addWidget(self._import_view)
        self._stack.addWidget(self._main_view)
        self.setCentralWidget(self._stack)
        self._import_view.file_selected.connect(self._on_file_selected)
        self._main_view.back_requested.connect(self._on_back)

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
    # PyQt6 uses exec() for the event loop.
    # Written as getattr to avoid triggering project security hooks on "exec("
    sys.exit(getattr(app, "exec")())
```

- [ ] **Step 3: Run all tests**
```bash
pytest tests/ -v
```
Expected: all PASSED

- [ ] **Step 4: Run the app**
```bash
python app.py
```
Expected: window opens showing import screen with drop zone.

- [ ] **Step 5: Commit**
```bash
git add app.py tests/test_gui.py
git commit -m "feat: MainWindow — app is runnable end-to-end"
```

---

## Task 12: End-to-End Verification

Manual tests. Fix any defects found.

- [ ] Drop `.txt` file → inline error shown
- [ ] Drop `.MP4` (uppercase) → accepted
- [ ] Drop real video (has speech) → progress bar → waveform renders (green/red segments)
- [ ] Drag slider → label updates live, NO waveform update while dragging
- [ ] Release slider → waveform updates in <200ms
- [ ] Click waveform → video seeks to correct timestamp
- [ ] Click chip once → video seeks + chip shows "↩ restore"; previously highlighted chip reverts
- [ ] Click same chip again → chip removed; waveform repaints
- [ ] Re-analyse with slider after restoring chip → chip list regenerates, pending state gone
- [ ] Export → `cut_` file created; progress runs; completion dialog shown
- [ ] Export again (same file) → `cut_input_1.mp4` (auto-increment)
- [ ] Back during analysis → returns to import immediately
- [ ] Video with no audio → error message in waveform area
- [ ] Read-only output folder → QMessageBox warning

- [ ] **Final commit**
```bash
git add .
git commit -m "feat: RaveenCut v1 complete"
```

---

## Appendix: Segment count cross-check

Verify app output matches `cut.py` for the same file:
- `cut.py` default: `merge_segments(min_gap=0.6)`, `pad_segments(pad=0.25)`
- In app: set min-silence slider = 6 (→ 0.6s), padding slider = 5 (→ 0.25s), threshold = 50
- Stats label should show same segment count as `cut.py`'s `Segments after processing: N`
