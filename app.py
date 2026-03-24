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

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    print("RaveenCut skeleton — OK")
    sys.exit(0)
