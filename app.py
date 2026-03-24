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
