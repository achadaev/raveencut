import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

from core.constants import SAMPLING_RATE


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


def _subprocess_hide_console():
    """Avoid flashing a console window when spawning ffmpeg/ffprobe on Windows."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


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
