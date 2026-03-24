import json
import os
import shutil
import subprocess

import numpy as np
import torch

from core.constants import SAMPLING_RATE
from core.utils import _ff_bin, _subprocess_hide_console


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
