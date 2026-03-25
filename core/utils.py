import os
import subprocess
import sys


def _ff_bin(name: str) -> str:
    if getattr(sys, "frozen", False):
        ext = ".exe" if os.name == "nt" else ""
        candidate = os.path.join(sys._MEIPASS, name + ext)
        if os.path.isfile(candidate):
            return candidate
    return name


def _subprocess_hide_console():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


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
