# conftest.py — must live at repo root so pytest loads it before any test module.
# Torch 2.x crashes on Python 3.14 when imported for the first time via
# pytest's importlib path (access violation in _load_dll_libraries).
# Pre-importing torch here, before pytest starts collecting, avoids the crash.
import torch  # noqa: F401
# Pre-import PyQt6 multimedia to avoid deferred-paint access violations on
# Python 3.14 / Windows when QVideoWidget is destroyed during test teardown.
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput  # noqa: F401
from PyQt6.QtMultimediaWidgets import QVideoWidget  # noqa: F401
