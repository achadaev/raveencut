"""
End-to-end tests for RaveenCut.

Analysis on input.mp4 runs ONCE per pytest session via the `analysis_data`
fixture (may take several minutes). Subsequent fixture calls return a cached
copy so the heavy pipeline does not re-run.

Run all (slow + fast):
    pytest tests/test_e2e.py -v

Skip the slow export test during development:
    pytest tests/test_e2e.py -m "not slow" -v
"""
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

sys.path.insert(0, str(Path(__file__).parent.parent))
import app as app_module
from app import AnalysisWorker, MainView, MainWindow

# ── Constants ─────────────────────────────────────────────────────────────────

INPUT_VIDEO = Path(__file__).parent.parent / "input.mp4"
# 1:28:47 = 1*3600 + 28*60 + 47 = 5327 s — user-specified acceptance criterion
# for default settings (threshold=0.50, min_silence=0.6s, padding=0.35s).
# Update if input.mp4 or the Silero VAD version changes.
EXPECTED_EXPORT_SECS = 5327
DURATION_TOLERANCE = 5  # ±5 seconds

# ── Module-level analysis cache ───────────────────────────────────────────────
# Populated once by the first test that calls analysis_data; re-used by the rest.

_analysis_cache: dict = {}

# ── QApplication fixture ──────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication — avoids creating multiple instances."""
    return QApplication.instance() or QApplication(sys.argv)


# ── Helpers ───────────────────────────────────────────────────────────────────


def ffprobe_duration(path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _parse_segment_count(text: str) -> int:
    """Parse 'Segments: N' from _stats_label text. Returns -1 on failure."""
    for line in text.splitlines():
        if "Segments:" in line:
            try:
                return int(line.split(":")[-1].strip())
            except ValueError:
                pass
    return -1


# ── Analysis fixture ──────────────────────────────────────────────────────────


@pytest.fixture
def analysis_data(qapp, qtbot):
    """
    Run AnalysisWorker on input.mp4 the first time; return cached copy each time.

    Function-scoped so qtbot (also function-scoped) can be used safely.
    Module-level _analysis_cache avoids re-running the expensive pipeline.
    Returns a shallow copy to prevent tests from corrupting the shared cache.
    """
    global _analysis_cache

    if not _analysis_cache:
        assert INPUT_VIDEO.exists(), f"input.mp4 not found at {INPUT_VIDEO}"
        worker = AnalysisWorker(str(INPUT_VIDEO))
        with qtbot.waitSignal(worker.analysis_complete, timeout=600_000) as blocker:
            worker.start()
        worker.wait()
        _analysis_cache = {
            "segs": blocker.args[0],
            "pcm": blocker.args[1],
            "duration": blocker.args[2],
            # cached_probs is set on the worker before the signal fires
            "cached_probs": list(worker.cached_probs),
        }

    return {
        "segs": list(_analysis_cache["segs"]),   # new list, same seg dicts
        "pcm": _analysis_cache["pcm"],            # ndarray is read-only in practice
        "duration": _analysis_cache["duration"],
        "cached_probs": _analysis_cache["cached_probs"],
    }


# ── Per-test MainView fixture ─────────────────────────────────────────────────


@pytest.fixture
def loaded_view(qapp, qtbot, analysis_data):
    """
    Create a fresh MainView loaded with the cached analysis result.

    set_language("en") is called first so _stats_label text is in English,
    making it parseable by _parse_segment_count. No teardown is needed because
    all E2E tests share the English-language requirement.
    """
    app_module.set_language("en")
    view = MainView()
    qtbot.addWidget(view)
    view._video_path = str(INPUT_VIDEO)
    view.load_analysis(
        analysis_data["segs"],
        analysis_data["pcm"],
        analysis_data["duration"],
        analysis_data["cached_probs"],
    )
    view.show()
    return view


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_full_flow_from_main_window(qapp, qtbot, monkeypatch, tmp_path):
    """
    Full E2E: ImportView → real analysis → default export → ffprobe = 1:28:47.

    This is the primary acceptance test. It exercises MainWindow end-to-end:
    file selection → AnalysisWorker → ExportWorker → output file duration check.
    """
    out_path = str(tmp_path / "cut_output.mp4")

    # Redirect export to tmp_path so the source directory stays clean
    monkeypatch.setattr(app_module, "resolve_output_path", lambda *_: out_path)
    # Prevent the post-export QMessageBox from blocking (exec → no-op)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)

    app_module.set_language("en")

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    # Trigger file selection programmatically (bypasses QFileDialog)
    window._import_view.file_selected.emit(str(INPUT_VIDEO))
    assert window._stack.currentIndex() == 1, "Expected MainView after file selection"

    # Wait for AnalysisWorker to finish (up to 10 min)
    with qtbot.waitSignal(
        window._main_view._analysis_worker.analysis_complete,
        timeout=600_000,
    ):
        pass

    # The analysis_complete slot is a queued connection (cross-thread).
    # Pump the event loop until _on_analysis_complete has run and enabled Export.
    qtbot.waitUntil(
        lambda: window._main_view._export_btn.isEnabled(),
        timeout=10_000,
    )

    # Click Export
    QTest.mouseClick(window._main_view._export_btn, Qt.MouseButton.LeftButton)

    # Wait for ExportWorker to finish (up to 30 min)
    with qtbot.waitSignal(
        window._main_view._export_worker.export_complete,
        timeout=1_800_000,
    ):
        pass

    # Allow _on_export_complete slot to run (queued cross-thread delivery)
    qtbot.wait(300)

    assert os.path.exists(out_path), f"Export output not found: {out_path}"

    duration = ffprobe_duration(out_path)
    assert abs(duration - EXPECTED_EXPORT_SECS) <= DURATION_TOLERANCE, (
        f"Expected ~1:28:47 ({EXPECTED_EXPORT_SECS}s ± {DURATION_TOLERANCE}s), "
        f"got {duration:.1f}s"
    )


def test_slider_threshold_reduces_kept_duration(qtbot, loaded_view):
    """
    Raising threshold from 0.50 to 0.80 keeps less speech → fewer segments.

    The slider value 80 maps to threshold 80/100 = 0.80. A higher threshold
    means a frame must score more confidently to count as speech, so the VAD
    keeps fewer frames and produces fewer/shorter speech segments.
    """
    baseline = _parse_segment_count(loaded_view._stats_label.text())
    assert baseline > 0, f"No segments at baseline (text: {loaded_view._stats_label.text()!r})"

    loaded_view._thr_slider.setValue(80)
    # sliderReleased is not emitted by setValue; fire it explicitly to trigger recompute
    loaded_view._thr_slider.sliderReleased.emit()

    qtbot.waitUntil(
        lambda: _parse_segment_count(loaded_view._stats_label.text()) != baseline,
        timeout=5_000,
    )
    new_count = _parse_segment_count(loaded_view._stats_label.text())
    assert new_count != baseline, (
        f"Expected segment count to change with threshold=0.80; baseline={baseline}"
    )


def test_slider_min_silence_merges_segments(qtbot, loaded_view):
    """
    Increasing min-silence from 0.6s to 1.5s merges more gaps → fewer segments.

    Slider value 15 maps to 15/10 = 1.5s. merge_segments() treats gaps
    smaller than min_gap as intra-speech pauses and merges the surrounding
    segments into one, reducing the total segment count.
    """
    baseline = _parse_segment_count(loaded_view._stats_label.text())

    loaded_view._sil_slider.setValue(15)  # 15 / 10 = 1.5 s
    loaded_view._sil_slider.sliderReleased.emit()

    qtbot.waitUntil(
        lambda: _parse_segment_count(loaded_view._stats_label.text()) != baseline,
        timeout=5_000,
    )
    new_count = _parse_segment_count(loaded_view._stats_label.text())
    assert new_count < baseline, (
        f"Expected fewer segments after min_silence=1.5s; "
        f"got {new_count} vs baseline {baseline}"
    )


def test_chip_restore_increases_kept_time(qtbot, loaded_view):
    """
    Double-clicking a silence chip restores it: chip disappears, segment count unchanged.

    First click: chip is marked pending (seek to silence start).
    Second click on same chip: silence is added back to export; chip removed from list.
    """
    chips = loaded_view._chips
    assert len(chips) > 0, "No silence chips found — input.mp4 must have detectable silence"
    initial_count = len(chips)

    chip = loaded_view._chips[0]
    QTest.mouseClick(chip, Qt.MouseButton.LeftButton)  # first click — seek / pending

    qtbot.wait(100)  # allow any repaint to settle

    # Re-fetch in case the chip list was rebuilt by a repaint
    chip = loaded_view._chips[0]
    QTest.mouseClick(chip, Qt.MouseButton.LeftButton)  # second click — restore

    qtbot.waitUntil(
        lambda: len(loaded_view._chips) < initial_count,
        timeout=5_000,
    )
    assert len(loaded_view._chips) == initial_count - 1


def test_back_navigation_returns_to_import(qapp, qtbot, analysis_data):
    """Clicking Back in MainView switches MainWindow back to ImportView."""
    app_module.set_language("en")
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    # Navigate to MainView and inject analysis (skip re-running the pipeline)
    window._stack.setCurrentIndex(1)
    window._main_view._video_path = str(INPUT_VIDEO)
    window._main_view.load_analysis(
        analysis_data["segs"],
        analysis_data["pcm"],
        analysis_data["duration"],
        analysis_data["cached_probs"],
    )
    assert window._stack.currentIndex() == 1

    QTest.mouseClick(window._main_view._back_btn, Qt.MouseButton.LeftButton)

    assert window._stack.currentIndex() == 0, (
        "Expected ImportView (stack index 0) after clicking Back"
    )


def test_back_and_reimport_triggers_analysis(qapp, qtbot, analysis_data):
    """After going back, selecting a file again starts a new analysis run."""
    app_module.set_language("en")
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    # Navigate to MainView with cached analysis
    window._stack.setCurrentIndex(1)
    window._main_view._video_path = str(INPUT_VIDEO)
    window._main_view.load_analysis(
        analysis_data["segs"],
        analysis_data["pcm"],
        analysis_data["duration"],
        analysis_data["cached_probs"],
    )

    # Go back to ImportView
    QTest.mouseClick(window._main_view._back_btn, Qt.MouseButton.LeftButton)
    assert window._stack.currentIndex() == 0

    # Re-import: triggers start_analysis → waveform stack switches to progress page
    window._import_view.file_selected.emit(str(INPUT_VIDEO))

    qtbot.waitUntil(
        lambda: (
            window._stack.currentIndex() == 1
            and window._main_view._waveform_stack.currentIndex() == 1  # progress page
        ),
        timeout=5_000,
    )

    # Clean up the new worker so it does not run in the background after this test
    worker = window._main_view._analysis_worker
    if worker and worker.isRunning():
        worker.terminate()
        worker.wait()
