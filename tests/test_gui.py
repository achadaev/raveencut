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
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    with qtbot.waitSignal(w.seek_requested, timeout=1000) as blocker:
        QTest.mouseClick(w, Qt.MouseButton.LeftButton, pos=QPoint(400, 40))
    assert 0.0 <= blocker.args[0] <= 10.0

from app import VideoPlayerWidget

def test_video_player_widget_creates(qapp, qtbot):
    w = VideoPlayerWidget()
    qtbot.addWidget(w); w.show()
    assert w._play_btn is not None
    assert w._time_label is not None

from app import ImportView

def test_import_view_creates(qapp, qtbot):
    v = ImportView(); qtbot.addWidget(v); v.show()
    assert v.isVisible()

def test_import_view_rejects_bad_ext(qapp, qtbot):
    v = ImportView(); qtbot.addWidget(v); v.show()
    v._handle_path("video.txt")
    assert v._error_label.isVisible()
    err = v._error_label.text()
    assert "txt" in err.lower() and "mp4" in err.lower()

def test_import_view_accepts_mp4(qapp, qtbot):
    v = ImportView(); qtbot.addWidget(v); v.show()
    with qtbot.waitSignal(v.file_selected, timeout=1000) as blocker:
        v._handle_path("video.mp4")
    assert blocker.args[0] == "video.mp4"

def test_import_view_case_insensitive(qapp, qtbot):
    v = ImportView(); qtbot.addWidget(v); v.show()
    with qtbot.waitSignal(v.file_selected, timeout=1000):
        v._handle_path("video.MP4")

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

from app import _CancelledError

def test_export_worker_emits_cancelled(qapp, qtbot, tmp_path):
    segs = [{"start": 0.0, "end": 1.0}]
    output = str(tmp_path / "out.mp4")
    worker = ExportWorker("input.mp4", segs, output, use_gpu=False)
    worker.request_cancel()  # flag set before start

    def fake_cut(vp, segs, tmpdir, progress_cb=None):
        if progress_cb:
            progress_cb(1, len(segs))  # triggers _CancelledError
        return []

    with patch("app.cut_segments_cpu", side_effect=fake_cut):
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start()
            worker.wait()

def test_export_worker_cancel_deletes_output(qapp, qtbot, tmp_path):
    segs = [{"start": 0.0, "end": 1.0}]
    output = str(tmp_path / "out.mp4")
    open(output, "w").close()  # pre-create partial output
    worker = ExportWorker("input.mp4", segs, output, use_gpu=False)
    worker.request_cancel()

    def fake_cut(vp, segs, tmpdir, progress_cb=None):
        if progress_cb:
            progress_cb(1, len(segs))
        return []

    with patch("app.cut_segments_cpu", side_effect=fake_cut):
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start()
            worker.wait()
    assert not os.path.exists(output)

def test_export_worker_cancel_does_not_emit_error(qapp, qtbot, tmp_path):
    segs = [{"start": 0.0, "end": 1.0}]
    output = str(tmp_path / "out.mp4")
    worker = ExportWorker("input.mp4", segs, output, use_gpu=False)
    worker.request_cancel()
    errors = []
    worker.error.connect(lambda m: errors.append(m))

    def fake_cut(vp, segs, tmpdir, progress_cb=None):
        if progress_cb:
            progress_cb(1, len(segs))
        return []

    with patch("app.cut_segments_cpu", side_effect=fake_cut):
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start()
            worker.wait()
    assert errors == []

def test_analysis_worker_emits_cancelled(qapp, qtbot):
    from unittest.mock import MagicMock
    wav = torch.zeros(SAMPLING_RATE)
    worker = AnalysisWorker("fake.mp4")
    worker.request_cancel()  # cancel before probs loop

    mock_model = MagicMock()
    mock_model.return_value.item.return_value = 0.0
    with patch.object(worker, "_extract_audio", return_value=(wav, 1.0)), \
         patch("app.load_silero_vad", return_value=mock_model):
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start()
            worker.wait()

def test_analysis_worker_cancel_does_not_emit_complete(qapp, qtbot):
    from unittest.mock import MagicMock
    wav = torch.zeros(SAMPLING_RATE)
    worker = AnalysisWorker("fake.mp4")
    worker.request_cancel()
    completed = []
    worker.analysis_complete.connect(lambda s, p, d: completed.append(True))

    mock_model = MagicMock()
    mock_model.return_value.item.return_value = 0.0
    with patch.object(worker, "_extract_audio", return_value=(wav, 1.0)), \
         patch("app.load_silero_vad", return_value=mock_model):
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start()
            worker.wait()
    assert completed == []

def test_analysis_worker_cancel_preserves_cached_probs(qapp, qtbot):
    from unittest.mock import MagicMock
    wav = torch.zeros(SAMPLING_RATE)
    n_frames = (SAMPLING_RATE + FRAME_SIZE - 1) // FRAME_SIZE
    prior_probs = [0.5] * n_frames
    worker = AnalysisWorker("fake.mp4")
    worker.cached_probs = prior_probs  # simulate a prior successful run
    worker.request_cancel()

    mock_model = MagicMock()
    mock_model.return_value.item.return_value = 0.0
    with patch.object(worker, "_extract_audio", return_value=(wav, 1.0)), \
         patch("app.load_silero_vad", return_value=mock_model):
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start()
            worker.wait()
    assert worker.cached_probs is prior_probs  # untouched

def test_main_view_export_cancel_btn_hidden_by_default(qapp, qtbot):
    v = MainView()
    qtbot.addWidget(v)
    assert not v._export_cancel_btn.isVisible()

def test_main_view_export_cancelled_resets_ui(qapp, qtbot):
    v = MainView()
    qtbot.addWidget(v)
    v.show()
    segs = [{"start": 0.0, "end": 5.0}]
    pcm = np.zeros(100, dtype=np.float32)
    v.load_analysis(segs, pcm, 10.0, [0.5] * 200)

    # Manually put UI into "export running" state
    v._export_btn.setVisible(False)
    v._export_progress.setVisible(True)
    v._export_cancel_btn.setVisible(True)
    v._export_cancel_btn.setEnabled(True)
    v._export_label.setText("Cutting 1/3")
    v._back_btn.setEnabled(False)

    v._on_export_cancelled()

    assert v._export_btn.isVisible()
    assert not v._export_progress.isVisible()
    assert not v._export_cancel_btn.isVisible()
    assert v._export_label.text() == ""
    assert v._back_btn.isEnabled()

def test_main_view_analysis_cancel_btn_hidden_by_default(qapp, qtbot):
    v = MainView()
    qtbot.addWidget(v)
    assert not v._analysis_cancel_btn.isVisible()

def test_main_view_reset_after_analysis_cancel_restores_ui(qapp, qtbot):
    from unittest.mock import MagicMock
    v = MainView()
    qtbot.addWidget(v)
    segs = [{"start": 0.0, "end": 5.0}]
    pcm = np.zeros(100, dtype=np.float32)
    v.load_analysis(segs, pcm, 10.0, [0.5] * 200)

    # Simulate UI entering analysis-running state
    v._analysis_worker = MagicMock()
    v._waveform_stack.setCurrentIndex(1)
    for sl in (v._thr_slider, v._sil_slider, v._pad_slider):
        sl.setEnabled(False)
    v._analysis_cancel_btn.setVisible(True)

    v._reset_after_analysis_cancel()

    assert v._analysis_worker is None
    assert v._waveform_stack.currentIndex() == 0
    assert v._thr_slider.isEnabled()
    assert v._sil_slider.isEnabled()
    assert v._pad_slider.isEnabled()
    assert v._export_btn.isEnabled()   # prior analysis data exists
    assert not v._analysis_cancel_btn.isVisible()

def test_main_view_reset_after_analysis_cancel_idempotent(qapp, qtbot):
    v = MainView()
    qtbot.addWidget(v)
    v._analysis_worker = None
    # Must not raise when called twice
    v._reset_after_analysis_cancel()
    v._reset_after_analysis_cancel()

def test_main_view_reset_after_analysis_cancel_no_prior_data(qapp, qtbot):
    from unittest.mock import MagicMock
    v = MainView()
    qtbot.addWidget(v)
    # No prior analysis: _cached_probs is still []
    v._analysis_worker = MagicMock()
    v._reset_after_analysis_cancel()
    assert not v._export_btn.isEnabled()

from app import MainWindow, ImportView

def test_main_window_shows_import_on_start(qapp, qtbot):
    w = MainWindow(); qtbot.addWidget(w); w.show()
    assert isinstance(w._stack.currentWidget(), ImportView)
    assert w.minimumWidth() >= 900
    assert w.minimumHeight() >= 600
