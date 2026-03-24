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
