import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Mock torch before importing app to avoid DLL issues
import sys
from unittest.mock import MagicMock

sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['silero_vad'] = MagicMock()

import pytest
from app import (
    merge_segments, pad_segments, probs_to_segments,
    silence_regions, export_segments_fn, fmt_time, resolve_output_path,
    run, probe_video_duration_sec, read_audio_from_video,
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

from unittest.mock import patch
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
    import numpy as np
    pcm = struct.pack("4f", 0.1, -0.1, 0.2, -0.2)
    mock_proc = MagicMock()
    mock_proc.stdout.read.return_value = pcm
    mock_proc.wait.return_value = None

    # Create a real tensor mock with shape and item methods
    import sys
    torch_mock = sys.modules['torch']
    tensor_mock = MagicMock()
    tensor_mock.shape = (4,)
    tensor_mock.__getitem__ = lambda self, i: MagicMock(item=lambda: [0.1, -0.1, 0.2, -0.2][i])
    torch_mock.from_numpy.return_value = tensor_mock

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("subprocess.Popen", return_value=mock_proc):
        t = read_audio_from_video("x.mp4")
    assert t.shape == (4,)
    assert abs(t[0].item() - 0.1) < 1e-5

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
