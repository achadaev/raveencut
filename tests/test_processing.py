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
