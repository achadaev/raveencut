import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.audio import read_audio_from_video
from core.constants import (DEFAULT_MIN_SILENCE, DEFAULT_PADDING,
                              DEFAULT_THRESHOLD, FRAME_SIZE, SAMPLING_RATE)
from core.exporter import concat_files, cut_segments_cpu, cut_segments_gpu
from core.segments import merge_segments, pad_segments, probs_to_segments
from core.utils import fmt_time
from core.vad import compute_vad_probs, downsample_pcm


class AnalysisWorker(QThread):
    progress          = pyqtSignal(int, str)
    analysis_complete = pyqtSignal(list, np.ndarray, float)
    error             = pyqtSignal(str)

    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.cached_probs = []

    def _extract_audio(self):
        self.progress.emit(5, _("Extracting audio\u2026"))
        wav = read_audio_from_video(self.video_path)
        return wav, len(wav) / SAMPLING_RATE

    def _compute_probs(self, wav):
        self.progress.emit(20, _("Loading VAD model\u2026"))
        n_frames = (len(wav) + FRAME_SIZE - 1) // FRAME_SIZE

        def _cb(done, n):
            pct = 20 + int(done / n * 70)
            self.progress.emit(
                pct, _("Detecting speech\u2026 {done}/{n} frames").format(done=done, n=n)
            )

        return compute_vad_probs(wav, progress_cb=_cb)

    def run(self):
        try:
            wav, duration = self._extract_audio()
            probs = self._compute_probs(wav)
            self.cached_probs = probs
            self.progress.emit(92, _("Computing segments\u2026"))
            segs = probs_to_segments(probs, DEFAULT_THRESHOLD)
            segs = merge_segments(segs, min_gap=DEFAULT_MIN_SILENCE)
            segs = pad_segments(segs, pad=DEFAULT_PADDING, max_duration=duration)
            pcm = downsample_pcm(wav)
            self.progress.emit(100, _("Done"))
            self.analysis_complete.emit(segs, pcm, duration)
        except Exception as exc:
            self.error.emit(str(exc))


class ExportWorker(QThread):
    progress        = pyqtSignal(int, str)
    export_complete = pyqtSignal(str)
    error           = pyqtSignal(str)

    def __init__(self, video_path, segments, output_path, use_gpu, parent=None):
        super().__init__(parent)
        self.video_path  = video_path
        self.segments    = segments
        self.output_path = output_path
        self.use_gpu     = use_gpu

    def run(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                start = time.monotonic()

                def cb(done, total_segs):
                    pct = int(done / total_segs * 90)
                    elapsed = time.monotonic() - start
                    if done > 0:
                        eta = elapsed / done * (total_segs - done)
                        eta_str = _("  \u2014  {eta} left").format(eta=fmt_time(eta))
                    else:
                        eta_str = ""
                    self.progress.emit(
                        pct,
                        _("Cutting {done}/{total}{eta}").format(
                            done=done, total=total_segs, eta=eta_str
                        ),
                    )

                cut_fn = cut_segments_gpu if self.use_gpu else cut_segments_cpu
                seg_files = cut_fn(self.video_path, self.segments, tmpdir, cb)
                self.progress.emit(91, _("Concatenating\u2026"))
                concat_files(seg_files, self.output_path, tmpdir)
                self.progress.emit(100, _("Done"))
                self.export_complete.emit(self.output_path)
            except Exception as exc:
                if os.path.exists(self.output_path):
                    os.remove(self.output_path)
                self.error.emit(str(exc))
