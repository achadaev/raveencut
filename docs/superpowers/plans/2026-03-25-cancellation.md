# Cancellation (Analysis & Export) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add user-initiated cancellation with confirmation dialogs for both the analysis and export operations in RaveenCut.

**Architecture:** Cooperative cancellation via `_cancel_requested` flag checked at loop checkpoints in both workers, with a `QThread.terminate()` fallback for the blocking audio-extraction phase. Cancel buttons appear in-context (progress page for analysis, export row for export) and call `request_cancel()` after a QMessageBox confirmation.

**Tech Stack:** PyQt6, QThread signals, pytest-qt

**Spec:** `docs/superpowers/specs/2026-03-25-cancellation-design.md`

---

## File Map

| File | Change |
|------|--------|
| `app.py:229-289` | Add `_cancel_requested`, `request_cancel()`, `cancelled` signal, cancel checks to `AnalysisWorker` |
| `app.py:291-330` | Add `_CancelledError`, `_cancel_requested`, `request_cancel()`, `cancelled` signal, cancel check in `cb` to `ExportWorker` |
| `app.py:669-730` | Add `_analysis_cancel_btn` to `progress_page`, add `_export_cancel_btn` to export row in `MainView.__init__` |
| `app.py:733-745` | Show `_analysis_cancel_btn` in `start_analysis()` |
| `app.py:764-778` | Wire `cancelled` signal in `start_analysis()`; add `_reset_after_analysis_cancel()` |
| `app.py:871-891` | Show `_export_cancel_btn` and clear `_export_label` in `_on_export()`; add `_on_export_cancel_clicked()` and `_on_export_cancelled()` |
| `tests/test_gui.py` | Add 8 new tests covering worker signals and UI state |

---

## Task 1: ExportWorker — cancellation signal and flag

**Files:**
- Modify: `app.py:291-330`
- Test: `tests/test_gui.py`

- [ ] **Step 1.1: Write failing tests**

Add to `tests/test_gui.py` after `test_export_worker_cleans_up_on_error`:

```python
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
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```
pytest tests/test_gui.py::test_export_worker_emits_cancelled tests/test_gui.py::test_export_worker_cancel_deletes_output tests/test_gui.py::test_export_worker_cancel_does_not_emit_error -v
```

Expected: ImportError (`_CancelledError` not found) or AttributeError (`request_cancel` not found).

- [ ] **Step 1.3: Implement in `app.py`**

After line 228 (before `class AnalysisWorker`), add:

```python
class _CancelledError(Exception):
    """Raised inside ExportWorker.run() to abort export cooperatively."""
```

In `ExportWorker.__init__` (after `self.use_gpu = use_gpu`), add:

```python
        self._cancel_requested = False
```

After `ExportWorker.__init__`, add:

```python
    def request_cancel(self):
        self._cancel_requested = True
```

In `ExportWorker` class declaration line, add `cancelled` signal alongside existing signals:

```python
    progress        = pyqtSignal(int, str)
    export_complete = pyqtSignal(str)
    error           = pyqtSignal(str)
    cancelled       = pyqtSignal()
```

In `ExportWorker.run()`, inside the `cb` closure, add the cancel check at the **top** of `cb`:

```python
            def cb(done, total_segs):
                if self._cancel_requested:
                    raise _CancelledError()
                pct = int(done / total_segs * 90)
                # ... rest of cb unchanged ...
```

In `ExportWorker.run()`, add a `except _CancelledError` block **before** the existing `except Exception` block:

```python
            except _CancelledError:
                if os.path.exists(self.output_path):
                    os.remove(self.output_path)
                self.cancelled.emit()
            except Exception as exc:
                # ... existing error handler unchanged ...
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```
pytest tests/test_gui.py::test_export_worker_emits_cancelled tests/test_gui.py::test_export_worker_cancel_deletes_output tests/test_gui.py::test_export_worker_cancel_does_not_emit_error -v
```

Expected: 3 PASSED. Also run the existing export tests to confirm no regression:

```
pytest tests/test_gui.py::test_export_worker_emits_complete tests/test_gui.py::test_export_worker_cleans_up_on_error -v
```

Expected: 2 PASSED.

- [ ] **Step 1.5: Commit**

```bash
git add app.py tests/test_gui.py
git commit -m "feat: add cooperative cancellation to ExportWorker"
```

---

## Task 2: AnalysisWorker — cancellation signal and flag

**Files:**
- Modify: `app.py:229-289`
- Test: `tests/test_gui.py`

- [ ] **Step 2.1: Write failing tests**

Add to `tests/test_gui.py` after `test_analysis_worker_emits_error_on_bad_file`:

```python
def test_analysis_worker_emits_cancelled(qapp, qtbot):
    wav = torch.zeros(SAMPLING_RATE)
    worker = AnalysisWorker("fake.mp4")
    worker.request_cancel()  # cancel before probs loop

    with patch.object(worker, "_extract_audio", return_value=(wav, 1.0)):
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start()
            worker.wait()

def test_analysis_worker_cancel_does_not_emit_complete(qapp, qtbot):
    wav = torch.zeros(SAMPLING_RATE)
    worker = AnalysisWorker("fake.mp4")
    worker.request_cancel()
    completed = []
    worker.analysis_complete.connect(lambda s, p, d: completed.append(True))

    with patch.object(worker, "_extract_audio", return_value=(wav, 1.0)):
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start()
            worker.wait()
    assert completed == []

def test_analysis_worker_cancel_preserves_cached_probs(qapp, qtbot):
    wav = torch.zeros(SAMPLING_RATE)
    n_frames = (SAMPLING_RATE + FRAME_SIZE - 1) // FRAME_SIZE
    prior_probs = [0.5] * n_frames
    worker = AnalysisWorker("fake.mp4")
    worker.cached_probs = prior_probs  # simulate a prior successful run
    worker.request_cancel()

    with patch.object(worker, "_extract_audio", return_value=(wav, 1.0)):
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start()
            worker.wait()
    assert worker.cached_probs is prior_probs  # untouched
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```
pytest tests/test_gui.py::test_analysis_worker_emits_cancelled tests/test_gui.py::test_analysis_worker_cancel_does_not_emit_complete tests/test_gui.py::test_analysis_worker_cancel_preserves_cached_probs -v
```

Expected: AttributeError (`request_cancel` not found on AnalysisWorker).

- [ ] **Step 2.3: Implement in `app.py`**

In `AnalysisWorker.__init__` (after `self.cached_probs = []`), add:

```python
        self._cancel_requested = False
```

After `AnalysisWorker.__init__`, add:

```python
    def request_cancel(self):
        self._cancel_requested = True
```

In `AnalysisWorker` class declaration, add `cancelled` signal:

```python
    progress          = pyqtSignal(int, str)
    analysis_complete = pyqtSignal(list, np.ndarray, float)
    error             = pyqtSignal(str)
    cancelled         = pyqtSignal()
```

In `_compute_probs()`, after the existing `if done % 500 == 0:` block, add:

```python
            if done % 500 == 0:
                pct = 20 + int(done / n_frames * 70)
                self.progress.emit(
                    pct, _("Detecting speech\u2026 {done}/{n} frames").format(done=done, n=n_frames)
                )
            if self._cancel_requested:
                return None
```

After the `for` loop ends (before `return probs`), add:

```python
        if self._cancel_requested:
            return None
        return probs
```

In `AnalysisWorker.run()`, change the probs assignment and subsequent lines to check for `None`:

```python
    def run(self):
        try:
            wav, duration = self._extract_audio()
            probs = self._compute_probs(wav)
            if probs is None:
                self.cancelled.emit()
                return
            self.cached_probs = probs
            self.progress.emit(92, _("Computing segments\u2026"))
            segs = probs_to_segments(probs, DEFAULT_THRESHOLD)
            segs = merge_segments(segs, min_gap=DEFAULT_MIN_SILENCE)
            segs = pad_segments(segs, pad=DEFAULT_PADDING, max_duration=duration)
            pcm = self._downsample_pcm(wav)
            self.progress.emit(100, _("Done"))
            self.analysis_complete.emit(segs, pcm, duration)
        except Exception as exc:
            self.error.emit(str(exc))
```

- [ ] **Step 2.4: Run tests to confirm they pass**

```
pytest tests/test_gui.py::test_analysis_worker_emits_cancelled tests/test_gui.py::test_analysis_worker_cancel_does_not_emit_complete tests/test_gui.py::test_analysis_worker_cancel_preserves_cached_probs -v
```

Expected: 3 PASSED. Also confirm no regression:

```
pytest tests/test_gui.py::test_analysis_worker_emits_complete tests/test_gui.py::test_analysis_worker_emits_error_on_bad_file -v
```

Expected: 2 PASSED.

- [ ] **Step 2.5: Commit**

```bash
git add app.py tests/test_gui.py
git commit -m "feat: add cooperative cancellation to AnalysisWorker"
```

---

## Task 3: MainView — export cancel button

**Files:**
- Modify: `app.py:669-730` (MainView.__init__), `app.py:871-891` (_on_export)
- Test: `tests/test_gui.py`

- [ ] **Step 3.1: Write failing tests**

Add to `tests/test_gui.py` after `test_main_view_two_step_restore`:

```python
def test_main_view_export_cancel_btn_hidden_by_default(qapp, qtbot):
    v = MainView()
    qtbot.addWidget(v)
    assert not v._export_cancel_btn.isVisible()

def test_main_view_export_cancelled_resets_ui(qapp, qtbot):
    v = MainView()
    qtbot.addWidget(v)
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
```

- [ ] **Step 3.2: Run tests to confirm they fail**

```
pytest tests/test_gui.py::test_main_view_export_cancel_btn_hidden_by_default tests/test_gui.py::test_main_view_export_cancelled_resets_ui -v
```

Expected: AttributeError (`_export_cancel_btn` not found).

- [ ] **Step 3.3: Implement in `app.py`**

**In `MainView.__init__`, export row section (~line 705):**

Add `_export_cancel_btn` between `_export_progress` and `_export_label`:

```python
        self._export_btn = QPushButton("")
        self._export_btn.setEnabled(False)
        self._export_progress = QProgressBar()
        self._export_progress.setRange(0, 100)
        self._export_progress.setVisible(False)
        self._export_cancel_btn = QPushButton(_("Cancel"))
        self._export_cancel_btn.setVisible(False)
        self._export_label = QLabel("")
        export_row = QHBoxLayout()
        export_row.addWidget(self._export_btn)
        export_row.addWidget(self._export_progress)
        export_row.addWidget(self._export_cancel_btn)
        export_row.addWidget(self._export_label)
        export_row.addStretch()
        self._export_btn.clicked.connect(self._on_export)
        self._export_cancel_btn.clicked.connect(self._on_export_cancel_clicked)
```

**In `_on_export` (~line 871):** show cancel button and clear label at export start:

```python
        self._export_btn.setVisible(False)
        self._export_label.setText("")
        self._export_progress.setVisible(True)
        self._export_cancel_btn.setVisible(True)
        self._back_btn.setEnabled(False)
        self._export_worker.start()
```

**Connect `cancelled` signal in `_on_export` (~line 887):**

```python
        self._export_worker.export_complete.connect(self._on_export_complete)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.cancelled.connect(self._on_export_cancelled)
```

**Add two new methods to `MainView`:**

```python
    def _on_export_cancel_clicked(self):
        reply = QMessageBox.question(
            self,
            _("Cancel export"),
            _("Cancel export? The partial output file will be deleted."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if self._export_worker is None or not self._export_worker.isRunning():
            return
        self._export_cancel_btn.setEnabled(False)
        self._export_worker.request_cancel()

    def _on_export_cancelled(self):
        self._export_worker = None
        self._export_progress.setVisible(False)
        self._export_cancel_btn.setVisible(False)
        self._export_label.setText("")
        self._export_btn.setVisible(True)
        self._back_btn.setEnabled(True)
```

- [ ] **Step 3.4: Run tests to confirm they pass**

```
pytest tests/test_gui.py::test_main_view_export_cancel_btn_hidden_by_default tests/test_gui.py::test_main_view_export_cancelled_resets_ui -v
```

Expected: 2 PASSED. Full suite check:

```
pytest tests/test_gui.py -v
```

Expected: all PASSED.

- [ ] **Step 3.5: Commit**

```bash
git add app.py tests/test_gui.py
git commit -m "feat: add export cancel button to MainView"
```

---

## Task 4: MainView — analysis cancel button

**Files:**
- Modify: `app.py:669-745` (MainView.__init__, start_analysis)
- Test: `tests/test_gui.py`

- [ ] **Step 4.1: Write failing tests**

Add to `tests/test_gui.py`:

```python
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
```

- [ ] **Step 4.2: Run tests to confirm they fail**

```
pytest tests/test_gui.py::test_main_view_analysis_cancel_btn_hidden_by_default tests/test_gui.py::test_main_view_reset_after_analysis_cancel_restores_ui tests/test_gui.py::test_main_view_reset_after_analysis_cancel_idempotent tests/test_gui.py::test_main_view_reset_after_analysis_cancel_no_prior_data -v
```

Expected: AttributeError (`_analysis_cancel_btn` not found).

- [ ] **Step 4.3: Implement in `app.py`**

**In `MainView.__init__`, progress page section (~line 669):**

Add `_analysis_cancel_btn` below `_progress_bar` in `progress_layout`:

```python
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._analysis_cancel_btn = QPushButton(_("Cancel"))
        self._analysis_cancel_btn.setVisible(False)
        self._analysis_cancel_btn.clicked.connect(self._on_analysis_cancel_clicked)
        progress_layout = QVBoxLayout()
        progress_layout.addStretch()
        progress_layout.addWidget(self._status_label)
        progress_layout.addWidget(self._progress_bar)
        progress_layout.addWidget(self._analysis_cancel_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        progress_layout.addStretch()
```

**In `start_analysis()` (~line 733):** show cancel button and connect `cancelled` signal:

```python
    def start_analysis(self, path: str):
        self._video_path = path
        self._title_label.setText(os.path.basename(path))
        self._video_player.load(path)
        self._waveform_stack.setCurrentIndex(1)
        self._analysis_cancel_btn.setEnabled(True)
        self._analysis_cancel_btn.setVisible(True)
        self._export_btn.setEnabled(False)
        for sl in (self._thr_slider, self._sil_slider, self._pad_slider):
            sl.setEnabled(False)
        self._analysis_worker = AnalysisWorker(path)
        self._analysis_worker.progress.connect(self._on_analysis_progress)
        self._analysis_worker.analysis_complete.connect(self._on_analysis_complete)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.cancelled.connect(self._reset_after_analysis_cancel)
        self._analysis_worker.start()
```

**In `_on_analysis_complete` (~line 768):** hide cancel button (normal completion path):

```python
    def _on_analysis_complete(self, segs, pcm, duration):
        self._analysis_cancel_btn.setVisible(False)
        worker = self._analysis_worker
        self._analysis_worker = None
        cached_probs = worker.cached_probs if worker else []
        self.load_analysis(segs, pcm, duration, cached_probs)
```

**In `_on_analysis_error` (~line 774):** hide cancel button (error path):

```python
    def _on_analysis_error(self, msg):
        self._analysis_cancel_btn.setVisible(False)
        self._analysis_worker = None
        self._waveform_stack.setCurrentIndex(0)
        self._warning_label.setText(_("Analysis failed: ") + msg)
        self._warning_label.setVisible(True)
```

**Add two new methods to `MainView`:**

```python
    def _on_analysis_cancel_clicked(self):
        reply = QMessageBox.question(
            self,
            _("Cancel analysis"),
            _("Cancel analysis?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if self._analysis_worker is None or not self._analysis_worker.isRunning():
            return
        self._analysis_cancel_btn.setEnabled(False)
        self._analysis_worker.request_cancel()
        self._analysis_worker.wait(300)
        if self._analysis_worker is not None and self._analysis_worker.isRunning():
            self._analysis_worker.terminate()
            self._analysis_worker.wait()
        self._reset_after_analysis_cancel()

    def _reset_after_analysis_cancel(self):
        if self._analysis_worker is None:
            return  # idempotent guard
        self._analysis_worker = None
        self._progress_bar.setValue(0)
        self._status_label.setText("")
        self._waveform_stack.setCurrentIndex(0)
        for sl in (self._thr_slider, self._sil_slider, self._pad_slider):
            sl.setEnabled(True)
        if self._cached_probs is not None and len(self._cached_probs) > 0:
            self._export_btn.setEnabled(True)
        self._analysis_cancel_btn.setVisible(False)
```

- [ ] **Step 4.4: Run tests to confirm they pass**

```
pytest tests/test_gui.py::test_main_view_analysis_cancel_btn_hidden_by_default tests/test_gui.py::test_main_view_reset_after_analysis_cancel_restores_ui tests/test_gui.py::test_main_view_reset_after_analysis_cancel_idempotent tests/test_gui.py::test_main_view_reset_after_analysis_cancel_no_prior_data -v
```

Expected: 4 PASSED. Full suite:

```
pytest tests/test_gui.py -v
```

Expected: all PASSED.

- [ ] **Step 4.5: Commit**

```bash
git add app.py tests/test_gui.py
git commit -m "feat: add analysis cancel button to MainView"
```

---

## Task 5: Full test suite verification

- [ ] **Step 5.1: Run all tests**

```
pytest -v
```

Expected: all tests PASSED. If any fail, investigate and fix before proceeding.

- [ ] **Step 5.2: Manual smoke test**

Run the app and verify:
- Analysis cancel button appears during analysis, disappears on completion
- Cancelling analysis mid-run (after clicking Yes) returns the waveform page; sliders re-enable; if a prior analysis existed, the Export button re-enables
- Export cancel button appears during export, disappears on completion or cancel
- Cancelling export (after clicking Yes) hides progress, re-enables Export and Back buttons, and deletes the partial output file
- Clicking No on either confirmation does nothing

- [ ] **Step 5.3: Final commit if any last fixes**

```bash
git add app.py tests/test_gui.py
git commit -m "fix: <describe any last-minute fixes>"
```
