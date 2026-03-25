# Cancellation Design — Analysis & Export

**Date:** 2026-03-25

## Overview

Add user-initiated cancellation for both the analysis and export operations in RaveenCut. Both operations run in background QThreads and can take significant time; users need a way to abort them without killing the app.

## Decisions

- Cancelling analysis keeps the user on MainView (not sent back to ImportView).
- Partial export output files are silently deleted on cancel.
- Both cancel actions require a confirmation dialog.
- Implementation uses cooperative cancellation (flag checked at loop checkpoints). When the cooperative flag is not enough (audio extraction blocking phase), `QThread.terminate()` + `wait()` is used as a fallback **only if the worker is still running** after a short wait.

## Worker Changes

### AnalysisWorker

Add `_cancel_requested: bool = False` and a `request_cancel()` method that sets it.

Add `cancelled = pyqtSignal()`.

In `_compute_probs()`, place the following two checks:
1. **Inside the loop**, immediately after the existing `if done % 500 == 0: self.progress.emit(...)` block: `if self._cancel_requested: return None`.
2. **After the loop** (post-loop, before the function returns `probs`): `if self._cancel_requested: return None`. This handles clips shorter than 500 VAD frames, where the in-loop check never fires.

In `run()`, check the return value of `_compute_probs()`. If `None`, emit `cancelled` and return without emitting `analysis_complete`. **Do not update `self.cached_probs` (the `AnalysisWorker` instance attribute)** — leave it at whatever value it held before this run (either `[]` from `__init__` or a prior successful run's probs).

**Audio extraction phase**: `_extract_audio()` blocks on `proc.stdout.read()` and cannot be cooperatively interrupted. The cancel button handler always calls `request_cancel()` first, then `worker.wait(300)`. If the worker is still running after 300 ms, call `terminate()` + `wait()`. This `wait(300)` blocks the Qt event loop for up to 300 ms — this is intentional and acceptable for this app. Because `terminate()` kills the thread before `cancelled` can be emitted, the cancel button handler calls `_reset_after_analysis_cancel()` directly after the fallback, rather than relying solely on the signal. The `cancelled` signal also connects to `_reset_after_analysis_cancel()` so the cooperative path is handled identically.

### ExportWorker

Add `_cancel_requested: bool = False` and `request_cancel()`.

Add `cancelled = pyqtSignal()`.

Define a module-level sentinel: `class _CancelledError(Exception): pass`.

In `ExportWorker.run()`, the existing `progress_cb` closure (`cb`) checks `self._cancel_requested` on every call and raises `_CancelledError` when set. This fires on every invocation, including the first call (before any segment is cut), which means a cancel request raised before the first segment is processed is caught immediately — this is correct behaviour.

The `_CancelledError` is caught in a dedicated `except _CancelledError` block placed **before** the existing `except Exception` block:

```python
except _CancelledError:
    if os.path.exists(self.output_path):
        os.remove(self.output_path)
    self.cancelled.emit()
    # do NOT re-raise — let the with block exit cleanly
except Exception as exc:
    ...  # existing error handler
```

Do not re-raise `_CancelledError` after emitting `cancelled`. The `tempfile.TemporaryDirectory` context manager cleans up segment files automatically because the `except _CancelledError` block catches the exception without re-raising, allowing the `with` block to exit normally.

## UI Changes

### Analysis cancel button

A "Cancel" `QPushButton` (`_analysis_cancel_btn`) is added to `progress_page` layout, below `_progress_bar`. It is only visible when analysis is running.

On click:
1. Show `QMessageBox`: "Cancel analysis?" — Yes / No.
2. If No: do nothing.
3. If Yes:
   a. Check `_analysis_worker is not None and _analysis_worker.isRunning()`. If not (worker already finished between dialog open and confirm), do nothing — the normal complete/error flow already handled it.
   b. Disable `_analysis_cancel_btn`.
   c. Call `_analysis_worker.request_cancel()`.
   d. Call `_analysis_worker.wait(300)`.
   e. If `_analysis_worker.isRunning()`: call `terminate()` + `wait()`.
   f. Call `_reset_after_analysis_cancel()`. Note: even if the worker finished cooperatively between steps 3d and 3e (making `isRunning()` return False), `_reset_after_analysis_cancel()` must still be called here — the `cancelled` signal may already be queued in the Qt event loop. The idempotent guard inside that method handles any double-call safely.

`_reset_after_analysis_cancel()`:
- If `_analysis_worker is None`: return immediately (idempotent guard — handles double-call from both the direct path above and the `cancelled` signal connection).
- Sets `_analysis_worker = None`.
- Resets `_progress_bar` to 0 and clears `_status_label`.
- Switches `_waveform_stack` to index 0.
- Re-enables the sliders (`_thr_slider`, `_sil_slider`, `_pad_slider`).
- Re-enables `_export_btn` if `MainView._cached_probs` is not None and `len(_cached_probs) > 0` (i.e., a prior analysis completed). Otherwise leaves `_export_btn` disabled.
- Hides `_analysis_cancel_btn`.
- **Does not touch `MainView._cached_probs`** (preserves prior analysis data for the waveform and sliders).

The `cancelled` signal from `AnalysisWorker` also connects to `_reset_after_analysis_cancel()`. The idempotent guard (`_analysis_worker is None`) means that if the direct path already ran, the signal connection is a no-op.

### Export cancel button

A "Cancel" `QPushButton` (`_export_cancel_btn`) is added to the export row, inserted **between `_export_progress` and `_export_label`** (preserving the layout order: `_export_btn` | `_export_progress` | `_export_cancel_btn` | `_export_label` | stretch). It is hidden by default (`setVisible(False)` in `__init__`).

When export starts (`_on_export`): hide `_export_btn`, clear `_export_label` text, show `_export_progress` and `_export_cancel_btn`. `_back_btn` remains disabled during export (unchanged from current behaviour).

On click:
1. Show `QMessageBox`: "Cancel export? The partial output file will be deleted." — Yes / No.
2. If No: do nothing.
3. If Yes: check `_export_worker is not None and _export_worker.isRunning()`. If not, do nothing. Otherwise: disable `_export_cancel_btn`, call `_export_worker.request_cancel()`.

`_on_export_cancelled()` (connected to `ExportWorker.cancelled`):
- Sets `_export_worker = None`.
- Hides `_export_progress` and `_export_cancel_btn`.
- Clears `_export_label` text.
- Shows `_export_btn`.
- Re-enables `_back_btn`.

## Edge Cases

- **`_on_back` during analysis**: unchanged — still calls `terminate()` + `wait()` directly (user is navigating away; `_reset_after_analysis_cancel()` is not called).
- **`_on_back` during export**: `_back_btn` is disabled during export (set in `_on_export`, re-enabled in `_on_export_complete`, `_on_export_error`, and `_on_export_cancelled`). `_on_back` does not need to handle this case.
- **Double-cancel**: cancel buttons are disabled immediately after the first confirmed click. `_reset_after_analysis_cancel()` is idempotent via the `_analysis_worker is None` guard.

## What Is Not Changed

- `cut_segments_gpu`, `cut_segments_cpu`, and `concat_files` remain pure functions with no cancellation awareness. The checkpoint lives entirely in `ExportWorker.run()` via the `progress_cb`.
- No new widget classes.
- No changes to `ImportView` or `MainWindow`.
