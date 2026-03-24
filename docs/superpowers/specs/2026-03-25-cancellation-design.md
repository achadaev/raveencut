# Cancellation Design — Analysis & Export

**Date:** 2026-03-25

## Overview

Add user-initiated cancellation for both the analysis and export operations in RaveenCut. Both operations run in background QThreads and can take significant time; users need a way to abort them without killing the app.

## Decisions

- Cancelling analysis keeps the user on MainView (not sent back to ImportView).
- Partial export output files are silently deleted on cancel.
- Both cancel actions require a confirmation dialog.
- Implementation uses cooperative cancellation (flag checked at loop checkpoints), with `QThread.terminate()` as a fallback only during audio extraction (a brief blocking phase).

## Worker Changes

### AnalysisWorker

Add `_cancel_requested: bool = False` attribute and a `request_cancel()` method that sets it.

Add `cancelled = pyqtSignal()`.

In `_compute_probs()`, check `_cancel_requested` every 500 frames (same cadence as the existing progress emit). If set, raise an internal sentinel or return `None` to signal cancellation.

In `run()`, detect cancellation return from `_compute_probs` and emit `cancelled` instead of `analysis_complete`.

The `_extract_audio()` phase blocks on `proc.stdout.read()` and cannot be cooperatively interrupted. If the user cancels during this brief phase, the MainView falls back to `QThread.terminate()` + `wait()` after calling `request_cancel()`.

### ExportWorker

Add `_cancel_requested: bool = False` and `request_cancel()`.

Add `cancelled = pyqtSignal()`.

In `cut_segments_gpu` / `cut_segments_cpu`, accept and check an optional `cancel_check` callable before each segment cut. If it returns `True`, raise a sentinel exception that the worker catches.

Alternatively (simpler): check `self._cancel_requested` directly inside `ExportWorker.run()` in the progress callback `cb`, raising a local sentinel. The `cut_segments_*` functions already accept a `progress_cb` — this callback can also serve as the cancellation checkpoint.

On cancellation, the worker:
1. Deletes `self.output_path` if it exists (same logic as the existing error handler).
2. Emits `cancelled`.

The `tempfile.TemporaryDirectory` context manager cleans up segment files automatically.

## UI Changes

### Analysis cancel button

A "Cancel" `QPushButton` is added to `progress_page` layout (below `_progress_bar`). It is shown only when analysis is running (i.e., while `_waveform_stack` is on index 1).

On click:
1. Show `QMessageBox`: "Cancel analysis?" — Yes / No.
2. If Yes: disable the cancel button (prevent double-click), call `_analysis_worker.request_cancel()`. If the worker is still running after the flag is set (audio extraction phase), also call `terminate()` + `wait()`.
3. The `cancelled` signal handler resets the progress bar to 0, clears the status label, and switches `_waveform_stack` to index 0. If prior analysis data exists (`_cached_probs` non-empty), the waveform and sliders remain intact. If no prior data, the export button stays disabled.

### Export cancel button

A "Cancel" `QPushButton` is added to the export row. It is hidden by default and shown when export starts (at the same time `_export_btn` is hidden and `_export_progress` is shown).

On click:
1. Show `QMessageBox`: "Cancel export? The partial output file will be deleted." — Yes / No.
2. If Yes: disable the cancel button, call `_export_worker.request_cancel()`.
3. The `cancelled` signal handler: hides `_export_progress` and the cancel button, shows `_export_btn`, re-enables `_back_btn`.

## Edge Cases

- **Worker finishes before confirmation**: guard with `worker is not None and worker.isRunning()` before calling `request_cancel()`. If already done, do nothing (the normal complete/error flow handles it).
- **`_on_back` during analysis**: unchanged — still calls `terminate()` + `wait()` directly (user is navigating away; the `cancelled` signal flow is not needed).
- **Double-cancel**: cancel button is disabled immediately after the first confirmed click.

## What Is Not Changed

- `cut_segments_gpu`, `cut_segments_cpu`, and `concat_files` remain pure functions with no cancellation awareness. The checkpoint is in `ExportWorker.run()` via the `progress_cb`.
- No new widget classes.
- No changes to `ImportView` or `MainWindow`.
