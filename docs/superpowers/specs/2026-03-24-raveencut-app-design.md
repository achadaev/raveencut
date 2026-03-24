# RaveenCut Desktop App — Design Spec

**Date:** 2026-03-24
**Status:** Approved
**Implements:** `docs/2026-03-24-raveencut-prd.md`

---

## Context

`cut.py` is a working CLI script that detects and removes silence from video files using Silero VAD + ffmpeg GPU encoding. It produces the correct output but has no UI. This spec defines the desktop GUI application that reimplements that logic with a GUI.

`cut.py` is **reference only** — it is not imported. `app.py` contains its own implementations of all processing functions, written from scratch using `cut.py` as the reference implementation.

Target timing (4h input video): audio extract ~3s, VAD ~45s, 1401 segment export ~6.5 minutes.

---

## File Structure

`cut.py` remains as reference. All runnable code lives in two new files:

```
raveencut-2/
├── cut.py              ← reference only, not imported
├── app.py              ← entire GUI + processing logic (~900 lines)
└── requirements.txt    ← new
```

`app.py` internal layout (top to bottom):

```
1.  Imports + constants
2.  ffmpeg helpers:  run(), probe_video_duration_sec(), read_audio_from_video()
3.  Segment helpers: merge_segments(), pad_segments(), probs_to_segments()
4.  Export helpers:  nvenc_available(), cut_segments_gpu(), cut_segments_cpu(), concat_files()
5.  WaveformWidget(QWidget)      — custom QPainter waveform
6.  VideoPlayerWidget(QWidget)   — QMediaPlayer wrapper
7.  ImportView(QWidget)          — drag-and-drop screen
8.  AnalysisWorker(QThread)      — audio extract + VAD background
9.  ExportWorker(QThread)        — ffmpeg export background
10. MainView(QWidget)            — Layout B assembly
11. MainWindow(QMainWindow)      — top-level, switches views
12. __main__ entry point
```

---

## Layout

**Layout B — Editor Style:**

```
┌─────────────────────────────────────────────────┐
│  input.mp4                          [← back]    │
├──────────────────────────┬──────────────────────┤
│                          │  Threshold   [====•] │
│     Video Player         │  Min silence [==•==] │
│     (QVideoWidget)       │  Padding     [===•=] │
│                          │                      │
│  [▶] 0:00:00 / 4:15:43   │  1,401 cuts · -1h12m │
├──────────────────────────┴──────────────────────┤
│  ≋≋≋ Waveform Timeline (scrollable + zoom) ≋≋≋  │
│  [scrollbar]                                    │
├─────────────────────────────────────────────────┤
│  Removed: [0:32–0:35] [1:14–1:16] [2:08–2:11]…  │
├─────────────────────────────────────────────────┤
│  [Export]   → cut_input.mp4                     │
└─────────────────────────────────────────────────┘
```

- Left/right split: fixed ratio ~65/35%; not a draggable splitter for v1
- Minimum window size: 900×600px
- Video player + controls pane is fixed height (video aspect ratio = 16:9, fills width of left column)

---

## Data Flow

### Analysis

1. User drops/browses file → `ImportView` validates extension → emits `file_selected(path)`
2. `MainWindow` switches to `MainView`; shows progress bar + status text in waveform area; video player loaded but paused; sliders and Export disabled
3. `AnalysisWorker`:
   - `read_audio_from_video(path, sampling_rate=16000)` — app.py's own implementation
   - Runs VAD probability forward pass (see **VAD Caching** below) → caches `probs: list[float]`
   - Applies default threshold → `merge_segments()` → `pad_segments()` → initial `speech_segments`
   - Emits `analysis_complete(speech_segments, audio_pcm_downsampled, duration_sec)`
4. `MainView` stores `speech_segments` and initialises `restored_indices = set()`, renders `WaveformWidget`

### Segment Data Model

```python
# After analysis — canonical state
speech_segments: list[dict]    # [{"start": 0.5, "end": 3.2}, ...]  — regions to KEEP
# Silence regions (to remove) = gaps between speech_segments
# restored_indices: set[int]   — indices into silence_regions that the user has restored
#                                 (these will also be kept in export)
```

Helper computed on demand:
```python
def silence_regions(speech_segs, duration):
    """Gaps between speech segments, including leading/trailing silence."""
    regions = []
    prev_end = 0.0
    for seg in speech_segs:
        if seg["start"] > prev_end + 0.001:
            regions.append({"start": prev_end, "end": seg["start"]})
        prev_end = seg["end"]
    if prev_end < duration - 0.001:
        regions.append({"start": prev_end, "end": duration})
    return regions

def export_segments(speech_segs, silence_regs, restored_indices):
    """Merge speech + restored silences, sorted by start time."""
    kept = list(speech_segs)
    for i in restored_indices:
        kept.append(silence_regs[i])
    kept.sort(key=lambda s: s["start"])
    return kept
```

On re-analysis (slider released):
- Recompute `speech_segments` from cached probs
- Clear `restored_indices = set()` (new analysis invalidates manual restores)
- Rerender waveform + chips

### VAD Probability Caching

**This is the mechanism that makes slider re-analysis instant (~50ms vs. 45s).**

In `AnalysisWorker`, instead of calling `get_speech_timestamps()`, call the Silero model directly to collect one probability per frame:

```python
FRAME_SIZE = 512   # Silero's native chunk at 16kHz = 32ms per frame

model.reset_states()
probs = []  # one float per frame, index i → time = i * FRAME_SIZE / 16000

for i in range(0, len(wav), FRAME_SIZE):
    chunk = wav[i : i + FRAME_SIZE]
    if len(chunk) < FRAME_SIZE:
        chunk = torch.nn.functional.pad(chunk, (0, FRAME_SIZE - len(chunk)))
    with torch.no_grad():
        prob = model(chunk, 16000).item()   # returns scalar speech probability
    probs.append(prob)

self.cached_probs = probs   # stored on the worker instance; handed to MainView
```

**Threshold application (fast path):**

```python
def probs_to_segments(probs, threshold, frame_size=512, sr=16000):
    """Convert cached probability array → speech segment list (seconds)."""
    frame_sec = frame_size / sr
    segments = []
    in_speech = False
    start = 0.0
    for i, p in enumerate(probs):
        t = i * frame_sec
        if p >= threshold and not in_speech:
            in_speech = True
            start = t
        elif p < threshold and in_speech:
            in_speech = False
            segments.append({"start": start, "end": t})
    if in_speech:
        segments.append({"start": start, "end": len(probs) * frame_sec})
    return segments
```

Re-analysis call on slider release:
```python
segs = probs_to_segments(cached_probs, threshold)
segs = merge_segments(segs, min_gap=min_silence)
segs = pad_segments(segs, pad=padding, max_duration=duration_sec)
speech_segments = segs
restored_indices.clear()
waveform_widget.update_segments(speech_segments)
```

### Export

1. User clicks Export
2. Pre-check: `os.access(output_dir, os.W_OK)` — if False, show `QMessageBox` with message and abort
3. Compute output path (see **Output Naming**)
4. Detect NVENC availability (see **GPU / CPU Fallback**)
5. Spawn `ExportWorker(video_path, export_segments, output_path, use_gpu)`
6. `ExportWorker`:
   - Create `tempfile.TemporaryDirectory()` — owned and cleaned up by the worker
   - Call `cut_segments_gpu(video_path, segments, tmpdir.name, progress_cb)` for all segments at once (no chunking)
   - Call `concat_files(seg_files, output_path, tmpdir.name)`
   - On exception: delete partial output (`os.path.exists(output_path) and os.remove(output_path)`), emit `error(msg)`
   - On success: emit `export_complete(output_path)`
7. On `export_complete`: `QMessageBox` with "Open File" / "Open Folder" / "Close"

**Export progress:** `progress = segments_done / total_segments * 90` for the cut phase, jumps to 100% after concat. No sub-progress on concat.

### GPU / CPU Fallback

Before export, check NVENC availability:

```python
def nvenc_available():
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True
    )
    return "h264_nvenc" in result.stdout
```

If `nvenc_available()` is True: use `cut_segments_gpu()` as-is (CUDA + h264_nvenc).
If False: use a CPU-codec variant with `-c:v libx264 -preset fast -crf 23` in place of the NVENC flags. `app.py` defines its own `cut_segments_cpu()` function with the same signature but CPU codec. This function is not imported from `cut.py`.

---

## Key Components

### WaveformWidget

- Custom `QWidget` drawn with `QPainter`
- **Downsampling:** audio PCM downsampled to max 2000 display bars; each bar = max absolute amplitude in its window; computed once on `update_audio(pcm_array)`
- Colors: speech (kept) = `#3a7a3a` green; silence (to remove) = `#7a2a2a` red; restored = `#3a5a7a` blue
- **Zoom:** 1× to 100×; mouse wheel ±1 tick = ×1.2 or ÷1.2; zoom centred on cursor x position; default zoom = 1× (full video visible)
- **Scroll:** horizontal `QScrollBar`; auto-scrolls during playback to keep playhead visible (scrolls only when playhead would exit the visible area, not continuously)
- Playhead: white vertical line driven by `QTimer` at 50ms interval polling `QMediaPlayer.position()`
- Click → maps pixel x to time: `t = scroll_offset_sec + (x / widget_width) * visible_duration_sec`; emits `seek_requested(float)`

### VideoPlayerWidget

- `QMediaPlayer` + `QVideoWidget`
- Play/pause toggle button + time label `H:MM:SS / H:MM:SS`
- `positionChanged` signal drives waveform playhead

### Parameters (three QSlider widgets)

| Slider | Display range | Default | QSlider range | Mapping | Maps to |
|---|---|---|---|---|---|
| Speech threshold | 0.10 – 0.90 | 0.50 | 10 – 90 | `value / 100` | `threshold` in `probs_to_segments()` |
| Min silence | 0.1s – 2.0s | 0.6s | 1 – 20 | `value / 10` | `merge_segments(min_gap=...)` |
| Padding | 0.0s – 1.0s | 0.35s | 0 – 20 | `value / 20` | `pad_segments(pad=...)` |

All fire re-analysis on `sliderReleased`. Value label next to each slider updates on `valueChanged` (display only, no re-analysis).

### Removed Segments Strip

- Horizontal scrollable `QScrollArea` containing a `QWidget` with `QHBoxLayout` of chip widgets
- Each chip: `QPushButton` styled as a pill; shows `H:MM:SS–H:MM:SS` for videos ≥ 1h; `M:SS–M:SS` for shorter
- **Two-step restore interaction:**
  1. First click → video seeks to segment start; chip highlights (border + text change to indicate "restore mode"); any previously highlighted chip reverts to normal; label changes to `↩ restore`
  2. Click the same highlighted chip → segment added back to `restored_indices`; chip removed from strip; waveform repaints; pending-restore state cleared
  3. Clicking a different chip while one is highlighted → previous chip reverts; new chip enters highlight state and video seeks

- If re-analysis fires while a chip is in pending-restore state → `restored_indices` cleared, all chips regenerated from new `silence_regions()` — pending state is lost (no warning needed)

### Timestamp display format

```python
def fmt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
```

Used for: chip labels, video player time display, waveform tick marks.

### Workers

Both inherit `QThread`, override `run()`:
- `AnalysisWorker`: emits `progress(int, str)`, `analysis_complete(list, np.ndarray, float)`, `error(str)`
- `ExportWorker`: emits `progress(int, str)`, `export_complete(str)`, `error(str)`
- On `error`: MainView shows inline error message; Export button re-enabled

**In-flight re-analysis policy:** Re-analysis is synchronous in the main thread (fast path, ~50ms). No worker is involved. If a slider is released while `AnalysisWorker` is still running (initial analysis), the slider event is ignored until `analysis_complete` fires.

---

## Screens

### Import Screen

- Large centered drop zone with dashed border: "Drop a video file here"
- "Browse…" button fallback opens `QFileDialog` filtered to `.mp4 .mov .mkv .avi .webm`
- Extension check at drop: if unsupported → inline error label below drop zone; drop zone stays active
- Extension check is case-insensitive

### Main Screen

See Layout section. Shown after `file_selected` fires.

### During Analysis

- Waveform area replaced by `QProgressBar` + status text (e.g. "Extracting audio… 0:42 / 4:15:43")
- Video player visible and loaded (user can preview the raw video while waiting)
- Sliders and Export disabled; back button enabled

### During Export

- Export button replaced by `QProgressBar`
- Sliders and waveform remain visible and interactive (view only, no re-analysis during export)
- Back button disabled during export

### Back Button Behaviour

- During analysis: clicking back aborts `AnalysisWorker` (call `worker.terminate(); worker.wait()`), returns to `ImportView`
- During export: back button is disabled
- Otherwise: returns to `ImportView` immediately, no confirmation dialog

---

## Output Naming

```python
def resolve_output_path(input_path: str) -> str:
    dir_ = os.path.dirname(input_path)
    stem, ext = os.path.splitext(os.path.basename(input_path))
    candidate = os.path.join(dir_, f"cut_{stem}{ext}")
    if not os.path.exists(candidate):
        return candidate
    n = 1
    while True:
        candidate = os.path.join(dir_, f"cut_{stem}_{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1
```

---

## Error Handling

| Situation | User-facing behaviour |
|---|---|
| No audio track | `AnalysisWorker` emits `error("No audio track found. Cannot process this file.")` → inline error in waveform area |
| No speech detected | `AnalysisWorker` emits `warning` → yellow banner above Export: "No speech detected. Try lowering the threshold." Export disabled. |
| Unsupported format | Extension check at drop → red inline label below drop zone |
| Corrupt/unreadable file | ffmpeg pipe empty or error → `error("Could not read file. Try converting to MP4 first.")` |
| Output folder not writable | `os.access` check pre-export → `QMessageBox.warning` with message |
| Export interrupted | Partial output deleted; `QMessageBox.warning` with "Retry" and "Cancel" buttons; Retry re-runs full export from scratch |

---

## Dependencies (requirements.txt)

```
PyQt6
torch
numpy
silero-vad
rich
```

ffmpeg and ffprobe must be on PATH. **Bundling ffmpeg is deferred to the PyInstaller packaging phase.** v1 requires ffmpeg installed separately.

---

## Distribution

Target: PyInstaller single-file `.exe` for Windows. Build pipeline to be defined separately after the app runs from source. This contradicts the PRD "zero additional software" goal for v1; the contradiction is accepted and will be resolved in the packaging phase.

---

## Verification

1. **Import** — drop supported file → transitions to main; drop unsupported → inline error; extension check is case-insensitive
2. **Analysis** — progress shown during extract + VAD; waveform renders after; segment count matches `cut.py` output for the same file and same default parameters
3. **Sliders (display)** — drag slider → value label updates live; no re-analysis fires
4. **Sliders (re-analysis)** — release slider → waveform updates in ~100ms; no VAD re-run (verify by timing: re-analysis < 200ms)
5. **Seek** — click waveform at a time position → video player jumps to correct timestamp (±0.1s)
6. **Restore flow** — click chip → video seeks + chip shows "↩ restore"; click same chip → segment disappears from list and waveform turns green at that region; click different chip while first is highlighted → first chip reverts to normal
7. **Re-analysis clears restore state** — restore a chip, then move a slider and release → chip list regenerates, restored segment no longer shown as blue on waveform
8. **Export (GPU)** — output file at correct path; no chunking (single concat call); duration matches expected; second export → `cut_input_1.mp4`
9. **Export (CPU fallback)** — simulate no NVENC by patching `nvenc_available()` to return False → export completes with libx264
10. **Error states** — video with no audio; read-only output folder; corrupt file
11. **Back during analysis** — click back while AnalysisWorker running → worker aborted, ImportView shown
