# RaveenCut

Automatically removes silence from video files. Import a video, analyze it, review the segments, and export a trimmed version — no video editing experience needed.

---

## Requirements

- Python 3.7+
- [FFmpeg](https://ffmpeg.org/download.html) — must be installed and available on your PATH
- NVIDIA GPU with CUDA (optional — speeds up export)

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python app.py
```

---

## How to Use

### 1. Import a Video

Drop a video file onto the import screen, or click **Browse…** to open a file dialog.

Supported formats: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`

### 2. Analyze

The app automatically analyzes your video after import. A progress bar shows the status. When complete, the waveform appears and silence regions are detected.

**Waveform colors:**
- **Green** — speech (kept)
- **Red** — silence (will be removed)
- **Blue** — silence you have manually restored

Click anywhere on the waveform to seek the video to that position. Use the scroll wheel to zoom in/out, and the scrollbar to navigate.

### 3. Adjust Parameters

Use the sliders on the right to tune detection:

| Slider | What it does | Default |
|--------|-------------|---------|
| **Threshold** | How confident the model must be to call something speech. Higher = stricter. | 0.50 |
| **Min silence** | Minimum gap length (in seconds) to remove. Shorter silences are kept. | 0.6s |
| **Padding** | Buffer of audio kept before and after each speech segment. | 0.35s |

Releasing a slider re-analyzes the video with the new parameters.

The **stats panel** shows the kept duration, percentage of original, and number of segments.

### 4. Restore Silence Segments (Optional)

Below the waveform, a strip of buttons shows each detected silence region by time range (e.g., `0:05–0:12`).

- **First click** — seeks the video to that segment so you can preview it
- **Second click** — restores the segment (keeps it in the final export)

Restored segments turn blue on the waveform and disappear from the strip.

### 5. Export

Click **Export** when you're ready. The app cuts each speech segment and concatenates them into a single output file.

- Output is saved in the same folder as the original, with a `cut_` prefix (e.g., `cut_myvideo.mp4`)
- If a file with that name already exists, a number is appended (e.g., `cut_myvideo_1.mp4`)
- GPU acceleration (NVENC) is used automatically if available

When export finishes, a dialog lets you **Open File** (opens the file selected in Explorer), **Open Folder**, or **Close**.

### 6. Start Over

Click **← Back** in the top-left to return to the import screen and load a different video.

---

## Running Tests

Install dev dependencies first:

```bash
pip install -r requirements-dev.txt
```

**Fast tests** (unit + widget + i18n, ~seconds):

```bash
pytest tests/test_processing.py tests/test_gui.py tests/test_i18n.py -v
```

**E2E fast scenarios** (real analysis on `input.mp4`, skips the slow export test, ~1–3 min):

```bash
pytest tests/test_e2e.py -m "not slow" -v
```

**Full E2E suite** including the export acceptance test (verifies the default export is 1:28:47, ~15–40 min depending on GPU):

```bash
pytest tests/test_e2e.py -v
```

**All tests**:

```bash
pytest -v
```

> The E2E tests require `input.mp4` to be present at the repo root, FFmpeg on PATH, and the Silero VAD model downloadable on first run.

---

## Command-Line Usage

For batch processing without the GUI:

```bash
python cut.py <video_file> [options]
```

Options mirror the GUI sliders — run `python cut.py --help` for details.
