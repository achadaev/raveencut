# RaveenCut — Product Requirements Document

**Status:** v1.0
**Platforms:** Windows, Linux

---

## Problem Statement

Content creators recording talking-head videos consistently produce raw footage bloated with dead air, filler pauses, and hesitation gaps. Trimming these silences manually is tedious, time-consuming, and requires proficiency in full-featured editing software. There is no lightweight, purpose-built tool that does this one job well.

---

## Product Vision

RaveenCut is a desktop application that automatically detects and removes silence from video files. A creator imports a recording, reviews and adjusts the proposed cuts, and exports a tightened video — in minutes, with no editing expertise required.

---

## Target Users

**Primary:** Independent content creators and YouTubers recording talking-head or voiceover-driven videos.

**Key pain point:** Dead air and hesitation pauses between sentences inflate recording length and require tedious manual trimming in general-purpose editors.

---

## Success Metrics

- Time from import to export under 2 minutes for a typical 10-minute recording
- Zero additional software required — works out of the box after installation
- Creators can recover segments they want to keep without re-importing

---

## Core Capabilities

### 1. Silent Segment Detection
The app analyzes the audio track of an imported video and automatically identifies silence regions. Users control two parameters: the volume threshold below which audio is considered silent, and the minimum duration a silent gap must hold before it qualifies for removal.

### 2. Interactive Waveform Timeline
Detected silence regions are visualized on a scrollable audio waveform. Speech is shown in one color; silences marked for removal in another. A playhead tracks the video position. Users can click any segment to seek the video player to that point.

### 3. Live Re-analysis
Adjusting either detection parameter triggers an immediate re-analysis. Any in-flight analysis is cancelled before the new one starts. The timeline updates in real time as results stream in.

### 4. Selective Restore
Removed segments appear as a scrollable list below the timeline. Clicking any item restores it — adding it back to the exported video. This lets users preserve intentional pauses, b-roll gaps, or any silence they want to keep.

### 5. One-click Export
Exporting produces a new video file with all marked silences removed. The output is saved alongside the original file with a `cut_` prefix. If a file by that name already exists, the output is auto-incremented. Progress is shown throughout; on completion the user can open the file or its containing folder directly.

---

## User Flow

1. Launch → Import screen
2. Drop a video file onto the window (or browse for one)
3. App analyses the file and renders the waveform timeline
4. Review detected silences; adjust sliders if needed
5. Restore any segments to keep
6. Export → receive trimmed video

---

## Key Constraints

- Works fully offline — no account, no cloud processing
- No admin rights required for installation
- Accepts: MP4, MOV, MKV, AVI, WebM
- Single file at a time (batch processing is out of scope for v1)
- Audio-only files are out of scope for v1

---

## Error States

| Situation | User-facing behavior |
|---|---|
| No audio track in video | Inline error; cannot proceed |
| No speech detected | Warning before export with guidance to lower threshold |
| Unsupported file format | Inline error at drop/import |
| Corrupt or unreadable file | Inline error; suggest converting to MP4 |
| Output folder not writable | Pre-export check with clear error message |
| Export interrupted | Partial output deleted; retry offered |

---

## Out of Scope (v1)

- Audio-only files
- Batch processing
- Custom output directory
- Undo/redo
- Auto-updates
- macOS
