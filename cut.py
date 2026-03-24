import json
import os  # filesystem operations
import shutil  # check ffmpeg availability
import subprocess  # run ffmpeg
import tempfile  # temp directories

import numpy as np  # buffer → array
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from silero_vad import get_speech_timestamps, load_silero_vad


# =========================
# CONFIG
# =========================
INPUT_VIDEO = r"C:/Users/Andrey/Downloads/input.mp4"
CHUNK_SIZE = 1000  # number of segments per batch (stability vs speed)


# =========================
# UTIL
# =========================
def run(cmd):
    """
    Execute ffmpeg command and surface errors.
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("FFmpeg failed")
    return result


def probe_video_duration_sec(video_path):
    """Return container duration in seconds (ffprobe), or None if unavailable."""
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(json.loads(result.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, TypeError, ValueError):
        return None


# =========================
# AUDIO VIA PIPE (NO FILE)
# =========================
def read_audio_from_video(video_path: str, sampling_rate: int = 16000, console=None) -> torch.Tensor:
    """
    Extract mono float32 PCM audio directly from video using ffmpeg pipe.
    Avoids writing large temporary WAV files.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")

    cmd = [
        "ffmpeg",
        "-i", video_path,

        # output raw PCM to stdout
        "-f", "f32le",          # float32 little-endian
        "-ac", "1",             # mono
        "-ar", str(sampling_rate),
        "-"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**6,
    )

    if console is None:
        raw_audio = process.stdout.read()
        process.wait()
        audio = np.frombuffer(raw_audio, dtype=np.float32).copy()
        return torch.from_numpy(audio)

    duration_sec = probe_video_duration_sec(video_path)
    total_units = int(duration_sec * sampling_rate) if duration_sec else None
    chunk_bytes = int(sampling_rate * 2.0) * 4
    parts = []
    total_samples = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Extract audio", total=total_units)
        while True:
            raw = process.stdout.read(chunk_bytes)
            if not raw:
                break
            parts.append(np.frombuffer(raw, dtype=np.float32).copy())
            n = len(raw) // 4
            total_samples += n
            sec = total_samples / sampling_rate
            desc = f"Extract audio — {sec:.1f}s"
            if duration_sec is not None:
                desc += f" / {duration_sec:.1f}s"
            if total_units is not None:
                progress.update(
                    task_id,
                    completed=min(total_samples, total_units),
                    description=desc,
                )
            else:
                progress.update(task_id, advance=n, description=desc)

    process.wait()
    if not parts:
        return torch.from_numpy(np.array([], dtype=np.float32))
    audio = np.concatenate(parts)
    return torch.from_numpy(audio)


# =========================
# SEGMENT PROCESSING
# =========================
def merge_segments(segments, min_gap=0.6):
    """
    Merge segments separated by small gaps.
    Reduces fragmentation → improves performance.
    """
    merged = []

    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]

        if seg["start"] - prev["end"] < min_gap:
            prev["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged


def pad_segments(segments, pad=0.35, max_duration=None):
    """
    Add padding to avoid cutting speech too tightly.
    """
    padded = []

    for seg in segments:
        start = max(0, seg["start"] - pad)
        end = seg["end"] + pad

        if max_duration:
            end = min(max_duration, end)

        padded.append({"start": start, "end": end})

    return padded


def chunk_segments(segments, chunk_size):
    """
    Split segments into batches.
    Prevents too many ffmpeg calls at once.
    """
    for i in range(0, len(segments), chunk_size):
        yield segments[i:i + chunk_size]


# =========================
# GPU CUTTING (NO FILTERS)
# =========================
def cut_segments_gpu(video_path, segments, tmpdir, progress=None, task_id=None):
    """
    Cut segments using GPU decode + encode.
    No filter_complex → no CPU frame processing.
    """
    outputs = []

    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]

        out_file = os.path.join(tmpdir, f"seg_{i}.mp4")

        cmd = [
            "ffmpeg", "-y",

            # GPU decode (NVDEC)
            "-hwaccel", "cuda",

            # fast seek (keyframe-based)
            "-ss", str(start),
            "-to", str(end),
            "-i", video_path,

            # GPU encode (NVENC)
            "-c:v", "h264_nvenc",
            "-preset", "p1",   # fastest preset
            "-cq", "28",       # quality/bitrate tradeoff
            "-rc", "vbr",

            # audio re-encode
            "-c:a", "aac",

            # playback optimization
            "-movflags", "+faststart",

            out_file
        ]

        run(cmd)
        outputs.append(out_file)
        if progress is not None and task_id is not None:
            progress.advance(task_id)

    return outputs


# =========================
# CONCAT
# =========================
def concat_files(files, output_path, tmpdir):
    """
    Concatenate video files without re-encoding.
    """
    list_file = os.path.join(tmpdir, "concat.txt")

    with open(list_file, "w", encoding="utf-8") as f:
        for path in files:
            f.write(f"file '{path}'\n")

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ])


# =========================
# PIPELINE
# =========================
def process_video(video_path, segments, output_path, console=None):
    """
    Full pipeline:
    - batch segments
    - GPU cut
    - concat batches
    """
    console = console or Console()
    num_batches = (len(segments) + CHUNK_SIZE - 1) // CHUNK_SIZE
    with tempfile.TemporaryDirectory() as tmpdir:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            cut_task = progress.add_task(
                "[cyan]Cut segments (GPU)",
                total=len(segments),
            )
            batch_task = progress.add_task(
                "[magenta]Merge batches",
                total=num_batches + 1,
            )
            batch_outputs = []
            for batch_idx, batch in enumerate(chunk_segments(segments, CHUNK_SIZE)):
                seg_files = cut_segments_gpu(
                    video_path, batch, tmpdir, progress=progress, task_id=cut_task
                )
                batch_file = os.path.join(tmpdir, f"batch_{batch_idx}.mp4")
                concat_files(seg_files, batch_file, tmpdir)
                progress.advance(batch_task)
                batch_outputs.append(batch_file)
            progress.update(
                batch_task,
                description="[magenta]Final concat",
            )
            concat_files(batch_outputs, output_path, tmpdir)
            progress.advance(batch_task)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    console = Console()
    model = load_silero_vad(onnx=True)

    console.print("[bold green]Extracting audio...[/]")
    wav = read_audio_from_video(INPUT_VIDEO, sampling_rate=16000, console=console)

    duration_sec = len(wav) / 16000

    console.print("[bold green]Detecting speech (VAD)...[/]")
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        vad_task = progress.add_task("Speech detection (VAD)", total=None)
        speech_segments = get_speech_timestamps(
            wav,
            model,
            sampling_rate=16000,
            return_seconds=True,
        )
        progress.update(
            vad_task,
            description="Speech detection (VAD) — done",
        )

    if not speech_segments:
        raise ValueError("No speech detected")

    speech_segments = merge_segments(speech_segments, min_gap=0.6)
    speech_segments = pad_segments(
        speech_segments,
        pad=0.25,
        max_duration=duration_sec,
    )

    console.print(f"[bold]Segments after processing:[/] {len(speech_segments)}")

    output_video = os.path.join(
        os.path.dirname(INPUT_VIDEO),
        "cut_" + os.path.basename(INPUT_VIDEO)
    )

    process_video(INPUT_VIDEO, speech_segments, output_video, console=console)
    console.print(f"[bold green]Saved:[/] {output_video}")