import os
import subprocess

from core.utils import _ff_bin, _subprocess_hide_console, run


def nvenc_available():
    try:
        result = subprocess.run(
            [_ff_bin("ffmpeg"), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            **_subprocess_hide_console(),
        )
        return "h264_nvenc" in result.stdout
    except FileNotFoundError:
        return False


def cut_segments_gpu(video_path, segments, tmpdir, progress_cb=None):
    outputs = []
    for i, seg in enumerate(segments):
        out = os.path.join(tmpdir, f"seg_{i:06d}.mp4")
        run([_ff_bin("ffmpeg"), "-y", "-hwaccel", "cuda",
             "-ss", str(seg["start"]), "-to", str(seg["end"]),
             "-i", video_path,
             "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "28", "-rc", "vbr",
             "-c:a", "aac", "-movflags", "+faststart", out])
        outputs.append(out)
        if progress_cb:
            progress_cb(i + 1, len(segments))
    return outputs


def cut_segments_cpu(video_path, segments, tmpdir, progress_cb=None):
    outputs = []
    for i, seg in enumerate(segments):
        out = os.path.join(tmpdir, f"seg_{i:06d}.mp4")
        run([_ff_bin("ffmpeg"), "-y",
             "-ss", str(seg["start"]), "-to", str(seg["end"]),
             "-i", video_path,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-c:a", "aac", "-movflags", "+faststart", out])
        outputs.append(out)
        if progress_cb:
            progress_cb(i + 1, len(segments))
    return outputs


def concat_files(files, output_path, tmpdir):
    list_file = os.path.join(tmpdir, "concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for path in files:
            escaped = path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    run([_ff_bin("ffmpeg"), "-y", "-f", "concat", "-safe", "0",
         "-i", list_file, "-c", "copy", output_path])
