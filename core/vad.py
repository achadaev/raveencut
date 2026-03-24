import numpy as np
import torch
import torch.nn.functional as F
from silero_vad import load_silero_vad

from core.constants import FRAME_SIZE, SAMPLING_RATE, WAVEFORM_BARS


def compute_vad_probs(wav, progress_cb=None):
    """Run Silero VAD on wav tensor, return per-frame probabilities.

    progress_cb(done, n_frames) is called every 500 frames if provided.
    """
    model = load_silero_vad(onnx=True)
    model.reset_states()
    probs = []
    n_frames = (len(wav) + FRAME_SIZE - 1) // FRAME_SIZE
    for i in range(0, len(wav), FRAME_SIZE):
        chunk = wav[i: i + FRAME_SIZE]
        if len(chunk) < FRAME_SIZE:
            chunk = F.pad(chunk, (0, FRAME_SIZE - len(chunk)))
        with torch.no_grad():
            probs.append(model(chunk, SAMPLING_RATE).item())
        if progress_cb and len(probs) % 500 == 0:
            progress_cb(len(probs), n_frames)
    return probs


def downsample_pcm(wav_tensor, n_bars=WAVEFORM_BARS):
    arr = wav_tensor.numpy()
    total = len(arr)
    if total == 0:
        return np.zeros(1, dtype=np.float32)
    step = max(1, total // n_bars)
    return np.array([
        np.max(np.abs(arr[i: i + step]))
        for i in range(0, total, step)
    ], dtype=np.float32)[:n_bars]
