from core.constants import FRAME_SIZE, SAMPLING_RATE


def probs_to_segments(probs, threshold, frame_size=FRAME_SIZE, sr=SAMPLING_RATE):
    frame_sec = frame_size / sr
    segments, in_speech, start = [], False, 0.0
    for i, p in enumerate(probs):
        t = i * frame_sec
        if p >= threshold and not in_speech:
            in_speech, start = True, t
        elif p < threshold and in_speech:
            in_speech = False
            segments.append({"start": start, "end": t})
    if in_speech:
        segments.append({"start": start, "end": len(probs) * frame_sec})
    return segments


def merge_segments(segments, min_gap=0.6):
    merged = []
    for seg in segments:
        if not merged:
            merged.append(dict(seg))
            continue
        prev = merged[-1]
        if seg["start"] - prev["end"] < min_gap:
            prev["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged


def pad_segments(segments, pad=0.35, max_duration=None):
    padded = []
    for seg in segments:
        start = max(0.0, seg["start"] - pad)
        end = seg["end"] + pad
        if max_duration is not None:
            end = min(max_duration, end)
        padded.append({"start": start, "end": end})
    return padded


def silence_regions(speech_segs, duration):
    regions, prev_end = [], 0.0
    for seg in speech_segs:
        if seg["start"] > prev_end + 0.001:
            regions.append({"start": prev_end, "end": seg["start"]})
        prev_end = seg["end"]
    if prev_end < duration - 0.001:
        regions.append({"start": prev_end, "end": duration})
    return regions


def export_segments_fn(speech_segs, silence_regs, restored_indices):
    kept = list(speech_segs) + [silence_regs[i] for i in restored_indices]
    kept.sort(key=lambda s: s["start"])
    return kept
