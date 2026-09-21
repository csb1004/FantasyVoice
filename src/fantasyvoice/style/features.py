"""Model-independent style definitions. Missing targets stay null."""
import math
from statistics import median

EMOTIONS = ('angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised')


def emotion_probabilities(labels, scores):
    labels = [label.split('/')[-1].strip().lower() for label in labels]
    labels = ['unknown' if label == '<unk>' else label for label in labels]
    if (len(labels) != 9 or len(scores) != 9 or len(set(labels)) != 9
            or set(labels) != set(EMOTIONS) | {'other', 'unknown'}):
        raise ValueError('Expected the nine emotion2vec labels')
    scores = [float(score) for score in scores]
    if any(not math.isfinite(s) or s < 0 or s > 1 for s in scores) or not math.isclose(sum(scores), 1, abs_tol=.001):
        raise ValueError('Invalid emotion probabilities')
    raw = dict(zip(labels, scores))
    top = max(raw, key=raw.get)
    total = sum(raw[label] for label in EMOTIONS)
    probs = {label: raw[label] / total for label in EMOTIONS} if top in EMOTIONS and total > 0 else None
    return {'raw_probabilities': raw, 'probabilities': probs, 'top_label': top}


def summarize(intervals, duration, phoneme_count, voiced_f0):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid duration')
    merged = []
    for start, end in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError('Invalid speech interval')
        start, end = max(0., start), min(duration, end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    speech = sum(end - start for start, end in merged)
    span = merged[-1][1] - merged[0][0] if merged else 0
    f0 = [float(f) for f in voiced_f0 if math.isfinite(f) and f > 0] if speech else []
    return {
        'speech_intervals_seconds': merged, 'speech_seconds': speech,
        'phoneme_count': phoneme_count,
        'speed_phonemes_per_second': phoneme_count / speech if speech and phoneme_count else None,
        'pause_ratio': max(0., min(1., 1 - speech / span)) if span else None,
        'pitch_f0_median_hz': median(f0) if f0 else None,
        'pitch_voiced_frames': len(f0), 'pitch_semitones': None,
        'pitch_normalization_status': 'awaiting_training_split',
    }
