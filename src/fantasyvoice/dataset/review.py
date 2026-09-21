"""Manual listening decisions are separate from ASR output."""
from datetime import datetime, timezone
from pathlib import Path

from .storage import read_jsonl, write_jsonl


def playback_audio(path):
    """Browser PCM preview without amplifying quiet audio or dividing by zero."""
    from io import BytesIO
    import numpy as np
    import soundfile as sf

    waveform, rate = sf.read(str(path), always_2d=True)
    result = {'duration_seconds': len(waveform) / rate, 'wav_bytes': None,
              'playback_gain': 1.0}
    if not len(waveform):
        return dict(result, status='empty')
    if not np.isfinite(waveform).all():
        raise ValueError('Audio contains non-finite samples')
    peak = float(np.max(np.abs(waveform)))
    if peak == 0:
        return dict(result, status='silent')
    # Values outside PCM's range are attenuated for playback only, never amplified.
    result['playback_gain'] = 1 / peak if peak > 1 else 1.0
    buffer = BytesIO()
    sf.write(buffer, waveform * result['playback_gain'], rate, format='WAV', subtype='PCM_16')
    return dict(result, status='playable', wav_bytes=buffer.getvalue())


def save_review(output, key, decision, corrected_text, note):
    output = Path(output)
    results = read_jsonl(output / 'transcripts.jsonl')
    latest = {r['key']: r for r in results}
    if key not in latest or latest[key]['status'] != 'success':
        raise ValueError('Review requires a successful transcript key')
    if decision not in {'accepted', 'rejected', 'pending'}:
        raise ValueError('Unknown review decision')
    if decision == 'accepted' and not corrected_text.strip():
        raise ValueError('Accepted transcript must contain text')
    record = {'key': key, 'decision': decision, 'corrected_text': corrected_text,
              'note': note, 'created_at': datetime.now(timezone.utc).isoformat()}
    records = read_jsonl(output / 'reviews.jsonl')
    records.append(record)
    write_jsonl(output / 'reviews.jsonl', records)
    return record
