"""Inventory WAV headers with libsndfile; never modify source audio."""
from collections import Counter, defaultdict
from pathlib import Path
import soundfile as sf

from .storage import read_jsonl, resolve_audio, sha256


def load_reference(path):
    """Load a local snapshot, NOT a verification of the remote Drive contents.

    run_pilot must still verify each selected source's SHA256 before use.
    """
    rows = read_jsonl(path)
    if not rows:
        raise ValueError('Reference inventory is missing or empty; rebuild the bundle locally')
    required = {'audio_path', 'character_id', 'sha256', 'status'}
    if any(not required.issubset(row) for row in rows):
        raise ValueError('Invalid reference inventory')
    return rows


def scan(root, progress=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Voice directory not found: {root}')
    paths = sorted((p for p in root.rglob('*') if p.suffix.lower() == '.wav' and p.is_file()),
                   key=lambda p: p.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f'No WAV files found: {root}')
    rows = []
    for index, path in enumerate(paths):
        relative = path.relative_to(root).as_posix()
        row = {'schema_version': 1, 'audio_path': relative,
               'character_id': path.relative_to(root).parts[0] if path.parent != root else None,
               'sha256': None, 'status': 'error'}
        try:
            path = resolve_audio(root, relative)
            row['sha256'] = sha256(path)
            info = sf.info(str(path))
            if row['character_id'] is None:
                raise ValueError('Expected a character subdirectory')
            row.update(status='ok', sample_rate=info.samplerate, channels=info.channels,
                       frames=info.frames, duration_seconds=info.duration, subtype=info.subtype,
                       format=info.format)
        except (OSError, RuntimeError, ValueError) as exc:
            message = exc.error_string if isinstance(exc, sf.LibsndfileError) else str(exc)
            row['error'] = f'{type(exc).__name__}: {message}'
        rows.append(row)
        if progress and ((index + 1) % 500 == 0 or index + 1 == len(paths)):
            progress(index + 1, len(paths))
    return rows


def select_pilot(rows, per_character=3):
    """Deterministic length coverage, plus one attack-type clip when present.

    Exclude recordings of 0.5 seconds or less before all sampling branches.
    This duration rule alone does not establish training suitability.
    """
    if per_character < 1:
        raise ValueError('per_character must be positive')
    groups = defaultdict(list)
    for row in rows:
        if row['status'] == 'ok' and row['duration_seconds'] > 0.5:
            groups[row['character_id']].append(row)
    selected = {}
    for character in sorted(groups):
        clips = sorted(groups[character], key=lambda r: (r['duration_seconds'], r['audio_path']))
        count = min(per_character, len(clips))
        indexes = [len(clips) // 2] if count == 1 else [round(i * (len(clips)-1) / (count-1)) for i in range(count)]
        for i in indexes:
            selected[clips[i]['audio_path']] = clips[i]
        attacks = [r for r in clips if any(t in r['audio_path'].lower() for t in ('attack', 'damage', 'death'))]
        if attacks:
            selected[attacks[0]['audio_path']] = attacks[0]
    return [selected[p] for p in sorted(selected)]


def summarize(rows):
    ok = [r for r in rows if r['status'] == 'ok']
    characters = defaultdict(lambda: {'files': 0, 'seconds': 0})
    for r in ok:
        entry = characters[r['character_id']]
        entry['files'] += 1
        entry['seconds'] += r['duration_seconds']
    return {'files': len(rows), 'readable_headers': len(ok), 'errors': len(rows)-len(ok),
            'duration_hours': sum(r['duration_seconds'] for r in ok)/3600,
            'formats': dict(Counter(f"{r['sample_rate']}Hz/{r['channels']}ch/{r['subtype']}" for r in ok)),
            'characters': dict(characters), 'validation': 'headers_and_sha256_only'}
