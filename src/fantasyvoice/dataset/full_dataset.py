"""Full-corpus ASR with bounded checkpoints and separate human review exports."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unicodedata

import numpy as np
import soundfile as sf

from .storage import check_output, read_jsonl, resolve_audio, sha256, write_json, write_jsonl


def eligible_rows(rows):
    return sorted((r for r in rows if r['status'] == 'ok' and r['duration_seconds'] > .5),
                  key=lambda r: r['audio_path'])


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _identity(config, row):
    return _digest({'config': config, 'audio_path': row['audio_path'], 'sha256': row['sha256']})


def _source_id(row):
    return unicodedata.normalize('NFC', row['audio_path']), row['sha256']


def _load_shards(output):
    return {p.name: read_jsonl(p) for p in sorted((Path(output) / 'results').glob('*.jsonl'))}


def _latest(shards):
    return {r['key']: r for records in shards.values() for r in records}


def export_dataset(rows, output, reviews=()):
    """ASR candidates are not automatically human-approved training targets."""
    output = Path(output)
    latest = _latest(_load_shards(output))
    review_by_key, review_by_source = {}, {}
    for review in sorted(reviews, key=lambda r: r.get('created_at', '')):
        if 'key' in review:
            review_by_key[review['key']] = review
        if review.get('audio_path') and review.get('sha256'):
            review_by_source[_source_id(review)] = review
    sources = {r['audio_path']: r for r in eligible_rows(rows)}
    candidates, reviewed, issues = [], [], []
    for result in latest.values():
        source = sources.get(result['audio_path'])
        if source is None or source['sha256'] != result['sha256']:
            continue
        possible = [r for r in (review_by_key.get(result['key']),
                                review_by_source.get(_source_id(result))) if r]
        review = max(possible, key=lambda r: r.get('created_at', '')) if possible else None
        decision = review['decision'] if review else 'pending'
        asr_text = result.get('asr', {}).get('text', '')
        record = {'key': result['key'], 'audio_path': source['audio_path'],
                  'character_id': source['character_id'], 'sha256': source['sha256'],
                  'duration_seconds': source['duration_seconds'], 'asr_text': asr_text,
                  'text': review['corrected_text'] if review else asr_text,
                  'review_status': decision, 'asr_status': result['status'],
                  'note': review.get('note', '') if review else '',
                  'style_status': 'not_generated', 'split': None}
        if result['status'] != 'success' or decision == 'rejected' or not record['text'].strip():
            record['reason'] = result.get('error') or result.get('reason') or ('human_rejected' if decision == 'rejected' else 'empty_transcript')
            issues.append(record)
        else:
            candidates.append(record)
            if decision == 'accepted':
                reviewed.append(record)
    write_jsonl(output / 'candidates.jsonl', candidates)
    write_jsonl(output / 'reviewed.jsonl', reviewed)
    write_jsonl(output / 'needs-review.jsonl', issues)
    report = {'candidate_count': len(candidates), 'human_accepted_count': len(reviewed),
              'issue_count': len(issues), 'candidate_review_states': dict(Counter(r['review_status'] for r in candidates)),
              'training_ready': False, 'remaining': ['pseudo style labels', 'split policy', 'ASR quality acceptance policy']}
    write_json(output / 'export-summary.json', report)
    return report


def run_dataset(root, rows, output, config, transcribe, *, reviews=(),
                pilot_output=None, checkpoint_every=20, progress=None):
    """Resume against an immutable reference snapshot; one writer per output.

    Checkpoint at most 20 new attempts by default. A hard runtime termination may
    require repeating the uncheckpointed tail. KeyboardInterrupt flushes it.
    Completed records are reused without rereading Drive; refresh the reference
    and start a new output if sources change.
    """
    output = Path(output)
    check_output(root, output)
    if checkpoint_every < 1:
        raise ValueError('checkpoint_every must be positive')
    if (config.get('model'), config.get('language'), config.get('task')) != ('large-v3', 'ko', 'transcribe') or not config.get('revision'):
        raise ValueError('config must select versioned large-v3 Korean transcription')
    config = json.loads(json.dumps(config))
    sources = eligible_rows(rows)
    if not sources:
        raise ValueError('No files longer than 0.5 seconds')
    signature = _digest([{k: r[k] for k in ('audio_path', 'character_id', 'sha256', 'duration_seconds')} for r in sources])
    for name, value in [('run.json', config), ('dataset.json', {'inventory_signature': signature, 'count': len(sources), 'duration_exclusion_seconds': .5})]:
        path = output / name
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
            kind = 'config' if name == 'run.json' else 'inventory'
            raise ValueError(f'{kind} changed; use a new dataset output directory')
    if not (output / 'run.json').exists() and (output / 'results').exists():
        raise ValueError('Missing config for existing checkpoints')
    write_json(output / 'run.json', config)
    write_json(output / 'dataset.json', {'inventory_signature': signature, 'count': len(sources), 'duration_exclusion_seconds': .5})
    write_jsonl(output / 'excluded-duration.jsonl', [r for r in rows if r['status'] == 'ok' and r['duration_seconds'] <= .5])
    shards = _load_shards(output)
    latest = _latest(shards)
    imported = {}
    if pilot_output:
        pilot = Path(pilot_output)
        if (pilot / 'run.json').exists() and json.loads((pilot / 'run.json').read_text(encoding='utf-8')) == config:
            for r in read_jsonl(pilot / 'transcripts.jsonl'):
                imported[_source_id(r)] = r
    dirty = set()
    pending_writes = 0

    def stats():
        counts = Counter(r['status'] for r in latest.values())
        return {'total': len(sources), 'success': counts['success'],
                'needs_review': counts['needs_review'], 'error': counts['error'],
                'unprocessed': len(sources)-len(latest)}

    def flush():
        for name in sorted(dirty):
            write_jsonl(output / 'results' / name, shards[name])
        dirty.clear()
        state = stats()
        write_json(output / 'progress.json', state)
        if progress:
            progress(state)

    try:
        with tempfile.TemporaryDirectory(prefix='fantasyvoice-asr-') as local_cache:
            local_audio = Path(local_cache) / 'source.wav'
            for index, source in enumerate(sources):
                key = _identity(config, source)
                if key in latest and latest[key]['status'] in {'success', 'needs_review'}:
                    continue
                result = {k: source[k] for k in ('audio_path', 'character_id', 'sha256')}
                result.update(key=key, created_at=datetime.now(timezone.utc).isoformat())
                fatal = None
                try:
                    remote = resolve_audio(root, source['audio_path'])
                    # Read Drive once; hash, decode and ASR subsequently read local bytes.
                    shutil.copyfile(remote, local_audio)
                    if sha256(local_audio) != source['sha256']:
                        raise ValueError('Source changed; refresh reference inventory')
                    waveform, _ = sf.read(str(local_audio), always_2d=True)
                    if not waveform.size or not np.isfinite(waveform).all() or not np.any(waveform):
                        result.update(status='needs_review', reason='empty_silent_or_nonfinite_audio')
                    else:
                        cached = imported.get(_source_id(source))
                        asr = cached['asr'] if cached and cached['status'] == 'success' else transcribe(local_audio)
                        if not isinstance(asr.get('text'), str) or not isinstance(asr.get('segments'), list):
                            raise ValueError('Invalid ASR result')
                        json.dumps(asr, allow_nan=False)
                        result.update(status='success' if asr['text'].strip() else 'needs_review', asr=asr)
                        if not asr['text'].strip():
                            result['reason'] = 'empty_transcript'
                except Exception as exc:
                    result.update(status='error', error=f'{type(exc).__name__}: {exc}')
                    if isinstance(exc, MemoryError) or 'out of memory' in str(exc).lower():
                        fatal = exc
                name = f'{index // 100:05d}.jsonl'
                shards.setdefault(name, []).append(result)
                latest[key] = result
                dirty.add(name)
                pending_writes += 1
                if pending_writes >= checkpoint_every:
                    flush()
                    pending_writes = 0
                if fatal:
                    raise fatal
    finally:
        flush()
        export_dataset(rows, output, reviews)
    return stats()
