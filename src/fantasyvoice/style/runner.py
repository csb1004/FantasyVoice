"""Resumable, single-writer labeling. Source WAVs and ASR outputs are immutable."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from fantasyvoice.dataset.storage import read_jsonl, write_json, write_jsonl


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def label_key(row, config):
    return digest({'config': config, 'audio_path': row['audio_path'],
                   'sha256': row['sha256'], 'text': row['text']})


def run_styles(rows, output, config, analyze, checkpoint_every=20, progress=None):
    if checkpoint_every < 1:
        raise ValueError('checkpoint_every must be positive')
    rows = list(rows)
    if len({r['audio_path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate audio paths')
    output = Path(output)
    config_path = output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Different config: use a new style output directory')
    write_json(config_path, config)
    cache = {}
    shards = {}
    for path in sorted((output / 'results').glob('*.jsonl')):
        records = read_jsonl(path)
        shards[path.name] = {r['style_key']: r for r in records}
        cache.update(shards[path.name])
    keys = [label_key(row, config) for row in rows]
    dirty = set()

    def flush():
        for name in sorted(dirty):
            write_jsonl(output / 'results' / name, shards[name].values())
        dirty.clear()

    def export():
        data = [dict(row, **cache.get(key, {'style_key': key, 'style_status': 'pending'}))
                for row, key in zip(rows, keys)]
        write_jsonl(output / 'style-labels.jsonl', data)
        write_jsonl(output / 'style-needs-review.jsonl',
                    [r for r in data if r['style_status'] in ('partial', 'error')])
        summary = {'total': len(rows), 'counts': dict(Counter(r['style_status'] for r in data)),
                   'training_ready': False, 'normalization': 'awaiting_training_split',
                   'source_signature': digest(rows), 'config_sha256': digest(config)}
        write_json(output / 'summary.json', summary)
        return summary

    try:
        for index, (row, key) in enumerate(zip(rows, keys), 1):
            cached = cache.get(key)
            if cached and cached['style_status'] in ('generated', 'partial', 'excluded'):
                continue
            record = {'style_key': key}
            try:
                if row['duration_seconds'] <= .5 or not row['text'].strip():
                    record.update(style_status='excluded', style_reason='short_or_empty_text')
                else:
                    values = analyze(row)
                    # Reject NaNs before checkpointing so a bad file cannot poison the run.
                    digest(values)
                    record.update(style_status='partial' if values.get('issues') else 'generated',
                                  pseudo_style=values)
            except Exception as exc:
                record.update(style_status='error', style_error=f'{type(exc).__name__}: {exc}')
                if 'out of memory' in str(exc).lower():
                    raise
            finally:
                if 'style_status' in record:
                    name = key[:2] + '.jsonl'
                    cache[key] = record
                    shards.setdefault(name, {})[key] = record
                    dirty.add(name)
            if index % checkpoint_every == 0:
                flush()
                if progress:
                    progress(index, len(rows), record['style_status'])
    finally:
        flush()
        summary = export()
    return summary
