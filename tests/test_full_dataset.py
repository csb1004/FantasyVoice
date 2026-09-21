import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fantasyvoice.dataset.inventory import scan
from fantasyvoice.dataset.storage import read_jsonl
from fantasyvoice.dataset.full_dataset import run_dataset, export_dataset


CONFIG = {'model': 'large-v3', 'revision': 'fixture', 'language': 'ko',
          'task': 'transcribe', 'options': {}}


def sources(root, lengths=(.5, .6, 1.0), silent=False):
    for i, length in enumerate(lengths):
        p = root / 'A' / f'{i}.wav'
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, np.full(int(length*16000), 0.0 if silent else .1), 16000)
    return scan(root)


def decode(path):
    assert path.exists()
    return {'text': '자동 전사', 'segments': [], 'language': 'ko'}


def test_full_run_filters_boundary_and_resumes_without_retranscription(tmp_path):
    root = tmp_path / 'Voice'
    rows = sources(root)
    out = tmp_path / 'dataset'
    stats = run_dataset(root, rows, out, CONFIG, decode, checkpoint_every=1)
    assert stats['total'] == 2 and stats['success'] == 2
    def forbidden(path):
        raise AssertionError('Completed checkpoint must not be retranscribed')
    stats = run_dataset(root, rows, out, CONFIG, forbidden)
    assert stats['success'] == 2
    candidates = read_jsonl(out / 'candidates.jsonl')
    assert len(candidates) == 2
    assert all(r['review_status'] == 'pending' for r in candidates)
    assert read_jsonl(out / 'reviewed.jsonl') == []
    assert not list(root.rglob('*.json*'))


def test_full_run_flushes_on_interrupt_and_retries_failed_file(tmp_path):
    root = tmp_path / 'Voice'
    rows = sources(root, (1., 1., 1.))
    out = tmp_path / 'dataset'
    calls = []
    def interrupted(path):
        calls.append(path)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return decode(path)
    with pytest.raises(KeyboardInterrupt):
        run_dataset(root, rows, out, CONFIG, interrupted, checkpoint_every=20)
    assert len(read_jsonl(out / 'candidates.jsonl')) == 1
    assert run_dataset(root, rows, out, CONFIG, decode)['success'] == 3


def test_full_run_detects_silence_and_preserves_reviewed_correction(tmp_path):
    root = tmp_path / 'Voice'
    rows = sources(root, (1., 1.))
    # The inventory must describe the actual silent bytes.
    sf.write(root / 'A/1.wav', np.zeros(16000), 16000)
    rows = scan(root)
    out = tmp_path / 'dataset'
    stats = run_dataset(root, rows, out, CONFIG, decode)
    assert stats['needs_review'] == 1 and stats['success'] == 1
    review = {'audio_path': rows[0]['audio_path'], 'sha256': rows[0]['sha256'],
              'decision': 'accepted', 'corrected_text': '수정한 대사',
              'created_at': '2026-09-20T00:00:00+00:00'}
    export_dataset(rows, out, [review])
    assert read_jsonl(out / 'reviewed.jsonl')[0]['text'] == '수정한 대사'
    assert read_jsonl(out / 'reviewed.jsonl')[0]['asr_text'] == '자동 전사'
    review['decision'] = 'rejected'
    export_dataset(rows, out, [review])
    assert read_jsonl(out / 'candidates.jsonl') == []


def test_full_run_refuses_changed_settings_or_inventory(tmp_path):
    root = tmp_path / 'Voice'
    rows = sources(root, (1.,))
    out = tmp_path / 'dataset'
    run_dataset(root, rows, out, CONFIG, decode)
    with pytest.raises(ValueError, match='config'):
        run_dataset(root, rows, out, dict(CONFIG, revision='different'), decode)
    rows[0]['sha256'] = 'changed'
    with pytest.raises(ValueError, match='inventory'):
        run_dataset(root, rows, out, CONFIG, decode)


def test_checkpoints_are_bounded_and_pilot_results_can_be_reused(tmp_path):
    from fantasyvoice.dataset.storage import write_json, write_jsonl
    root = tmp_path / 'Voice'
    rows = sources(root, (1.,) * 105)
    pilot = tmp_path / 'pilot'
    write_json(pilot / 'run.json', CONFIG)
    write_jsonl(pilot / 'transcripts.jsonl', [{
        **{k: rows[0][k] for k in ('audio_path', 'character_id', 'sha256')},
        'key': 'old-pilot-key', 'status': 'success',
        'asr': {'text': '이미 전사함', 'segments': [], 'language': 'ko'}}])
    out = tmp_path / 'dataset'
    run_dataset(root, rows, out, CONFIG, decode, pilot_output=pilot)
    shards = sorted((out / 'results').glob('*.jsonl'))
    assert len(shards) == 2
    assert sorted(len(read_jsonl(p)) for p in shards) == [5, 100]
    assert sum(r['asr_text'] == '이미 전사함' for r in read_jsonl(out/'candidates.jsonl')) == 1


def test_full_run_records_error_then_retries_without_losing_history(tmp_path):
    root = tmp_path / 'Voice'
    rows = sources(root, (1.,))
    out = tmp_path / 'dataset'
    def fail(path):
        raise RuntimeError('decode failure')
    assert run_dataset(root, rows, out, CONFIG, fail)['error'] == 1
    assert run_dataset(root, rows, out, CONFIG, decode)['success'] == 1
    records = read_jsonl(out / 'results/00000.jsonl')
    assert [r['status'] for r in records] == ['error', 'success']


def test_review_timestamp_prevents_old_pilot_rejection_overriding_confirmation(tmp_path):
    root = tmp_path / 'Voice'
    rows = sources(root, (1.,))
    out = tmp_path / 'dataset'
    run_dataset(root, rows, out, CONFIG, decode)
    record = read_jsonl(out / 'candidates.jsonl')[0]
    reviews = [
        {'audio_path': record['audio_path'], 'sha256': record['sha256'],
         'decision': 'accepted', 'corrected_text': '확인한 대사', 'created_at': '2026-09-20'},
        {'key': record['key'], 'decision': 'rejected', 'corrected_text': '이전 대사',
         'created_at': '2026-09-19'}]
    export_dataset(rows, out, reviews)
    assert read_jsonl(out / 'reviewed.jsonl')[0]['text'] == '확인한 대사'
