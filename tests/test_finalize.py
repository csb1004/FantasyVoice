import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from fantasyvoice.dataset.inventory import scan
from fantasyvoice.dataset.full_dataset import run_dataset
from fantasyvoice.dataset.finalize import finalize_dataset
from fantasyvoice.dataset.storage import read_jsonl, write_jsonl


CONFIG = {'model': 'large-v3', 'revision': 'fixture', 'language': 'ko', 'task': 'transcribe'}


def setup_run(tmp_path, fail=False):
    root = tmp_path / 'Voice'
    (root / 'A').mkdir(parents=True)
    for i in range(2):
        sf.write(root / 'A' / f'{i}.wav', np.full(16000, .1), 16000)
    rows = scan(root)
    out = tmp_path / 'dataset'
    calls = []
    def decode(path):
        calls.append(path)
        if fail and len(calls) == 2:
            raise RuntimeError('temporary decoder error')
        return {'text': '대사', 'segments': [{'end': 1.5}], 'language': 'ko'}
    run_dataset(root, rows, out, CONFIG, decode)
    return rows, out


def test_finalize_full_run_rebuilds_exports_and_packages_no_audio(tmp_path):
    rows, out = setup_run(tmp_path)
    (out / 'accidental.wav').write_bytes(b'not for export')
    (out / 'candidates.jsonl').write_text('stale', encoding='utf-8')
    report = finalize_dataset(rows, out, tmp_path / 'report.zip')
    assert report['asr_pass_complete'] is True
    assert report['training_ready'] is False
    assert report['total'] == report['success'] == 2
    assert report['segment_endpoint_overruns'] == 2
    with zipfile.ZipFile(tmp_path / 'report.zip') as z:
        assert 'dataset-v1/candidates.jsonl' in z.namelist()
        assert not any(n.endswith('.wav') for n in z.namelist())
        assert len(z.read('dataset-v1/candidates.jsonl').splitlines()) == 2


def test_finalize_distinguishes_errors_missing_and_review(tmp_path):
    rows, out = setup_run(tmp_path, fail=True)
    report = finalize_dataset(rows, out, tmp_path / 'partial.zip')
    assert report['asr_pass_complete'] is False
    assert report['error'] == 1 and report['unprocessed'] == 0
    assert len(read_jsonl(out / 'retry.jsonl')) == 1
    shard = out / 'results/00000.jsonl'
    write_jsonl(shard, read_jsonl(shard)[:1])
    report = finalize_dataset(rows, out, tmp_path / 'partial.zip')
    assert report['error'] == 0 and report['unprocessed'] == 1
    assert report['asr_pass_complete'] is False


def test_finalize_applies_new_review_without_changing_asr(tmp_path):
    rows, out = setup_run(tmp_path)
    raw_before = (out / 'results/00000.jsonl').read_bytes()
    review = {'audio_path': rows[0]['audio_path'], 'sha256': rows[0]['sha256'],
              'decision': 'accepted', 'corrected_text': '수정 대사', 'created_at': '2026-09-20'}
    report = finalize_dataset(rows, out, tmp_path / 'report.zip', [review])
    assert report['human_accepted_count'] == 1
    assert read_jsonl(out / 'reviewed.jsonl')[0]['text'] == '수정 대사'
    assert (out / 'results/00000.jsonl').read_bytes() == raw_before


def test_finalize_rejects_mismatched_snapshot_before_rewriting_exports(tmp_path):
    rows, out = setup_run(tmp_path)
    before = (out / 'candidates.jsonl').read_bytes()
    rows[0]['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='inventory'):
        finalize_dataset(rows, out, tmp_path / 'report.zip')
    assert (out / 'candidates.jsonl').read_bytes() == before
