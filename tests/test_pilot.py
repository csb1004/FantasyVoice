import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fantasyvoice.dataset.inventory import scan, select_pilot
from fantasyvoice.dataset.storage import read_jsonl, resolve_audio
from fantasyvoice.dataset.transcribe import run_pilot
from fantasyvoice.dataset.review import save_review


def audio(root, name='캐릭터 A/a.wav', subtype='PCM_16', channels=1):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((1600, channels)), 16000, subtype=subtype)
    return path


def config(revision='fixture'):
    return {'model': 'large-v3', 'revision': revision,
            'language': 'ko', 'task': 'transcribe', 'options': {'fp16': False}}


def decode(path):
    return {'text': '안녕하세요.', 'segments': [], 'language': 'ko'}


def test_inventory_preserves_float_stereo_corrupt_and_relative_paths(tmp_path):
    root = tmp_path / 'Voice'
    good = audio(root, '캐릭터 A/Float.WAV', 'FLOAT', 2)
    bad = root / '캐릭터 A/broken.wav'
    bad.write_bytes(b'not a wave')
    before = hashlib.sha256(good.read_bytes()).hexdigest()
    rows = scan(root)
    assert len(rows) == 2
    ok = next(r for r in rows if r['status'] == 'ok')
    assert ok['audio_path'] == '캐릭터 A/Float.WAV'
    assert ok['character_id'] == '캐릭터 A'
    assert ok['channels'] == 2 and ok['subtype'] == 'FLOAT'
    assert ok['duration_seconds'] == pytest.approx(.1)
    assert next(r for r in rows if r['status'] == 'error')['error']
    assert hashlib.sha256(good.read_bytes()).hexdigest() == before == ok['sha256']
    moved = tmp_path / 'Other root'
    root.rename(moved)
    assert scan(moved) == rows


def test_pilot_is_deterministic_and_contains_each_character(tmp_path):
    root = tmp_path / 'Voice'
    for c in ['A', 'B']:
        for i in range(5):
            path = audio(root, f'{c}/{i}.wav')
            sf.write(path, np.zeros(16000), 16000)
    rows = scan(root)
    selected = select_pilot(rows, per_character=3)
    assert selected == select_pilot(list(reversed(rows)), per_character=3)
    assert len(selected) == 6
    assert {r['character_id'] for r in selected} == {'A', 'B'}
    with pytest.raises(ValueError):
        select_pilot(rows, per_character=0)


def test_resume_preserves_success_and_retries_failure(tmp_path):
    root = tmp_path / 'Voice'
    audio(root, 'A/a.wav')
    audio(root, 'A/b.wav')
    rows = scan(root)
    out = tmp_path / 'pilot'
    def fail_b(path):
        if path.name == 'b.wav':
            raise RuntimeError('decoder failed')
        return decode(path)
    result = run_pilot(root, rows, out, config(), fail_b)
    assert [r['status'] for r in result] == ['success', 'error']
    def retry(path):
        assert path.name == 'b.wav', 'successful file must not be retranscribed'
        return decode(path)
    result = run_pilot(root, rows, out, config(), retry)
    assert [r['status'] for r in result] == ['success', 'error', 'success']
    assert read_jsonl(out / 'transcripts.jsonl') == result
    original = (out / 'transcripts.jsonl').read_bytes()
    with pytest.raises(ValueError, match='config'):
        run_pilot(root, rows, out, config('changed'), decode)
    assert (out / 'transcripts.jsonl').read_bytes() == original


def test_changed_audio_requires_inventory_refresh(tmp_path):
    root = tmp_path / 'Voice'
    path = audio(root)
    rows = scan(root)
    path.write_bytes(b'changed')
    result = run_pilot(root, rows, tmp_path / 'pilot', config(), decode)
    assert result[0]['status'] == 'error'
    assert 'changed' in result[0]['error']


def test_outputs_cannot_be_inside_original_data(tmp_path):
    root = tmp_path / 'Voice'
    audio(root)
    with pytest.raises(ValueError):
        run_pilot(root, scan(root), root / 'outputs', config(), decode)
    assert not (root / 'outputs').exists()


@pytest.mark.parametrize('relative', ['../outside.wav', '/tmp/outside.wav', 'C:/outside.wav'])
def test_audio_paths_cannot_escape_root(tmp_path, relative):
    with pytest.raises(ValueError):
        resolve_audio(tmp_path, relative)


def test_interrupt_keeps_previous_results(tmp_path):
    root = tmp_path / 'Voice'
    audio(root, 'A/a.wav'); audio(root, 'A/b.wav')
    def interrupt(path):
        if path.name == 'b.wav':
            raise KeyboardInterrupt
        return decode(path)
    out = tmp_path / 'pilot'
    with pytest.raises(KeyboardInterrupt):
        run_pilot(root, scan(root), out, config(), interrupt)
    assert len(read_jsonl(out / 'transcripts.jsonl')) == 1


def test_review_keeps_asr_original_and_history(tmp_path):
    root = tmp_path / 'Voice'; audio(root)
    out = tmp_path / 'pilot'
    row = run_pilot(root, scan(root), out, config(), decode)[0]
    save_review(out, row['key'], 'accepted', '수정한 대사', '청취 완료')
    save_review(out, row['key'], 'rejected', '', '비언어 발성')
    reviews = read_jsonl(out / 'reviews.jsonl')
    assert len(reviews) == 2 and reviews[0]['corrected_text'] == '수정한 대사'
    assert read_jsonl(out / 'transcripts.jsonl')[0]['asr']['text'] == '안녕하세요.'
    with pytest.raises(ValueError):
        save_review(out, 'missing', 'accepted', '대사', '')
    with pytest.raises(ValueError):
        save_review(out, row['key'], 'accepted', '', '')


def test_out_of_memory_is_saved_and_stops_run(tmp_path):
    root = tmp_path / 'Voice'; audio(root)
    out = tmp_path / 'pilot'
    def oom(path):
        raise RuntimeError('CUDA out of memory')
    with pytest.raises(RuntimeError, match='out of memory'):
        run_pilot(root, scan(root), out, config(), oom)
    assert read_jsonl(out / 'transcripts.jsonl')[0]['status'] == 'error'


def test_restored_source_can_recover_after_a_stale_inventory_error(tmp_path):
    root = tmp_path / 'Voice'
    path = audio(root)
    original = path.read_bytes()
    rows = scan(root)
    out = tmp_path / 'pilot'
    run_pilot(root, rows, out, config(), decode)
    path.write_bytes(b'changed')
    assert run_pilot(root, rows, out, config(), decode)[-1]['status'] == 'error'
    path.write_bytes(original)
    assert run_pilot(root, rows, out, config(), decode)[-1]['status'] == 'success'


def test_failed_snapshot_replacement_preserves_previous_results(tmp_path, monkeypatch):
    from fantasyvoice.dataset import storage
    target = tmp_path / 'results.jsonl'
    storage.write_jsonl(target, [{'text': 'saved'}])
    def fail_replace(*args):
        raise OSError('filesystem unavailable')
    monkeypatch.setattr(storage.os, 'replace', fail_replace)
    with pytest.raises(OSError):
        storage.write_jsonl(target, [{'text': 'new'}])
    assert storage.read_jsonl(target) == [{'text': 'saved'}]
    assert list(tmp_path.iterdir()) == [target]


def test_resolve_audio_handles_drive_decomposed_korean(tmp_path):
    import unicodedata
    decomposed = unicodedata.normalize('NFD', '캐릭터/수아.wav')
    actual = audio(tmp_path, decomposed)
    assert resolve_audio(tmp_path, '캐릭터/수아.wav') == actual.resolve()


def test_reference_inventory_load_does_not_access_voice(tmp_path):
    from fantasyvoice.dataset.inventory import load_reference
    from fantasyvoice.dataset.storage import write_jsonl
    ref = tmp_path / 'inventory.jsonl'
    write_jsonl(ref, [{'audio_path': 'A/absent.wav', 'character_id': 'A',
                      'sha256': 'a'*64, 'status': 'ok', 'duration_seconds': 1.5}])
    rows = load_reference(ref)
    assert rows[0]['audio_path'] == 'A/absent.wav'
    assert not (tmp_path / 'A').exists()
    with pytest.raises(ValueError):
        load_reference(tmp_path / 'missing.jsonl')


def test_pilot_excludes_half_second_boundary_even_attack_clips():
    rows = [{'audio_path': f'A/attack_{i}.wav', 'character_id': 'A',
             'duration_seconds': d, 'status': 'ok'}
            for i, d in enumerate([.1, .5, .5001, 1., 2.])]
    selected = select_pilot(rows)
    assert {r['audio_path'] for r in selected} == {
        'A/attack_2.wav', 'A/attack_3.wav', 'A/attack_4.wav'}
    assert select_pilot(rows[:2]) == []
