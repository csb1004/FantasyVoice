import json
import zipfile

import pytest

from fantasyvoice.dataset.storage import read_jsonl, write_json, write_jsonl
from fantasyvoice.style.features import emotion_probabilities, summarize
from fantasyvoice.style.runner import label_key
from fantasyvoice.style.finalize import finalize_styles


def test_emotion_validation_tolerates_float_rounding_not_changed_labels(tmp_path):
    from fantasyvoice.style.finalize import validate_style
    _, _, _, value = fixture(tmp_path)
    value['emotion']['probabilities']['neutral'] += 1e-15
    assert validate_style(value) == []
    value['emotion']['probabilities']['neutral'] -= .1
    assert any('emotion' in error for error in validate_style(value))


def fixture(tmp_path):
    source = tmp_path / 'source'
    config = {'extraction': {'pitch': {'fmin': 50, 'fmax': 1100}}}
    write_json(source / 'config.json', config)
    rows = [dict(audio_path=f'c/{i}.wav', sha256=str(i), text='대사', duration_seconds=2,
                 character_id='c', review_status='pending', split=None) for i in range(3)]
    value = summarize([(0, 2)], 2, 4, [200, 210])
    value.update(emotion=emotion_probabilities(
        ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'other', 'unknown'],
        [.025] * 4 + [.8] + [.025] * 4), decoded_duration_seconds=2,
        valid_targets={'emotion': True, 'speed': True, 'pitch_raw': True,
                       'pitch_normalized': False, 'pause': True}, issues=[])
    return source, config, rows, value


def test_report_recovers_from_checkpoints_preserves_pending_and_source(tmp_path):
    source, config, rows, value = fixture(tmp_path)
    write_jsonl(source / 'results/ab.jsonl', [
        dict(style_key=label_key(rows[0], config), style_status='generated', pseudo_style=value),
        dict(style_key=label_key(rows[1], config), style_status='error', style_error='bad audio')])
    write_jsonl(source / 'style-labels.jsonl', [])  # stale export must not be trusted
    before = (source / 'results/ab.jsonl').read_bytes()
    output = tmp_path / 'report'
    report = finalize_styles(rows, source, output)
    assert report['counts'] == {'generated': 1, 'error': 1, 'pending': 1}
    assert not report['style_pass_complete'] and not report['training_ready']
    assert report['numeric']['pitch_f0_median_hz']['median'] == 205
    assert len(read_jsonl(output / 'retry.jsonl')) == 2
    assert read_jsonl(output / 'style-labels.jsonl')[0]['review_status'] == 'pending'
    assert (source / 'results/ab.jsonl').read_bytes() == before
    with zipfile.ZipFile(output / 'style-report.zip') as archive:
        assert 'style-v1/completion-report.json' in archive.namelist()
        assert not any(n.endswith('.wav') for n in archive.namelist())


def test_invalid_value_is_flagged_not_silently_accepted(tmp_path):
    source, config, rows, value = fixture(tmp_path)
    value['pause_ratio'] = 1.5
    write_jsonl(source / 'results/ab.jsonl', [
        dict(style_key=label_key(rows[0], config), style_status='generated', pseudo_style=value)])
    report = finalize_styles(rows[:1], source, tmp_path / 'report')
    assert report['invalid_count'] == 1
    assert not report['style_pass_complete']
    assert 'pause_ratio' in read_jsonl(tmp_path / 'report/retry.jsonl')[0]['validation_errors'][0]


def test_partial_emotion_can_be_complete_without_becoming_accepted(tmp_path):
    source, config, rows, value = fixture(tmp_path)
    value['emotion'] = emotion_probabilities(
        ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'other', 'unknown'],
        [.025] * 8 + [.8])
    value['valid_targets']['emotion'] = False
    value['issues'] = ['emotion_unavailable', 'emotion_other_or_unknown']
    write_jsonl(source / 'results/ab.jsonl', [
        dict(style_key=label_key(rows[0], config), style_status='partial', pseudo_style=value)])
    report = finalize_styles(rows[:1], source, tmp_path / 'report')
    assert report['style_pass_complete']
    assert report['counts'] == {'partial': 1}
    assert not report['training_ready']
    assert len(read_jsonl(tmp_path / 'report/style-needs-review.jsonl')) == 1


def test_edited_text_cannot_reuse_old_labels(tmp_path):
    source, config, rows, value = fixture(tmp_path)
    write_jsonl(source / 'results/ab.jsonl', [
        dict(style_key=label_key(rows[0], config), style_status='generated', pseudo_style=value)])
    rows[0]['text'] = '수정한 대사'
    report = finalize_styles(rows[:1], source, tmp_path / 'report')
    assert report['counts'] == {'pending': 1}
    assert report['unmatched_checkpoint_count'] == 1


def test_refuses_output_inside_source(tmp_path):
    source, _, rows, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        finalize_styles(rows, source, source / 'report')
