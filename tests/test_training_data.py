import copy

import pytest

from fantasyvoice.dataset.training_data import prepare_training_data
from fantasyvoice.dataset.storage import read_jsonl


def examples():
    return [dict(audio_path=f'c/{i}.wav', sha256=str(i), character_id='c', text=f'대사 {i}',
        review_status='pending', style_status='generated', validation_errors=[], split=None,
        pseudo_style=dict(emotion={'probabilities': {'neutral': 1.}},
            speed_phonemes_per_second=float(i+1), pitch_f0_median_hz=100.+i*10,
            pause_ratio=.1*i, valid_targets={'emotion': True, 'speed': True,
            'pitch_raw': True, 'pitch_normalized': False, 'pause': True})) for i in range(10)]


def settings():
    return dict(validation_fraction=.2, seed=42, allow_pending_transcripts=True,
                label_policy='complete_only', pitch_baseline='median',
                grouping='character_text_and_audio_hash', no_pitch_policy='hold_out')


def test_requires_explicit_settings_and_does_not_create_output(tmp_path):
    with pytest.raises(ValueError, match='settings'):
        prepare_training_data(examples(), {}, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_speed_limit_is_strict_and_applied_before_splitting(tmp_path):
    rows = examples()
    rows[0]['pseudo_style']['speed_phonemes_per_second'] = 30.
    rows[1]['pseudo_style']['speed_phonemes_per_second'] = 30.0001
    config = settings()
    config['max_speed_phonemes_per_second'] = 30.
    report = prepare_training_data(rows, config, tmp_path)
    data = read_jsonl(tmp_path / 'train.jsonl') + read_jsonl(tmp_path / 'validation.jsonl')
    assert rows[0]['audio_path'] in {r['audio_path'] for r in data}
    assert rows[1]['audio_path'] not in {r['audio_path'] for r in data}
    assert report['hold_reasons'] == {'speed_above_limit': 1}


@pytest.mark.parametrize('limit', [-1, 0, float('nan'), float('inf')])
def test_invalid_speed_limit_refused_before_output(tmp_path, limit):
    config = settings()
    config['max_speed_phonemes_per_second'] = limit
    with pytest.raises(ValueError, match='speed'):
        prepare_training_data(examples(), config, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_deterministic_split_and_training_only_statistics(tmp_path):
    rows=examples()
    report=prepare_training_data(rows, settings(), tmp_path / 'one')
    train=read_jsonl(tmp_path / 'one/train.jsonl')
    validation=read_jsonl(tmp_path / 'one/validation.jsonl')
    assert len(train)==8 and len(validation)==2
    # Extreme validation labels must not change fitted training statistics.
    val_paths={r['audio_path'] for r in validation}
    modified=copy.deepcopy(rows)
    for row in modified:
        if row['audio_path'] in val_paths:
            row['pseudo_style']['pitch_f0_median_hz']=1000.
            row['pseudo_style']['speed_phonemes_per_second']=1000.
    second=prepare_training_data(modified, settings(), tmp_path / 'two')
    assert report['normalization']==second['normalization']
    assert report['pitch_baselines_hz']==second['pitch_baselines_hz']
    assert all(r['review_status']=='pending' for r in train+validation)
    assert all(r['split'] is None for r in rows)
    assert abs(sum(r['training_style']['speed_z'] for r in train))<1e-10
    assert not report['trainer_implemented']


def test_duplicate_text_and_audio_never_cross_splits(tmp_path):
    rows=examples()
    rows[1]['text']=rows[0]['text']
    rows[2]['sha256']=rows[1]['sha256']
    prepare_training_data(rows, settings(), tmp_path)
    data=read_jsonl(tmp_path / 'train.jsonl')+read_jsonl(tmp_path / 'validation.jsonl')
    locations={r['audio_path']:r['split'] for r in data}
    assert len({locations[rows[i]['audio_path']] for i in (0,1,2)})==1


def test_missing_targets_are_masked_and_rejected_reviews_held(tmp_path):
    rows=examples()
    rows[0]['pseudo_style']['speed_phonemes_per_second']=None
    rows[0]['pseudo_style']['valid_targets']['speed']=False
    rows[0]['style_status']='partial'
    rows[1]['review_status']='rejected'
    config=settings();config['label_policy']='masked_partial'
    prepare_training_data(rows, config, tmp_path)
    data=read_jsonl(tmp_path / 'train.jsonl')+read_jsonl(tmp_path / 'validation.jsonl')
    zero=next(r for r in data if r['audio_path']==rows[0]['audio_path'])
    assert zero['training_style']['speed_z'] is None
    assert not zero['training_style']['valid_targets']['speed']
    assert rows[1]['audio_path'] not in {r['audio_path'] for r in data}
