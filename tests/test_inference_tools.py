import copy
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.inference_tools import validate_completed_joint, load_joint_weights, save_generation, verify_inference_sources


def records():
    config = dict(stage='joint', epochs=3, batch_size=8, accumulation_steps=1)
    identity = {'predictor_sha256':'p', 'dataset_metadata':{'character_map':{'A':0}}}
    checkpoint = {'fingerprint':{'config':config,'identity':identity,'size':17},
                  'updates':6,'mel':1.2,'modules':{'g':{},'predictor':{}}}
    report = {'complete':True,'epoch':3,'updates':9,'total_updates':9,
              'best_update':6,'best_mel':1.2,'pending_eval':None}
    return checkpoint, report, {'config':copy.deepcopy(config),'identity':copy.deepcopy(identity)}


def test_joint_completion_uses_saved_batch_settings():
    args = records()
    assert validate_completed_joint(*args, 'p')['selected_update'] == 6


@pytest.mark.parametrize('change', ['incomplete','warmup','mismatch','predictor','missing_model'])
def test_rejects_wrong_inference_inputs(change):
    checkpoint, report, run = records()
    predictor_sha = 'p'
    if change == 'incomplete': report['complete'] = False
    if change == 'warmup': checkpoint['fingerprint']['config']['stage'] = 'warmup'
    if change == 'mismatch': report['best_update'] = 3
    if change == 'predictor': predictor_sha = 'other'
    if change == 'missing_model': checkpoint['modules'].pop('predictor')
    with pytest.raises(ValueError):
        validate_completed_joint(checkpoint, report, run, predictor_sha)


def test_joint_predictor_weights_replace_old_warmup_weights():
    g, predictor = torch.nn.Linear(2,1), torch.nn.Linear(2,1)
    checkpoint = {'modules':{'g':{k:torch.ones_like(v)*2 for k,v in g.state_dict().items()},
                              'predictor':{k:torch.ones_like(v)*3 for k,v in predictor.state_dict().items()}}}
    load_joint_weights(g,predictor,checkpoint)
    assert torch.equal(g.weight,torch.full_like(g.weight,2))
    assert torch.equal(predictor.weight,torch.full_like(predictor.weight,3))
    assert not g.training and not predictor.training


def test_export_preserves_raw_audio_and_records_playback_gain(tmp_path):
    wave = torch.tensor([0., 2., -2., .25])
    result = save_generation(tmp_path,wave,{'text':'hello','seed':42},{'checkpoint_sha256':'abc'})
    raw, rate = sf.read(result['raw'])
    assert rate == 44100 and np.array_equal(raw,wave.numpy())
    preview, _ = sf.read(result['preview'])
    assert abs(preview).max() <= 1
    info = json.loads(result['metadata'].read_text(encoding='utf-8'))
    assert info['playback_gain'] == .5 and info['model']['checkpoint_sha256'] == 'abc'
    with zipfile.ZipFile(result['archive']) as z:
        assert set(z.namelist()) == {'voice.wav','playback.wav','generation.json'}
    again = save_generation(tmp_path,wave,{}, {})
    assert again['raw'] != result['raw'] and result['raw'].is_file()


def test_nonfinite_audio_creates_no_output(tmp_path):
    with pytest.raises(ValueError):
        save_generation(tmp_path,torch.tensor([float('nan')]),{}, {})
    assert list(tmp_path.iterdir()) == []


def test_source_identity_mismatch_is_rejected(tmp_path):
    from fantasyvoice.dataset.storage import sha256
    name = 'src/fantasyvoice/inference/tts.py'
    path = tmp_path / name
    path.parent.mkdir(parents=True); path.write_text('original')
    identity = {'source_sha256':{name:sha256(path)}}
    verify_inference_sources(tmp_path, identity)
    path.write_text('changed')
    with pytest.raises(ValueError): verify_inference_sources(tmp_path,identity)


def test_silent_audio_is_saved_without_a_broken_player(tmp_path):
    result = save_generation(tmp_path,torch.zeros(32),{}, {})
    assert result['playback_status'] == 'silent' and result['preview'] is None
    assert result['raw'].is_file()
    with zipfile.ZipFile(result['archive']) as z:
        assert set(z.namelist()) == {'voice.wav','generation.json'}
