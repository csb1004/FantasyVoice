import copy

import pytest

from scripts.joint_inputs import validate_completed_warmup


def inputs():
    config = dict(stage='warmup', epochs=3, batch_size=1, accumulation_steps=4,
                  model_revision='m', melo_revision='code', frontend_revision='f',
                  add_blank=True, model_artifact_sha256={'checkpoint.pth':'weights'})
    identity = {'dataset_sha256':'data', 'dataset_metadata':{'character_map':{'A':0}},
                'predictor_sha256':'predictor'}
    checkpoint = {'fingerprint':{'config':config, 'identity':identity, 'size':8},
                  'updates':4, 'mel':1.5, 'modules':{'g':{}}}
    report = {'complete':True, 'epoch':3, 'updates':6, 'total_updates':6,
              'best_update':4, 'best_mel':1.5, 'pending_eval':None}
    run = {'config':copy.deepcopy(config), 'identity':copy.deepcopy(identity)}
    current = {**config, 'stage':'joint'}
    return checkpoint, report, run, current, identity


def test_accepts_completed_warmup_best_and_preserves_inputs():
    args = inputs()
    before = copy.deepcopy(args)
    result = validate_completed_warmup(*args)
    assert result == {'warmup_updates':6, 'selected_update':4, 'best_mel':1.5}
    assert args == before


@pytest.mark.parametrize('field,value', [('complete',False), ('epoch',2),
    ('updates',5), ('total_updates',7), ('best_update',2), ('best_mel',2.), ('pending_eval',{'full':True})])
def test_rejects_incomplete_or_mismatched_report(field, value):
    checkpoint, report, run, config, identity = inputs()
    report[field] = value
    with pytest.raises(ValueError):
        validate_completed_warmup(checkpoint, report, run, config, identity)


@pytest.mark.parametrize('field,value', [('dataset_sha256','other'),
    ('predictor_sha256','new'), ('dataset_metadata',{'character_map':{'B':0}})])
def test_rejects_changed_training_inputs(field, value):
    args = list(inputs())
    args[4] = {**args[4], field:value}
    with pytest.raises(ValueError, match='identity'):
        validate_completed_warmup(*args)


def test_rejects_joint_checkpoint_and_incompatible_frontend():
    args = inputs()
    args[0]['fingerprint']['config']['stage'] = 'joint'
    with pytest.raises(ValueError):
        validate_completed_warmup(*args)
    args = inputs()
    args[3]['frontend_revision'] = 'new'
    with pytest.raises(ValueError, match='frontend_revision'):
        validate_completed_warmup(*args)
