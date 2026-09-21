import pytest
import torch
from fantasyvoice.training.tts_runtime import warm_start_joint, make_optimizers


def test_joint_warm_start_copies_weights_and_refuses_other_dataset(tmp_path):
    source = torch.nn.Linear(2,1)
    modules = {k:torch.nn.Linear(2,1) for k in ('g','d','dur','predictor')}
    path = tmp_path / 'best.pt'
    torch.save({'updates':12,'fingerprint':{'config':{'stage':'warmup','model_revision':'r'},
                'identity':{'dataset_sha256':'a','dataset_metadata':{'characters':2}}},
                'modules':{'g':source.state_dict()}},path)
    predictor_before = modules['predictor'].weight.clone()
    info = warm_start_joint(path, modules, 'a', {'characters':2}, 'r')
    assert torch.equal(modules['g'].weight,source.weight)
    assert torch.equal(modules['predictor'].weight,predictor_before)
    assert not info['discriminators_reused']
    with pytest.raises(ValueError,match='identity'):
        warm_start_joint(path,modules,'wrong',{'characters':2},'r')


def test_warmup_optimizer_excludes_predictor_and_joint_includes_it():
    config = dict(stage='warmup',learning_rate=1e-4,predictor_lr=1e-5,eps=1e-9,
                  lr_decay=.99,betas=[.8,.99],weight_decay=.01)
    modules = {k:torch.nn.Linear(2,1) for k in ('g','d','dur')}
    opts, _ = make_optimizers(modules,config)
    assert len(opts['g'].param_groups) == 1
    modules['predictor'] = torch.nn.Linear(2,1)
    with pytest.raises(ValueError,match='Predictor'):
        make_optimizers(modules,config)
    opts, _ = make_optimizers(modules,{**config,'stage':'joint'})
    assert opts['g'].param_groups[1]['lr'] == 1e-5
    assert {id(p) for p in opts['g'].param_groups[1]['params']} == {id(p) for p in modules['predictor'].parameters()}
