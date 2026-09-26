import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from emotion_feedback import EmotionAnalyzer, emotion_indices, emotion_kl, generated_emotion_loss, generate_prior
from test_conditional_tts import tiny_tts, tiny_batch


LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', '<unk>']


class FakeEmotion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(normalize=True)
        self.proj = torch.nn.Linear(2, 9)
        self.modality_encoders = {'AUDIO': SimpleNamespace(local_grad_mult=0.)}

    def extract_features(self, source, padding_mask=None, mask=False):
        return {'x': torch.stack((source, source.square()), -1)}


def analyzer():
    return EmotionAnalyzer(FakeEmotion(), LABELS, lambda x: x)


def test_label_order_not_assumed_and_unknown_labels_rejected():
    assert emotion_indices(list(reversed(LABELS))) == [8, 7, 6, 5, 4, 2, 1]
    with pytest.raises(ValueError):
        emotion_indices(LABELS[:-1] + ['happy'])


def test_kl_has_zero_at_target_and_rejects_invalid_distributions():
    target = torch.tensor([[.1, .1, .1, .2, .2, .1, .2]])
    assert abs(emotion_kl(target.log(), target).item()) < 1e-6
    with pytest.raises(ValueError):
        emotion_kl(torch.zeros(1, 7), target * 2)


def test_frozen_analyzer_passes_waveform_gradient_without_parameter_updates():
    model = analyzer().requires_grad_(False)
    fake = torch.randn(1, 1200, requires_grad=True)
    target = torch.tensor([[1., 0, 0, 0, 0, 0, 0]])
    before = {k:v.clone() for k,v in model.state_dict().items()}
    generated_emotion_loss(model, fake, target).backward()
    assert fake.grad is not None and fake.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())
    assert model.model.modality_encoders['AUDIO'].local_grad_mult == 1.


def test_analyzer_matches_utterance_normalization_and_subset_logits():
    model = analyzer().eval()
    wave = torch.randn(1, 1200)
    x = torch.nn.functional.layer_norm(wave, (1200,))
    expected = model.model.proj(torch.stack((x, x.square()), -1).mean(1))[:, [0,1,2,3,4,6,7]]
    assert torch.allclose(model(wave), expected)


def test_full_prior_has_no_reference_input_and_reaches_character_and_decoder():
    model = tiny_tts().eval()
    batch = tiny_batch()
    del batch['spec'], batch['spec_lengths']
    probs = torch.full((1, 7), 1/7)
    wave = generate_prior(model, batch, probs, torch.zeros(1,3), max_frames=500)
    wave.square().mean().backward()
    assert model.conditioner.character.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.base.dec.parameters())
    with pytest.raises(ValueError, match='length'):
        generate_prior(model, batch, probs, torch.zeros(1,3), max_frames=1)



def test_scoring_restores_flags_on_error():
    from emotion_feedback import scoring
    model = analyzer().train()
    model.model.proj.bias.requires_grad_(False)
    with pytest.raises(RuntimeError):
        with scoring(model):
            assert not any(p.requires_grad for p in model.parameters())
            raise RuntimeError('abort')
    assert model.training
    assert model.model.proj.weight.requires_grad
    assert not model.model.proj.bias.requires_grad


def test_analyzer_training_resume_and_completed_handoff(tmp_path):
    from emotion_training import fit_emotion, validate_emotion_checkpoint
    torch.set_num_threads(1)
    data = [(torch.linspace(-1,1,1200).unsqueeze(0)*(i+1),
             torch.tensor([[.1,.2,.1,.1,.3,.1,.1]])) for i in range(5)]
    config = dict(epochs=2, accumulation_steps=2, learning_rate=.001,
                  gradient_clip=1., seed=42, save_every=1, eval_every=2,
                  validation_samples=2, lr_decay=.99)
    identity = {'dataset_sha256':'d', 'emotion_order':LABELS[:5]+LABELS[6:8]}
    def setup():
        torch.manual_seed(32)
        return analyzer()
    full = setup()
    expected = fit_emotion(full,data,data,config,tmp_path/'full',identity,'cpu')
    fit_emotion(setup(),data,data,config,tmp_path/'resume',identity,'cpu',max_updates=1)
    resumed=setup()
    actual=fit_emotion(resumed,data,data,config,tmp_path/'resume',identity,'cpu')
    assert actual['updates']==expected['updates']==6
    assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in full.state_dict().items())
    best=torch.load(tmp_path/'resume/best-model.pt',weights_only=True)
    validate_emotion_checkpoint(best,actual,identity)
    with pytest.raises(ValueError):
        validate_emotion_checkpoint(best,{**actual,'complete':False},identity)
    with pytest.raises(ValueError):
        validate_emotion_checkpoint(best,actual,{**identity,'dataset_sha256':'wrong'})


def test_analyzer_nonfinite_does_not_commit(tmp_path):
    from emotion_training import fit_emotion
    model=analyzer()
    before={k:v.clone() for k,v in model.state_dict().items()}
    data=[(torch.full((1,1200),float('nan')),torch.full((1,7),1/7))]
    config=dict(epochs=1,accumulation_steps=1,learning_rate=.001,gradient_clip=1.,
                seed=42,save_every=1,eval_every=1,validation_samples=1,lr_decay=.99)
    with pytest.raises(ValueError):
        fit_emotion(model,data,data,config,tmp_path/'fail',{},'cpu')
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_real_supervision_updates_analyzer_without_any_generator():
    model=analyzer()
    before=model.model.proj.weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    loss=emotion_kl(model(torch.randn(1,1200)),torch.tensor([[1.,0,0,0,0,0,0]]))
    loss.backward(); optimizer.step()
    assert not torch.equal(before,model.model.proj.weight)


def test_duration_selection_preserves_split_and_characters():
    from emotion_training import select_duration
    rows=[{'character_id':'a','duration_seconds':2.}, {'character_id':'a','duration_seconds':20.}]
    selected, report=select_duration({'train':rows},15.)
    assert len(selected['train'])==1 and len(rows)==2
    assert report['train']['excluded_over_duration']==1
    with pytest.raises(ValueError,match='loses a character'):
        select_duration({'train':rows+[{'character_id':'b','duration_seconds':20.}]},15.)


def test_interrupted_analyzer_validation_resumes_before_training(tmp_path):
    from emotion_training import fit_emotion
    model=analyzer()
    data=[(torch.randn(1,1200),torch.full((1,7),1/7))]
    config=dict(epochs=1,accumulation_steps=1,learning_rate=.001,gradient_clip=1.,
                seed=42,save_every=1,eval_every=1,validation_samples=1,lr_decay=.99)
    class Interrupted:
        def __len__(self): return 1
        def __getitem__(self,i): raise KeyboardInterrupt('validation')
    with pytest.raises(KeyboardInterrupt):
        fit_emotion(model,data,Interrupted(),config,tmp_path,{},'cpu')
    saved=torch.load(tmp_path/'latest.pt',weights_only=True)
    assert saved['pending_eval'] is True and saved['updates']==0
    report=fit_emotion(analyzer(),data,data,config,tmp_path,{},'cpu')
    assert [r['updates'] for r in report['history']]==[0,1]
