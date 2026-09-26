import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from gpu_batches import batch_preset, emotion_batch_logits
from test_emotion_feedback import analyzer


def test_gpu_presets_distinguish_real_and_effective_batch():
    assert batch_preset('Tesla T4', 15, 11) == (4,1)
    assert batch_preset('NVIDIA L4', 22, 11) == (8,1)
    assert batch_preset('Tesla T4', 15, 12) == (4,1)
    assert batch_preset('NVIDIA L4', 22, 12) == (8,1)
    assert batch_preset('NVIDIA L4', 10, 11) == (1,1)
    assert batch_preset('unknown', 80, 11) == (1,1)


def test_padding_does_not_enter_normalization_or_pooling():
    model=analyzer()
    # Fake backbone preserves time resolution and supplies its valid-frame mask.
    def features(source,padding_mask=None,mask=False):
        return {'x':torch.stack((source,source.square()),-1),'padding_mask':padding_mask}
    model.model.extract_features=features
    waves=[torch.randn(1,500),torch.randn(1,1200)*2+3]
    expected=torch.cat([model(w) for w in waves])
    actual=emotion_batch_logits(model,waves,'cpu')
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    actual.square().mean().backward()
    assert model.model.proj.weight.grad.abs().sum()>0


def test_batch_training_resume_and_partial_window(tmp_path):
    from emotion_training import fit_emotion, validate_emotion_checkpoint
    def setup():
        torch.manual_seed(22)
        model=analyzer()
        model.model.extract_features=lambda source,padding_mask=None,mask=False: {
            'x':torch.stack((source,source.square()),-1),'padding_mask':padding_mask}
        return model
    data=[(torch.linspace(-1,1,500+i*20)[None],torch.tensor([[.1,.2,.1,.1,.3,.1,.1]])) for i in range(9)]
    config=dict(epochs=2,batch_size=2,accumulation_steps=2,learning_rate=.001,
        gradient_clip=1.,seed=42,save_every=1,eval_every=2,validation_samples=2,lr_decay=.99)
    full=setup();fit_emotion(full,data,data,config,tmp_path/'full',{},'cpu')
    fit_emotion(setup(),data,data,config,tmp_path/'resume',{},'cpu',max_updates=1)
    resumed=setup();report=fit_emotion(resumed,data,data,config,tmp_path/'resume',{},'cpu')
    assert report['updates']==6
    for k,v in full.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
    validate_emotion_checkpoint(torch.load(tmp_path/'resume/best-model.pt',weights_only=True),report,{})


def test_prior_generates_multiple_lengths_and_backpropagates():
    from gpu_batches import generate_prior_batch
    from test_conditional_tts import tiny_tts, tiny_batch
    model=tiny_tts().eval()
    original=tiny_batch()
    batch={k:torch.cat([v,v],dim=0) for k,v in original.items()}
    batch['character_ids']=torch.tensor([0,1])
    batch['x_lengths']=torch.tensor([4,2])
    model.base.dp.forward=lambda hidden,mask,g=None:torch.zeros_like(mask)
    waves=generate_prior_batch(model,batch,torch.full((2,7),1/7),torch.zeros(2,3),500)
    assert len(waves)==2 and all(w.shape[0]==1 and w.numel()>0 for w in waves)
    assert [w.shape[1] for w in waves]==[16,8]
    sum(w.square().mean() for w in waves).backward()
    assert (model.conditioner.character.weight.grad.abs().sum(1)>0).all()


def test_oom_replays_entire_window_without_duplicate_gradients():
    from gpu_batches import backward_with_batch_retry
    from fantasyvoice.training.predictor import rng_state
    model=torch.nn.Linear(1,1)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    def objective(indices):
        if len(indices)>1 and indices[0]>=3:raise torch.cuda.OutOfMemoryError('simulated after partial backward')
        return {'g':model(torch.tensor([[float(i)] for i in indices])).square().mean()}
    totals,size=backward_with_batch_retry(objective,[1,2,3,4],2,scaler,{'g':optimizer},rng_state())
    grads=[p.grad.clone() for p in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    sum(objective([i])['g']/4 for i in [1,2,3,4]).backward()
    assert size==1
    for expected,p in zip(grads,model.parameters()):torch.testing.assert_close(expected,p.grad)


def test_feedback_engine_resume_with_saved_batch_size(tmp_path):
    from feedback_engine import fit_engine
    from test_tts_training import setup_run,CONFIG
    config={**CONFIG,'batch_size':4,'accumulation_steps':1}
    full=setup_run();fit_engine(*full,7,config,tmp_path/'full',{},'cpu')
    fit_engine(*setup_run(),7,config,tmp_path/'resumed',{},'cpu',max_updates=1)
    resumed=setup_run();report=fit_engine(*resumed,7,config,tmp_path/'resumed',{},'cpu')
    assert report['microbatch_size']==4
    for key in full[0]:
        for name,value in full[0][key].state_dict().items():
            assert torch.equal(value,resumed[0][key].state_dict()[name])


def test_feedback_engine_preserves_original_transaction_guards(tmp_path,monkeypatch):
    import test_tts_training as original
    from feedback_engine import fit_engine
    monkeypatch.setattr(original,'fit_engine',fit_engine)
    original.test_interruption_inside_second_optimizer_preserves_checkpoint(tmp_path/'optimizer')
    original.test_interrupted_evaluation_retried_before_next_update(tmp_path/'evaluation')
    original.test_nonfinite_one_optimizer_prevents_all_updates(tmp_path/'nonfinite')
