import copy
from types import SimpleNamespace

import pytest
import torch

from fantasyvoice.models.predictor import StylePredictor
from fantasyvoice.training.predictor import fit, loss_sums, read_prepared, tokenize_rows


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = torch.nn.Embedding(32, 8)
        self.layer = torch.nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=self.layer(self.embedding(input_ids)))


class Tokenizer:
    pad_token_id = 0
    def __call__(self, texts, **kwargs):
        return {'input_ids': [[1] + [2] * len(t) + [3] for t in texts],
                'attention_mask': [[1] * (len(t) + 2) for t in texts]}


def rows(n=7):
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    return [dict(audio_path=f'c/{i}.wav', text='대사', character_id='not_model_input',
                 training_style={'emotion_probabilities': {e: 1/7 for e in emotions},
                    'speed_z': i/10, 'pitch_z': -i/10, 'pause_z': .2,
                    'valid_targets': dict(emotion=True, speed=True, pitch=True, pause=True)})
            for i in range(n)]


def config():
    return dict(seed=42, epochs=2, batch_size=2, accumulation_steps=2, eval_batch_size=3,
                bert_lr=.001, head_lr=.002, weight_decay=.01, betas=[.9, .999], eps=1e-8,
                warmup_fraction=.1, gradient_clip=1., eval_every=1, save_every=1,
                precision='fp32', loss_weights=dict(emotion=1., speed=1., pitch=1., pause=1.))


def test_all_missing_targets_zero_loss_and_full_gradient_path():
    model=StylePredictor(TinyEncoder(), dropout=0.)
    outputs=model(torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.long))
    targets=torch.full((2, 10), float('nan'))
    masks=torch.zeros(2, 4, dtype=torch.bool)
    sums, counts=loss_sums(outputs, targets, masks)
    loss=sum(sums.values())
    assert loss.item()==0 and sum(counts.values())==0
    loss.backward()
    model.zero_grad()
    targets=torch.zeros(2,10); targets[:,:7]=1/7
    sums, _=loss_sums(model(torch.ones(2,3,dtype=torch.long),torch.ones(2,3,dtype=torch.long)),
                      targets,torch.ones(2,4,dtype=torch.bool))
    sum(sums.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_tokenization_rejects_truncation_and_keeps_only_text_input():
    data=tokenize_rows(rows(), Tokenizer(), 16)
    assert set(data[0]['inputs'])=={'input_ids', 'attention_mask'}
    with pytest.raises(ValueError, match='token'):
        tokenize_rows(rows(), Tokenizer(), 2)


def test_resume_matches_uninterrupted_with_dropout_and_partial_batch(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    initial=copy.deepcopy(StylePredictor(TinyEncoder(), dropout=.2).state_dict())
    data=tokenize_rows(rows(), Tokenizer(), 16)
    def fresh():
        model=StylePredictor(TinyEncoder(), dropout=.2)
        model.load_state_dict(initial)
        return model
    whole=fresh()
    fit(whole, data, data[:3], config(), tmp_path/'whole', {'input':'same'}, 0, 'cpu')
    partial=fresh()
    fit(partial, data, data[:3], config(), tmp_path/'resume', {'input':'same'}, 0, 'cpu', stop_after=1)
    restored=fresh()
    result=fit(restored, data, data[:3], config(), tmp_path/'resume', {'input':'same'}, 0, 'cpu')
    assert result['complete'] and result['updates']==4
    for key,value in whole.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
    with pytest.raises(ValueError, match='identity'):
        fit(fresh(),data,data[:3],config(),tmp_path/'resume',{'input':'changed'},0,'cpu')


def test_accumulation_matches_single_batch(tmp_path):
    torch.manual_seed(1)
    first=StylePredictor(TinyEncoder(),dropout=0.)
    second=copy.deepcopy(first)
    data=tokenize_rows(rows(5),Tokenizer(),16)
    # Different missing counts must be normalized across the whole accumulation window.
    data[0]['masks'][1]=False
    a=config(); a.update(epochs=1,batch_size=2,accumulation_steps=3)
    b=dict(a,batch_size=5,accumulation_steps=1)
    fit(first,data,data,a,tmp_path/'a',{},0,'cpu')
    fit(second,data,data,b,tmp_path/'b',{},0,'cpu')
    for k,v in first.state_dict().items():
        torch.testing.assert_close(v,second.state_dict()[k],rtol=1e-5,atol=1e-7)


def test_real_small_bert_forward_backward_and_training(tmp_path):
    transformers=pytest.importorskip('transformers')
    from fantasyvoice.training.predictor import preflight, evaluate, constant_baseline
    torch.set_num_threads(1)
    encoder=transformers.BertModel(transformers.BertConfig(vocab_size=32,hidden_size=16,
        num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=32),
        add_pooling_layer=False)
    model=StylePredictor(encoder)
    data=tokenize_rows(rows(),Tokenizer(),16)
    before=copy.deepcopy(model.state_dict()); rng=torch.get_rng_state().clone()
    preflight(model,data,config(),0,'cpu')
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in before.items():
        torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    result=fit(model,data,data[:3],config(),tmp_path,{},0,'cpu')
    assert result['complete']
    before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    metrics, predictions=evaluate(model,data,config(),0,'cpu',predictions=True)
    baseline=constant_baseline(data,data,config(),0,'cpu')
    assert len(predictions)==len(data)
    assert torch.equal(rng,torch.get_rng_state()) and model.training
    assert metrics['total_loss']>=0 and baseline['total_loss']>=0
    for key,value in before.items():
        torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)


def test_interrupt_inside_optimizer_keeps_last_durable_checkpoint(tmp_path, monkeypatch):
    data=tokenize_rows(rows(),Tokenizer(),16)
    model=StylePredictor(TinyEncoder(),dropout=0.)
    before=copy.deepcopy(model.state_dict())
    original=torch.optim.AdamW.step
    def interrupt(optimizer,*args,**kwargs):
        original(optimizer,*args,**kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(torch.optim.AdamW,'step',interrupt)
    with pytest.raises(KeyboardInterrupt):
        fit(model,data,data,config(),tmp_path,{},0,'cpu')
    state=torch.load(tmp_path/'latest.pt',weights_only=True)
    assert state['updates']==0
    for key,value in before.items():
        torch.testing.assert_close(value,state['model'][key],rtol=0,atol=0)


def test_overflow_replays_window_without_advancing_scheduler(tmp_path, monkeypatch):
    data=tokenize_rows(rows(),Tokenizer(),16)
    torch.manual_seed(3)
    reference=StylePredictor(TinyEncoder(),dropout=.2)
    replay=copy.deepcopy(reference)
    fit(reference,data,data,config(),tmp_path/'reference',{},0,'cpu')
    class SimulatedScaler:
        def __init__(self,*args,**kwargs):
            self.attempts=0
            self.overflow=False
        def is_enabled(self): return True
        def scale(self,loss): return loss
        def unscale_(self,optimizer):
            self.overflow=self.attempts==0
            if self.overflow:
                optimizer.param_groups[0]['params'][0].grad.fill_(float('inf'))
        def step(self,optimizer):
            if not self.overflow: optimizer.step()
        def update(self): self.attempts+=1
        def get_scale(self): return 32768.
        def state_dict(self): return {'attempts':self.attempts}
        def load_state_dict(self,state): self.attempts=state['attempts']
    monkeypatch.setattr(torch.amp,'GradScaler',SimulatedScaler)
    with pytest.warns(RuntimeWarning,match='FP16 overflow'):
        result=fit(replay,data,data,config(),tmp_path/'replay',{},0,'cpu')
    assert result['updates']==4
    for key,value in reference.state_dict().items():
        torch.testing.assert_close(value,replay.state_dict()[key],rtol=0,atol=0)


def test_interrupted_evaluation_is_retried_on_resume(tmp_path, monkeypatch):
    import fantasyvoice.training.predictor as trainer
    data=tokenize_rows(rows(),Tokenizer(),16)
    model=StylePredictor(TinyEncoder(),dropout=.2)
    original=trainer.evaluate
    def fail(*args,**kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(trainer,'evaluate',fail)
    with pytest.raises(KeyboardInterrupt):
        fit(model,data,data,config(),tmp_path,{},0,'cpu')
    monkeypatch.setattr(trainer,'evaluate',original)
    fit(model,data,data,config(),tmp_path,{},0,'cpu')
    state=torch.load(tmp_path/'latest.pt',weights_only=True)
    assert [r['updates'] for r in state['history']]==[1,2,3,4]
