import copy
import pytest
import torch

from fantasyvoice.training.tts_engine import fit_engine


def setup_run():
    torch.manual_seed(123)
    models = {name: torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.2),
             torch.nn.Linear(3, 1)) for name in ('g', 'd', 'dur')}
    opts = {k: torch.optim.AdamW(m.parameters(), lr=.01) for k, m in models.items()}
    sched = {k: torch.optim.lr_scheduler.ExponentialLR(o, gamma=.99) for k, o in opts.items()}
    def objective(indices):
        x = torch.tensor([[i / 10, 1.] for i in indices])
        return {k: (m(x) - .5).square().mean() for k, m in models.items()}
    return models, opts, sched, objective


CONFIG = dict(epochs=2, batch_size=2, accumulation_steps=2, precision='fp32',
              seed=42, gradient_clip=1., save_every=1, eval_every=2)


def test_exact_resume_all_optimizers_partial_accumulation_and_evaluation(tmp_path):
    a = setup_run()
    expected = fit_engine(*a, 7, CONFIG, tmp_path / 'full', {'dataset':'a'}, 'cpu')
    b = setup_run()
    fit_engine(*b, 7, CONFIG, tmp_path / 'resume', {'dataset':'a'}, 'cpu', max_updates=1)
    c = setup_run()
    actual = fit_engine(*c, 7, CONFIG, tmp_path / 'resume', {'dataset':'a'}, 'cpu')
    assert expected['updates'] == actual['updates'] == 4
    for key in a[0]:
        for name, value in a[0][key].state_dict().items():
            assert torch.equal(value, c[0][key].state_dict()[name])
    with pytest.raises(ValueError, match='identity'):
        fit_engine(*setup_run(), 7, CONFIG, tmp_path / 'resume', {'dataset':'b'}, 'cpu')


def test_interruption_inside_second_optimizer_preserves_checkpoint(tmp_path):
    parts = setup_run()
    path = tmp_path / 'run'
    fit_engine(*parts, 7, CONFIG, path, {}, 'cpu', max_updates=1)
    durable = (path / 'latest.pt').read_bytes()
    resumed = setup_run()
    def broken_step(*args, **kwargs):
        raise KeyboardInterrupt('during discriminator commit')
    resumed[1]['d'].step = broken_step
    with pytest.raises(KeyboardInterrupt):
        fit_engine(*resumed, 7, CONFIG, path, {}, 'cpu')
    assert (path / 'latest.pt').read_bytes() == durable
    fit_engine(*setup_run(), 7, CONFIG, path, {}, 'cpu')


def test_interrupted_evaluation_retried_before_next_update(tmp_path):
    path = tmp_path / 'run'
    def broken_eval(state):
        raise KeyboardInterrupt('evaluation')
    with pytest.raises(KeyboardInterrupt):
        fit_engine(*setup_run(), 7, CONFIG, path, {}, 'cpu', evaluate=broken_eval)
    saved = torch.load(path / 'latest.pt', weights_only=True)
    assert saved['updates'] == 2 and saved['pending_eval']
    seen = []
    def evaluate(state):
        seen.append(state['updates'])
        return {'mel': 1. / state['updates']}
    fit_engine(*setup_run(), 7, CONFIG, path, {}, 'cpu', evaluate=evaluate)
    assert seen == [2, 4]


def test_nonfinite_one_optimizer_prevents_all_updates(tmp_path):
    parts = setup_run()
    before = copy.deepcopy({k:m.state_dict() for k,m in parts[0].items()})
    original = parts[3]
    def bad(indices):
        losses = original(indices)
        losses['dur'] *= float('nan')
        return losses
    with pytest.raises(RuntimeError, match='finite'):
        fit_engine(*parts[:3], bad, 7, CONFIG, tmp_path / 'run', {}, 'cpu')
    for key, module in parts[0].items():
        for name, value in module.state_dict().items():
            assert torch.equal(value, before[key][name])


def test_amp_overflow_replays_window_before_any_commit(tmp_path, monkeypatch):
    # Exercise all-or-none retry without pretending CPU verifies CUDA kernels.
    expected = setup_run()
    fit_engine(*expected, 7, CONFIG, tmp_path / 'baseline', {}, 'cpu')
    class SimulatedScaler:
        def __init__(self, *args, **kwargs):
            self.value = 16.
            self.injected = False
        def is_enabled(self): return True
        def scale(self, loss): return loss
        def unscale_(self, optimizer):
            if not self.injected:
                self.injected = True
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.fill_(float('inf'))
                            return
        def step(self, optimizer): optimizer.step()
        def update(self, new_scale=None):
            if new_scale is not None: self.value = new_scale
        def get_scale(self): return self.value
        def state_dict(self): return {'value':self.value, 'injected':self.injected}
        def load_state_dict(self, value): self.value=value['value']; self.injected=value['injected']
    monkeypatch.setattr(torch.amp, 'GradScaler', SimulatedScaler)
    actual = setup_run()
    fit_engine(*actual, 7, CONFIG, tmp_path / 'replayed', {}, 'cpu')
    for key in expected[0]:
        for name, value in expected[0][key].state_dict().items():
            assert torch.equal(value, actual[0][key].state_dict()[name])
