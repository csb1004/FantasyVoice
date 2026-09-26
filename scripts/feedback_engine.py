"""Stage 12 durable GAN engine with actual-batch OOM retries.

Based on the existing training engine; historical src hashes remain unchanged.
"""
import math
from pathlib import Path

import torch

from fantasyvoice.training.predictor import atomic_checkpoint, rng_state, restore_rng
from fantasyvoice.dataset.storage import write_json
from gpu_batches import backward_with_batch_retry


def fit_engine(modules, optimizers, schedulers, objective, size, config, output,
               identity, device, evaluate=None, progress=None, max_updates=None):
    for key in ('epochs', 'batch_size', 'accumulation_steps', 'save_every', 'eval_every'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'Invalid {key}')
    if size < 1 or not math.isfinite(config['gradient_clip']) or config['gradient_clip'] <= 0:
        raise ValueError('Invalid dataset/gradient clip')
    if config['precision'] not in ('fp32', 'fp16'):
        raise ValueError('Invalid precision')
    if config['precision'] == 'fp16' and not str(device).startswith('cuda'):
        raise ValueError('FP16 requires CUDA')
    if set(optimizers) != set(schedulers):
        raise ValueError('Missing optimizer scheduler')
    output = Path(output)
    path = output / 'latest.pt'
    if output.exists() and any(output.iterdir()) and not path.exists():
        raise ValueError('Nonempty output without a resumable checkpoint; choose a new OUTPUT')
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = {'config': config, 'identity': identity, 'size': size}
    scaler = torch.amp.GradScaler('cuda', enabled=config['precision'] == 'fp16')
    state = {'epoch': 0, 'cursor': 0, 'updates': 0, 'pending_eval': None,
             'best_mel': None, 'best_update': None, 'history': [], 'microbatch_size':config['batch_size']}

    def save():
        atomic_checkpoint(path, {**state, 'fingerprint': fingerprint,
            'modules': {k:m.state_dict() for k,m in modules.items()},
            'optimizers': {k:o.state_dict() for k,o in optimizers.items()},
            'schedulers': {k:s.state_dict() for k,s in schedulers.items()},
            'scaler': scaler.state_dict(), 'rng': rng_state()})
        write_json(output / 'progress.json', {k:v for k,v in state.items() if k != 'history'})
        write_json(output / 'history.json', state['history'])

    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Checkpoint identity/config differs; choose a new OUTPUT')
        if set(saved['modules']) != set(modules):
            raise ValueError('Checkpoint module identity differs')
        for key, module in modules.items():
            module.load_state_dict(saved['modules'][key], strict=True)
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(saved['optimizers'][key])
            schedulers[key].load_state_dict(saved['schedulers'][key])
        scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng'])
        state = {k:saved[k] for k in state}
        del saved
    else:
        save()  # Even an interruption in the first multi-optimizer commit can resume.

    def run_evaluation():
        if not state['pending_eval']:
            return
        before, modes = rng_state(), {k:m.training for k,m in modules.items()}
        try:
            if hasattr(objective, 'set_update'):
                objective.set_update(max(state['updates'] - 1, 0))
            for module in modules.values():
                module.eval()
            torch.manual_seed(config['seed'])
            metrics = evaluate(state) if evaluate else {}
        finally:
            for key, module in modules.items():
                module.train(modes[key])
            restore_rng(before)
        if 'mel' in metrics and not math.isfinite(metrics['mel']):
            raise RuntimeError('Nonfinite validation mel')
        state['history'].append({'updates':state['updates'], 'epoch':state['epoch'], **metrics})
        improved = ('mel' in metrics and (state['best_mel'] is None or metrics['mel'] < state['best_mel']))
        if improved:
            state['best_mel'], state['best_update'] = metrics['mel'], state['updates']
            # Inference export is atomic and not a resumable optimizer checkpoint.
            atomic_checkpoint(output / 'best-model.pt', {'fingerprint':fingerprint,
                'updates':state['updates'], 'mel':metrics['mel'],
                'modules':{k:m.state_dict() for k,m in modules.items() if k in ('g', 'predictor')}})
        state['pending_eval'] = None
        save()

    run_evaluation()
    window_size = config['batch_size'] * config['accumulation_steps']
    total_updates = math.ceil(size / window_size) * config['epochs']
    for module in modules.values():
        module.train()
    while state['epoch'] < config['epochs']:
        order = torch.randperm(size, generator=torch.Generator().manual_seed(
            config['seed'] + state['epoch'])).tolist()
        while state['cursor'] < size:
            indices = order[state['cursor']:state['cursor'] + window_size]
            if hasattr(objective, 'set_update'):
                objective.set_update(state['updates'])
            window_rng = rng_state()
            committed = False
            totals = {}
            for attempt in range(9):
                restore_rng(window_rng)
                totals = {key:0. for key in optimizers}
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                totals, state['microbatch_size'] = backward_with_batch_retry(
                    objective, indices, state['microbatch_size'], scaler, optimizers, window_rng)
                norms = {}
                for key, optimizer in optimizers.items():
                    scaler.unscale_(optimizer)
                    parameters = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
                    if not parameters:
                        raise RuntimeError(f'No gradient for optimizer {key}')
                    norms[key] = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'])
                if all(torch.isfinite(value) for value in norms.values()):
                    # No checkpoint can be written inside this transaction. If any
                    # step raises, disk retains the preceding complete boundary.
                    for optimizer in optimizers.values():
                        scaler.step(optimizer)
                    scaler.update()
                    committed = True
                    break
                if not scaler.is_enabled() or attempt == 8:
                    raise RuntimeError('Nonfinite gradients; no optimizer committed')
                scaler.update(new_scale=scaler.get_scale() / 2)
            if not committed:
                raise RuntimeError('Optimizer window not committed')
            state['updates'] += 1
            state['cursor'] += len(indices)
            epoch_end = state['cursor'] == size
            if epoch_end:
                for scheduler in schedulers.values():
                    scheduler.step()
                state['epoch'] += 1
                state['cursor'] = 0
            if evaluate and (epoch_end or state['updates'] % config['eval_every'] == 0):
                state['pending_eval'] = {'full':epoch_end}
            if (state['pending_eval'] or epoch_end or state['updates'] % config['save_every'] == 0
                    or max_updates is not None and state['updates'] >= max_updates):
                save()
            run_evaluation()
            if progress:
                progress({'updates':state['updates'], 'total_updates':total_updates,
                          'epoch':state['epoch'], 'losses':totals, 'microbatch_size':state['microbatch_size']})
            if max_updates is not None and state['updates'] >= max_updates:
                return {**state, 'complete':state['epoch'] == config['epochs']}
            if epoch_end:
                break
    return {**state, 'complete':True, 'total_updates':total_updates}
