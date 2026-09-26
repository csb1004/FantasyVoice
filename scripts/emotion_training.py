"""Stage 11 supervised emotion training and the completed-checkpoint handoff."""
import math
from pathlib import Path

import torch

from fantasyvoice.dataset.storage import write_json
from fantasyvoice.style.features import EMOTIONS
from fantasyvoice.training.predictor import atomic_checkpoint, rng_state, restore_rng
from emotion_feedback import emotion_kl


class EmotionData:
    def __init__(self, rows, audio_paths):
        if len(rows) != len(audio_paths):
            raise ValueError('Audio/label count mismatch')
        self.rows, self.audio_paths = rows, audio_paths

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        wave = torch.load(self.audio_paths[i], map_location='cpu', weights_only=True)['waveform']
        style = self.rows[i]['training_style']
        if style['valid_targets']['emotion'] is not True:
            raise ValueError('Missing emotion label')
        target = torch.tensor([[style['emotion_probabilities'][key] for key in EMOTIONS]])
        return wave.unsqueeze(0), target


def select_duration(splits, maximum):
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError('Invalid maximum duration')
    selected, report = {}, {}
    for split, rows in splits.items():
        if any(not math.isfinite(r['duration_seconds']) or r['duration_seconds'] <= 0 for r in rows):
            raise ValueError('Invalid source duration')
        kept = [r for r in rows if r['duration_seconds'] <= maximum]
        if not kept or {r['character_id'] for r in kept} != {r['character_id'] for r in rows}:
            raise ValueError(f'{split}: duration filtering loses a character; increase MAX_SECONDS')
        selected[split] = kept
        report[split] = {'included':len(kept), 'excluded_over_duration':len(rows)-len(kept)}
    return selected, report


def validate_emotion_checkpoint(best, report, identity):
    saved = best['fingerprint']
    if saved['identity'] != identity:
        raise ValueError('Emotion checkpoint identity mismatch')
    config = saved['config']
    total = math.ceil(saved['size'] / config['accumulation_steps']) * config['epochs']
    if (report.get('complete') is not True or report.get('pending_eval') is not None
            or report['epoch'] != config['epochs'] or report['updates'] != total
            or report['total_updates'] != total):
        raise ValueError('Complete stage 11 and its validation before stage 12')
    if (not 0 <= best['updates'] <= total or best['updates'] != report['best_update']
            or not math.isfinite(best['validation_kl']) or best['validation_kl'] != report['best_kl']):
        raise ValueError('Emotion best checkpoint/report mismatch')


def fit_emotion(model, train, validation, config, output, identity, device,
                progress=None, max_updates=None):
    """One supervised optimizer, weighted accumulation, durable pending evaluation.

    FP32 intentionally. Baseline is eligible for selection: pseudo labels were
    produced by this pretrained analyzer, so fine-tuning need not improve it.
    """
    for key in ('epochs','accumulation_steps','save_every','eval_every','validation_samples'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'Invalid {key}')
    for key in ('learning_rate','gradient_clip','lr_decay'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not len(train) or not len(validation):
        raise ValueError('Empty train/validation split')
    output = Path(output)
    path = output / 'latest.pt'
    if output.exists() and any(output.iterdir()) and not path.exists():
        raise ValueError('Nonempty output without a resumable checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])
    fingerprint = {'config':config, 'identity':identity, 'size':len(train)}
    state = dict(epoch=0, cursor=0, updates=0, pending_eval=True,
                 best_kl=None, best_update=None, history=[])
    total = math.ceil(len(train) / config['accumulation_steps']) * config['epochs']

    def save():
        atomic_checkpoint(path, {**state, 'fingerprint':fingerprint, 'model':model.state_dict(),
            'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'rng':rng_state()})
        write_json(output/'progress.json', {k:v for k,v in state.items() if k != 'history'})
        write_json(output/'history.json', state['history'])

    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Analyzer resume identity/config mismatch; choose a new experiment')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'])
        state = {key:saved[key] for key in state}
        del saved
    else:
        save()

    def evaluate():
        if state['pending_eval'] is None:
            return
        full = state['pending_eval']
        before = rng_state()
        try:
            model.eval()
            count = len(validation) if full else min(len(validation), config['validation_samples'])
            score = 0.
            with torch.no_grad():
                for i in range(count):
                    wave, target = validation[i]
                    score += float(emotion_kl(model(wave.to(device)), target.to(device)))
            score /= count
            state['history'].append({'updates':state['updates'], 'validation_kl':score,
                                     'validation_count':count, 'full_validation':full})
            if full and (state['best_kl'] is None or score < state['best_kl']):
                state['best_kl'], state['best_update'] = score, state['updates']
                atomic_checkpoint(output/'best-model.pt', {'fingerprint':fingerprint,
                    'model':model.state_dict(), 'updates':state['updates'], 'validation_kl':score})
        finally:
            restore_rng(before)
        state['pending_eval'] = None
        save()

    evaluate()
    while state['epoch'] < config['epochs']:
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(
            config['seed'] + state['epoch'])).tolist()
        while state['cursor'] < len(train):
            indices = order[state['cursor']:state['cursor']+config['accumulation_steps']]
            optimizer.zero_grad(set_to_none=True)
            model.train()
            loss_sum = 0.
            for i in indices:
                wave, target = train[i]
                loss = emotion_kl(model(wave.to(device)), target.to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite analyzer loss; no optimizer update')
                (loss / len(indices)).backward()
                loss_sum += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite analyzer gradient; no optimizer update')
            optimizer.step()
            state['updates'] += 1
            state['cursor'] += len(indices)
            epoch_end = state['cursor'] == len(train)
            if epoch_end:
                scheduler.step()
                state['epoch'] += 1
                state['cursor'] = 0
            if epoch_end or state['updates'] % config['eval_every'] == 0:
                state['pending_eval'] = epoch_end
            if (state['pending_eval'] is not None or state['updates'] % config['save_every'] == 0
                    or max_updates is not None and state['updates'] >= max_updates):
                save()
            evaluate()
            if progress:
                progress({'updates':state['updates'], 'total_updates':total,
                          'epoch':state['epoch'], 'train_kl':loss_sum/len(indices)})
            if max_updates is not None and state['updates'] >= max_updates:
                return {**state, 'total_updates':total, 'complete':state['epoch'] == config['epochs']}
            if epoch_end:
                break
    return {**state, 'total_updates':total, 'complete':True}
