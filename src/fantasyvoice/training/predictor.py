"""Text-only supervised warmup with update-boundary resume and auditable evaluation."""
import contextlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unicodedata
import warnings
import zipfile

import torch
from torch.nn import functional as F

from fantasyvoice.dataset.storage import write_json, write_jsonl
from fantasyvoice.style.features import EMOTIONS
from fantasyvoice.style.runner import digest

TARGETS = ('emotion', 'speed', 'pitch', 'pause')


def read_prepared(path):
    with zipfile.ZipFile(path) as archive:
        prefix = 'dataset-prepared-v1/'
        def read(name):
            return json.loads(archive.read(prefix + name))
        report = read('dataset-preparation.json')
        splits = {split: [json.loads(line) for line in archive.read(prefix + split + '.jsonl').decode('utf-8').splitlines() if line.strip()]
                  for split in ('train', 'validation')}
        metadata = {'preparation': report, 'character_map': read('character-map.json'),
                    'analyzer_config': read('analyzer-config.json'), 'source_report': read('source-report.json')}
    if report['statistics_fitted_on'] != 'train_only':
        raise ValueError('Expected train-only normalization')
    for split, rows in splits.items():
        if not rows or len(rows) != report[split + '_count']:
            raise ValueError('Prepared split count mismatch')
        if len({row['audio_path'] for row in rows}) != len(rows):
            raise ValueError('Duplicate input paths')
        if any(row['split'] != split or row.get('validation_errors') != [] for row in rows):
            raise ValueError('Unexpected split or invalid style row')
    for key in ('audio_path', 'sha256', 'split_group'):
        if {r[key] for r in splits['train']} & {r[key] for r in splits['validation']}:
            raise ValueError(f'Train/validation leakage: {key}')
    def text_key(row):
        return row['character_id'], ' '.join(unicodedata.normalize('NFC', row['text']).split())
    if {text_key(r) for r in splits['train']} & {text_key(r) for r in splits['validation']}:
        raise ValueError('Train/validation leakage: character text')
    return splits, metadata


def tokenize_rows(rows, tokenizer, max_tokens):
    """No truncation. Targets are CPU tensors, model inputs contain token fields only."""
    result = []
    for start in range(0, len(rows), 512):
        chunk = rows[start:start + 512]
        if any(not isinstance(r['text'], str) or not r['text'].strip() for r in chunk):
            raise ValueError('Empty text')
        encoded = tokenizer([r['text'] for r in chunk], truncation=False, padding=False,
                            add_special_tokens=True, return_attention_mask=True)
        for i, row in enumerate(chunk):
            inputs = {key: encoded[key][i] for key in ('input_ids', 'attention_mask', 'token_type_ids') if key in encoded}
            if len(inputs['input_ids']) > max_tokens:
                raise ValueError(f"Text exceeds token limit ({max_tokens}): {row['audio_path']}")
            style = row['training_style']
            masks = [style['valid_targets'][key] for key in TARGETS]
            if any(type(mask) is not bool for mask in masks):
                raise ValueError('Target masks must be boolean')
            probs = style['emotion_probabilities']
            if masks[0]:
                if not isinstance(probs, dict) or set(probs) != set(EMOTIONS):
                    raise ValueError('Wrong emotion labels')
                values = [probs[key] for key in EMOTIONS]
                if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values) or not math.isclose(sum(values), 1., abs_tol=1e-5):
                    raise ValueError('Invalid emotion probabilities')
            else:
                values = [0.] * 7  # storage placeholders only; masks control all losses
            for valid, key in zip(masks[1:], ('speed_z', 'pitch_z', 'pause_z')):
                value = style[key]
                if valid and (value is None or not math.isfinite(value)):
                    raise ValueError('Invalid continuous target')
                values.append(float(value) if valid else 0.)
            result.append({'inputs': inputs, 'targets': torch.tensor(values, dtype=torch.float32),
                           'masks': torch.tensor(masks, dtype=torch.bool),
                           'audio_path': row['audio_path'], 'text': row['text']})
    return result


def collate(rows, pad_id, device):
    length = max(len(r['inputs']['input_ids']) for r in rows)
    inputs = {}
    for key in rows[0]['inputs']:
        fill = pad_id if key == 'input_ids' else 0
        inputs[key] = torch.tensor([r['inputs'][key] + [fill] * (length - len(r['inputs'][key])) for r in rows],
                                   dtype=torch.long, device=device)
    return inputs, torch.stack([r['targets'] for r in rows]).to(device), torch.stack([r['masks'] for r in rows]).to(device)


def loss_sums(outputs, targets, masks):
    logits, continuous = outputs['emotion_logits'].float(), outputs['continuous'].float()
    sums, counts = {}, {}
    for i, key in enumerate(TARGETS):
        mask = masks[:, i]
        counts[key] = int(mask.sum().item())
        if i == 0:
            sums[key] = -(targets[mask, :7] * F.log_softmax(logits[mask], -1)).sum() if counts[key] else logits.sum() * 0
        else:
            sums[key] = F.smooth_l1_loss(continuous[mask, i - 1], targets[mask, i + 6], reduction='sum', beta=1.) if counts[key] else continuous[:, i - 1].sum() * 0
    return sums, counts


def rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt.partial', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def amp_context(device, precision):
    return torch.autocast('cuda', dtype=torch.float16) if precision == 'fp16' else contextlib.nullcontext()


def evaluate(model, data, config, pad_id, device, predictions=False):
    before = rng_state()
    was_training = model.training
    totals, counts, absolute = {k: 0. for k in TARGETS}, {k: 0 for k in TARGETS}, {k: 0. for k in TARGETS[1:]}
    exported = []
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(data), config['eval_batch_size']):
                rows = data[start:start + config['eval_batch_size']]
                inputs, targets, masks = collate(rows, pad_id, device)
                with amp_context(device, config['precision']):
                    outputs = model(**inputs)
                sums, numbers = loss_sums(outputs, targets, masks)
                for key in TARGETS:
                    value = float(sums[key])
                    if not math.isfinite(value):
                        raise RuntimeError('Non-finite validation loss')
                    totals[key] += value
                    counts[key] += numbers[key]
                for j, key in enumerate(TARGETS[1:]):
                    mask = masks[:, j + 1]
                    absolute[key] += float((outputs['continuous'][mask, j].float() - targets[mask, j + 7]).abs().sum())
                if predictions:
                    probs = outputs['emotion_logits'].float().softmax(-1).cpu().tolist()
                    cont = outputs['continuous'].float().cpu().tolist()
                    for row, probability, continuous in zip(rows, probs, cont):
                        exported.append({'audio_path': row['audio_path'], 'text': row['text'],
                                         'emotion_probabilities': dict(zip(EMOTIONS, probability)),
                                         'speed_z': continuous[0], 'pitch_z': continuous[1], 'pause_z': continuous[2],
                                         'target_values': row['targets'].tolist(),
                                         'target_masks': row['masks'].tolist()})
    finally:
        model.train(was_training)
        restore_rng(before)
    losses = {key: totals[key] / counts[key] if counts[key] else None for key in TARGETS}
    return {'losses': losses, 'valid_counts': counts,
            'total_loss': sum(config['loss_weights'][k] * (losses[k] or 0.) for k in TARGETS),
            'continuous_mae_z': {key: absolute[key] / counts[key] if counts[key] else None for key in TARGETS[1:]}}, exported


def check_config(config, device):
    for key in ('epochs', 'batch_size', 'accumulation_steps', 'eval_batch_size', 'save_every', 'eval_every'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'Invalid {key}')
    for key in ('bert_lr', 'head_lr', 'eps', 'gradient_clip'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not 0 <= config['warmup_fraction'] < 1 or config['precision'] not in ('fp32', 'fp16'):
        raise ValueError('Invalid scheduler/precision')
    if config['precision'] == 'fp16' and not str(device).startswith('cuda'):
        raise ValueError('FP16 warmup requires a CUDA runtime')
    if type(config['seed']) is not int or not math.isfinite(config['weight_decay']) or config['weight_decay'] < 0:
        raise ValueError('Invalid seed/weight decay')
    if len(config['betas']) != 2 or any(not 0 <= beta < 1 for beta in config['betas']):
        raise ValueError('Invalid Adam betas')
    if set(config['loss_weights']) != set(TARGETS) or any(not math.isfinite(v) or v <= 0 for v in config['loss_weights'].values()):
        raise ValueError('All four loss weights must be finite and positive')


def preflight(model, data, config, pad_id, device):
    """Small real forward/backward without an optimizer update or consuming training RNG."""
    before = rng_state()
    was_training = model.training
    model.to(device)
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        # FP32 loss, matching normal training; this is not a claim that a full batch fits.
        inputs, targets, masks = collate(data[:min(2, len(data))], pad_id, device)
        with amp_context(device, config['precision']):
            outputs = model(**inputs)
        sums, counts = loss_sums(outputs, targets, masks)
        loss = sum(sums[k] / max(1, counts[k]) for k in TARGETS)
        if not torch.isfinite(loss):
            raise RuntimeError('Preflight loss is not finite')
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f'Preflight gradient failed: {name}')
        return {'loss': float(loss.detach()), 'samples': len(targets)}
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        restore_rng(before)


def constant_baseline(train, validation, config, pad_id, device):
    """Baseline fitted only to train, evaluated using the same masked losses."""
    targets, masks = torch.stack([r['targets'] for r in train]), torch.stack([r['masks'] for r in train])
    emotion = targets[masks[:, 0], :7].mean(0) if masks[:, 0].any() else torch.full((7,), 1/7)
    continuous = torch.stack([targets[masks[:, i+1], i+7].mean() if masks[:, i+1].any() else torch.tensor(0.) for i in range(3)])
    class Constant(torch.nn.Module):
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            n = input_ids.shape[0]
            return {'emotion_logits': emotion.clamp_min(1e-12).log().to(input_ids.device).expand(n, -1),
                    'continuous': continuous.to(input_ids.device).expand(n, -1)}
    metrics, _ = evaluate(Constant(), validation, config, pad_id, device)
    return metrics


def fit(model, train, validation, config, output, identity, pad_id, device, *, stop_after=None, progress=None):
    """Resume only at optimizer boundaries. stop_after is an absolute update limit for tests."""
    check_config(config, device)
    if not train or not validation:
        raise ValueError('Empty training/validation data')
    output = Path(output)
    fingerprint = digest({'identity': identity, 'config': config})
    latest = output / 'latest.pt'
    state = torch.load(latest, map_location='cpu', weights_only=True) if latest.exists() else None
    if state and state['fingerprint'] != fingerprint:
        raise ValueError('Checkpoint identity/config mismatch: use a new experiment directory')
    if not state and output.exists() and any(output.iterdir()):
        raise ValueError('Output is nonempty but has no latest checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config['seed']); torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    model.to(device)
    model.train()
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError('07 requires the full encoder and both heads to be trainable')
        lr = config['bert_lr'] if name.startswith('encoder.') else config['head_lr']
        decay = 0. if name.endswith('bias') or 'layernorm' in name.lower() else config['weight_decay']
        groups.setdefault((lr, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([{'params': params, 'lr': lr, 'weight_decay': decay} for (lr, decay), params in groups.items()],
                                 betas=tuple(config['betas']), eps=config['eps'])
    window_size = config['batch_size'] * config['accumulation_steps']
    windows = math.ceil(len(train) / window_size)
    total_updates = windows * config['epochs']
    warmup = math.ceil(total_updates * config['warmup_fraction'])
    def lr_factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        return max(0., (total_updates - step) / max(1, total_updates - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler('cuda', enabled=config['precision'] == 'fp16')
    updates, best, history = 0, float('inf'), []
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        updates, best, history = state['updates'], state['best_loss'], state['history']
        restore_rng(state['rng'])
        del state

    def save(path):
        atomic_checkpoint(path, {'schema_version': 1, 'fingerprint': fingerprint, 'identity': identity,
            'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state(),
            'updates': updates, 'best_loss': best, 'history': history, 'total_updates': total_updates,
            'next_epoch': updates // windows, 'next_window': updates % windows})
        write_json(output / 'progress.json', {'updates': updates, 'total_updates': total_updates,
                   'complete': updates == total_updates, 'best_validation_loss': best if math.isfinite(best) else None})
        write_json(output / 'history.json', history)

    if not latest.exists():
        save(latest)

    def evaluate_current(window_loss):
        nonlocal best
        metrics, _ = evaluate(model, validation, config, pad_id, device)
        history.append({'updates': updates, 'epoch': updates / windows,
                        'train_window_loss': window_loss, 'validation': metrics})
        if metrics['total_loss'] < best:
            best = metrics['total_loss']
            save(output / 'best.pt')
        return metrics

    boundary = rng_state()
    step_in_progress = False
    overflow_retries = 0
    try:
        if (updates and (updates % config['eval_every'] == 0 or updates % windows == 0)
                and (not history or history[-1]['updates'] != updates)):
            # An interruption during validation must not silently drop a scheduled evaluation.
            evaluate_current(None)
            save(latest)
        while updates < total_updates and (stop_after is None or updates < stop_after):
            epoch, window = divmod(updates, windows)
            permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(config['seed'] + epoch)).tolist()
            selected = [train[i] for i in permutation[window * window_size:(window + 1) * window_size]]
            boundary = rng_state()
            optimizer.zero_grad(set_to_none=True)
            denominator = torch.stack([row['masks'] for row in selected]).sum(0).tolist()
            window_loss = 0.
            for start in range(0, len(selected), config['batch_size']):
                inputs, targets, masks = collate(selected[start:start + config['batch_size']], pad_id, device)
                with amp_context(device, config['precision']):
                    outputs = model(**inputs)
                sums, _ = loss_sums(outputs, targets, masks)
                loss = sum(config['loss_weights'][key] * sums[key] / max(1, denominator[i]) for i, key in enumerate(TARGETS))
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                window_loss += float(loss.detach())
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite = True
            for parameter in model.parameters():
                if parameter.grad is None:
                    raise RuntimeError('Missing gradient: model graph is disconnected')
                finite = finite and bool(torch.isfinite(parameter.grad).all())
            if not finite:
                if not scaler.is_enabled():
                    raise RuntimeError('Non-finite FP32 gradient; inspect input and learning rate')
                # GradScaler skips the optimizer step on inf/nan and reduces its scale.
                # Replay the SAME data/RNG; do not consume an update or schedule step.
                scaler.step(optimizer)
                scaler.update()
                restore_rng(boundary)
                overflow_retries += 1
                warnings.warn(f'FP16 overflow: retrying update {updates + 1}, scale={scaler.get_scale()}', RuntimeWarning)
                if overflow_retries >= 8:
                    raise RuntimeError('Repeated FP16 gradient overflow; inspect precision and learning rate')
                continue
            overflow_retries = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            step_in_progress = True
            scaler.step(optimizer); scaler.update(); scheduler.step()
            updates += 1
            boundary = rng_state()
            step_in_progress = False
            metrics = None
            if updates % config['eval_every'] == 0 or updates % windows == 0:
                metrics = evaluate_current(window_loss)
            if updates % config['save_every'] == 0 or updates % windows == 0:
                save(latest)
            if progress:
                progress({'updates': updates, 'total_updates': total_updates, 'epoch': epoch + 1,
                          'train_window_loss': window_loss, 'validation': metrics})
        save(latest)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        if not step_in_progress:
            restore_rng(boundary)
            save(latest)
        # If interrupted inside optimizer.step, keep the last durable checkpoint unchanged.
        raise
    return {'complete': updates == total_updates, 'updates': updates, 'total_updates': total_updates,
            'best_validation_loss': best if math.isfinite(best) else None}
