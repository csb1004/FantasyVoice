"""Melo GAN objectives, explicit pseudo/predicted style stages, and audio reports."""
import contextlib
import math
from pathlib import Path

import soundfile as sf
import torch
from torch.nn import functional as F

from melo import commons
from melo.losses import discriminator_loss, generator_loss, feature_loss, kl_loss
from fantasyvoice.dataset.storage import write_json
from fantasyvoice.dataset.tts_data import collate_tts
from .predictor import collate, loss_sums, rng_state, restore_rng, amp_context
from .tts_audio import SpectralFeatures


@contextlib.contextmanager
def frozen_parameters(*modules):
    parameters = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in parameters:
            parameter.requires_grad_(enabled)


class TTSObjective:
    def __init__(self, modules, data, config, device, pad_id):
        self.modules, self.data, self.config, self.device = modules, data, config, device
        self.pad_id = pad_id
        self.spectral = SpectralFeatures().to(device)

    def set_update(self, update):
        base = self.modules['g'].base
        base.current_mas_noise_scale = max(base.mas_noise_scale_initial - base.noise_scale_delta * update, 0.)

    def style(self, batch):
        if self.config['stage'] == 'warmup':
            return batch['targets'][:, :7], batch['targets'][:, 7:], None
        inputs, targets, masks = collate(batch['rows'], self.pad_id, self.device)
        with amp_context(self.device, self.config['precision']):
            prediction = self.modules['predictor'](**inputs)
        return prediction['emotion_logits'].float().softmax(-1), prediction['continuous'].float(), prediction

    def forward(self, batch):
        probs, z, prediction = self.style(batch)
        outputs, values = self.modules['g'](batch, probs, z)
        fake, duration, _, ids, x_mask, z_mask, latent, duration_features = outputs
        real = commons.slice_segments(batch['waveform'], ids * 512, 16384)
        target_mel = commons.slice_segments(self.spectral.mel(batch['spec']), ids, 32)
        predicted_mel = self.spectral.mel(self.spectral.spectrogram(fake.squeeze(1)))
        z_value, z_p, m_p, logs_p, m_q, logs_q = latent
        reconstruction = {'mel': F.l1_loss(predicted_mel, target_mel),
                          'duration':duration.float().sum(),
                          'kl':kl_loss(z_p, logs_q, m_p, logs_p, z_mask)}
        return real, fake, x_mask, duration_features, reconstruction, prediction, values

    def __call__(self, indices):
        batch = collate_tts([self.data[i] for i in indices], self.device)
        real, fake, mask, (hidden, logw, target_logw), rec, prediction, _ = self.forward(batch)
        d, dur = self.modules['d'], self.modules['dur']
        with amp_context(self.device, self.config['precision']):
            dr, df, _, _ = d(real, fake.detach())
            loss_d = discriminator_loss(dr, df)[0]
            # Reference duration is real; the predicted duration is fake.
            rr, rf = dur(hidden.detach(), mask.detach(), target_logw.detach(), logw.detach())
            loss_dur_d = discriminator_loss([rr], [rf])[0]
        with frozen_parameters(d, dur), amp_context(self.device, self.config['precision']):
            _, gf, fr, fg = d(real, fake)
            _, duration_fake = dur(hidden, mask, target_logw, logw)
            g_loss = (generator_loss(gf)[0] + feature_loss(fr, fg)
                      + generator_loss([duration_fake])[0])
        g_loss = g_loss + 45 * rec['mel'] + rec['duration'] + rec['kl']
        if prediction is not None:
            masks = torch.ones(len(indices), 4, device=self.device, dtype=torch.bool)
            sums, counts = loss_sums(prediction, batch['targets'], masks)
            g_loss = g_loss + sum(sums[k] / counts[k] for k in sums)
        return {'g':g_loss, 'd':loss_d, 'dur':loss_dur_d}

    def gradient_preflight(self, row, predictor=None):
        """TTS reconstruction+duration+KL only. No style loss can hide a disconnected path."""
        before = rng_state()
        modes = {key:module.training for key,module in self.modules.items()}
        try:
            batch = collate_tts([row], self.device)
            if predictor is not None:
                predictor.zero_grad(set_to_none=True)
                inputs, _, _ = collate([row], self.pad_id, self.device)
                prediction = predictor(**inputs)
                logits, z = prediction['emotion_logits'], prediction['continuous']
                logits.retain_grad(); z.retain_grad()
            else:
                logits = batch['targets'][:, :7].clamp_min(1e-8).log().detach().requires_grad_()
                z = batch['targets'][:, 7:].detach().requires_grad_()
            result, _ = self.modules['g'](batch, logits.float().softmax(-1), z)
            fake, duration, _, ids, _, mask, latent, _ = result
            target_mel = commons.slice_segments(self.spectral.mel(batch['spec']), ids, 32)
            mel = self.spectral.mel(self.spectral.spectrogram(fake.squeeze(1)))
            _, zp, mp, lp, _, lq = latent
            loss = 45 * F.l1_loss(mel, target_mel) + duration.sum() + kl_loss(zp, lq, mp, lp, mask)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite TTS-only preflight loss')
            loss.backward()
            if logits.grad is None or z.grad is None:
                raise RuntimeError('Disconnected style gradient')
            norms = {'emotion':float(logits.grad.abs().sum()), **{
                key:float(z.grad[:, i].abs().sum()) for i,key in enumerate(('speed','pitch','pause'))}}
            if any(not torch.isfinite(torch.tensor(v)) or v <= 0 for v in norms.values()):
                raise RuntimeError(f'TTS-only style gradient check failed: {norms}')
            if predictor is not None:
                for name in ('encoder', 'emotion_head', 'continuous_head'):
                    grads = [p.grad for p in getattr(predictor, name).parameters()]
                    if any(g is None or not torch.isfinite(g).all() for g in grads) or sum(g.abs().sum() for g in grads) <= 0:
                        raise RuntimeError(f'No finite Predictor gradient: {name}')
            return {'tts_only_gradient_l1':norms, 'predictor_checked':predictor is not None}
        finally:
            for key, module in self.modules.items():
                module.zero_grad(set_to_none=True)
                module.train(modes[key])
            if predictor is not None:
                predictor.zero_grad(set_to_none=True)
            restore_rng(before)

    def evaluate(self, validation, output, state):
        full = state['pending_eval']['full']
        indices = range(len(validation)) if full else range(min(8, len(validation)))
        metrics = {key:0. for key in ('mel', 'duration', 'kl')}
        with torch.inference_mode():
            for i in indices:
                batch = collate_tts([validation[i]], self.device)
                *_, rec, _, _ = self.forward(batch)
                for key in metrics:
                    value = float(rec[key])
                    if not math.isfinite(value):
                        raise RuntimeError(f'Nonfinite validation {key}')
                    metrics[key] += value
            metrics = {key:value / len(indices) for key,value in metrics.items()}
            # Fixed examples use no reference waveform for generation. Warmup
            # uses pseudo conditions, joint uses text-predicted conditions.
            self.samples(validation, Path(output) / 'samples' / f"step-{state['updates']:07d}")
        return {**({k:v for k,v in metrics.items()} if full else {'sample_'+k:v for k,v in metrics.items()}),
                'validation_count':len(indices), 'full_validation':full}

    def samples(self, validation, output):
        output.mkdir(parents=True, exist_ok=True)
        rows = [validation[i] for i in range(min(3, len(validation)))]
        metadata = []
        for i, row in enumerate(rows):
            batch = collate_tts([row], self.device)
            probs, z, _ = self.style(batch)
            conditions = [('base', probs, z)]
            if i == 0:
                for emotion in range(7):
                    p = torch.zeros_like(probs); p[:, emotion] = 1
                    conditions.append((f'emotion-{emotion}', p, z))
                for j,key in enumerate(('speed','pitch','pause')):
                    for delta in (-1., 1.):
                        changed = z.clone(); changed[:,j] += delta
                        conditions.append((f'{key}-{delta:+.0f}sd', probs, changed))
            for name, p, value in conditions:
                torch.manual_seed(self.config['seed'])
                wav, values = self.modules['g'].infer(batch, p, value)
                if not torch.isfinite(wav).all() or wav.numel() == 0:
                    raise RuntimeError('Invalid generated waveform')
                filename = f'{i:02d}-{name}.wav'
                # FLOAT preserves amplitudes; never normalize a bad sample into sounding valid.
                sf.write(output / filename, wav[0,0].cpu().numpy(), 44100, subtype='FLOAT')
                metadata.append({'file':filename, 'text':row['text'], 'character_id':row['character_id'],
                    'condition_source':self.config['stage'], 'emotion_probabilities':p[0].cpu().tolist(),
                    'raw_style':values['raw'][0].cpu().tolist(),
                    'bounded_style':values['bounded'][0].cpu().tolist(),
                    'peak_amplitude':float(wav.abs().max())})
        write_json(output / 'samples.json', metadata)
