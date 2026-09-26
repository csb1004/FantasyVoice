"""Shared emotion adapter and stage 12 feedback through a frozen stage 11 analyzer."""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from fantasyvoice.style.features import EMOTIONS
from fantasyvoice.training.tts import TTSObjective, frozen_parameters
from fantasyvoice.dataset.tts_data import collate_tts


def emotion_indices(labels):
    labels = [s.split('/')[-1].strip().lower() for s in labels]
    labels = ['unknown' if s == '<unk>' else s for s in labels]
    if len(labels) != 9 or set(labels) != set(EMOTIONS) | {'other', 'unknown'}:
        raise ValueError('Expected the original nine emotion2vec classes')
    return [labels.index(name) for name in EMOTIONS]


def emotion_kl(logits, target):
    if logits.shape != target.shape or logits.ndim != 2 or logits.shape[1] != 7:
        raise ValueError('Expected seven emotion probabilities')
    if (not torch.isfinite(logits).all() or not torch.isfinite(target).all()
            or (target < 0).any() or not torch.allclose(target.sum(-1),
                torch.ones_like(target[:, 0]), atol=1e-5)):
        raise ValueError('Invalid emotion distribution')
    return F.kl_div(logits.float().log_softmax(-1), target.detach().float(), reduction='batchmean')


class EmotionAnalyzer(nn.Module):
    """Same normalization/mean-pooling/head as pinned FunASR utterance inference.

    No generate(), numpy(), detach() or inference_mode in the waveform path.
    The seven retained logits renormalize exactly like the existing label table.
    local_grad_mult=1 is essential: upstream can otherwise disable audio gradients.
    """
    def __init__(self, model, labels, resample):
        super().__init__()
        self.model, self.resample = model, resample
        self.register_buffer('indices', torch.tensor(emotion_indices(labels)))
        if model.proj is None or model.proj.out_features != 9:
            raise ValueError('Missing original emotion2vec nine-class head')
        model.modality_encoders['AUDIO'].local_grad_mult = 1.

    def forward(self, wave):
        if wave.ndim != 2 or wave.shape[0] != 1 or not torch.isfinite(wave).all():
            raise ValueError('Emotion feedback requires one finite unpadded utterance')
        # eval disables dropout/masking drift, not parameter or waveform gradients.
        self.model.eval()
        with torch.autocast(device_type=wave.device.type, enabled=False):
            source = self.resample(wave.float())
            if source.shape[-1] < 400:
                raise ValueError('Generated utterance too short for emotion2vec')
            if self.model.cfg.normalize:
                source = F.layer_norm(source, (source.shape[-1],))
            features = self.model.extract_features(source, padding_mask=None, mask=False)['x']
            return self.model.proj(features.mean(1)).index_select(-1, self.indices)


def load_analyzer(config, device):
    from funasr import AutoModel
    from huggingface_hub import snapshot_download
    from torchaudio.transforms import Resample
    directory = snapshot_download(config['repo'], revision=config['revision'],
                                  allow_patterns=['*.json', '*.yaml', '*.txt', '*.pt'])
    backend = AutoModel(model=directory, hub='hf', device=device, disable_update=True,
                        disable_pbar=True, trust_remote_code=False)
    tokenizer = backend.kwargs['tokenizer']
    labels = tokenizer.token_list
    model = backend.model
    model.requires_grad_(True)
    return EmotionAnalyzer(model, labels, Resample(44100, 16000)).to(device)


@contextmanager
def scoring(model):
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with frozen_parameters(model):
            yield
    finally:
        for module, mode in modes.items():
            module.training = mode


def generated_emotion_loss(analyzer, generated, target):
    with scoring(analyzer):
        return emotion_kl(analyzer(generated), target)


def generate_prior(tts, batch, probabilities, z_style, max_frames):
    """Full text-prior synthesis with gradients; no reference waveform/posterior.

    Mirrors pinned Melo infer's default sdp_ratio=0 (deterministic duration net).
    A length guard runs BEFORE attention/decoder allocation. Rounded duration is
    not differentiable; reconstruction duration supervision is retained separately.
    See docs/melotts-LICENSE for the underlying MIT inference formulation.
    """
    from melo import commons
    base = tts.base
    if batch['x'].shape[0] != 1:
        raise ValueError('Full-utterance feedback requires microbatch 1')
    g, _ = tts.conditioner(batch['character_ids'], probabilities, z_style)
    with torch.autocast(device_type=g.device.type, enabled=False):
        hidden, mean, logs, mask = base.enc_p(batch['x'], batch['x_lengths'],
            batch['tone'], batch['language'], batch['bert'].float(), batch['ja_bert'].float(),
            g=None if base.use_vc else g.float())
        logw = base.dp(hidden, mask, g=g.float())
        durations = torch.ceil(torch.exp(logw) * mask)
        total = durations.sum()
        if not torch.isfinite(total) or not 1 <= total.item() <= max_frames:
            raise ValueError('Generated length exceeds feedback limit; no truncation or optimizer update')
        lengths = durations.sum((1, 2)).long()
        y_mask = commons.sequence_mask(lengths).unsqueeze(1).to(mask.dtype)
        attention = commons.generate_path(durations, mask.unsqueeze(2) * y_mask.unsqueeze(-1))
        mean = torch.matmul(attention.squeeze(1), mean.transpose(1, 2)).transpose(1, 2)
        logs = torch.matmul(attention.squeeze(1), logs.transpose(1, 2)).transpose(1, 2)
        prior = mean + torch.randn_like(mean) * logs.exp() * .667
        latent = base.flow(prior, y_mask, g=g.float(), reverse=True)
        # FP32 through the complete audio path; no numpy/resampling detach.
        wave = base.dec(latent * y_mask, g=g.float())
    if not torch.isfinite(wave).all():
        raise ValueError('Nonfinite generated utterance')
    return wave[:, 0]


class FeedbackObjective(TTSObjective):
    def __init__(self, *args, emotion, **kwargs):
        super().__init__(*args, **kwargs)
        self.emotion = emotion.eval().requires_grad_(False)

    def style(self, batch):
        # Explicit targets; a frozen text Predictor cannot move the goalposts.
        return batch['targets'][:, :7], batch['targets'][:, 7:], None

    def generated(self, batch):
        model = self.modules['g']
        modes = {m: m.training for m in model.modules()}
        try:
            model.eval()
            return generate_prior(model, batch, *self.style(batch)[:2],
                max_frames=int(self.config['feedback']['max_seconds'] * 44100 / 512))
        finally:
            for m, mode in modes.items():
                m.training = mode

    def __call__(self, indices):
        if len(indices) != 1:
            raise ValueError('Use batch_size=1 and accumulation for emotion feedback')
        losses = super().__call__(indices)
        batch = collate_tts([self.data[indices[0]]], self.device)
        generated = generated_emotion_loss(self.emotion, self.generated(batch), batch['targets'][:, :7])
        losses['g'] = losses['g'] + self.config['feedback']['weight'] * generated
        return losses

    def feedback_preflight(self, row):
        """Fail early if analyzer wrappers silently disconnect generator gradients."""
        batch = collate_tts([row], self.device)
        with scoring(self.emotion):
            wave = self.generated(batch)
            loss = emotion_kl(self.emotion(wave), batch['targets'][:, :7])
        generator = self.modules['g']
        parameters = [generator.conditioner.character.weight,
                      next(generator.base.dec.parameters())]
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        norms = [float(g.norm()) if g is not None else 0. for g in gradients]
        if not all(math.isfinite(n) and n > 0 for n in norms):
            raise RuntimeError('Emotion loss does not reach character/decoder parameters')
        if any(p.requires_grad or p.grad is not None for p in self.emotion.parameters()):
            raise RuntimeError('Stage 12 analyzer must remain frozen')
        return {'generator_character_decoder_grad_norms':norms,
                'generated_emotion_kl':float(loss.detach()), 'analyzer_frozen':True}

    def evaluate(self, validation, output, state):
        metrics = super().evaluate(validation, output, state)
        # Fixed held-out subset for expensive full-utterance diagnostics.
        count = min(self.config['feedback']['validation_samples'], len(validation))
        totals = {'generated_emotion_kl':0., 'real_emotion_kl':0.}
        with torch.no_grad():
            for i in range(count):
                batch = collate_tts([validation[i]], self.device)
                target = batch['targets'][:, :7]
                totals['generated_emotion_kl'] += float(emotion_kl(
                    self.emotion(self.generated(batch)), target))
                totals['real_emotion_kl'] += float(emotion_kl(
                    self.emotion(batch['waveform'][:, 0]), target))
        return {**metrics, **{k:v/count for k,v in totals.items()},
                'emotion_validation_count':count, 'emotion_evaluator':'frozen_stage_11'}

    def samples(self, validation, output):
        import json
        from pathlib import Path
        from fantasyvoice.dataset.storage import write_json
        super().samples(validation, output)
        path = Path(output) / 'samples.json'
        samples = json.loads(path.read_text(encoding='utf-8'))
        for sample in samples:
            sample['condition_source'] = 'dataset_pseudo_labels_with_named_style_perturbations'
        write_json(path, samples)
