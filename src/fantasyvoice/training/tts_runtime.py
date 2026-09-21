"""Pinned model loading and stage identities, separate from the training loop."""
import json
import math
from pathlib import Path

import torch

from fantasyvoice.dataset.storage import sha256
from fantasyvoice.models.predictor import StylePredictor
from fantasyvoice.style.features import EMOTIONS


def verify_melo(config):
    import melo
    root = Path(melo.__file__).parent
    for name, expected in config['melo_source_sha256'].items():
        # Git installations may normalize newlines; package artifacts must match exactly.
        if sha256(root / name) != expected:
            raise ValueError(f'Unexpected Melo source: {name}; reinstall the pinned source ZIP')


def load_tts(config, metadata, device):
    from huggingface_hub import hf_hub_download
    from melo.models import SynthesizerTrn
    from fantasyvoice.models.conditional_tts import ConditionalTTS
    verify_melo(config)
    paths = {name:Path(hf_hub_download(config['model_repo'], name, revision=config['model_revision']))
             for name in ('config.json', 'checkpoint.pth')}
    artifacts = {name:sha256(path) for name,path in paths.items()}
    if artifacts != config['model_artifact_sha256']:
        raise ValueError('Pretrained artifact SHA mismatch')
    hps = json.loads(paths['config.json'].read_text(encoding='utf-8'))
    if hps['data']['sampling_rate'] != 44100 or hps['data']['hop_length'] != 512 or hps['data']['filter_length'] != 2048:
        raise ValueError('Unsupported spectral configuration')
    if hps['train']['segment_size'] != 16384 or hps['data']['add_blank'] != config['add_blank']:
        raise ValueError('Unsupported segment/frontend configuration')
    character_map = metadata['character_map']
    if sorted(character_map.values()) != list(range(len(character_map))):
        raise ValueError('Character map must be contiguous')
    base = SynthesizerTrn(len(hps['symbols']), 1025, 32, n_speakers=hps['data']['n_speakers'],
            num_languages=hps['num_languages'], num_tones=hps['num_tones'], **hps['model'])
    model = ConditionalTTS(base, metadata['preparation']['normalization'], len(character_map))
    checkpoint = torch.load(paths['checkpoint.pth'], map_location='cpu', weights_only=True)
    model.load_base(checkpoint['model'])
    del checkpoint
    model.base.decoder_amp = config['precision'] == 'fp16' and str(device).startswith('cuda')
    return model.to(device), hps, artifacts


def load_predictor(path, dataset_sha, metadata, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    identity = saved['identity']
    stats = {'normalization':metadata['preparation']['normalization'],
             'pitch_baselines_hz':metadata['preparation']['pitch_baselines_hz'],
             'character_map':metadata['character_map']}
    if identity['dataset_sha256'] != dataset_sha or identity['dataset_statistics'] != stats:
        raise ValueError('Predictor dataset/statistics identity mismatch')
    if identity['emotion_order'] != list(EMOTIONS):
        raise ValueError('Predictor emotion order mismatch')
    config = saved['config']
    model = StylePredictor.from_pretrained(config['model_name'], config['model_revision'], config['dropout'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device), config


def make_optimizers(modules, config):
    if config['stage'] not in ('warmup', 'joint'):
        raise ValueError('Choose warmup or joint explicitly')
    if ('predictor' in modules) != (config['stage'] == 'joint'):
        raise ValueError('Predictor must be present only in joint training')
    for key in ('learning_rate', 'predictor_lr', 'eps', 'lr_decay'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Invalid {key}')
    generators = [{'params':[p for p in modules['g'].parameters() if p.requires_grad],
                   'lr':config['learning_rate']}]
    if 'predictor' in modules:
        generators.append({'params':modules['predictor'].parameters(), 'lr':config['predictor_lr']})
    optimizers = {'g':torch.optim.AdamW(generators, betas=tuple(config['betas']), eps=config['eps'],
                                       weight_decay=config['weight_decay'])}
    for key in ('d','dur'):
        optimizers[key] = torch.optim.AdamW(modules[key].parameters(), lr=config['learning_rate'],
            betas=tuple(config['betas']), eps=config['eps'], weight_decay=config['weight_decay'])
    schedulers = {k:torch.optim.lr_scheduler.ExponentialLR(o, gamma=config['lr_decay']) for k,o in optimizers.items()}
    return optimizers, schedulers


def warm_start_joint(path, modules, dataset_sha, metadata, model_revision):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    identity = saved['fingerprint']['identity']
    if saved['fingerprint']['config']['stage'] != 'warmup':
        raise ValueError('Joint warm start requires a warmup checkpoint')
    if identity['dataset_sha256'] != dataset_sha or identity['dataset_metadata'] != metadata:
        raise ValueError('Warmup dataset identity mismatch')
    if saved['fingerprint']['config']['model_revision'] != model_revision:
        raise ValueError('Warmup base model differs')
    modules['g'].load_state_dict(saved['modules']['g'], strict=True)
    # Best-model exports contain generator only. Discriminators start fresh in
    # that case; a latest.pt input carries matching discriminator weights too.
    for key in ('d','dur'):
        if key in saved['modules']:
            modules[key].load_state_dict(saved['modules'][key], strict=True)
    return {'source_sha256':sha256(path), 'source_update':saved['updates'],
            'discriminators_reused':all(k in saved['modules'] for k in ('d','dur'))}
