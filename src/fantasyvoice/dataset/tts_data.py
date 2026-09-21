"""Content-addressed local derivatives. Never trim, normalize, or edit source audio."""
import math
import hashlib
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .storage import resolve_audio
from fantasyvoice.style.runner import digest
from fantasyvoice.training.predictor import atomic_checkpoint, tokenize_rows

SAMPLE_RATE = 44100


def cache_audio(row, root, cache):
    identity = {'sha256': row['sha256'], 'sample_rate': SAMPLE_RATE,
                'mono': 'channel_mean', 'trim': False, 'normalize': False, 'version': 1}
    target = Path(cache) / 'audio' / (digest(identity) + '.pt')
    if target.is_file():
        return target  # Immutable derivative of this hash, not a claim about a changed Drive file.
    source = resolve_audio(Path(root), row['audio_path'])
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row['sha256']:
        raise ValueError(f"Audio SHA mismatch: {row['audio_path']}")
    # Hash and decode the very same bytes, with only one read over Drive.
    samples, rate = sf.read(io.BytesIO(payload), dtype='float32', always_2d=True)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"Empty/nonfinite audio: {row['audio_path']}")
    samples = samples.mean(axis=1)
    if not np.any(samples):
        raise ValueError(f"Exact silent audio: {row['audio_path']}")
    if rate != SAMPLE_RATE:
        factor = math.gcd(rate, SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // factor, rate // factor).astype(np.float32)
    if not np.isfinite(samples).all():
        raise ValueError('Nonfinite resampled audio')
    atomic_checkpoint(target, {'identity': identity, 'waveform': torch.from_numpy(samples.copy())})
    return target


class KoreanFrontend:
    """Official Korean phonemes and third-last BERT layer, with pinned artifacts."""
    def __init__(self, config, symbols, device='cpu'):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from melo.text import korean
        from fantasyvoice.style.backends import prepare_korean
        self.config, self.device = config, device
        self.tokenizer = AutoTokenizer.from_pretrained('kykim/bert-kor-base',
            revision=config['frontend_revision'], trust_remote_code=False)
        korean.tokenizer = self.tokenizer
        prepare_korean(korean)
        self.korean = korean
        self.model = AutoModelForMaskedLM.from_pretrained('kykim/bert-kor-base',
            revision=config['frontend_revision'], trust_remote_code=False).to(device).eval()
        self.model.requires_grad_(False)
        self.symbol_to_id = {value: i for i, value in enumerate(symbols)}

    def __call__(self, text):
        from melo.text import cleaned_text_to_sequence
        from melo.commons import intersperse
        normalized = self.korean.text_normalize(text)
        if '[UNK]' in self.tokenizer.tokenize(normalized):
            raise ValueError('Unsupported Korean frontend token; no silent replacement')
        phones, tones, word2ph = self.korean.g2p(normalized)
        phones, tones, language = cleaned_text_to_sequence(phones, tones, 'KR', self.symbol_to_id)
        if self.config['add_blank']:
            phones, tones, language = [intersperse(values, 0) for values in (phones, tones, language)]
            word2ph = [n * 2 for n in word2ph]
            word2ph[0] += 1
        inputs = self.tokenizer(normalized, return_tensors='pt', truncation=False)
        if inputs['input_ids'].shape[1] != len(word2ph):
            raise ValueError('Korean token/phoneme alignment mismatch')
        if len(word2ph) > self.model.config.max_position_embeddings:
            raise ValueError('Korean text exceeds encoder length; refusing truncation')
        with torch.inference_mode():
            hidden = self.model(**{k: v.to(self.device) for k, v in inputs.items()},
                                output_hidden_states=True).hidden_states[-3][0].cpu()
            features = torch.repeat_interleave(hidden, torch.tensor(word2ph), dim=0).T.contiguous()
        if features.shape != (768, len(phones)) or not torch.isfinite(features).all():
            raise ValueError('Invalid Korean BERT features')
        return {'x': torch.tensor(phones), 'tone': torch.tensor(tones),
                'language': torch.tensor(language), 'ja_bert': features, 'normalized_text': normalized}


def prepare_cache(splits, root, cache, frontend, predictor_tokenizer, metadata, identity, progress=None):
    """Build all derivatives once on local disk; checkpoints remain on Drive."""
    cache = Path(cache)
    result = {}
    total = sum(len(rows) for rows in splits.values())
    done = 0
    for split, rows in splits.items():
        tokenized = tokenize_rows(rows, predictor_tokenizer, 256)
        entries = []
        for row, tokens in zip(rows, tokenized):
            if not tokens['masks'].all():
                raise ValueError('TTS v1 requires the approved complete-label dataset')
            audio = cache_audio(row, root, cache)
            feature_key = digest({'text': row['text'], 'frontend': identity})
            feature_path = cache / 'text' / (feature_key + '.pt')
            if not feature_path.exists():
                atomic_checkpoint(feature_path, frontend(row['text']))
            entries.append({'audio': str(audio), 'features': str(feature_path),
                'character_id': metadata['character_map'][row['character_id']], **tokens})
            done += 1
            if progress and (done % 20 == 0 or done == total):
                progress(done, total)
        result[split] = TTSData(entries)
    return result


class TTSData:
    def __init__(self, entries):
        self.entries = entries
        from fantasyvoice.training.tts_audio import SpectralFeatures
        self.spectral = SpectralFeatures()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row = self.entries[index]
        wave = torch.load(row['audio'], map_location='cpu', weights_only=True)['waveform']
        features = torch.load(row['features'], map_location='cpu', weights_only=True)
        spec = self.spectral.spectrogram(wave.unsqueeze(0)).squeeze(0)
        if spec.shape[1] < 32 or spec.shape[1] < len(features['x']):
            raise ValueError(f"Audio too short for segment/alignment: {row['audio_path']}")
        return {**row, **features, 'waveform': wave, 'spec': spec}


def collate_tts(rows, device):
    if not rows:
        raise ValueError('Empty TTS batch')
    from torch.nn.utils.rnn import pad_sequence
    result = {}
    for key in ('x', 'tone', 'language', 'waveform'):
        result[key] = pad_sequence([r[key] for r in rows], batch_first=True).to(device)
    for key in ('ja_bert', 'spec'):
        result[key] = pad_sequence([r[key].T for r in rows], batch_first=True).transpose(1, 2).to(device)
    result['waveform'] = result['waveform'].unsqueeze(1)
    result['x_lengths'] = torch.tensor([len(r['x']) for r in rows], device=device)
    result['spec_lengths'] = torch.tensor([r['spec'].shape[-1] for r in rows], device=device)
    result['character_ids'] = torch.tensor([r['character_id'] for r in rows], device=device)
    # Official Melo Korean uses ja_bert (768 channels), with unused zh_bert zero.
    result['bert'] = torch.zeros(len(rows), 1024, result['x'].shape[1], device=device)
    result['targets'] = torch.stack([r['targets'] for r in rows]).to(device)
    result['rows'] = rows
    return result
