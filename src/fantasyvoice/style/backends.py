"""Frozen analyzers. Heavy optional dependencies load only at initialization."""
import importlib.metadata
import math
from pathlib import Path
import shutil
import tempfile
import unicodedata

from fantasyvoice.dataset.storage import resolve_audio, sha256
from .features import emotion_probabilities, summarize


def freeze_analyzer(model):
    """ScriptModules support eval and parameter freezing, but not module.requires_grad_."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def prepare_korean(korean):
    """Check MeCab explicitly; g2pkk otherwise swallows initialization failures."""
    import platform
    try:
        if platform.system() == 'Windows':
            from eunjeon import Mecab
            analyzer = Mecab()
        else:
            from mecab import MeCab
            analyzer = MeCab()
        if not analyzer.pos('음성 확인'):
            raise ValueError('Empty MeCab output')
    except Exception as exc:
        raise RuntimeError(
            'Korean MeCab initialization failed. Install the Python-compatible MeCab '
            'binding and Korean dictionary from the updated [style] dependencies.'
        ) from exc
    # Notebook retries retain Melo's module even after fantasyvoice is reloaded.
    if getattr(getattr(korean, 'g2p_kr', None), 'mecab', None) is None:
        korean.g2p_kr = None
    korean.g2p(korean.text_normalize('음성 확인'))


def count_phones(phones, tokens):
    if '[UNK]' in tokens:
        return None
    count = 0
    for phone in phones:
        if len(phone) == 1 and '\u1100' <= phone <= '\u11ff':
            count += 1
        elif phone == '_' or phone.isspace() or all(unicodedata.category(c).startswith('P') for c in phone):
            continue
        else:
            return None
    return count or None


class FrozenAnalyzers:
    def __init__(self, root, config):
        import torch
        import torchcrepe
        from funasr import AutoModel
        from huggingface_hub import snapshot_download
        from silero_vad import load_silero_vad
        from transformers import AutoTokenizer
        from melo.text import korean

        self.root, self.config = Path(root), config
        if config['sample_rate'] != 16000:
            raise ValueError('The analyzers require 16000 Hz')
        self.device = config['device']
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Colab GPU runtime is required by this configuration')
        korean.tokenizer = AutoTokenizer.from_pretrained(
            korean.model_id, revision=config['tokenizer_revision'])
        self.korean = korean
        # Initialize the G2P dependencies before a long dataset run.
        prepare_korean(korean)
        model_dir = snapshot_download(config['emotion']['repo'],
                                      revision=config['emotion']['revision'],
                                      allow_patterns=['*.json', '*.yaml', '*.txt', '*.pt'])
        self.emotion = AutoModel(model=model_dir, hub='hf', device=self.device,
                                 disable_update=True, disable_pbar=True, trust_remote_code=False)
        freeze_analyzer(self.emotion.model)
        self.vad = load_silero_vad(onnx=False)
        freeze_analyzer(self.vad)
        torchcrepe.load.model(self.device, config['pitch']['model'])
        freeze_analyzer(torchcrepe.infer.model)
        self.provenance = {
            'packages': {name: importlib.metadata.version(name) for name in (
                'torch', 'torchaudio', 'numpy', 'scipy', 'soundfile', 'funasr',
                'torchcrepe', 'silero-vad', 'transformers', 'huggingface-hub',
                'g2pkk', 'jamo', 'anyascii', 'num2words', 'nltk')},
            'melo_revision': config['melo_revision'],
            'emotion_revision': config['emotion']['revision'],
            'tokenizer_revision': config['tokenizer_revision'],
            'melo_frontend_sha256': {
                name: sha256(Path(korean.__file__).parent / name)
                for name in ('korean.py', 'ko_dictionary.py', 'symbols.py')
            },
        }

    def __call__(self, row):
        import numpy as np
        import soundfile as sf
        import torch
        import torchcrepe
        from scipy.signal import resample_poly
        from silero_vad import get_speech_timestamps

        source = resolve_audio(self.root, row['audio_path'])
        # Read Drive once, then hash and decode locally; never rewrite source audio.
        with tempfile.TemporaryDirectory(prefix='fantasyvoice-style-') as folder:
            local = Path(folder) / 'input.wav'
            shutil.copyfile(source, local)
            if sha256(local) != row['sha256']:
                raise ValueError('Source SHA256 differs from ASR inventory')
            audio, sample_rate = sf.read(local, dtype='float32', always_2d=True)
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError('Empty or non-finite audio')
        audio = audio.mean(axis=1)
        if not np.any(audio):
            raise ValueError('Silent audio')
        if len(audio) / sample_rate <= .5:
            raise ValueError('Decoded audio is <= 0.5 seconds')
        divisor = math.gcd(sample_rate, 16000)
        if sample_rate != 16000:
            audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype('float32')
        waveform = torch.from_numpy(np.ascontiguousarray(audio))
        duration = len(audio) / 16000
        normalized = self.korean.text_normalize(row['text'])
        phones, _, _ = self.korean.g2p(normalized)
        phone_count = count_phones(phones, self.korean.tokenizer.tokenize(normalized))
        pitch_config = self.config['pitch']
        with torch.inference_mode():
            stamps = get_speech_timestamps(waveform, self.vad, sampling_rate=16000,
                                           return_seconds=False, **self.config['vad'])
            intervals = [(s['start'] / 16000, s['end'] / 16000) for s in stamps]
            if intervals:
                prediction = self.emotion.generate(input=audio, granularity='utterance',
                                                  extract_embedding=False)[0]
                emotion = emotion_probabilities(prediction['labels'], prediction['scores'])
                pitch, periodicity = torchcrepe.predict(
                    waveform[None], 16000, hop_length=pitch_config['hop_length'],
                    fmin=pitch_config['fmin'], fmax=pitch_config['fmax'],
                    model=pitch_config['model'], batch_size=pitch_config['batch_size'],
                    device=self.device, return_periodicity=True)
                periodicity = torchcrepe.threshold.Silence(pitch_config['silence_db'])(
                    periodicity, waveform[None], 16000, pitch_config['hop_length'])
                times = torch.arange(pitch.shape[-1]) * pitch_config['hop_length'] / 16000
                speech_mask = torch.zeros_like(times, dtype=torch.bool)
                for start, end in intervals:
                    speech_mask |= (times >= start) & (times < end)
                mask = speech_mask[None] & (periodicity >= pitch_config['periodicity_threshold'])
                f0 = pitch[mask].cpu().tolist()
            else:
                emotion = {'raw_probabilities': None, 'probabilities': None, 'top_label': None}
                f0 = []
        result = summarize(intervals, duration, phone_count, f0)
        result.update(emotion=emotion, normalized_text=normalized, phones=phones,
                      decoded_duration_seconds=duration, label_source='frozen_analyzer_pseudo_labels')
        masks = {
            'emotion': emotion['probabilities'] is not None,
            'speed': result['speed_phonemes_per_second'] is not None,
            'pitch_raw': result['pitch_f0_median_hz'] is not None,
            'pitch_normalized': False,
            'pause': result['pause_ratio'] is not None,
        }
        result['valid_targets'] = masks
        result['issues'] = [f'{name}_unavailable' for name, valid in masks.items()
                            if not valid and name != 'pitch_normalized']
        if not intervals:
            result['issues'].append('no_vad_speech')
        if emotion['top_label'] in ('other', 'unknown'):
            result['issues'].append('emotion_other_or_unknown')
        return result
