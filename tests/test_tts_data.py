import numpy as np
import pytest
import soundfile as sf
import torch

from fantasyvoice.dataset.storage import sha256
from fantasyvoice.dataset.tts_data import cache_audio, collate_tts, prepare_cache


def test_audio_cache_preserves_amplitude_silence_and_checks_hash(tmp_path):
    root = tmp_path / 'Voice'
    root.mkdir()
    wav = root / 'a.wav'
    signal = np.zeros((16000, 2), dtype=np.float32)
    signal[4000:12000] = .2
    sf.write(wav, signal, 16000, subtype='FLOAT')
    row = {'audio_path': 'a.wav', 'sha256': sha256(wav)}
    path = cache_audio(row, root, tmp_path / 'cache')
    audio = torch.load(path, weights_only=True)['waveform']
    assert audio.shape == (44100,)
    assert audio[12000:30000].mean().item() == pytest.approx(.2, abs=1e-4)
    assert audio[:5000].abs().max() == 0
    bad = dict(row, sha256='0' * 64)
    with pytest.raises(ValueError, match='SHA'):
        cache_audio(bad, root, tmp_path / 'other')
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    silent = dict(row, sha256=sha256(wav))
    with pytest.raises(ValueError, match='silent'):
        cache_audio(silent, root, tmp_path / 'other')


def test_collator_lengths_and_targets():
    def row(n, frames):
        return dict(x=torch.ones(n, dtype=torch.long), tone=torch.ones(n, dtype=torch.long),
            language=torch.ones(n, dtype=torch.long), ja_bert=torch.ones(768, n),
            waveform=torch.ones(frames * 512 + 100), spec=torch.ones(1025, frames),
            character_id=0, targets=torch.tensor([1., 0, 0, 0, 0, 0, 0, 1., 2., 3.]),
            inputs={'input_ids':[1,2], 'attention_mask':[1,1]}, audio_path='a.wav', text='test')
    batch = collate_tts([row(7, 50), row(5, 40)], 'cpu')
    assert batch['x_lengths'].tolist() == [7, 5]
    assert batch['spec_lengths'].tolist() == [50, 40]
    assert batch['ja_bert'][1, :, 5:].sum() == 0
    assert batch['bert'].shape == (2, 1024, 7)
    assert batch['targets'].shape == (2, 10)


def test_text_cache_invalidates_corrected_transcript(tmp_path):
    from fantasyvoice.style.features import EMOTIONS
    sf.write(tmp_path / 'voice.wav', np.ones(44100, dtype=np.float32) * .1, 44100)
    row = {'audio_path':'voice.wav', 'sha256':sha256(tmp_path / 'voice.wav'),
        'character_id':'A', 'text':'first', 'training_style':{
            'valid_targets':{k:True for k in ('emotion','speed','pitch','pause')},
            'emotion_probabilities':dict(zip(EMOTIONS,[1.,0,0,0,0,0,0])),
            'speed_z':0., 'pitch_z':0., 'pause_z':0.}}
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {'input_ids':[[1,2] for _ in texts], 'attention_mask':[[1,1] for _ in texts]}
    def frontend(text):
        n = 4 if text == 'first' else 7
        return {'x':torch.ones(n,dtype=torch.long), 'tone':torch.zeros(n,dtype=torch.long),
                'language':torch.ones(n,dtype=torch.long), 'ja_bert':torch.ones(768,n)}
    args = (tmp_path, tmp_path / 'cache', frontend, Tokenizer(), {'character_map':{'A':0}}, {'revision':'a'})
    before = prepare_cache({'train':[row]}, *args)['train'][0]
    after = prepare_cache({'train':[{**row,'text':'corrected'}]}, *args)['train'][0]
    assert before['x'].numel() == 4 and after['x'].numel() == 7
    assert torch.equal(before['waveform'], after['waveform'])
