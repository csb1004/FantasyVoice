import pytest

from fantasyvoice.style.features import summarize, emotion_probabilities
from fantasyvoice.style.runner import run_styles
from fantasyvoice.dataset.storage import read_jsonl


@pytest.mark.parametrize('scripted', [False, True])
def test_freeze_analyzer_supports_torchscript(scripted):
    import torch
    from fantasyvoice.style.backends import freeze_analyzer
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.5))
    if scripted:
        model = torch.jit.script(model)
    model.train()
    freeze_analyzer(model)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    with torch.inference_mode():
        assert torch.isfinite(model(torch.ones(1, 3))).all()


def test_korean_preflight_resets_broken_g2p_cache(monkeypatch):
    import sys
    from types import SimpleNamespace
    from fantasyvoice.style.backends import prepare_korean
    monkeypatch.setattr('platform.system', lambda: 'Linux')
    monkeypatch.setitem(sys.modules, 'mecab', SimpleNamespace(
        MeCab=lambda: SimpleNamespace(pos=lambda text: [('음성', 'NNG')])) )
    korean = SimpleNamespace(g2p_kr=SimpleNamespace(mecab=None), text_normalize=lambda text: text)
    def g2p(text):
        assert korean.g2p_kr is None
        return (['_', 'ᄋ', 'ᅡ', '_'], [], [])
    korean.g2p = g2p
    prepare_korean(korean)


def test_korean_preflight_reports_mecab_failure(monkeypatch):
    import sys
    from types import SimpleNamespace
    from fantasyvoice.style.backends import prepare_korean
    monkeypatch.setattr('platform.system', lambda: 'Linux')
    def fail():
        raise ImportError('missing dictionary')
    monkeypatch.setitem(sys.modules, 'mecab', SimpleNamespace(MeCab=fail))
    with pytest.raises(RuntimeError, match='MeCab'):
        prepare_korean(SimpleNamespace())


def test_features_exclude_edge_silence_and_defer_pitch_normalization():
    result = summarize([(1, 2), (3, 5)], 6, 12, [100, 200, 300])
    assert result['speech_seconds'] == 3
    assert result['pause_ratio'] == .25
    assert result['speed_phonemes_per_second'] == 4
    assert result['pitch_f0_median_hz'] == 200
    assert result['pitch_semitones'] is None


def test_empty_speech_has_no_targets():
    result = summarize([], 2, 5, [])
    for field in ('pause_ratio', 'speed_phonemes_per_second', 'pitch_f0_median_hz'):
        assert result[field] is None


def test_intervals_are_clamped_and_merged():
    result = summarize([(-1, 1), (.5, 2), (3, 9)], 4, 9, [])
    assert result['speech_seconds'] == 3
    assert result['pause_ratio'] == .25


def test_unknown_emotion_is_not_neutral():
    labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'other', 'unknown']
    result = emotion_probabilities(labels, [.025] * 8 + [.8])
    assert result['probabilities'] is None
    assert result['top_label'] == 'unknown'
    result = emotion_probabilities(labels, [.8] + [.025] * 8)
    assert sum(result['probabilities'].values()) == pytest.approx(1)
    with pytest.raises(ValueError):
        emotion_probabilities(labels, [float('nan')] * 9)
    labels[-1] = '<unk>'
    assert emotion_probabilities(labels, [.025] * 8 + [.8])['top_label'] == 'unknown'


def test_phoneme_count_rejects_unknown_instead_of_undercounting():
    from fantasyvoice.style.backends import count_phones
    assert count_phones(['_', 'ᄋ', 'ᅡ', '!', '_'], ['아', '!']) == 2
    assert count_phones(['_', '_', '_'], ['[UNK]']) is None
    assert count_phones(['_', 'x', '_'], ['x']) is None


def test_resume_reuses_labels_but_preserves_current_review(tmp_path):
    rows = [dict(key='asr', audio_path='c/a.wav', character_id='c', sha256='hash',
                 duration_seconds=1, text='안녕', review_status='pending', split=None)]
    calls = []
    def analyze(row):
        calls.append(row['text'])
        return {'value': 1, 'issues': []}
    output = tmp_path / 'style'
    run_styles(rows, output, {'v': 1}, analyze)
    rows[0]['review_status'] = 'accepted'
    run_styles(rows, output, {'v': 1}, analyze)
    assert calls == ['안녕']
    assert read_jsonl(output / 'style-labels.jsonl')[0]['review_status'] == 'accepted'
    rows[0]['text'] = '수정'
    run_styles(rows, output, {'v': 1}, analyze)
    assert calls == ['안녕', '수정']
    with pytest.raises(ValueError, match='config'):
        run_styles(rows, output, {'v': 2}, analyze)


def test_failure_is_retryable_and_short_audio_excluded(tmp_path):
    rows = [dict(audio_path='c/a.wav', sha256='h', text='아', duration_seconds=1),
            dict(audio_path='c/b.wav', sha256='h2', text='아', duration_seconds=.5)]
    def fail(row):
        raise ValueError('bad audio')
    run_styles(rows, tmp_path, {}, fail)
    assert read_jsonl(tmp_path / 'style-labels.jsonl')[0]['style_status'] == 'error'
    run_styles(rows, tmp_path, {}, lambda row: {'issues': []})
    data = read_jsonl(tmp_path / 'style-labels.jsonl')
    assert [r['style_status'] for r in data] == ['generated', 'excluded']


def test_interrupt_saves_previous_result(tmp_path):
    rows = [dict(audio_path=f'c/{i}.wav', sha256=str(i), text='대사', duration_seconds=1)
            for i in range(2)]
    def analyze(row):
        if row['sha256'] == '1':
            raise KeyboardInterrupt()
        return {'issues': []}
    with pytest.raises(KeyboardInterrupt):
        run_styles(rows, tmp_path, {}, analyze)
    data = read_jsonl(tmp_path / 'style-labels.jsonl')
    assert [r['style_status'] for r in data] == ['generated', 'pending']


def test_backend_audio_and_analyzer_contract(tmp_path, monkeypatch):
    """Exercise real decoding/resampling/tensors with substituted model outputs."""
    import sys
    from types import SimpleNamespace
    import numpy as np
    import soundfile as sf
    import torch
    from fantasyvoice.style.backends import FrozenAnalyzers
    from fantasyvoice.dataset.storage import sha256

    source = tmp_path / 'sample.wav'
    wave = .2 * np.sin(2 * np.pi * 200 * np.arange(24000) / 24000)
    sf.write(source, wave, 24000)
    row = dict(audio_path='sample.wav', sha256=sha256(source), text='아')
    analyzer = FrozenAnalyzers.__new__(FrozenAnalyzers)
    analyzer.root = tmp_path
    analyzer.device = 'cpu'
    analyzer.config = {'vad': {}, 'pitch': dict(model='full', hop_length=160,
        fmin=50, fmax=1100, batch_size=512, periodicity_threshold=.21, silence_db=-60)}
    analyzer.korean = SimpleNamespace(text_normalize=lambda text: text,
        g2p=lambda text: (['_', 'ᄋ', 'ᅡ', '_'], [], []),
        tokenizer=SimpleNamespace(tokenize=lambda text: ['아']))
    def emotion_generate(**kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs['input'].shape == (16000,)
        return [dict(labels=['angry', 'disgusted', 'fearful', 'happy', 'neutral',
                            'sad', 'surprised', 'other', '<unk>'], scores=[.025] * 4 + [.8] + [.025] * 4)]
    analyzer.emotion = SimpleNamespace(generate=emotion_generate)
    analyzer.vad = object()
    def predict(audio, sample_rate, **kwargs):
        assert sample_rate == 16000 and audio.shape == (1, 16000)
        pitch = torch.full((1, 101), 200.)
        # Values outside speech must not affect the utterance median.
        pitch[:, :20] = 1000
        pitch[:, 80:] = 1000
        return pitch, torch.ones_like(pitch)
    monkeypatch.setitem(sys.modules, 'torchcrepe', SimpleNamespace(predict=predict,
        threshold=SimpleNamespace(Silence=lambda threshold: lambda periodicity, *args: periodicity)))
    monkeypatch.setitem(sys.modules, 'silero_vad', SimpleNamespace(
        get_speech_timestamps=lambda *args, **kwargs: [dict(start=3200, end=12800)]))
    result = analyzer(row)
    assert result['pitch_f0_median_hz'] == 200
    assert result['speech_seconds'] == pytest.approx(.6)
    assert result['speed_phonemes_per_second'] == pytest.approx(2 / .6)
    assert result['emotion']['top_label'] == 'neutral'
    assert result['issues'] == []
    assert result['valid_targets']['pitch_normalized'] is False
    row['sha256'] = 'changed'
    with pytest.raises(ValueError, match='SHA256'):
        analyzer(row)


def test_oom_stops_and_checkpoints_error(tmp_path):
    calls = []
    rows = [dict(audio_path=f'{i}.wav', sha256=str(i), text='대사', duration_seconds=1)
            for i in range(2)]
    def oom(row):
        calls.append(row)
        raise RuntimeError('CUDA out of memory')
    with pytest.raises(RuntimeError, match='out of memory'):
        run_styles(rows, tmp_path, {}, oom)
    assert len(calls) == 1
    assert [r['style_status'] for r in read_jsonl(tmp_path / 'style-labels.jsonl')] == ['error', 'pending']
