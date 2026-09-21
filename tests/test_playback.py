from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from fantasyvoice.dataset.review import playback_audio


def test_silent_audio_reports_silence_without_normalization(tmp_path):
    path = tmp_path / 'silent.wav'
    sf.write(path, np.zeros(2772), 44100)
    result = playback_audio(path)
    assert result['status'] == 'silent'
    assert result['duration_seconds'] == pytest.approx(2772 / 44100)
    assert result['wav_bytes'] is None


def test_playback_preserves_quiet_audio_volume(tmp_path):
    path = tmp_path / 'quiet.wav'
    sf.write(path, np.full((100, 2), .125), 16000, subtype='FLOAT')
    original = path.read_bytes()
    result = playback_audio(path)
    x, rate = sf.read(BytesIO(result['wav_bytes']), always_2d=True)
    assert rate == 16000 and x.shape == (100, 2)
    assert np.max(x) == pytest.approx(.125)
    assert result['playback_gain'] == 1
    assert path.read_bytes() == original


def test_empty_and_nonfinite_audio_are_not_normalized(tmp_path):
    path = tmp_path / 'empty.wav'
    sf.write(path, np.zeros(0), 16000)
    assert playback_audio(path)['status'] == 'empty'
    sf.write(path, np.array([np.nan, .5]), 16000, subtype='FLOAT')
    with pytest.raises(ValueError, match='non-finite'):
        playback_audio(path)
