"""Melo's spectral definition using complex STFT, without per-file Fourier matrices."""
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=1)
def mel_basis():
    from librosa.filters import mel
    return torch.from_numpy(mel(sr=44100, n_fft=2048, n_mels=128, fmin=0., fmax=None))


class SpectralFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('window', torch.hann_window(2048), persistent=False)
        self.register_buffer('basis', mel_basis().clone(), persistent=False)

    def spectrogram(self, waveform):
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            wave = F.pad(waveform.float().unsqueeze(1), (768, 768), mode='reflect').squeeze(1)
            spec = torch.stft(wave, 2048, hop_length=512, win_length=2048,
                window=self.window, center=False, normalized=False, onesided=True, return_complex=True)
            return (spec.real.square() + spec.imag.square() + 1e-6).sqrt()

    def mel(self, spec):
        with torch.autocast(device_type=spec.device.type, enabled=False):
            return (self.basis @ spec.float()).clamp_min(1e-5).log()
