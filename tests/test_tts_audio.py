import torch
from fantasyvoice.training.tts_audio import SpectralFeatures


def test_spectrogram_matches_melo_convolution_and_mel_gradient():
    torch.set_num_threads(1)
    wave = (torch.randn(1, 16384) * .05).requires_grad_()
    features = SpectralFeatures()
    actual = features.spectrogram(wave)
    # Independent direct convolution implements the pinned Melo Fourier basis.
    basis = torch.view_as_real(torch.fft.fft(torch.eye(2048)))[:1025]
    basis = basis.permute(2, 0, 1).reshape(-1, 1, 2048) * torch.hann_window(2048)
    padded = torch.nn.functional.pad(wave[:, None], (768, 768), mode='reflect')
    conv = torch.nn.functional.conv1d(padded, basis, stride=512)
    expected = (conv[:, :1025].square() + conv[:, 1025:].square() + 1e-6).sqrt()
    assert torch.allclose(actual, expected, atol=1e-4)
    features.mel(actual).mean().backward()
    assert torch.isfinite(wave.grad).all() and wave.grad.abs().sum() > 0
