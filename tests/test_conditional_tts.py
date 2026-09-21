import pytest
import torch

from fantasyvoice.models.conditional_tts import ConditionalTTS


def tiny_tts():
    torch.set_num_threads(1)
    from melo.models import SynthesizerTrn
    base = SynthesizerTrn(20, 9, 4, 8, 8, 16, 2, 3, 3, 0., '1', [3],
                          [[1, 3, 5]], [2, 2], 16, [4, 4], n_speakers=2,
                          gin_channels=8, n_layers_trans_flow=3, n_flow_layer=2,
                          num_languages=2, num_tones=2, use_noise_scaled_mas=False)
    norm = {k: {'mean': m, 'scale': s} for k, m, s in
            [('speed', 12., 3.), ('pitch', 0., 3.), ('pause', .08, .12)]}
    return ConditionalTTS(base, norm, 2)


def tiny_batch():
    return dict(x=torch.tensor([[1, 2, 3, 4]]), x_lengths=torch.tensor([4]),
                spec=torch.rand(1, 9, 16), spec_lengths=torch.tensor([16]),
                character_ids=torch.tensor([0]), tone=torch.zeros(1, 4, dtype=torch.long),
                language=torch.zeros(1, 4, dtype=torch.long), bert=torch.zeros(1, 1024, 4),
                ja_bert=torch.randn(1, 768, 4))


def test_actual_melo_tts_only_backward_and_reference_free_generation():
    model = tiny_tts()
    batch = tiny_batch()
    logits = torch.randn(1, 7, requires_grad=True)
    z = torch.zeros(1, 3, requires_grad=True)
    result, _ = model(batch, logits.softmax(-1), z)
    loss = result[0].square().mean() + result[1].sum()
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert torch.isfinite(z.grad).all() and (z.grad.abs().sum(0) > 0).all()
    # Duration supervision alone must also reach style, not only the decoder.
    model.zero_grad(set_to_none=True)
    z2 = torch.zeros(1, 3, requires_grad=True)
    result, _ = model(batch, logits.detach().softmax(-1), z2)
    result[1].sum().backward()
    assert (z2.grad.abs().sum(0) > 0).all()
    text_only = {k: v for k, v in batch.items() if k not in ('spec', 'spec_lengths')}
    model.eval()
    with torch.no_grad():
        wav, _ = model.infer(text_only, logits.detach().softmax(-1), z.detach())
    assert wav.shape[0] == 1 and wav.shape[-1] > 0 and torch.isfinite(wav).all()


def test_strict_pretrained_loading():
    model = tiny_tts()
    state = model.base.state_dict()
    model.load_base(state)
    broken = dict(state)
    broken.pop('emb_g.weight')
    with pytest.raises(RuntimeError):
        model.load_base(broken)
