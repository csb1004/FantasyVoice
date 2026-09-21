import pytest
import torch

from fantasyvoice.models.conditioning import StyleConditioner, bound_style


def test_bounds_extremes_and_observed_negative_gradient():
    raw = torch.tensor([[12., -3., -.0453], [12., 2., 0.],
                        [12., 3., 1.], [12., 4., 1e20], [12., 5., -1e20]], requires_grad=True)
    out = bound_style(raw)
    assert torch.isfinite(out).all()
    assert ((out[:, 2] >= 0) & (out[:, 2] <= 1)).all()
    assert torch.equal(out[:, 1], raw[:, 1])
    assert (out[:, 0] > 0).all()
    out.sum().backward()
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[0, 2] > 0
    assert raw.grad[1, 2].item() == pytest.approx(.5)
    assert raw.grad[2, 2].item() == pytest.approx(.5)
    assert out[1, 2].item() == pytest.approx(.02 * torch.log(torch.tensor(2.)).item())


def test_conditioning_gradients_and_character_contract():
    norm = {k: {'mean': m, 'scale': s} for k, m, s in
            [('speed', 12., 3.), ('pitch', 0., 3.), ('pause', .08, .12)]}
    adapter = StyleConditioner(norm, 2, torch.ones(8))
    logits = torch.randn(2, 7, requires_grad=True)
    z = torch.zeros(2, 3, requires_grad=True)
    g, values = adapter(torch.tensor([0, 1]), logits.softmax(-1), z)
    assert g.shape == (2, 8, 1)
    assert values['raw'][0, 2].item() == pytest.approx(.08)
    g.square().sum().backward()
    assert logits.grad.abs().sum() > 0
    assert (z.grad.abs().sum(0) > 0).all()
    with pytest.raises(ValueError, match='character'):
        adapter(torch.tensor([0, 2]), logits.softmax(-1), z)
    with pytest.raises(ValueError, match='probabilit'):
        adapter(torch.tensor([0, 1]), logits, z)


def test_extreme_negative_speed_remains_representably_positive():
    raw = torch.tensor([[-1e20, 0., .5], [-100., 0., .5]], requires_grad=True)
    bounded = bound_style(raw)
    assert (bounded[:, 0] > 0).all()
    assert torch.isfinite(bounded).all()
