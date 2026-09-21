"""Differentiable physical bounds at the TTS boundary, leaving Predictor v1 intact."""
import torch
from torch import nn
from torch.nn import functional as F


def bound_style(raw):
    raw = raw.float()
    speed, pitch, pause = raw.unbind(-1)
    # Evaluate each half near its closest endpoint to avoid cancellation at
    # large positive values; both expressions have the same value/derivative.
    t = .02
    low, high = pause.clamp(max=.5), pause.clamp(min=.5)
    bounded_pause = torch.where(pause <= .5,
        t * (F.softplus(low / t) - F.softplus((low - 1) / t)),
        1 + t * (F.softplus(-high / t) - F.softplus((1 - high) / t)))
    bounded_speed = (.1 * F.softplus(speed / .1)).clamp_min(torch.finfo(torch.float32).tiny)
    return torch.stack((bounded_speed, pitch, bounded_pause), -1)


class StyleConditioner(nn.Module):
    def __init__(self, normalization, character_count, base_embedding):
        super().__init__()
        if character_count < 1 or base_embedding.ndim != 1:
            raise ValueError('Invalid character embedding')
        keys = ('speed', 'pitch', 'pause')
        mean = torch.tensor([normalization[k]['mean'] for k in keys], dtype=torch.float32)
        scale = torch.tensor([normalization[k]['scale'] for k in keys], dtype=torch.float32)
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid normalization')
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        channels = base_embedding.numel()
        self.character = nn.Embedding(character_count, channels)
        self.emotion = nn.Parameter(torch.empty(7, channels))
        self.continuous = nn.Linear(3, channels)
        with torch.no_grad():
            self.character.weight.copy_(base_embedding.float().expand(character_count, -1))
            nn.init.normal_(self.emotion, std=.01)
            nn.init.normal_(self.continuous.weight, std=.01)
            nn.init.zeros_(self.continuous.bias)

    def forward(self, character_ids, probabilities, continuous_z):
        n = character_ids.numel()
        if character_ids.shape != (n,) or character_ids.dtype != torch.long or n == 0:
            raise ValueError('Invalid character IDs')
        if (character_ids < 0).any() or (character_ids >= self.character.num_embeddings).any():
            raise ValueError('Unregistered character ID')
        if probabilities.shape != (n, 7) or not torch.isfinite(probabilities).all():
            raise ValueError('Invalid emotion probabilities')
        if (probabilities < 0).any() or not torch.allclose(probabilities.sum(-1).float(),
                                                         torch.ones(n, device=probabilities.device), atol=1e-5):
            raise ValueError('Invalid emotion probabilities')
        if continuous_z.shape != (n, 3) or not torch.isfinite(continuous_z).all():
            raise ValueError('Invalid continuous style')
        with torch.autocast(device_type=continuous_z.device.type, enabled=False):
            raw = continuous_z.float() * self.scale + self.mean
            bounded = bound_style(raw)
            bounded_z = (bounded - self.mean) / self.scale
            g = (self.character(character_ids) + probabilities.float() @ self.emotion
                 + self.continuous(bounded_z))
        return g.unsqueeze(-1), {'raw': raw, 'bounded': bounded, 'bounded_z': bounded_z}
