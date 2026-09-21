"""Explicit condition routing around pinned Melo; no global monkey patches."""
import torch
from torch import nn

from .conditioning import StyleConditioner
from .melo_connected import ConnectedDP, ConnectedSDP, forward_with_condition


class ConditionalTTS(nn.Module):
    def __init__(self, base, normalization, character_count):
        super().__init__()
        self.base = base
        self.conditioner = StyleConditioner(normalization, character_count,
                                           base.emb_g.weight[0].detach())
        for name, cls, channels in [('sdp', ConnectedSDP, 192), ('dp', ConnectedDP, 256)]:
            original = getattr(base, name)
            connected = cls(base.hidden_channels, channels, 3, .5, gin_channels=base.gin_channels)
            connected.load_state_dict(original.state_dict(), strict=True)
            setattr(base, name, connected)
        # Retain the original speaker table for strict base checkpoint compatibility.
        # It is unused: character embeddings are owned by the conditioner.
        base.emb_g.requires_grad_(False)

    def load_base(self, state):
        self.base.load_state_dict(state, strict=True)
        with torch.no_grad():
            self.conditioner.character.weight.copy_(self.base.emb_g.weight[0].expand_as(
                self.conditioner.character.weight))

    def forward(self, batch, probabilities, continuous_z):
        g, values = self.conditioner(batch['character_ids'], probabilities, continuous_z)
        # Flow distributions, duration and MAS are fragile in FP16. The trainer
        # uses autocast for discriminators/Predictor; preserve these computations.
        with torch.autocast(device_type=g.device.type, enabled=False):
            result = forward_with_condition(
                self.base, batch['x'], batch['x_lengths'], batch['spec'].float(),
                batch['spec_lengths'], batch['tone'], batch['language'],
                batch['bert'].float(), batch['ja_bert'].float(), g.float())
        return result, values

    def infer(self, batch, probabilities, continuous_z):
        g, values = self.conditioner(batch['character_ids'], probabilities, continuous_z)
        with torch.autocast(device_type=g.device.type, enabled=False):
            result = self.base.infer(batch['x'], batch['x_lengths'], None,
                batch['tone'], batch['language'], batch['bert'].float(), batch['ja_bert'].float(),
                g=g.float())
        return result[0], values
