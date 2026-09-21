"""Current-text-only Style Predictor; never accepts character or audio inputs."""
import torch
from torch import nn


class StylePredictor(nn.Module):
    def __init__(self, encoder, dropout=.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(hidden, 7)
        self.continuous_head = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            inputs['token_type_ids'] = token_type_ids
        hidden = self.encoder(**inputs).last_hidden_state[:, 0]
        hidden = self.dropout(hidden)
        return {'emotion_logits': self.emotion_head(hidden),
                'continuous': self.continuous_head(hidden)}

    @classmethod
    def from_pretrained(cls, model_name, revision, dropout=.1):
        from transformers import AutoModel
        return cls(AutoModel.from_pretrained(model_name, revision=revision, add_pooling_layer=False,
                    trust_remote_code=False, use_safetensors=True), dropout)
