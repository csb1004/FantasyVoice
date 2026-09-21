"""Inference accepts only current text, character ID, and an explicit random seed."""
import torch

from fantasyvoice.training.predictor import rng_state, restore_rng


def synthesize(text, character, frontend, tokenizer, predictor, tts, character_map,
               device='cpu', seed=42):
    if character not in character_map:
        raise ValueError(f'Unregistered character: {character}')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Text must be nonempty')
    features = frontend(text)
    inputs = tokenizer(text, return_tensors='pt', truncation=False)
    if inputs['input_ids'].shape[1] > 256:
        raise ValueError('Text exceeds Predictor length; refusing truncation')
    batch = {k:features[k].unsqueeze(0).to(device) for k in ('x','tone','language','ja_bert')}
    batch['x_lengths'] = torch.tensor([len(features['x'])], device=device)
    batch['character_ids'] = torch.tensor([character_map[character]], device=device)
    batch['bert'] = torch.zeros(1, 1024, len(features['x']), device=device)
    before, modes = rng_state(), (predictor.training, tts.training)
    try:
        predictor.eval(); tts.eval()
        torch.manual_seed(seed)
        with torch.inference_mode():
            prediction = predictor(**{k:v.to(device) for k,v in inputs.items()})
            probabilities = prediction['emotion_logits'].float().softmax(-1)
            wave, values = tts.infer(batch, probabilities, prediction['continuous'])
        if not torch.isfinite(wave).all() or wave.numel() == 0:
            raise RuntimeError('Invalid generated waveform')
        return wave[0,0].cpu(), {'text':text, 'character':character, 'seed':seed,
            'emotion_probabilities':probabilities[0].cpu().tolist(),
            'raw_style':values['raw'][0].cpu().tolist(),
            'bounded_style':values['bounded'][0].cpu().tolist()}
    finally:
        predictor.train(modes[0]); tts.train(modes[1]); restore_rng(before)
