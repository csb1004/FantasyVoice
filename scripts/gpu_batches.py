"""Conservative GPU presets and padded emotion batches; single inference unchanged."""
import math
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def batch_preset(name, memory_gib, stage):
    if stage not in (11,12) or not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('Invalid GPU/stage')
    if 'L4' in name.upper().split() and memory_gib >= 20:
        return (8,1)
    if 'T4' in name.upper().split() and memory_gib >= 14:
        return (4,1)
    return (1,1)


def emotion_batch_logits(analyzer, waves, device):
    if not waves:
        raise ValueError('Empty emotion batch')
    if len(waves) == 1:
        return analyzer(waves[0].to(device))
    analyzer.model.eval()
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        sources=[]
        for wave in waves:
            if wave.ndim != 2 or wave.shape[0] != 1 or not torch.isfinite(wave).all():
                raise ValueError('Expected finite unpadded mono utterance')
            source=analyzer.resample(wave.to(device).float())[0]
            if source.numel() < 400:
                raise ValueError('Utterance too short for emotion2vec')
            if analyzer.model.cfg.normalize:
                source=F.layer_norm(source,source.shape)
            sources.append(source)
        source=pad_sequence(sources,batch_first=True)
        lengths=torch.tensor([x.numel() for x in sources],device=device)
        padding=torch.arange(source.shape[1],device=device)[None] >= lengths[:,None]
        features=analyzer.model.extract_features(source,padding_mask=padding,mask=False)
        x=features['x']
        mask=features.get('padding_mask')
        if mask is None:
            if bool(padding.any()):
                raise ValueError('Backbone did not return a padding mask')
            pooled=x.mean(1)
        else:
            if mask.shape != x.shape[:2]:
                raise ValueError('Invalid feature padding mask')
            valid=~mask
            count=valid.sum(1,keepdim=True)
            if (count == 0).any():
                raise ValueError('No valid emotion frames')
            pooled=x.masked_fill(mask[:,:,None],0).sum(1)/count
        return analyzer.model.proj(pooled).index_select(-1,analyzer.indices)


def generate_prior_batch(tts, batch, probabilities, z_style, max_frames):
    """Pinned Melo text prior, padded across samples; return each unpadded waveform.

    Same MIT Melo inference formulation as emotion_feedback.generate_prior.
    Length guard uses EACH utterance, not the sum of durations across a batch.
    """
    from melo import commons
    base=tts.base
    g,_=tts.conditioner(batch['character_ids'],probabilities,z_style)
    with torch.autocast(device_type=g.device.type,enabled=False):
        hidden,mean,logs,mask=base.enc_p(batch['x'],batch['x_lengths'],batch['tone'],
            batch['language'],batch['bert'].float(),batch['ja_bert'].float(),
            g=None if base.use_vc else g.float())
        durations=torch.ceil(base.dp(hidden,mask,g=g.float()).exp()*mask)
        totals=durations.sum((1,2))
        if not torch.isfinite(totals).all() or (totals<1).any() or (totals>max_frames).any():
            raise ValueError('Generated length exceeds feedback limit; no truncation or optimizer update')
        lengths=totals.long()
        y_mask=commons.sequence_mask(lengths).unsqueeze(1).to(mask.dtype)
        attention=commons.generate_path(durations,mask.unsqueeze(2)*y_mask.unsqueeze(-1))
        mean=torch.matmul(attention.squeeze(1),mean.transpose(1,2)).transpose(1,2)
        logs=torch.matmul(attention.squeeze(1),logs.transpose(1,2)).transpose(1,2)
        prior=mean+torch.randn_like(mean)*logs.exp()*.667
        latent=base.flow(prior,y_mask,g=g.float(),reverse=True)
        wave=base.dec(latent*y_mask,g=g.float())[:,0]
    if not torch.isfinite(wave).all():
        raise ValueError('Nonfinite generated utterance')
    hop=math.prod(base.dec.upsample_rates) if hasattr(base.dec,'upsample_rates') else None
    # Melo's decoder does not expose upsample_rates in every pinned build.
    if hop is None:
        hop=math.prod(layer.stride[0] for layer in base.dec.ups)
    return [wave[i:i+1,:int(n)*hop] for i,n in enumerate(lengths.tolist())]


from emotion_feedback import FeedbackObjective, emotion_kl, scoring
from fantasyvoice.training.tts import TTSObjective
from fantasyvoice.dataset.tts_data import collate_tts


class BatchedFeedbackObjective(FeedbackObjective):
    def __call__(self, indices):
        losses=TTSObjective.__call__(self,indices)
        batch=collate_tts([self.data[i] for i in indices],self.device)
        tts=self.modules['g']
        modes={m:m.training for m in tts.modules()}
        try:
            tts.eval()
            waves=generate_prior_batch(tts,batch,*self.style(batch)[:2],
                max_frames=int(self.config['feedback']['max_seconds']*44100/512))
        finally:
            for m,mode in modes.items():m.training=mode
        with scoring(self.emotion):
            logits=emotion_batch_logits(self.emotion,waves,self.device)
            feedback=emotion_kl(logits,batch['targets'][:,:7])
        losses['g']=losses['g']+self.config['feedback']['weight']*feedback
        return losses


def backward_with_batch_retry(objective, indices, batch_size, scaler, optimizers, before):
    """Replay a whole update on forward/backward OOM, before any optimizer step."""
    import gc
    from fantasyvoice.training.predictor import restore_rng
    def run(size):
        totals={key:0. for key in optimizers}
        for start in range(0,len(indices),size):
            chunk=indices[start:start+size]
            losses=objective(chunk)
            if set(losses)!=set(optimizers):
                raise ValueError('Loss/optimizer names differ')
            weight=len(chunk)/len(indices)
            for key,loss in losses.items():
                if loss.numel()!=1 or not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite {key} loss; no optimizer committed')
                scaler.scale(loss*weight).backward()
                totals[key]+=float(loss.detach())*weight
        return totals
    while True:
        for optimizer in optimizers.values():optimizer.zero_grad(set_to_none=True)
        restore_rng(before)
        oom=False
        try:
            totals=run(batch_size)
        except torch.cuda.OutOfMemoryError:
            oom=True
        if not oom:
            return totals,batch_size
        for optimizer in optimizers.values():optimizer.zero_grad(set_to_none=True)
        gc.collect();torch.cuda.empty_cache()
        if batch_size==1:
            raise RuntimeError('One utterance exceeds GPU memory; no optimizer committed')
        batch_size=max(1,batch_size//2)
        print(f'CUDA OOM: retrying full update with actual batch {batch_size}',flush=True)
