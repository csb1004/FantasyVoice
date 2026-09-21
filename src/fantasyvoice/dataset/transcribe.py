"""Resumable ASR candidates. No automatic training acceptance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .storage import check_output, read_jsonl, resolve_audio, sha256, write_json, write_jsonl


def run_pilot(root, rows, output, config, transcribe, progress=None):
    output = Path(output)
    check_output(root, output)
    if config.get('model') != 'large-v3' or config.get('language') != 'ko' or config.get('task') != 'transcribe':
        raise ValueError('Pilot requires large-v3 Korean transcription')
    if not config.get('revision'):
        raise ValueError('Model revision is required')
    config = json.loads(json.dumps(config, allow_nan=False))
    run_file = output / 'run.json'
    if run_file.exists():
        if json.loads(run_file.read_text(encoding='utf-8')) != config:
            raise ValueError('Run config changed; use a new output directory')
    else:
        if (output / 'transcripts.jsonl').exists():
            raise ValueError('Missing run config for existing results')
        write_json(run_file, config)
    history = read_jsonl(output / 'transcripts.jsonl')
    latest = {r['key']: r for r in history}
    done = {key for key, row in latest.items() if row['status'] == 'success'}
    for index, source in enumerate(rows):
        identity = {'config': config, 'audio_path': source['audio_path'], 'sha256': source['sha256']}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        # Verify bytes even when resuming; stale inventories must not silently reuse results.
        result = {'schema_version': 1, 'key': key, 'audio_path': source['audio_path'],
                  'character_id': source['character_id'], 'sha256': source['sha256'],
                  'review_status': 'pending', 'created_at': datetime.now(timezone.utc).isoformat()}
        fatal = None
        try:
            if source['status'] != 'ok':
                raise ValueError('Inventory entry is not readable')
            path = resolve_audio(root, source['audio_path'])
            if sha256(path) != source['sha256']:
                raise ValueError('Source audio changed; refresh inventory and use a new pilot selection')
            if key in done:
                if progress:
                    progress(index + 1, len(rows), 'cached')
                continue
            asr = transcribe(path)
            if not isinstance(asr.get('text'), str) or not isinstance(asr.get('segments'), list):
                raise ValueError('Invalid ASR result')
            json.dumps(asr, allow_nan=False)
            result.update(status='success', asr=asr,
                          review_reasons=['empty_transcript'] if not asr['text'].strip() else ['pilot_listening_required'])
        except Exception as exc:
            result.update(status='error', error=f'{type(exc).__name__}: {exc}', review_reasons=['asr_error'])
            if isinstance(exc, MemoryError) or 'out of memory' in str(exc).lower():
                fatal = exc
        history.append(result)
        write_jsonl(output / 'transcripts.jsonl', history)
        if result['status'] == 'success':
            done.add(key)
        else:
            done.discard(key)
        if progress:
            progress(index + 1, len(rows), result['status'])
        if fatal:
            raise fatal
    return history


class WhisperBackend:
    """Load the approved model explicitly; never fall back to a smaller model."""

    def __init__(self, options, download_root):
        import importlib.metadata
        import shutil
        import torch
        import whisper

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required for this Colab pilot; select a GPU runtime')
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('ffmpeg is required for Whisper audio decoding')
        self.options = dict(options)
        if self.options.get('language') != 'ko' or self.options.get('task') != 'transcribe':
            raise ValueError('Expected Korean transcription options')
        self.model = whisper.load_model('large-v3', device='cuda', download_root=str(download_root))
        self.config = {'model': 'large-v3', 'revision': sha256(Path(download_root) / 'large-v3.pt'),
                       'language': 'ko', 'task': 'transcribe', 'options': self.options,
                       'whisper_version': importlib.metadata.version('openai-whisper'),
                       'pipeline_version': importlib.metadata.version('fantasyvoice'),
                       'torch_version': str(torch.__version__)}

    def __call__(self, path):
        return self.model.transcribe(str(path), **self.options)
