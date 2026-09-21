"""Post-joint validation and audio export, separate from training source identities."""
from datetime import datetime, timezone
import math
from pathlib import Path
import uuid
import zipfile

import numpy as np
import soundfile as sf

from fantasyvoice.dataset.review import playback_audio
from fantasyvoice.dataset.storage import sha256, write_json


def validate_completed_joint(checkpoint, report, run, predictor_sha):
    fingerprint = checkpoint['fingerprint']
    config, identity = fingerprint['config'], fingerprint['identity']
    if config.get('stage') != 'joint' or set(checkpoint['modules']) != {'g','predictor'}:
        raise ValueError('10 requires the joint best-model.pt containing TTS and Predictor')
    if report.get('complete') is not True or report.get('pending_eval') is not None:
        raise ValueError('09 공동 학습과 마지막 검증을 먼저 완료하세요.')
    if run['config'] != config or run['identity'] != identity:
        raise ValueError('09 run/checkpoint identity mismatch')
    if predictor_sha != identity['predictor_sha256']:
        raise ValueError('Original 07 Predictor identity mismatch')
    for value in (fingerprint['size'], config['epochs'], config['batch_size'], config['accumulation_steps']):
        if type(value) is not int or value < 1:
            raise ValueError('Invalid saved training counters')
    updates = math.ceil(fingerprint['size'] / (config['batch_size'] * config['accumulation_steps'])) * config['epochs']
    if report['epoch'] != config['epochs'] or report['updates'] != updates or report['total_updates'] != updates:
        raise ValueError('09 completion counters do not match saved settings')
    selected = checkpoint['updates']
    if type(selected) is not int or not 1 <= selected <= updates or report['best_update'] != selected:
        raise ValueError('09 best-model/report update mismatch')
    if not math.isfinite(checkpoint['mel']) or report['best_mel'] != checkpoint['mel']:
        raise ValueError('09 best-model/report metric mismatch')
    return {'selected_update':selected, 'training_updates':updates, 'best_mel':checkpoint['mel']}


def verify_inference_sources(project, identity):
    project = Path(project).resolve()
    expected = identity['source_sha256']
    if 'src/fantasyvoice/inference/tts.py' not in expected:
        raise ValueError('Missing inference source identity')
    for name, digest in expected.items():
        path = project / name
        if not name.startswith('src/fantasyvoice/') or not path.resolve().is_relative_to(project):
            raise ValueError('Invalid source identity path')
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'09 모델과 코드 버전이 다릅니다: {name}. 학습에 사용한 코드 ZIP이 필요합니다.')


def load_joint_weights(tts, predictor, checkpoint):
    # The 07 file only reconstructs the correct architecture/configuration.
    # Always replace BOTH networks with the matched joint checkpoint.
    tts.load_state_dict(checkpoint['modules']['g'], strict=True)
    predictor.load_state_dict(checkpoint['modules']['predictor'], strict=True)
    tts.eval(); predictor.eval()


def save_generation(output, waveform, info, model_info):
    wave = waveform.detach().cpu().float().numpy()
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError('Invalid generated audio')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:12]
    directory = Path(output) / stamp
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory / 'voice.wav'
    sf.write(raw, wave, 44100, subtype='FLOAT')
    playback = playback_audio(raw)
    preview = None
    if playback['wav_bytes'] is not None:
        preview = directory / 'playback.wav'
        preview.write_bytes(playback['wav_bytes'])
    metadata = directory / 'generation.json'
    write_json(metadata, {**info, 'model':model_info, 'sample_rate':44100,
        'duration_seconds':len(wave)/44100, 'peak_amplitude':float(np.max(np.abs(wave))),
        'playback_status':playback['status'], 'playback_gain':playback['playback_gain'],
        'raw_sha256':sha256(raw)})
    archive = directory.with_suffix('.zip')
    partial = archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in (raw, preview, metadata):
            if path is not None:
                zipped.write(path, path.name)
    partial.replace(archive)
    return {'raw':raw, 'preview':preview, 'metadata':metadata, 'archive':archive,
            'playback_status':playback['status']}
