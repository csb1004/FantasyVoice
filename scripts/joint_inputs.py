"""Read-only handoff checks for Colab 09; no dependency or model loading side effects."""
import math


def validate_completed_warmup(checkpoint, report, run, joint_config, current_identity):
    fingerprint = checkpoint['fingerprint']
    config, identity = fingerprint['config'], fingerprint['identity']
    if config.get('stage') != 'warmup' or joint_config.get('stage') != 'joint':
        raise ValueError('09 requires a warmup checkpoint and joint configuration')
    if report.get('complete') is not True or report.get('pending_eval') is not None:
        raise ValueError('08 준비 학습과 마지막 검증을 먼저 완료하세요.')
    if run['config'] != config or run['identity'] != identity:
        raise ValueError('08 run/report/checkpoint identity mismatch')
    for key in ('dataset_sha256', 'dataset_metadata', 'predictor_sha256'):
        if identity[key] != current_identity[key]:
            raise ValueError(f'08 input identity mismatch: {key}')
    for key in ('model_revision', 'melo_revision', 'frontend_revision', 'add_blank', 'model_artifact_sha256'):
        if config[key] != joint_config[key]:
            raise ValueError(f'08/09 configuration mismatch: {key}')
    for key in ('epochs', 'batch_size', 'accumulation_steps'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'Invalid warmup {key}')
    size = fingerprint['size']
    if type(size) is not int or size < 1:
        raise ValueError('Invalid warmup dataset size')
    updates = math.ceil(size / (config['batch_size'] * config['accumulation_steps'])) * config['epochs']
    if report['epoch'] != config['epochs'] or report['updates'] != updates or report['total_updates'] != updates:
        raise ValueError('08 completion counters do not match its configuration')
    selected = checkpoint['updates']
    if type(selected) is not int or not 1 <= selected <= updates or report['best_update'] != selected:
        raise ValueError('Select the best-model.pt matching the completed 08 report')
    if not math.isfinite(checkpoint['mel']) or checkpoint['mel'] != report['best_mel']:
        raise ValueError('08 best validation metric mismatch')
    if 'g' not in checkpoint['modules'] or 'predictor' in checkpoint['modules']:
        raise ValueError('Expected warmup generator weights')
    return {'warmup_updates':updates, 'selected_update':selected, 'best_mel':checkpoint['mel']}
