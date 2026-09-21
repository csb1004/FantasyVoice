"""07: actual text-only Style Predictor warmup. Use a fresh T4 Colab runtime."""
from pathlib import Path
import importlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from google.colab import drive, files

drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/FantasyVoice')
BUNDLE = BASE / 'fantasyvoice-pilot.zip'
DATASET = BASE / 'dataset-prepared-v1.zip'
OUTPUT = BASE / 'predictor-v1'
PROJECT = Path('/content/fantasyvoice-predictor')
# Experiment controls. Same settings + OUTPUT resume; changed settings need a new OUTPUT.
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EPOCHS = 3
EVAL_BATCH_SIZE = 16
BERT_LR = 2e-5
HEAD_LR = 1e-4
DOWNLOAD_REPORT = True

if not DATASET.is_file():
    raise FileNotFoundError('MyDrive/FantasyVoice/dataset-prepared-v1.zip을 업로드하세요. 06 재실행은 필요 없습니다.')
with zipfile.ZipFile(BUNDLE) as archive:
    if 'src/fantasyvoice/training/predictor.py' not in archive.namelist():
        raise RuntimeError('07 학습 코드가 포함된 최신 fantasyvoice-pilot.zip으로 교체하세요.')
    for item in archive.infolist():
        if not (PROJECT / item.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
print('Predictor 학습 의존성을 설치합니다. 음성 분석기는 설치하지 않습니다.', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT) + '[predictor]'], check=True)
if 'transformers' in sys.modules and sys.modules['transformers'].__version__ != '4.57.6':
    raise RuntimeError('이전 Transformers가 메모리에 남아 있습니다. 새 런타임에서 07을 실행하세요.')
sys.path.insert(0, str(PROJECT / 'src'))
for name in list(sys.modules):
    if name == 'fantasyvoice' or name.startswith('fantasyvoice.'):
        del sys.modules[name]
importlib.invalidate_caches()

import torch
from transformers import AutoTokenizer
from fantasyvoice.dataset.storage import sha256, write_json, write_jsonl
from fantasyvoice.models.predictor import StylePredictor
from fantasyvoice.style.features import EMOTIONS
from fantasyvoice.style.runner import digest
from fantasyvoice.training.predictor import (
    read_prepared, tokenize_rows, check_config, preflight, fit, evaluate,
    constant_baseline, atomic_checkpoint)

if not torch.cuda.is_available():
    raise RuntimeError('07은 실제 모델 학습입니다. Colab 런타임에서 T4 GPU를 선택하세요.')
device = 'cuda'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
config = json.loads((PROJECT / 'config/predictor.json').read_text(encoding='utf-8'))
config.update(batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS,
              epochs=EPOCHS, eval_batch_size=EVAL_BATCH_SIZE, bert_lr=BERT_LR, head_lr=HEAD_LR)
check_config(config, device)
if OUTPUT.exists() and any(OUTPUT.iterdir()) and not (OUTPUT / 'latest.pt').exists():
    raise RuntimeError('출력 폴더에 파일이 있지만 latest.pt가 없습니다. 내용을 보존하고 새 OUTPUT 이름을 지정하세요.')
splits, metadata = read_prepared(DATASET)
identity = {
    'dataset_sha256': sha256(DATASET), 'model_revision': config['model_revision'],
    'emotion_order': list(EMOTIONS),
    'dataset_statistics': {'normalization': metadata['preparation']['normalization'],
                           'pitch_baselines_hz': metadata['preparation']['pitch_baselines_hz'],
                           'character_map': metadata['character_map']},
    'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'safetensors')},
    'implementation_sha256': {p.relative_to(PROJECT / 'src').as_posix(): sha256(p)
        for folder in ('models', 'training') for p in sorted((PROJECT / 'src/fantasyvoice' / folder).glob('*.py'))},
}
run_identity = {'identity': identity, 'config': config}
if (OUTPUT / 'run.json').exists():
    if json.loads((OUTPUT / 'run.json').read_text(encoding='utf-8')) != run_identity:
        raise RuntimeError('데이터·설정·코드·패키지 버전이 이전 실행과 다릅니다. OUTPUT을 predictor-v2 등으로 바꾸세요.')
print(f"학습 {len(splits['train']):,} / 검증 {len(splits['validation']):,}. 대사를 토큰화합니다.", flush=True)
tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['model_revision'], trust_remote_code=False)
if tokenizer.pad_token_id is None:
    raise RuntimeError('Tokenizer requires a padding token')
data = {split: tokenize_rows(rows, tokenizer, config['max_tokens']) for split, rows in splits.items()}
print('최대 토큰 수:', max(len(r['inputs']['input_ids']) for rows in data.values() for r in rows), flush=True)
random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
print('KLUE BERT와 스타일 출력층을 로딩합니다.', flush=True)
model = StylePredictor.from_pretrained(config['model_name'], config['model_revision'], config['dropout'])
print('작은 배치의 실제 forward/backward를 점검합니다.', flush=True)
probe = preflight(model, data['train'], config, tokenizer.pad_token_id, device)
print('사전 점검 통과:', probe, flush=True)
print('전체 설정으로 학습을 시작합니다. 작은 배치 점검은 전체 배치 메모리 보장이 아닙니다.', flush=True)
started = time.monotonic()
def progress(info):
    if info['updates'] % 10 == 0 or info['validation'] is not None:
        val = info['validation']['total_loss'] if info['validation'] else None
        print(f"update {info['updates']}/{info['total_updates']} | epoch {info['epoch']} | "
              f"train {info['train_window_loss']:.4f} | validation {val} | "
              f"이번 실행 {(time.monotonic()-started)/60:.1f}분", flush=True)
    # Persist run identity after the first durable checkpoint exists.
    if not (OUTPUT / 'run.json').exists():
        write_json(OUTPUT / 'run.json', run_identity)

result = fit(model, data['train'], data['validation'], config, OUTPUT, identity,
             tokenizer.pad_token_id, device, progress=progress)
write_json(OUTPUT / 'run.json', run_identity)
write_json(OUTPUT / 'dataset-metadata.json', metadata)
write_json(OUTPUT / 'runtime.json', {'python': sys.version, 'torch': str(torch.__version__),
                                  'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)})
print('학습 종료. 최적 체크포인트를 평가하고 결과를 내보냅니다.', flush=True)
state = torch.load(OUTPUT / 'best.pt', map_location='cpu', weights_only=True)
best_update = state['updates']
model.load_state_dict(state['model'])
del state
metrics, predictions = evaluate(model, data['validation'], config, tokenizer.pad_token_id, device, predictions=True)
baseline = constant_baseline(data['train'], data['validation'], config, tokenizer.pad_token_id, device)
statistics = metadata['preparation']['normalization']
for prediction in predictions:
    prediction['continuous_raw'] = {key: prediction[key + '_z'] * statistics[key]['scale'] + statistics[key]['mean']
                                    for key in ('speed', 'pitch', 'pause')}
report = {**result, 'best_update': best_update, 'best_validation': metrics,
          'constant_train_baseline': baseline,
          'mae_original_units': {key: metrics['continuous_mae_z'][key] * statistics[key]['scale']
              if metrics['continuous_mae_z'][key] is not None else None for key in ('speed', 'pitch', 'pause')},
          'units': {'speed': 'phonemes/second', 'pitch': 'semitones relative to character baseline', 'pause': 'ratio'},
          'scope': 'Text-only predictor warmup against pseudo labels; no TTS training or speech quality evaluation'}
write_json(OUTPUT / 'training-report.json', report)
write_jsonl(OUTPUT / 'validation-predictions.jsonl', predictions)
atomic_checkpoint(OUTPUT / 'best-model.pt', {'model': model.state_dict(), 'config': config,
                  'identity': identity, 'dataset_metadata': metadata, 'best_update': best_update})
tokenizer.save_pretrained(OUTPUT / 'tokenizer')
with tempfile.TemporaryDirectory(prefix='fantasyvoice-predictor-report-') as temp:
    local_zip = Path(temp) / 'report.zip'
    with zipfile.ZipFile(local_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('run.json', 'runtime.json', 'progress.json', 'history.json', 'dataset-metadata.json',
                     'training-report.json', 'validation-predictions.jsonl'):
            archive.write(OUTPUT / name, 'predictor-v1/' + name)
    destination = BASE / (OUTPUT.name + '-report.zip')
    pending = destination.with_suffix('.zip.partial')
    shutil.copyfile(local_zip, pending)
    os.replace(pending, destination)
print(json.dumps(report, ensure_ascii=False, indent=2))
print('가중치:', OUTPUT / 'best-model.pt')
print('검토용 결과 ZIP:', destination)
if DOWNLOAD_REPORT:
    try:
        files.download(str(destination))
    except Exception as exc:
        print('자동 다운로드 실패. Drive에서 결과 ZIP을 직접 받을 수 있습니다:', exc)
