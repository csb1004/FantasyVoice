"""One-cell Colab entry point for ALL duration-eligible Voice recordings."""
from pathlib import Path
import importlib
import json
import platform
import subprocess
import sys
import time
import zipfile

from google.colab import drive

drive.mount('/content/drive')
VOICE_ROOT = Path('/content/drive/MyDrive/Voice')
BUNDLE = Path('/content/drive/MyDrive/FantasyVoice/fantasyvoice-pilot.zip')
PROJECT = Path('/content/fantasyvoice-dataset')
OUTPUT = Path('/content/drive/MyDrive/FantasyVoice/dataset-v1')
PILOT = Path('/content/drive/MyDrive/FantasyVoice/pilot-v1')
MODEL_CACHE = Path('/content/whisper-models')

if not VOICE_ROOT.is_dir():
    raise FileNotFoundError(VOICE_ROOT)
with zipfile.ZipFile(BUNDLE) as archive:
    if 'src/fantasyvoice/dataset/full_dataset.py' not in archive.namelist():
        raise RuntimeError('최신 전체 데이터셋 지원 ZIP으로 교체해주세요.')
    for item in archive.infolist():
        if not (PROJECT / item.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e',
                str(PROJECT) + '[asr]'], check=True)
sys.path.insert(0, str(PROJECT / 'src'))
# Previous pilot imports must not keep running an older installed copy.
for module_name in list(sys.modules):
    if module_name == 'fantasyvoice' or module_name.startswith('fantasyvoice.'):
        del sys.modules[module_name]
importlib.invalidate_caches()

import torch
from fantasyvoice.dataset.inventory import load_reference
from fantasyvoice.dataset.storage import check_output, read_jsonl, write_json, write_jsonl
from fantasyvoice.dataset.transcribe import WhisperBackend
from fantasyvoice.dataset.full_dataset import eligible_rows, run_dataset

check_output(VOICE_ROOT, OUTPUT)
rows = load_reference(PROJECT / 'reference/inventory.jsonl')
eligible = eligible_rows(rows)
print(f'로컬 기준 목록: {len(rows):,}개 / 0.5초 초과 전사 대상: {len(eligible):,}개', flush=True)
print('12개 제한 없음. Drive 파일은 처리할 때 해시를 확인합니다.', flush=True)
print('출력:', OUTPUT, flush=True)

# Preserve original review files. Join legacy key-only reviews to source hashes.
known_reviews = read_jsonl(PROJECT / 'reference/reviews.jsonl')
pilot_transcripts = read_jsonl(PILOT / 'transcripts.jsonl')
by_key = {r['key']: r for r in known_reviews + pilot_transcripts}
reviews = list(known_reviews)
for review in read_jsonl(PILOT / 'reviews.jsonl'):
    record = dict(review)
    source = by_key.get(record['key'])
    if source:
        record.update(audio_path=source['audio_path'], sha256=source['sha256'])
    reviews.append(record)
write_jsonl(OUTPUT / 'imported-reviews.jsonl', reviews)
options = json.loads((PROJECT / 'config/pilot.json').read_text(encoding='utf-8'))['whisper_options']
print('Whisper large-v3를 준비합니다.', flush=True)
backend = WhisperBackend(options, MODEL_CACHE)
from datetime import datetime, timezone
write_json(OUTPUT / ('runtime-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json'),
           {'python': platform.python_version(), 'torch': str(torch.__version__),
            'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
            'source': 'bundled_local_inventory', 'remote_full_scan_performed': False})
started = time.monotonic()
def report(state):
    finished = state['success'] + state['needs_review']
    print(f"{finished:,}/{state['total']:,} 처리 | 전사 {state['success']:,} | "
          f"검수대기 {state['needs_review']:,} | 오류 {state['error']:,} | "
          f"이번 실행 {(time.monotonic()-started)/60:.1f}분", flush=True)

try:
    state = run_dataset(VOICE_ROOT, rows, OUTPUT, backend.config, backend,
                        reviews=reviews, pilot_output=PILOT, checkpoint_every=20,
                        progress=report)
finally:
    del backend
    torch.cuda.empty_cache()
print('이번 실행 종료:', state)
print('전체 전사 후보:', OUTPUT / 'candidates.jsonl')
print('사람이 확인한 대사:', OUTPUT / 'reviewed.jsonl')
print('무음/오류/부적합:', OUTPUT / 'needs-review.jsonl')
print('진행 기록:', OUTPUT / 'progress.json')
print('감정 라벨과 학습 분할은 아직 생성하지 않았습니다.')
