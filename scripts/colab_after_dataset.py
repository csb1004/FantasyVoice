"""Run AFTER transcription ends, on CPU if desired; no ASR/model download."""
from pathlib import Path
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

from google.colab import drive, files

drive.mount('/content/drive')
BUNDLE = Path('/content/drive/MyDrive/FantasyVoice/fantasyvoice-pilot.zip')
DATASET = Path('/content/drive/MyDrive/FantasyVoice/dataset-v1')
PILOT = Path('/content/drive/MyDrive/FantasyVoice/pilot-v1')
PROJECT = Path('/content/fantasyvoice-postprocess')
DOWNLOAD_REPORT = True

if not (DATASET / 'run.json').is_file() or not (DATASET / 'dataset.json').is_file():
    raise FileNotFoundError('전체 전사 결과 폴더에 run.json/dataset.json이 없습니다.')
with zipfile.ZipFile(BUNDLE) as archive:
    if 'src/fantasyvoice/dataset/finalize.py' not in archive.namelist():
        raise RuntimeError('후속 처리 코드가 포함된 최신 ZIP으로 교체해주세요.')
    for item in archive.infolist():
        if not (PROJECT / item.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT)], check=True)
sys.path.insert(0, str(PROJECT / 'src'))
for name in list(sys.modules):
    if name == 'fantasyvoice' or name.startswith('fantasyvoice.'):
        del sys.modules[name]
importlib.invalidate_caches()

from fantasyvoice.dataset.inventory import load_reference
from fantasyvoice.dataset.storage import read_jsonl
from fantasyvoice.dataset.finalize import finalize_dataset

rows = load_reference(PROJECT / 'reference/inventory.jsonl')
print('전사 결과 파일을 복사합니다. 원본 음성은 읽지 않습니다.', flush=True)
progress_path = DATASET / 'progress.json'
progress_before = progress_path.read_bytes() if progress_path.exists() else None
with tempfile.TemporaryDirectory(prefix='fantasyvoice-report-') as temp:
    local = Path(temp) / 'dataset'
    local.mkdir()
    metadata = sorted(DATASET.glob('*.json')) + sorted(DATASET.glob('*.jsonl'))
    metadata += sorted((DATASET / 'results').glob('*.jsonl'))
    for index, path in enumerate(metadata, 1):
        destination = local / path.relative_to(DATASET)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if index % 50 == 0 or index == len(metadata):
            print(f'결과 파일 {index}/{len(metadata)}', flush=True)
    progress_after = progress_path.read_bytes() if progress_path.exists() else None
    if progress_before != progress_after:
        raise RuntimeError('전사가 아직 진행 중인 것으로 보입니다. 종료 후 다시 실행하세요.')
    known = read_jsonl(PROJECT / 'reference/reviews.jsonl')
    by_key = {r['key']: r for r in known + read_jsonl(PILOT / 'transcripts.jsonl')}
    current_reviews = list(known)
    for r in read_jsonl(PILOT / 'reviews.jsonl'):
        review = dict(r)
        match = by_key.get(review['key'])
        if match:
            review.update(audio_path=match['audio_path'], sha256=match['sha256'])
        current_reviews.append(review)
    print('완료 상태 확인 및 수정 대사 반영 중...', flush=True)
    local_zip = Path(temp) / 'report.zip'
    report = finalize_dataset(rows, local, local_zip, current_reviews)
    print((local / 'completion-report.md').read_text(encoding='utf-8'))
    name = 'dataset-report.zip' if report['asr_pass_complete'] else 'dataset-report-partial.zip'
    destination = DATASET.parent / name
    pending_copy = destination.with_suffix('.zip.partial')
    shutil.copyfile(local_zip, pending_copy)
    os.replace(pending_copy, destination)
    print('Drive 저장 완료:', destination, flush=True)
    if not report['asr_pass_complete']:
        print('오류/미처리가 남았습니다. ZIP에 retry.jsonl이 포함됐습니다.')
        print('기존 전체 전사 코드로 재개하면 오류/미처리 항목을 다시 처리합니다.')
    print('이 ZIP을 전달해주시면 데이터 품질과 다음 스타일 라벨 생성 설정을 확인할 수 있습니다.')
    if DOWNLOAD_REPORT:
        try:
            files.download(str(destination))
        except Exception as exc:
            print('브라우저 다운로드를 시작하지 못했습니다. 위 Drive 파일을 직접 내려받으세요:', exc)
