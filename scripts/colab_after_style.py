"""CPU-only validation/export after 04; source audio and models are not loaded."""
from pathlib import Path
import importlib
import json
import os
import shutil
import sys
import tempfile
import zipfile

from google.colab import drive, files

drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/FantasyVoice')
BUNDLE = BASE / 'fantasyvoice-pilot.zip'
DATASET_REPORT = BASE / 'dataset-report.zip'
STYLE = BASE / 'style-v1'
PROJECT = Path('/content/fantasyvoice-style-report')
CANDIDATES_OVERRIDE = None  # Set the SAME override as 04, if you used one.
DOWNLOAD_REPORT = True

if not (STYLE / 'config.json').is_file():
    raise FileNotFoundError('style-v1/config.json이 없습니다. 04 출력 폴더를 확인하세요.')
with zipfile.ZipFile(BUNDLE) as archive:
    if 'src/fantasyvoice/style/finalize.py' not in archive.namelist():
        raise RuntimeError('05 결과 점검 코드가 포함된 최신 fantasyvoice-pilot.zip을 업로드하세요.')
    for item in archive.infolist():
        if not (PROJECT / item.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
sys.path.insert(0, str(PROJECT / 'src'))
for name in list(sys.modules):
    if name == 'fantasyvoice' or name.startswith('fantasyvoice.'):
        del sys.modules[name]
importlib.invalidate_caches()

from fantasyvoice.dataset.storage import sha256
from fantasyvoice.style.finalize import finalize_styles

with zipfile.ZipFile(DATASET_REPORT) as archive:
    candidate_text = archive.read('dataset-v1/candidates.jsonl').decode('utf-8')
if CANDIDATES_OVERRIDE:
    candidate_text = Path(CANDIDATES_OVERRIDE).read_text(encoding='utf-8')
rows = [json.loads(line) for line in candidate_text.splitlines() if line.strip()]
print(f'기준 후보 {len(rows):,}개. 스타일 체크포인트를 복사합니다.', flush=True)

def source_files():
    paths = [STYLE / 'config.json'] + sorted((STYLE / 'results').glob('*.jsonl'))
    if (STYLE / 'summary.json').exists():
        paths.append(STYLE / 'summary.json')
    return paths

with tempfile.TemporaryDirectory(prefix='fantasyvoice-style-report-') as temp:
    snapshot, report_dir = Path(temp) / 'snapshot', Path(temp) / 'report'
    paths = source_files()
    hashes = {}
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(STYLE)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[relative] = sha256(target)
        if index % 50 == 0 or index == len(paths):
            print(f'체크포인트 복사 {index}/{len(paths)}', flush=True)
    print('복사 중 결과가 변경되지 않았는지 확인합니다.', flush=True)
    if source_files() != paths or any(sha256(STYLE / name) != value for name, value in hashes.items()):
        raise RuntimeError('스타일 결과가 변경 중입니다. 04가 끝난 뒤 다시 실행하세요.')
    report = finalize_styles(rows, snapshot, report_dir)
    print((report_dir / 'completion-report.md').read_text(encoding='utf-8'))
    if report.get('previous_summary_matches') is False:
        print('기존 summary와 다릅니다. 입력 후보 변경 또는 이전 내보내기 중단 여부를 확인하세요.')
    name = 'style-report.zip' if report['style_pass_complete'] else 'style-report-partial.zip'
    destination = BASE / name
    pending = destination.with_suffix('.zip.partial')
    shutil.copyfile(report_dir / 'style-report.zip', pending)
    os.replace(pending, destination)
    print('Drive 저장 완료:', destination, flush=True)
    print('이 ZIP을 전달해주세요. 라벨 분포를 확인한 뒤 데이터 분할·학습 설정을 진행합니다.')
    if DOWNLOAD_REPORT:
        try:
            files.download(str(destination))
        except Exception as exc:
            print('자동 다운로드 실패. Drive에서 직접 받을 수 있습니다:', exc)
