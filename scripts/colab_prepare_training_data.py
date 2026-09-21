"""06: CPU-only manifest preparation, using explicitly confirmed dataset settings."""
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
REPORT = BASE / 'style-report.zip'
OUTPUT_ZIP = BASE / 'dataset-prepared-v1.zip'
PROJECT = Path('/content/fantasyvoice-prepare-training')
# Fill ONLY after confirming policies. Otherwise use config/training-data.json in the bundle.
SETTINGS_OVERRIDE = None

with zipfile.ZipFile(BUNDLE) as archive:
    if 'src/fantasyvoice/dataset/training_data.py' not in archive.namelist():
        raise RuntimeError('06 코드가 포함된 최신 코드 ZIP을 업로드하세요.')
    for item in archive.infolist():
        if not (PROJECT / item.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
sys.path.insert(0, str(PROJECT / 'src'))
for name in list(sys.modules):
    if name == 'fantasyvoice' or name.startswith('fantasyvoice.'):
        del sys.modules[name]
importlib.invalidate_caches()
from fantasyvoice.dataset.training_data import check_settings, prepare_training_data
from fantasyvoice.dataset.storage import sha256, write_json

config = SETTINGS_OVERRIDE if SETTINGS_OVERRIDE is not None else json.loads(
    (PROJECT / 'config/training-data.json').read_text(encoding='utf-8'))
try:
    check_settings(config)
except ValueError as exc:
    raise ValueError('06 설정 확인이 필요합니다. 분할 비율·데이터 사용 범위·피치 기준·seed를 '
                     '확정한 뒤 config/training-data.json 또는 SETTINGS_OVERRIDE에 반영하세요.') from exc
if OUTPUT_ZIP.exists():
    raise FileExistsError('기존 준비 결과를 보존합니다. 새 버전은 OUTPUT_ZIP 이름을 바꿔주세요.')
with zipfile.ZipFile(REPORT) as archive:
    report = json.loads(archive.read('style-v1/completion-report.json'))
    if not report['style_pass_complete'] or report['invalid_count']:
        raise RuntimeError('05에서 누락·오류·형식 이상을 먼저 해결해주세요.')
    rows = [json.loads(line) for line in archive.read('style-v1/style-labels.jsonl').decode('utf-8').splitlines() if line.strip()]
    analyzer_config = json.loads(archive.read('style-v1/config.json'))
print(f'입력 {len(rows):,}개. 확인된 설정으로 목록을 분할하고 학습 통계를 계산합니다.', flush=True)
with tempfile.TemporaryDirectory(prefix='fantasyvoice-prepare-') as temp:
    output = Path(temp) / 'dataset-prepared-v1'
    summary = prepare_training_data(rows, config, output)
    write_json(output / 'analyzer-config.json', analyzer_config)
    write_json(output / 'source-report.json', {'path': str(REPORT), 'sha256': sha256(REPORT)})
    local_zip = Path(temp) / 'dataset.zip'
    with zipfile.ZipFile(local_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, 'dataset-prepared-v1/' + path.name)
    pending = OUTPUT_ZIP.with_suffix('.zip.partial')
    shutil.copyfile(local_zip, pending)
    os.replace(pending, OUTPUT_ZIP)
    print(f"학습 {summary['train_count']:,} / 검증 {summary['validation_count']:,} / 보류 {summary['held_count']:,}")
    print('실제 검증 비율:', summary['actual_validation_fraction'])
    print('검증 데이터 없는 캐릭터:', summary['characters_without_validation'])
    print('저장:', OUTPUT_ZIP)
    print('목록 준비 완료입니다. 원본 음원 변환이나 모델 학습은 실행하지 않았습니다.')
    try:
        files.download(str(OUTPUT_ZIP))
    except Exception as exc:
        print('자동 다운로드 실패. Drive에서 직접 받을 수 있습니다:', exc)
