"""Run in a fresh GPU Colab AFTER ASR and colab_after_dataset.py finish."""
from pathlib import Path
import importlib
import json
import subprocess
import sys
import time
import zipfile

from google.colab import drive

drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/FantasyVoice')
VOICE_ROOT = Path('/content/drive/MyDrive/Voice')
BUNDLE = BASE / 'fantasyvoice-pilot.zip'
REPORT = BASE / 'dataset-report.zip'
OUTPUT = BASE / 'style-v1'
PROJECT = Path('/content/fantasyvoice-style')
# Start with the full finalized candidate list, including pending human reviews.
# To use a subsequently corrected list, set an absolute JSONL path here.
CANDIDATES_OVERRIDE = None

if not VOICE_ROOT.is_dir():
    raise FileNotFoundError(VOICE_ROOT)
with zipfile.ZipFile(BUNDLE) as archive:
    if 'scripts/colab_style_labels.py' not in archive.namelist():
        raise RuntimeError('스타일 라벨 코드가 포함된 최신 ZIP으로 교체해주세요.')
    for item in archive.infolist():
        if not (PROJECT / item.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
with zipfile.ZipFile(REPORT) as archive:
    report = json.loads(archive.read('dataset-v1/completion-report.json'))
    if not report['asr_pass_complete']:
        raise RuntimeError('전사 오류/미처리를 완료한 뒤 후속 점검 코드를 다시 실행하세요.')
    candidate_text = archive.read('dataset-v1/candidates.jsonl').decode('utf-8')
if CANDIDATES_OVERRIDE:
    candidate_text = Path(CANDIDATES_OVERRIDE).read_text(encoding='utf-8')
rows = [json.loads(line) for line in candidate_text.splitlines() if line.strip()]
if not rows:
    raise RuntimeError('스타일 분석 대상이 없습니다.')
config = json.loads((PROJECT / 'config/style.json').read_text(encoding='utf-8'))
print(f'스타일 분석 대상 {len(rows):,}개. 의존성을 설치합니다.', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e',
                str(PROJECT) + '[style]'], check=True)
# Install only Melo's source; the Korean frontend dependencies are in [style].
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
                f'https://github.com/myshell-ai/MeloTTS/archive/{config["melo_revision"]}.zip'], check=True)
sys.path.insert(0, str(PROJECT / 'src'))
for name in list(sys.modules):
    if name == 'fantasyvoice' or name.startswith('fantasyvoice.'):
        del sys.modules[name]
importlib.invalidate_caches()

from fantasyvoice.dataset.storage import check_output, sha256
from fantasyvoice.style.backends import FrozenAnalyzers
from fantasyvoice.style.runner import run_styles

check_output(VOICE_ROOT, OUTPUT)
print('고정 분석기를 로딩합니다. 최초 실행 시 모델 다운로드가 필요합니다.', flush=True)
backend = FrozenAnalyzers(VOICE_ROOT, config)
run_config = {'extraction': config, 'provenance': backend.provenance,
              'implementation_sha256': {p.name: sha256(p) for p in
                  sorted((PROJECT / 'src/fantasyvoice/style').glob('*.py'))}}
# A mismatched configuration must fail before spending time on analysis.
saved_config = OUTPUT / 'config.json'
if saved_config.exists() and json.loads(saved_config.read_text(encoding='utf-8')) != run_config:
    raise RuntimeError('분석 설정/버전이 바뀌었습니다. OUTPUT을 style-v2 등 새 폴더로 지정하세요.')
probe = next((r for r in rows if r['duration_seconds'] > .5 and r['text'].strip()), None)
if probe:
    print('첫 음원으로 분석기 연결을 확인합니다:', probe['audio_path'], flush=True)
    backend(probe)
started = time.monotonic()
def progress(done, total, status):
    print(f'{done:,}/{total:,} 처리 | 최근 {status} | 이번 실행 {(time.monotonic()-started)/60:.1f}분', flush=True)

summary = run_styles(rows, OUTPUT, run_config, backend, checkpoint_every=20, progress=progress)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('저장:', OUTPUT / 'style-labels.jsonl')
print('확인 필요:', OUTPUT / 'style-needs-review.jsonl')
print('감정/운율은 자동 추정값입니다. 검수 상태는 그대로이며 상대 피치 정규화는 분할 이후입니다.')
