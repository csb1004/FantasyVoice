"""11: supervised real-audio emotion training, completed BEFORE TTS feedback."""
from pathlib import Path
import importlib
import importlib.metadata
import json
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile

from google.colab import drive, files

drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/FantasyVoice')
VOICE_ROOT = Path('/content/drive/MyDrive/Voice')
BUNDLE = BASE / 'fantasyvoice-pilot.zip'
DATASET = BASE / 'dataset-prepared-v1.zip'
PROJECT = Path('/content/fantasyvoice-emotion')
CACHE = Path('/content/fantasyvoice-tts-cache')
CACHE_CHECKPOINTS = BASE / 'tts-cache-checkpoints'
EXPERIMENT = 'v2-batch'
OUTPUT = BASE / f'emotion-analyzer-{EXPERIMENT}'
# Provisional starting values, not quality-validated hyperparameters.
EPOCHS = 1
BATCH_SIZE = 'auto'  # T4: 4, L4: 8; or set a positive integer.
ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-6
MAX_SECONDS = 15.0
DOWNLOAD_REPORT = True

for path in (BUNDLE, DATASET):
    if not path.is_file():
        raise FileNotFoundError(path)
if not VOICE_ROOT.is_dir():
    raise FileNotFoundError(VOICE_ROOT)
with zipfile.ZipFile(BUNDLE) as archive:
    if not {'scripts/emotion_training.py','scripts/gpu_batches.py'}.issubset(archive.namelist()):
        raise ValueError('11/12가 포함된 최신 코드 ZIP으로 교체하세요.')
    for item in archive.infolist():
        if not (PROJECT / item.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT)+'[tts]'], check=True)
torch_release = importlib.metadata.version('torch').split('+')[0]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'funasr==1.4.16',
                'torchaudio=='+torch_release], check=True)
melo_revision = json.loads((PROJECT/'config/tts.json').read_text())['melo_revision']
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
    f'https://github.com/myshell-ai/MeloTTS/archive/{melo_revision}.zip'], check=True)
for package in ('torch','torchaudio','numpy','transformers','numba','librosa'):
    if package in sys.modules and sys.modules[package].__version__ != importlib.metadata.version(package):
        raise RuntimeError(f'{package}: 런타임을 다시 시작하고 셀을 실행하세요.')
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT/'scripts')]
for name in list(sys.modules):
    if (name == 'fantasyvoice' or name.startswith('fantasyvoice.')
            or name in ('emotion_feedback','emotion_training','gpu_batches','cache_checkpoints')):
        del sys.modules[name]
importlib.invalidate_caches()

import torch
from fantasyvoice.dataset.storage import sha256, write_json
from fantasyvoice.dataset.tts_data import cache_audio
from fantasyvoice.training.predictor import read_prepared
from fantasyvoice.style.features import EMOTIONS
from cache_checkpoints import CacheCheckpoints
from emotion_feedback import load_analyzer
from emotion_training import EmotionData, select_duration, fit_emotion
from gpu_batches import batch_preset

if not torch.cuda.is_available():
    raise RuntimeError('감정 분석기 전체 학습에는 GPU 런타임을 선택하세요.')
device = 'cuda'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
if BATCH_SIZE == 'auto':
    props = torch.cuda.get_device_properties(0)
    BATCH_SIZE, _ = batch_preset(props.name, props.total_memory / 2**30, 11)
if type(BATCH_SIZE) is not int or BATCH_SIZE < 1:
    raise ValueError("BATCH_SIZE must be 'auto' or a positive integer")
print(f'실제 배치 {BATCH_SIZE}, 누적 {ACCUMULATION_STEPS}, 유효 배치 {BATCH_SIZE*ACCUMULATION_STEPS}', flush=True)

config = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE, max_seconds=MAX_SECONDS, gradient_clip=1.,
    seed=42, save_every=100, eval_every=100, validation_samples=16, lr_decay=.999875,
    precision='fp32', training='all_used_parameters_real_audio_pseudo_labels')
random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
splits, metadata = read_prepared(DATASET)
splits, selection = select_duration(splits, MAX_SECONDS)
analyzer_config = metadata['analyzer_config']['extraction']['emotion']
identity = {'dataset_sha256':sha256(DATASET), 'dataset_metadata':metadata,
    'emotion_order':list(EMOTIONS), 'emotion_model':analyzer_config,
    'duration_selection':selection,
    'entrypoint_sha256':sha256(PROJECT/'scripts/colab_train_emotion.py'),
    'adapter_sha256':sha256(PROJECT/'scripts/emotion_feedback.py'),
    'training_sha256':sha256(PROJECT/'scripts/emotion_training.py'),
    'batch_helper_sha256':sha256(PROJECT/'scripts/gpu_batches.py'),
    'cache_sha256':sha256(PROJECT/'scripts/cache_checkpoints.py'),
    'source_sha256':{p.relative_to(PROJECT).as_posix():sha256(p)
        for p in sorted((PROJECT/'src/fantasyvoice').rglob('*.py'))},
    'versions':{k:importlib.metadata.version(k) for k in
        ('torch','torchaudio','funasr','numpy','transformers','omegaconf','soundfile','scipy')}}
fingerprint = {'config':config, 'identity':identity, 'size':len(splits['train'])}
if (OUTPUT/'latest.pt').exists():
    saved = torch.load(OUTPUT/'latest.pt', map_location='cpu', weights_only=True, mmap=True)
    if saved['fingerprint'] != fingerprint:
        raise ValueError('설정/코드/데이터 변경 시 새 EXPERIMENT를 사용하세요.')
    del saved
elif OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise ValueError('출력 폴더에 재개 가능한 체크포인트가 없습니다. 새 EXPERIMENT를 사용하세요.')

print('길이 조건별 대상:', selection, flush=True)
cache_checkpoints = CacheCheckpoints(CACHE, CACHE_CHECKPOINTS)
cache_checkpoints.restore()
data = {}
done = 0
try:
    for split, rows in splits.items():
        paths = []
        for row in rows:
            paths.append(cache_audio(row, VOICE_ROOT, CACHE))
            done += 1
            if done % 500 == 0:
                cache_checkpoints.save()
                print('음원 캐시:', done, flush=True)
        data[split] = EmotionData(rows, paths)
except BaseException:
    cache_checkpoints.save()
    raise
cache_checkpoints.save()
model = load_analyzer(analyzer_config, device)

def progress(info):
    if info['updates'] == 1 or info['updates'] % 10 == 0:
        print(info, flush=True)
    if not (OUTPUT/'run.json').exists():
        write_json(OUTPUT/'run.json', {'config':config, 'identity':identity})

print('원본 음성 + 감정 라벨 지도학습 시작. 최초 전체 검증이 먼저 실행됩니다.', flush=True)
report = fit_emotion(model, data['train'], data['validation'], config, OUTPUT, identity, device, progress)
write_json(OUTPUT/'run.json', {'config':config, 'identity':identity})
write_json(OUTPUT/'training-report.json', {**report,
    'label_source':'existing automatically extracted pseudo labels',
    'selection':'lowest full validation KL including pretrained baseline at update 0',
    'duration_selection':selection})
write_json(OUTPUT/'runtime.json', {'torch':str(torch.__version__), 'cuda':torch.version.cuda,
    'gpu':torch.cuda.get_device_name(0), 'python':sys.version})
report_path = BASE / f'{OUTPUT.name}-report.zip'
with tempfile.TemporaryDirectory(prefix='fantasyvoice-emotion-report-') as temp:
    local = Path(temp)/'report.zip'
    with zipfile.ZipFile(local, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('run.json','training-report.json','runtime.json','progress.json','history.json'):
            archive.write(OUTPUT/name, OUTPUT.name+'/'+name)
    partial = report_path.with_suffix('.zip.partial')
    shutil.copy2(local, partial)
    partial.replace(report_path)
print('11 완료:', OUTPUT, '\n다음은 12_train_tts_emotion_feedback.ipynb입니다.', flush=True)
if DOWNLOAD_REPORT:
    files.download(str(report_path))
