"""08: approved MeloTTS warmup; explicit STAGE='joint' is a separate experiment."""
from pathlib import Path
import gc
import importlib
import importlib.metadata
import json
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
VOICE_ROOT = Path('/content/drive/MyDrive/Voice')
BUNDLE = BASE / 'fantasyvoice-pilot.zip'
DATASET = BASE / 'dataset-prepared-v1.zip'
PREDICTOR = BASE / 'predictor-v1/best-model.pt'
PROJECT = Path('/content/fantasyvoice-tts')
CACHE = Path('/content/fantasyvoice-tts-cache')

# First run: warmup. Joint training is a separate, explicitly selected stage.
STAGE = 'warmup'
EXPERIMENT = 'v1'
WARMUP_MODEL = BASE / 'tts-warmup-v1/best-model.pt'
BATCH_SIZE = 1
ACCUMULATION_STEPS = 4
EPOCHS = 3
LEARNING_RATE = 1e-4
DOWNLOAD_REPORT = True
OUTPUT = BASE / f'tts-{STAGE}-{EXPERIMENT}'

if STAGE not in ('warmup', 'joint'):
    raise ValueError("STAGE는 'warmup' 또는 'joint'입니다.")
for path in (BUNDLE, DATASET, PREDICTOR):
    if not path.is_file():
        raise FileNotFoundError(f'필요한 파일이 없습니다: {path}')
if STAGE == 'joint' and not WARMUP_MODEL.is_file():
    raise FileNotFoundError('TTS 준비 학습의 best-model.pt를 먼저 생성하세요.')
if not VOICE_ROOT.is_dir():
    raise FileNotFoundError(VOICE_ROOT)
with zipfile.ZipFile(BUNDLE) as archive:
    if 'src/fantasyvoice/training/tts_engine.py' not in archive.namelist():
        raise RuntimeError('08이 포함된 최신 fantasyvoice-pilot.zip으로 교체하세요.')
    for item in archive.infolist():
        if not (PROJECT / item.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
config = json.loads((PROJECT / 'config/tts.json').read_text(encoding='utf-8'))
config.update(stage=STAGE, batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS,
              epochs=EPOCHS, learning_rate=LEARNING_RATE)
print('TTS 의존성을 설치합니다. 새 T4 런타임에서 실행하세요.', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT) + '[tts]'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
    f'https://github.com/myshell-ai/MeloTTS/archive/{config["melo_revision"]}.zip'], check=True)
for package in ('torch','numpy','transformers','numba','librosa'):
    if package in sys.modules and sys.modules[package].__version__ != importlib.metadata.version(package):
        raise RuntimeError(f'{package} 이전 버전이 메모리에 있습니다. 런타임을 다시 시작하고 이 셀을 실행하세요.')
sys.path.insert(0, str(PROJECT / 'src'))
for name in list(sys.modules):
    if name == 'fantasyvoice' or name.startswith('fantasyvoice.'):
        del sys.modules[name]
importlib.invalidate_caches()

import torch
import soundfile as sf
from transformers import AutoTokenizer
from melo.models import MultiPeriodDiscriminator, DurationDiscriminator
from fantasyvoice.dataset.storage import sha256, write_json
from fantasyvoice.dataset.tts_data import KoreanFrontend, prepare_cache
from fantasyvoice.training.predictor import read_prepared
from fantasyvoice.training.tts_runtime import load_tts, load_predictor, make_optimizers, warm_start_joint
from fantasyvoice.training.tts import TTSObjective
from fantasyvoice.training.tts_engine import fit_engine
from fantasyvoice.inference.tts import synthesize

if not torch.cuda.is_available():
    raise RuntimeError('08은 실제 음성 모델 학습입니다. GPU 런타임을 선택하세요.')
device = 'cuda'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
torch.backends.cudnn.benchmark = False
splits, metadata = read_prepared(DATASET)
dataset_sha = sha256(DATASET)
print('한국어 MeloTTS 사전학습 가중치를 로딩합니다.', flush=True)
tts, hps, artifacts = load_tts(config, metadata, 'cpu')
predictor, predictor_config = load_predictor(PREDICTOR, dataset_sha, metadata, 'cpu')
tokenizer = AutoTokenizer.from_pretrained(predictor_config['model_name'],
    revision=predictor_config['model_revision'], trust_remote_code=False)
if tokenizer.pad_token_id is None:
    raise ValueError('Predictor tokenizer has no pad ID')
frontend_identity = {key:config[key] for key in ('frontend_revision','melo_revision','add_blank')}
frontend_identity['model_artifacts'] = artifacts
package_versions = {name:importlib.metadata.version(name) for name in
    ('torch','transformers','tokenizers','safetensors','huggingface-hub','numpy','numba','librosa',
     'scipy','soundfile','g2pkk','jamo','anyascii','num2words','nltk','python-mecab-ko-dic',
     'python-mecab-ko-py313' if sys.version_info >= (3,13) else 'python-mecab-ko')}
frontend_identity['versions'] = package_versions
frontend_identity['implementation_sha256'] = sha256(PROJECT / 'src/fantasyvoice/dataset/tts_data.py')
identity = {'dataset_sha256':dataset_sha, 'dataset_metadata':metadata,
    'entrypoint_sha256':sha256(PROJECT / 'scripts/colab_train_tts.py'),
    'predictor_sha256':sha256(PREDICTOR), 'pretrained_artifacts':artifacts,
    'warmup_sha256':sha256(WARMUP_MODEL) if STAGE == 'joint' else None,
    'versions':package_versions,
    'source_sha256':{p.relative_to(PROJECT).as_posix():sha256(p)
        for p in sorted((PROJECT / 'src/fantasyvoice').rglob('*.py'))}}
if (OUTPUT / 'latest.pt').exists():
    saved = torch.load(OUTPUT / 'latest.pt', map_location='cpu', weights_only=True, mmap=True)
    if saved['fingerprint'] != {'config':config, 'identity':identity, 'size':len(splits['train'])}:
        raise ValueError('설정/데이터/코드가 기존 실행과 다릅니다. 새 EXPERIMENT 이름으로 실행하세요.')
    del saved
elif OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise ValueError('출력 폴더에 체크포인트가 아닌 파일이 있습니다. 새 EXPERIMENT 이름을 지정하세요.')

print(f"학습 {len(splits['train']):,} / 검증 {len(splits['validation']):,}. 로컬 음원·텍스트 캐시를 준비합니다.", flush=True)
print('최초 준비는 Drive에서 각 음원을 읽습니다. 런타임 재시작으로 /content가 지워지면 캐시도 다시 준비합니다.', flush=True)
frontend = KoreanFrontend(config, hps['symbols'], device)
started = time.monotonic()
data = prepare_cache(splits, VOICE_ROOT, CACHE, frontend, tokenizer, metadata, frontend_identity,
    progress=lambda done,total: print(f'캐시 {done:,}/{total:,} | {(time.monotonic()-started)/60:.1f}분', flush=True))
frontend.model.to('cpu')
del frontend
gc.collect(); torch.cuda.empty_cache()

tts = tts.to(device)
tts.base.decoder_amp = config['precision'] == 'fp16'
modules = {'g':tts, 'd':MultiPeriodDiscriminator(hps['model']['use_spectral_norm']).to(device),
    # Official trainer does not pass g to its duration discriminator.
    'dur':DurationDiscriminator(hps['model']['hidden_channels'], hps['model']['hidden_channels'], 3, .1).to(device)}
if STAGE == 'joint':
    warm_start_joint(WARMUP_MODEL, modules, dataset_sha, metadata, config['model_revision'])
    modules['predictor'] = predictor.to(device)
objective = TTSObjective(modules, data['train'], config, device, tokenizer.pad_token_id)
print('실제 음원과 Predictor로 TTS-only gradient 사전 점검을 실행합니다.', flush=True)
probe = objective.gradient_preflight(data['train'][0], predictor.to(device))
print('스타일 역전파 점검:', probe, flush=True)
if STAGE == 'warmup':
    predictor.to('cpu')
    del predictor
    gc.collect(); torch.cuda.empty_cache()
optimizers, schedulers = make_optimizers(modules, config)
print('TTS 학습 시작. OOM이면 batch/누적 설정을 확인하고 새 실험으로 실행하세요.', flush=True)
started = time.monotonic()
def progress(info):
    if info['updates'] % 10 == 0 or info['updates'] == 1:
        print(f"update {info['updates']}/{info['total_updates']} | epoch {info['epoch']} | "
              f"G {info['losses']['g']:.3f} D {info['losses']['d']:.3f} | "
              f"{(time.monotonic()-started)/60:.1f}분", flush=True)
    if not (OUTPUT / 'run.json').exists():
        write_json(OUTPUT / 'run.json', {'config':config, 'identity':identity, 'preflight':probe})

result = fit_engine(modules, optimizers, schedulers, objective, len(data['train']), config,
    OUTPUT, identity, device, progress=progress,
    evaluate=lambda state:objective.evaluate(data['validation'], OUTPUT, state))
write_json(OUTPUT / 'run.json', {'config':config, 'identity':identity, 'preflight':probe})
write_json(OUTPUT / 'runtime.json', {'python':sys.version, 'torch':str(torch.__version__),
    'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0)})
write_json(OUTPUT / 'training-report.json', {**result,
    'scope':'Conditional TTS ' + STAGE + '; best means lowest full validation reconstruction mel, not listening quality',
    'preflight':probe, 'sample_rate':44100})

print('최적 모델로 text + character → Predictor → TTS 예시를 생성합니다.', flush=True)
# Release optimizer moments and discriminators before loading the text frontend.
del optimizers, schedulers, objective
for key in ('d','dur'):
    del modules[key]
gc.collect(); torch.cuda.empty_cache()
best = torch.load(OUTPUT / 'best-model.pt', map_location='cpu', weights_only=True)
tts.load_state_dict(best['modules']['g'], strict=True)
if STAGE == 'joint':
    predictor = modules['predictor']
    predictor.load_state_dict(best['modules']['predictor'], strict=True)
else:
    predictor, _ = load_predictor(PREDICTOR, dataset_sha, metadata, device)
del best
frontend = KoreanFrontend(config, hps['symbols'], 'cpu')
preview = splits['validation'][0]
wave, info = synthesize(preview['text'], preview['character_id'], frontend, tokenizer,
    predictor, tts, metadata['character_map'], device, config['seed'])
sf.write(OUTPUT / 'predicted-style-preview.wav', wave.numpy(), 44100, subtype='FLOAT')
write_json(OUTPUT / 'predicted-style-preview.json', info)

with tempfile.TemporaryDirectory(prefix='fantasyvoice-tts-report-') as temp:
    local_zip = Path(temp) / 'report.zip'
    with zipfile.ZipFile(local_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('run.json','runtime.json','progress.json','history.json','training-report.json',
                     'predicted-style-preview.wav','predicted-style-preview.json'):
            archive.write(OUTPUT / name, OUTPUT.name + '/' + name)
        for update in sorted({result['updates'], result['best_update']}):
            folder = OUTPUT / 'samples' / f'step-{update:07d}'
            for path in sorted(folder.glob('*')):
                archive.write(path, OUTPUT.name + '/' + path.relative_to(OUTPUT).as_posix())
    report_path = BASE / f'{OUTPUT.name}-report.zip'
    partial = report_path.with_suffix('.zip.partial')
    shutil.copy2(local_zip, partial)
    partial.replace(report_path)
print('완료:', OUTPUT, '\n결과 보고서:', report_path, flush=True)
if DOWNLOAD_REPORT:
    files.download(str(report_path))
