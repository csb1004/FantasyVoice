"""12: TTS feedback through a completed frozen stage 11 analyzer, starting from 09."""
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
PROJECT = Path('/content/fantasyvoice-feedback')
CACHE = Path('/content/fantasyvoice-tts-cache')
CACHE_CHECKPOINTS = BASE / 'tts-cache-checkpoints'
CACHE_CHECKPOINT_EVERY = 500

# Experimental defaults, not tuned values. Change EXPERIMENT when changing any setting.
STAGE = 'joint'
EXPERIMENT = 'v2-batch'
SOURCE_MODEL = BASE / 'tts-joint-v1/best-model.pt'
EMOTION_MODEL = BASE / 'emotion-analyzer-v2-batch/best-model.pt'
BATCH_SIZE = 'auto'  # T4: 4, L4: 8; or set a positive integer.
ACCUMULATION_STEPS = 1
EPOCHS = 1
LEARNING_RATE = 1e-5
DOWNLOAD_REPORT = True
OUTPUT = BASE / f'tts-emotion-feedback-{EXPERIMENT}'
EMOTION_WEIGHT = 1.0
MAX_SECONDS = 15.0
EMOTION_VALIDATION_SAMPLES = 16

for path in (BUNDLE, DATASET, PREDICTOR, SOURCE_MODEL,
             SOURCE_MODEL.parent / 'training-report.json', SOURCE_MODEL.parent / 'run.json',
             EMOTION_MODEL, EMOTION_MODEL.parent/'training-report.json', EMOTION_MODEL.parent/'run.json'):
    if not path.is_file():
        raise FileNotFoundError(f'필요한 파일이 없습니다: {path}')
if not VOICE_ROOT.is_dir():
    raise FileNotFoundError(VOICE_ROOT)
with zipfile.ZipFile(BUNDLE) as archive:
    if not {'scripts/emotion_feedback.py', 'scripts/gpu_batches.py', 'scripts/feedback_engine.py', 'scripts/emotion_training.py', 'scripts/inference_tools.py', 'scripts/cache_checkpoints.py'}.issubset(archive.namelist()):
        raise RuntimeError('11/12가 포함된 최신 fantasyvoice-pilot.zip으로 교체하세요.')
    for item in archive.infolist():
        if not (PROJECT / item.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
source_run = json.loads((SOURCE_MODEL.parent / 'run.json').read_text(encoding='utf-8'))
config = dict(source_run['config'])
config.update(stage=STAGE, batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS,
              epochs=EPOCHS, learning_rate=LEARNING_RATE)
config['feedback'] = dict(weight=EMOTION_WEIGHT,
    max_seconds=MAX_SECONDS, validation_samples=EMOTION_VALIDATION_SAMPLES,
    predictor_frozen=True, analyzer_training='frozen_completed_stage_11')
for value in (EMOTION_WEIGHT, MAX_SECONDS):
    if not isinstance(value, (int, float)) or not 0 < value < float('inf'):
        raise ValueError('Feedback weights, learning rate and duration limit must be positive finite values')
if type(EMOTION_VALIDATION_SAMPLES) is not int or EMOTION_VALIDATION_SAMPLES < 1:
    raise ValueError('EMOTION_VALIDATION_SAMPLES must be a positive integer')
print('TTS 의존성을 설치합니다. 새 GPU 런타임에서 실행하세요. T4 메모리에 들어간다고 보장하지 않습니다.', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT) + '[tts]'], check=True)
# Match torchaudio exactly to the installed torch release; fail on mismatch below.
torch_release = importlib.metadata.version('torch').split('+')[0]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'funasr==1.4.16',
                'torchaudio==' + torch_release], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
    f'https://github.com/myshell-ai/MeloTTS/archive/{config["melo_revision"]}.zip'], check=True)
for package in ('torch','torchaudio','numpy','transformers','numba','librosa'):
    if package in sys.modules and sys.modules[package].__version__ != importlib.metadata.version(package):
        raise RuntimeError(f'{package} 이전 버전이 메모리에 있습니다. 런타임을 다시 시작하고 이 셀을 실행하세요.')
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.modules.pop('emotion_feedback', None)
sys.modules.pop('emotion_training', None)
sys.modules.pop('gpu_batches', None)
sys.modules.pop('feedback_engine', None)
sys.modules.pop('inference_tools', None)
sys.modules.pop('cache_checkpoints', None)
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
from fantasyvoice.training.tts_runtime import load_tts, load_predictor
from feedback_engine import fit_engine
from fantasyvoice.inference.tts import synthesize
from inference_tools import validate_completed_joint, verify_inference_sources, load_joint_weights
from emotion_feedback import load_analyzer
from gpu_batches import batch_preset, BatchedFeedbackObjective
from emotion_training import validate_emotion_checkpoint, select_duration
from cache_checkpoints import CacheCheckpoints

if not torch.cuda.is_available():
    raise RuntimeError('12는 감정 피드백을 사용하는 TTS 학습입니다. GPU 런타임을 선택하세요.')
device = 'cuda'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
if BATCH_SIZE == 'auto':
    props = torch.cuda.get_device_properties(0)
    BATCH_SIZE, _ = batch_preset(props.name, props.total_memory / 2**30, 12)
if type(BATCH_SIZE) is not int or BATCH_SIZE < 1:
    raise ValueError("BATCH_SIZE must be 'auto' or a positive integer")
print(f'실제 배치 {BATCH_SIZE}, 누적 {ACCUMULATION_STEPS}, 유효 배치 {BATCH_SIZE*ACCUMULATION_STEPS}', flush=True)
config['batch_size'] = BATCH_SIZE

random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
torch.backends.cudnn.benchmark = False
splits, metadata = read_prepared(DATASET)
dataset_sha = sha256(DATASET)
# Verify the completed source checkpoint before cache preparation or optimizer setup.
source_checkpoint = torch.load(SOURCE_MODEL, map_location='cpu', weights_only=True, mmap=True)
source_report = json.loads((SOURCE_MODEL.parent / 'training-report.json').read_text(encoding='utf-8'))
handoff = validate_completed_joint(source_checkpoint, source_report, source_run, sha256(PREDICTOR))
source_identity = source_checkpoint['fingerprint']['identity']
if source_identity['dataset_sha256'] != dataset_sha or source_identity['dataset_metadata'] != metadata:
    raise ValueError('09 checkpoint and prepared dataset differ')
verify_inference_sources(PROJECT, source_identity)
if 'feedback' in source_run['config']:
    raise ValueError('Use completed 09 as source; resume 12 through its existing OUTPUT/latest.pt')
splits, selection = select_duration(splits, MAX_SECONDS)
emotion_run = json.loads((EMOTION_MODEL.parent/'run.json').read_text(encoding='utf-8'))
emotion_report = json.loads((EMOTION_MODEL.parent/'training-report.json').read_text(encoding='utf-8'))
emotion_saved = torch.load(EMOTION_MODEL, map_location='cpu', weights_only=True, mmap=True)
validate_emotion_checkpoint(emotion_saved, emotion_report, emotion_run['identity'])
if emotion_run['config'] != emotion_saved['fingerprint']['config']:
    raise ValueError('11 run/checkpoint config mismatch')
if (emotion_run['identity']['dataset_sha256'] != dataset_sha
        or emotion_run['identity']['dataset_metadata'] != metadata
        or emotion_run['identity']['adapter_sha256'] != sha256(PROJECT/'scripts/emotion_feedback.py')):
    raise ValueError('11 analyzer dataset or adapter differs from current experiment')
del emotion_saved
print('09 completion:', handoff, 'duration selection:', selection, flush=True)
tts, hps, artifacts = load_tts(config, metadata, 'cpu')
predictor, predictor_config = load_predictor(PREDICTOR, dataset_sha, metadata, 'cpu')
load_joint_weights(tts, predictor, source_checkpoint)
predictor.requires_grad_(False)
del source_checkpoint
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
    'entrypoint_sha256':sha256(PROJECT / 'scripts/colab_emotion_feedback.py'),
    'handoff_check_sha256':sha256(PROJECT / 'scripts/inference_tools.py'),
    'feedback_code_sha256':sha256(PROJECT / 'scripts/emotion_feedback.py'),
    'analyzer_handoff_sha256':sha256(PROJECT / 'scripts/emotion_training.py'),
    'batch_helper_sha256':sha256(PROJECT/'scripts/gpu_batches.py'),
    'engine_sha256':sha256(PROJECT/'scripts/feedback_engine.py'),
    'trained_analyzer_sha256':sha256(EMOTION_MODEL),
    'feedback_versions':{k:importlib.metadata.version(k) for k in ('funasr','torchaudio','omegaconf')},
    'duration_selection':selection,
    'cache_checkpoint_sha256':sha256(PROJECT / 'scripts/cache_checkpoints.py'),
    'predictor_sha256':sha256(PREDICTOR), 'pretrained_artifacts':artifacts,
    'source_joint_sha256':sha256(SOURCE_MODEL),
    'versions':package_versions,
    'source_sha256':{p.relative_to(PROJECT).as_posix():sha256(p)
        for p in sorted((PROJECT / 'src/fantasyvoice').rglob('*.py'))}}
analyzer_config = metadata['analyzer_config']['extraction']['emotion']
identity['emotion_model'] = analyzer_config
if (OUTPUT / 'latest.pt').exists():
    saved = torch.load(OUTPUT / 'latest.pt', map_location='cpu', weights_only=True, mmap=True)
    if saved['fingerprint'] != {'config':config, 'identity':identity, 'size':len(splits['train'])}:
        raise ValueError('설정/데이터/코드가 기존 실행과 다릅니다. 새 EXPERIMENT 이름으로 실행하세요.')
    del saved
elif OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise ValueError('출력 폴더에 체크포인트가 아닌 파일이 있습니다. 새 EXPERIMENT 이름을 지정하세요.')

print(f"학습 {len(splits['train']):,} / 검증 {len(splits['validation']):,}. 로컬 음원·텍스트 캐시를 준비합니다.", flush=True)
if type(CACHE_CHECKPOINT_EVERY) is not int or CACHE_CHECKPOINT_EVERY < 1:
    raise ValueError('CACHE_CHECKPOINT_EVERY must be a positive integer')
cache_checkpoints = CacheCheckpoints(CACHE, CACHE_CHECKPOINTS)
restored = cache_checkpoints.restore()
print(f'08/09 Drive 캐시 복원: {restored:,}개 파일. 현재 해시에 맞는 캐시를 재사용합니다.', flush=True)
frontend = KoreanFrontend(config, hps['symbols'], device)
started = time.monotonic()
last_cache_checkpoint = 0

def cache_progress(done, total):
    global last_cache_checkpoint
    print(f'캐시 {done:,}/{total:,} | {(time.monotonic()-started)/60:.1f}분', flush=True)
    if done - last_cache_checkpoint >= CACHE_CHECKPOINT_EVERY:
        cache_checkpoints.save()
        last_cache_checkpoint = done

try:
    data = prepare_cache(splits, VOICE_ROOT, CACHE, frontend, tokenizer, metadata, frontend_identity,
                         progress=cache_progress)
except BaseException:
    # On an ordinary interruption preserve complete local cache entries. A hard
    # runtime termination can only recover the previously published tar parts.
    try:
        cache_checkpoints.save()
    except Exception as cache_error:
        print(f'마지막 캐시 저장 실패; 기존 Drive 파트는 보존됩니다: {cache_error}', flush=True)
    raise
cache_checkpoints.save()
print('캐시 준비 및 최종 증분 저장 완료.', flush=True)
frontend.model.to('cpu')
del frontend
gc.collect(); torch.cuda.empty_cache()

tts = tts.to(device)
tts.base.decoder_amp = config['precision'] == 'fp16'
modules = {'g':tts, 'd':MultiPeriodDiscriminator(hps['model']['use_spectral_norm']).to(device),
    # Official trainer does not pass g to its duration discriminator.
    'dur':DurationDiscriminator(hps['model']['hidden_channels'], hps['model']['hidden_channels'], 3, .1).to(device)}
modules['predictor'] = predictor.to(device)
print('Loading completed 11 analyzer; freezing its weights for TTS feedback.', flush=True)
analyzer = load_analyzer(analyzer_config, device)
emotion_saved = torch.load(EMOTION_MODEL, map_location='cpu', weights_only=True, mmap=True)
analyzer.load_state_dict(emotion_saved['model'], strict=True)
analyzer.eval().requires_grad_(False)
del emotion_saved
objective = BatchedFeedbackObjective(modules, data['train'], config, device, tokenizer.pad_token_id, emotion=analyzer)
# Preflight is a check, not an optimizer update. Engine restores RNG on resume.
probe = objective.feedback_preflight(data['train'][0])
print('Emotion-only gradient preflight:', probe, flush=True)
# Only generator and existing GAN discriminators have optimizers in stage 12.
optimizers = {key:torch.optim.AdamW(
    [p for p in modules[key].parameters() if p.requires_grad], lr=LEARNING_RATE,
    betas=tuple(config['betas']), eps=config['eps'], weight_decay=config['weight_decay'])
    for key in ('g','d','dur')}
schedulers = {key:torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config['lr_decay'])
              for key,opt in optimizers.items()}
print('12 training: existing TTS loss + generated-emotion KL; analyzer stays frozen.', flush=True)
started = time.monotonic()
def progress(info):
    if info['updates'] % 10 == 0 or info['updates'] == 1:
        print(f"update {info['updates']}/{info['total_updates']} | epoch {info['epoch']} | "
              f"batch {info['microbatch_size']} | G {info['losses']['g']:.3f} D {info['losses']['d']:.3f} | "
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
    'scope':'Emotion feedback experiment; best is full validation mel; emotion metrics use frozen stage 11 evaluator; labels are pseudo labels',
    'preflight':probe, 'sample_rate':44100, 'source_joint_handoff':handoff, 'duration_selection':selection})

print('최적 모델로 text + character → Predictor → TTS 예시를 생성합니다.', flush=True)
# Release optimizer moments and discriminators before loading the text frontend.
del optimizers, schedulers, objective, analyzer
for key in ('d','dur'):
    del modules[key]
gc.collect(); torch.cuda.empty_cache()
best = torch.load(OUTPUT / 'best-model.pt', map_location='cpu', weights_only=True)
tts.load_state_dict(best['modules']['g'], strict=True)
predictor = modules['predictor']
predictor.load_state_dict(best['modules']['predictor'], strict=True)
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
print('10번 JOINT_DIR을 이 결과 폴더로 바꾸면 추론할 수 있습니다.', flush=True)
print('완료:', OUTPUT, '\n결과 보고서:', report_path, flush=True)
if DOWNLOAD_REPORT:
    files.download(str(report_path))
