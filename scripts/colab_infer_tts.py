"""10 cell 1: load the completed joint model once; no optimizer or audio dataset."""
from pathlib import Path
import gc
import importlib
import importlib.metadata
import json
import subprocess
import sys
import zipfile

from google.colab import drive, files

drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/FantasyVoice')
BUNDLE = BASE / 'fantasyvoice-pilot.zip'
JOINT_DIR = BASE / 'tts-joint-v1'  # Change to the completed 09 experiment.
PREDICTOR_CONFIG_MODEL = BASE / 'predictor-v1/best-model.pt'
OUTPUT = BASE / 'inference-v1'
PROJECT = Path('/content/fantasyvoice-inference')
BEST = JOINT_DIR / 'best-model.pt'

for path in (BUNDLE, BEST, PREDICTOR_CONFIG_MODEL, JOINT_DIR / 'run.json', JOINT_DIR / 'training-report.json'):
    if not path.is_file():
        raise FileNotFoundError(f'필요한 파일이 없습니다: {path}')
with zipfile.ZipFile(BUNDLE) as archive:
    if 'scripts/inference_tools.py' not in archive.namelist():
        raise RuntimeError('10이 포함된 최신 fantasyvoice-pilot.zip으로 교체하세요.')
    for item in archive.infolist():
        if not (PROJECT / item.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
# Install only the dependencies/source required by the saved training configuration.
run = json.loads((JOINT_DIR / 'run.json').read_text(encoding='utf-8'))
config = run['config']
if config.get('stage') != 'joint':
    raise ValueError('09의 tts-joint 결과 폴더를 지정하세요.')
print('추론 의존성을 설치합니다. 학습/음원 캐시 준비는 실행하지 않습니다.', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT) + '[tts]'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
    f'https://github.com/myshell-ai/MeloTTS/archive/{config["melo_revision"]}.zip'], check=True)
for package in ('torch','numpy','transformers','numba','librosa'):
    if package in sys.modules and sys.modules[package].__version__ != importlib.metadata.version(package):
        raise RuntimeError(f'{package} 이전 버전이 메모리에 있습니다. 런타임을 다시 시작하세요.')
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'scripts'))
for name in list(sys.modules):
    if name == 'inference_tools' or name == 'fantasyvoice' or name.startswith('fantasyvoice.'):
        del sys.modules[name]
importlib.invalidate_caches()

import torch
from IPython.display import Audio, display
from transformers import AutoTokenizer
from fantasyvoice.dataset.storage import sha256
from fantasyvoice.dataset.tts_data import KoreanFrontend
from fantasyvoice.training.tts_runtime import load_tts, load_predictor
from fantasyvoice.inference.tts import synthesize
from inference_tools import validate_completed_joint, verify_inference_sources, load_joint_weights, save_generation

saved = torch.load(BEST, map_location='cpu', weights_only=True, mmap=True)
report = json.loads((JOINT_DIR / 'training-report.json').read_text(encoding='utf-8'))
completion = validate_completed_joint(saved, report, run, sha256(PREDICTOR_CONFIG_MODEL))
identity = saved['fingerprint']['identity']
verify_inference_sources(PROJECT, identity)
metadata = identity['dataset_metadata']
character_map = metadata['character_map']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('09 완료 확인:', completion, flush=True)
print('추론 장치:', torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU (생성이 느릴 수 있습니다)', flush=True)
tts, hps, artifacts = load_tts(config, metadata, 'cpu')
predictor, predictor_config = load_predictor(PREDICTOR_CONFIG_MODEL, identity['dataset_sha256'], metadata, 'cpu')
load_joint_weights(tts, predictor, saved)
del saved
tts.to(device); predictor.to(device)
tokenizer = AutoTokenizer.from_pretrained(predictor_config['model_name'],
    revision=predictor_config['model_revision'], trust_remote_code=False)
frontend = KoreanFrontend(config, hps['symbols'], device)
model_info = {'checkpoint_sha256':sha256(BEST), 'joint_directory':JOINT_DIR.name,
    **completion, 'dataset_sha256':identity['dataset_sha256'],
    'inference_script_sha256':sha256(PROJECT / 'scripts/colab_infer_tts.py'),
    'export_helper_sha256':sha256(PROJECT / 'scripts/inference_tools.py'),
    'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','soundfile')},
    'device':device, 'emotion_order':['angry','disgusted','fearful','happy','neutral','sad','surprised']}
gc.collect()
if device == 'cuda':
    torch.cuda.empty_cache()
print('사용 가능한 캐릭터:', ', '.join(sorted(character_map)), flush=True)


def generate_voice(text, character, seed=42, download=True):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('SEED는 0 이상 2**63 미만의 정수여야 합니다.')
    print(f'음성 생성: {character}', flush=True)
    wave, info = synthesize(text, character, frontend, tokenizer, predictor, tts,
                            character_map, device, seed)
    result = save_generation(OUTPUT, wave, info, model_info)
    if result['preview'] is not None:
        display(Audio(data=result['preview'].read_bytes()))
    else:
        print('생성 결과가 무음입니다. 원본 WAV와 생성 정보는 저장했습니다.', flush=True)
    print('원본 WAV:', result['raw'], '\n생성 정보:', result['metadata'], flush=True)
    if download:
        files.download(str(result['archive']))
    return result


print('로딩 완료. 아래 생성 셀에서 TEXT와 CHARACTER를 바꿔 실행하세요.', flush=True)
