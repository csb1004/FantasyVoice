"""Paste into one Colab cell after selecting a T4 GPU runtime.

Upload the latest fantasyvoice-pilot.zip to MyDrive/FantasyVoice first.
Restart the runtime before switching from an older bundle.
"""
from pathlib import Path
import ast
import json
import zipfile
from google.colab import drive

drive.mount('/content/drive')
bundle = Path('/content/drive/MyDrive/FantasyVoice/fantasyvoice-pilot.zip')
if not bundle.is_file():
    raise FileNotFoundError(f'최신 ZIP을 먼저 올려주세요: {bundle}')

with zipfile.ZipFile(bundle) as archive:
    if 'reference/inventory.jsonl' not in archive.namelist():
        raise RuntimeError('이 ZIP은 이전 버전입니다. 최신 ZIP으로 교체해주세요.')
    notebook = json.loads(archive.read('notebooks/01_transcription_pilot.ipynb'))

# Parse all cells before starting. Execute in the shared notebook namespace so
# the listening widgets retain their callbacks and data after this cell ends.
stages = [compile(ast.parse(''.join(cell['source'])), f'pilot-stage-{index}', 'exec')
          for index, cell in enumerate(notebook['cells']) if cell['cell_type'] == 'code']
for index, stage in enumerate(stages, 1):
    print(f'[{index}/{len(stages)}] 실행 중', flush=True)
    exec(stage, globals())
print('전사 단계가 끝났습니다. 위 검수 화면에서 듣고 판정을 저장하세요.')
