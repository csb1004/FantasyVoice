"""Refresh only the listening UI in an already configured Colab session."""
from pathlib import Path
import importlib
import json
import zipfile

PROJECT = Path(globals().get('PROJECT', '/content/fantasyvoice-pilot'))
VOICE_ROOT = Path(globals().get('VOICE_ROOT', '/content/drive/MyDrive/Voice'))
OUTPUT = Path(globals().get('OUTPUT', '/content/drive/MyDrive/FantasyVoice/pilot-v1'))
bundle = Path('/content/drive/MyDrive/FantasyVoice/fantasyvoice-pilot.zip')
with zipfile.ZipFile(bundle) as archive:
    for name in ['src/fantasyvoice/dataset/review.py', 'notebooks/01_transcription_pilot.ipynb']:
        target = PROJECT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
import fantasyvoice.dataset.review as review_module
importlib.invalidate_caches()
importlib.reload(review_module)
from fantasyvoice.dataset.storage import read_jsonl
notebook = json.loads((PROJECT / 'notebooks/01_transcription_pilot.ipynb').read_text(encoding='utf-8'))
cell = next(''.join(c['source']) for c in notebook['cells']
            if c['cell_type'] == 'code' and 'def show(change=None)' in ''.join(c['source']))
exec(compile(cell, 'review-ui', 'exec'), globals())
