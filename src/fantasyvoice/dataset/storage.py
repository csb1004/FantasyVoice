"""UTF-8 snapshots and source path boundaries."""
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import tempfile
import unicodedata


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audio(root, relative):
    path = Path(relative)
    if path.is_absolute() or PureWindowsPath(relative).drive or '..' in path.parts:
        raise ValueError(f'Unsafe audio path: {relative}')
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Audio path leaves root: {relative}')
    if not target.exists():
        current = root
        for part in path.parts:
            candidate = current / part
            if not candidate.exists() and current.is_dir():
                matches = [p for p in current.iterdir()
                           if unicodedata.normalize('NFC', p.name) == unicodedata.normalize('NFC', part)]
                if len(matches) > 1:
                    raise ValueError(f'Ambiguous normalized audio path: {relative}')
                if matches:
                    candidate = matches[0]
            current = candidate.resolve()
            if not current.is_relative_to(root):
                raise ValueError(f'Audio path leaves root: {relative}')
        target = current
    return target


def check_output(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError('Output must be outside the original Voice directory')


def atomic_text(path, text):
    """One writer per output directory. Keep the old snapshot until replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                              for row in rows))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]
