"""Drive tar parts compatible with the user's resumable-cache-v2 notebook.

Cache keys remain owned by tts_data.py; no training fingerprints change here.
Use one writer per Drive checkpoint directory.
"""
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile


MEMBER = re.compile(r'(audio|text)/[0-9a-f]{64}\.pt\Z')
PART = re.compile(r'cache-part-([0-9]+)\.tar\.gz\Z')


class CacheCheckpoints:
    def __init__(self, local, drive, log=print):
        self.local, self.drive, self.log = Path(local), Path(drive), log
        self.local.mkdir(parents=True, exist_ok=True)
        self.drive.mkdir(parents=True, exist_ok=True)
        self.saved = {}
        self.next_index = 1

    def _parts(self):
        return sorted((p for p in self.drive.iterdir() if PART.fullmatch(p.name)),
                      key=lambda p:int(PART.fullmatch(p.name)[1]))

    @staticmethod
    def _signature(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    def _target(self, name):
        if not MEMBER.fullmatch(name):
            raise ValueError(f'Unexpected cache archive member: {name}')
        target = self.local / name
        if target.is_symlink() or not target.resolve().is_relative_to(self.local.resolve()):
            raise ValueError(f'Cache path escapes local directory: {name}')
        return target

    def restore(self):
        restored = set()
        parts = self._parts()
        for part in parts:
            with tarfile.open(part, 'r:gz') as archive:
                for member in archive:
                    target = self._target(member.name)
                    if not member.isfile():
                        raise ValueError(f'Non-file cache member: {member.name}')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = None
                    try:
                        with archive.extractfile(member) as source, tempfile.NamedTemporaryFile(
                                dir=target.parent, suffix='.partial', delete=False) as dest:
                            temporary = Path(dest.name)
                            shutil.copyfileobj(source, dest)
                        if temporary.stat().st_size != member.size:
                            raise ValueError(f'Incomplete cache member: {member.name}')
                        os.replace(temporary, target)
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
                    restored.add(member.name)
            self.log(f'캐시 복원: {part.name}', flush=True)
        # Existing local files absent from the archives still need a backup.
        self.saved = {name:self._signature(self.local / name) for name in restored}
        self.next_index = max((int(PART.fullmatch(p.name)[1]) for p in parts), default=0) + 1
        return len(restored)

    def save(self):
        current = {}
        for folder in ('audio', 'text'):
            for path in (self.local / folder).glob('*.pt'):
                name = path.relative_to(self.local).as_posix()
                if MEMBER.fullmatch(name):
                    self._target(name)
                    current[name] = self._signature(path)
        changed = sorted(name for name,sig in current.items() if self.saved.get(name) != sig)
        if not changed:
            return None
        final = self.drive / f'cache-part-{self.next_index:06d}.tar.gz'
        partial = self.drive / f'.cache-part-{self.next_index:06d}.partial'
        if final.exists():
            raise FileExistsError('Another cache writer published this part; use one runtime at a time')
        with tempfile.TemporaryDirectory(dir=self.local.parent, prefix='tts-cache-export-') as temp:
            local_archive = Path(temp) / final.name
            with tarfile.open(local_archive, 'w:gz', compresslevel=1) as archive:
                for name in changed:
                    path = self._target(name)
                    archive.add(path, arcname=name, recursive=False)
                    if self._signature(path) != current[name]:
                        raise RuntimeError(f'Cache file changed during checkpoint: {name}')
            shutil.copyfile(local_archive, partial)
            if final.exists():
                raise FileExistsError('Concurrent cache writer detected')
            os.replace(partial, final)
        self.saved.update({name:current[name] for name in changed})
        self.next_index += 1
        self.log(f'캐시 저장: {final.name} ({len(changed):,}개 파일)', flush=True)
        return final
