import io
import tarfile
from pathlib import Path

import pytest

from scripts.cache_checkpoints import CacheCheckpoints

NAME = 'audio/' + 'a' * 64 + '.pt'
TEXT = 'text/' + 'b' * 64 + '.pt'


def legacy_part(folder, entries, number=1):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'cache-part-{number:06d}.tar.gz'
    with tarfile.open(path, 'w:gz') as tar:
        for name, value in entries.items():
            member = tarfile.TarInfo(name); member.size = len(value)
            tar.addfile(member, io.BytesIO(value))
    return path


def test_legacy_restore_incremental_save_and_resume(tmp_path):
    drive, local = tmp_path / 'drive', tmp_path / 'local'
    legacy_part(drive, {NAME:b'original'})
    manager = CacheCheckpoints(local, drive)
    assert manager.restore() == 1
    assert (local / NAME).read_bytes() == b'original'
    assert manager.save() is None
    (local / TEXT).parent.mkdir(parents=True)
    (local / TEXT).write_bytes(b'new')
    part = manager.save()
    assert part.name == 'cache-part-000002.tar.gz'
    with tarfile.open(part) as tar:
        assert tar.getnames() == [TEXT]
    (drive / '.cache-part-000003.partial').write_bytes(b'incomplete')
    restored = tmp_path / 'restored'
    assert CacheCheckpoints(restored, drive).restore() == 2
    assert (restored / TEXT).read_bytes() == b'new'


def test_existing_local_cache_is_not_mistaken_for_persisted_data(tmp_path):
    local, drive = tmp_path / 'local', tmp_path / 'drive'
    (local / NAME).parent.mkdir(parents=True)
    (local / NAME).write_bytes(b'not-backed-up')
    manager = CacheCheckpoints(local, drive)
    manager.restore()
    assert manager.save() is not None
    assert manager.save() is None


@pytest.mark.parametrize('name', ['../outside.pt', '/outside.pt', 'text/../../outside.pt', 'run.py'])
def test_rejects_unsafe_archive_members(tmp_path, name):
    drive = tmp_path / 'drive'
    legacy_part(drive, {name:b'bad'})
    with pytest.raises(ValueError):
        CacheCheckpoints(tmp_path / 'local', drive).restore()
    assert not (tmp_path / 'outside.pt').exists()


def test_rejects_tar_links(tmp_path):
    drive = tmp_path / 'drive'; drive.mkdir()
    with tarfile.open(drive / 'cache-part-000001.tar.gz', 'w:gz') as tar:
        member = tarfile.TarInfo(NAME); member.type = tarfile.SYMTYPE; member.linkname = '../outside'
        tar.addfile(member)
    with pytest.raises(ValueError):
        CacheCheckpoints(tmp_path / 'local', drive).restore()


def test_failed_upload_keeps_local_changes_pending(tmp_path, monkeypatch):
    import scripts.cache_checkpoints as module
    local = tmp_path / 'local'; (local / NAME).parent.mkdir(parents=True)
    (local / NAME).write_bytes(b'retry me')
    manager = CacheCheckpoints(local, tmp_path / 'drive')
    manager.restore()
    copy = module.shutil.copyfile
    def fail(*args, **kwargs): raise OSError('Drive disconnected')
    monkeypatch.setattr(module.shutil, 'copyfile', fail)
    with pytest.raises(OSError): manager.save()
    monkeypatch.setattr(module.shutil, 'copyfile', copy)
    assert manager.save().name == 'cache-part-000001.tar.gz'
