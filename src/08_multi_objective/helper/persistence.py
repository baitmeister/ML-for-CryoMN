"""Serialized CSV publication with durable, roll-forward recovery.

Writers hold the repository lock from reading inputs through publication. A
prepared journal is a commit decision: recovery finishes it before another
workflow may read the tables. Unrelated edits cause a conflict, never rollback.
"""
from contextlib import contextmanager
from functools import wraps
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid

from .paths import PROJECT_ROOT

_mutex = threading.RLock()
_local = threading.local()


class PersistenceConflict(ValueError):
    pass


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    path = Path(path)
    return digest_bytes(path.read_bytes()) if path.exists() else None


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


@contextmanager
def writer_lock():
    # Outside campaign outputs: even a failed preflight must leave them untouched.
    key = digest_bytes(str(PROJECT_ROOT.resolve()).encode())[:20]
    with _mutex:
        if getattr(_local, 'locked', False):
            yield
            return
        with (Path(tempfile.gettempdir()) / f'cryomn-v2-{key}.lock').open('a+b') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            _local.locked = True
            try:
                yield
            finally:
                _local.locked = False
                fcntl.flock(handle, fcntl.LOCK_UN)


def serialized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with writer_lock():
            return function(*args, **kwargs)
    return wrapped


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path):
    missing = []
    current = Path(path)
    while not current.exists():
        missing.append(current)
        current = current.parent
    Path(path).mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        _sync_directory(directory.parent)


def publish_bytes(destination, data, *, frozen=False):
    """Atomic single-file publish; hard-link creation cannot clobber a winner."""
    destination = Path(destination)
    _mkdir_durable(destination.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.' + destination.name,
                                         suffix='.staged', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if frozen:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.read_bytes() != data:
                    raise PersistenceConflict(f'Refusing to overwrite frozen artifact: {destination}')
        else:
            os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def journal_root(formulations_path):
    return Path(formulations_path).resolve().parent / '.v2-transactions'


def _finish(directory, manifest):
    # Validate the entire remaining write set before publishing any more files.
    for entry in manifest['outputs']:
        payload = directory / entry['payload']
        if file_hash(payload) != entry['after']:
            raise PersistenceConflict(f'Corrupt transaction payload: {payload}')
        actual = file_hash(entry['path'])
        allowed = {entry['before'], entry['after']}
        if actual not in allowed:
            raise PersistenceConflict(f'Interrupted commit conflicts with external edit: {entry["path"]}')
    for entry in manifest['outputs']:
        if file_hash(entry['path']) != entry['after']:
            publish_bytes(entry['path'], (directory / entry['payload']).read_bytes(),
                          frozen=entry['frozen'])
    manifest['state'] = 'committed'
    publish_bytes(directory / 'manifest.json', json_bytes(manifest))


@serialized
def recover_transactions(root, *, read_only=False):
    for path in sorted(Path(root).glob('*/manifest.json')):
        manifest = json.loads(path.read_text())
        if manifest['state'] == 'prepared':
            if read_only:
                raise PersistenceConflict('Interrupted commit requires recovery before validation: ' + str(path))
            _finish(path.parent, manifest)


@serialized
def commit_files(root, outputs, *, inputs=None, label='v2 commit'):
    """outputs is (path, bytes, frozen) tuples. Retain payloads for recovery/audit."""
    recover_transactions(root)
    inputs = inputs or {}
    for path, expected in inputs.items():
        if file_hash(path) != expected:
            raise PersistenceConflict(f'Input changed during workflow: {path}')
    entries = []
    seen = set()
    for index, (path, data, frozen) in enumerate(outputs):
        path = str(Path(path).resolve())
        if path in seen:
            raise ValueError('Duplicate commit destination: ' + path)
        seen.add(path)
        before, after = file_hash(path), digest_bytes(data)
        if frozen and before is not None and before != after:
            raise PersistenceConflict('Refusing to overwrite frozen artifact: ' + path)
        entries.append(dict(path=path, before=before, after=after, frozen=frozen,
                            payload=f'{index:04d}.payload'))
    directory = Path(root) / uuid.uuid4().hex
    _mkdir_durable(directory)
    for entry, (_, data, _) in zip(entries, outputs):
        publish_bytes(directory / entry['payload'], data, frozen=True)
    manifest = dict(version=1, state='prepared', label=label, inputs=inputs, outputs=entries)
    publish_bytes(directory / 'manifest.json', json_bytes(manifest), frozen=True)
    _sync_directory(directory.parent)
    _finish(directory, manifest)
    return directory / 'manifest.json'
