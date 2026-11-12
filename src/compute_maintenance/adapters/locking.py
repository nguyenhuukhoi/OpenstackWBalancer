"""locking — extracted domain responsibilities."""
import logging
import os
from datetime import datetime

try:
    import fcntl
except ImportError:
    fcntl = None


def acquire_single_instance_lock(lock_file):
    """Try to take an exclusive, non-blocking lock for apply mode.

    Returns (ok, handle):
    - (True, handle): lock acquired. Keep ``handle`` referenced for the whole run
      so the lock is held; the OS releases it when the process exits.
    - (False, None): another instance already holds the lock -- caller must abort.
    """
    if fcntl is None:
        logging.error(
            "File locking is unavailable on this platform; refusing apply mode "
            "without a single-instance guard."
        )
        return False, None

    try:
        directory = os.path.dirname(lock_file) or "."
        os.makedirs(directory, exist_ok=True)
        handle = open(lock_file, "a+", encoding="utf-8")  # noqa: SIM115 - the lock handle must outlive this function (returned to the caller)
    except OSError as e:
        logging.error(
            "Could not open lock file %s (%s); refusing apply mode.",
            lock_file, e,
        )
        return False, None

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False, None

    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} acquired_at={datetime.now().isoformat()}\n")
    handle.flush()
    return True, handle


from contextlib import contextmanager

from compute_maintenance.adapters.persistence import StateError


@contextmanager
def coordinator_lock(path):
    if fcntl is None:
        raise StateError("maintenance apply requires POSIX flock; no safe lock is available")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise StateError(f"another maintenance run holds coordinator lock {path}") from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def shared_apply_lock(path, inherited_fd=None):
    """Child shares its parent's open lock description; independent runs acquire it."""
    if inherited_fd is None:
        with coordinator_lock(path) as stream:
            yield stream
        return
    if fcntl is None:
        raise StateError("POSIX flock required")
    fd = int(inherited_fd)
    expected, actual = os.stat(path), os.fstat(fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise StateError("inherited coordinator descriptor does not match lock path")
    # With an inherited descriptor this is the same open-file description.
    # Do not unlock it on child exit: the parent owns the maintenance lifetime.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    yield fd

