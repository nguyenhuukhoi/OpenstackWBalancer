"""execution.cooldown — extracted domain responsibilities."""
import json
import logging
import math
import os
import tempfile
import time

import compute_maintenance.config as _config
import compute_maintenance.domain.errors as _errors


def load_active_cooldown_entries(raw, max_age, label, now=None):
    if not isinstance(raw, dict):
        logging.warning("Cooldown %s data is not a JSON object. Ignoring it.", label)
        return {}

    if now is None:
        now = time.time()

    loaded = {}
    for vm_id, recorded_at in raw.items():
        try:
            recorded_at = float(recorded_at)
        except (TypeError, ValueError):
            logging.debug("Ignore invalid %s cooldown timestamp for VM %s: %r", label, vm_id, recorded_at)
            continue

        if not math.isfinite(recorded_at):
            logging.debug("Ignore non-finite %s cooldown timestamp for VM %s", label, vm_id)
            continue
        if recorded_at > now:
            # The coordinator clock stepped backwards (NTP). Dropping the entry
            # would let a just-moved VM bounce straight back; keep it active as
            # if it had been recorded now.
            logging.warning(
                "Future %s cooldown timestamp for VM %s (%.0fs ahead); clock stepped back? "
                "Keeping it active from now.", label, vm_id, recorded_at - now,
            )
            recorded_at = now

        if now - recorded_at < max_age:
            loaded[vm_id] = recorded_at

    return loaded


class CooldownStore:
    """One invocation owns its persistent cooldown data; planners receive snapshots."""
    def __init__(self, path=None):
        self.path = path
        self.moved = {}
        self.failed = {}
        # Short cooldown after a planned move was rejected at revalidation.
        self.rejected = {}

    def load(self, filename=None):
        # state belongs to this store

        if filename is None:
            filename = self.path or _config.COOLDOWN_STATE_FILE

        try:
            if not os.path.exists(filename):
                self.moved = {}
                self.failed = {}
                self.rejected = {}
                return

            with open(filename, 'r', encoding='utf-8') as f:
                raw = json.load(f)

            if not isinstance(raw, dict):
                raise _errors.ConfigError(
                    f"Cooldown state file {filename} is not a JSON object; refusing to ignore it."
                )

            now = time.time()
            if "moved" in raw or "failed" in raw:
                moved_raw = raw.get("moved", {})
                failed_raw = raw.get("failed", {})
                rejected_raw = raw.get("rejected", {})
            else:
                # Backward compatibility with the old flat {vm_id: moved_at} format.
                moved_raw = raw
                failed_raw = {}
                rejected_raw = {}

            if (not isinstance(moved_raw, dict) or not isinstance(failed_raw, dict)
                    or not isinstance(rejected_raw, dict)):
                raise _errors.ConfigError(
                    f"Cooldown state file {filename} has malformed moved/failed sections."
                )

            self.moved = load_active_cooldown_entries(
                moved_raw, _config.VM_COOLDOWN_SECONDS, "moved", now=now
            )
            self.failed = load_active_cooldown_entries(
                failed_raw, _config.VM_FAILED_MOVE_COOLDOWN_SECONDS, "failed-move", now=now
            )
            self.rejected = load_active_cooldown_entries(
                rejected_raw, _config.VM_REJECTED_MOVE_COOLDOWN_SECONDS, "rejected-move", now=now
            )
            logging.info(
                "Loaded %d moved and %d failed-move VM cooldown entries from %s.",
                len(self.moved),
                len(self.failed),
                filename,
            )
        except FileNotFoundError:
            self.moved, self.failed, self.rejected = {}, {}, {}
        except _errors.ConfigError:
            self.moved, self.failed, self.rejected = {}, {}, {}
            raise
        except Exception as e:
            self.moved, self.failed, self.rejected = {}, {}, {}
            raise _errors.ConfigError(
                f"Failed to load cooldown state from {filename}: {e}"
            ) from e

    def save(self, filename=None):
        if filename is None:
            filename = self.path or _config.COOLDOWN_STATE_FILE
        tmp_file = None
        try:
            self.prune()
            directory = os.path.dirname(filename) or "."
            os.makedirs(directory, exist_ok=True)

            # A unique same-directory temporary file keeps os.replace atomic while
            # avoiding collisions with a stale .tmp file from a killed process.
            fd, tmp_file = tempfile.mkstemp(
                prefix=f".{os.path.basename(filename)}.", suffix=".tmp", dir=directory
            )
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "moved": self.moved,
                        "failed": self.failed,
                        "rejected": self.rejected,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, filename)
            tmp_file = None
            # Make the rename itself durable across a power loss.
            try:
                dir_fd = os.open(directory, os.O_RDONLY)
            except OSError:
                dir_fd = None  # not supported on this platform (e.g. Windows)
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)

            logging.debug(
                "Saved %d moved and %d failed-move VM cooldown entries to %s.",
                len(self.moved),
                len(self.failed),
                filename,
            )
            return True
        except Exception as e:  # noqa: BLE001 - any write failure returns False; callers fail closed
            logging.warning("Failed to save cooldown state to %s: %s", filename, e)
            return False
        finally:
            if tmp_file:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass

    def prune(self, now=None):
        if now is None:
            now = time.time()

        expired = [
            vm_id for vm_id, moved_at in self.moved.items()
            if now - moved_at >= _config.VM_COOLDOWN_SECONDS
        ]
        failed_expired = [
            vm_id for vm_id, failed_at in self.failed.items()
            if now - failed_at >= _config.VM_FAILED_MOVE_COOLDOWN_SECONDS
        ]

        for vm_id in expired:
            self.moved.pop(vm_id, None)
        for vm_id in failed_expired:
            self.failed.pop(vm_id, None)
        for vm_id in [
            vm for vm, at in self.rejected.items()
            if now - at >= _config.VM_REJECTED_MOVE_COOLDOWN_SECONDS
        ]:
            self.rejected.pop(vm_id, None)

        if expired:
            logging.debug("Pruned %d expired moved VM cooldown entries.", len(expired))
        if failed_expired:
            logging.debug("Pruned %d expired failed-move VM cooldown entries.", len(failed_expired))

    def remaining(self, vm_id, now=None):
        """Return seconds of cooldown still active for a VM (0 if none).

        Considers both the post-move cooldown and the failed-move cooldown and
        returns the larger remaining value, so a VM is skippable while either is
        still active.
        """
        if now is None:
            now = time.time()

        remaining = 0.0
        moved_at = self.moved.get(vm_id)
        if moved_at is not None:
            remaining = max(remaining, _config.VM_COOLDOWN_SECONDS - (now - moved_at))
        failed_at = self.failed.get(vm_id)
        if failed_at is not None:
            remaining = max(remaining, _config.VM_FAILED_MOVE_COOLDOWN_SECONDS - (now - failed_at))
        rejected_at = self.rejected.get(vm_id)
        if rejected_at is not None:
            remaining = max(remaining, _config.VM_REJECTED_MOVE_COOLDOWN_SECONDS - (now - rejected_at))
        return remaining

default_store = CooldownStore()  # compatibility for library callers without a config


def store_for(cfg):
    return getattr(cfg, "cooldown_store", default_store)


# Compatibility API for existing library callers; CLI uses its own store.
def load_cooldown_state(filename=None):
    return default_store.load(filename)


def save_cooldown_state(filename=None):
    return default_store.save(filename)


def prune_expired_cooldowns(now=None):
    return default_store.prune(now)


def vm_cooldown_remaining(vm_id, now=None):
    return default_store.remaining(vm_id, now)
