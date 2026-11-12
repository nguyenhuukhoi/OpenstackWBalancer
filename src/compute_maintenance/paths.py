"""Default on-disk locations (FHS): config in /etc/wbalancer, state in
/var/lib/wbalancer, logs in /var/log/wbalancer, locks in /var/lock.

Versions up to 1.0.0rc8 used the same layout with a "-v1" suffix
(/var/lib/wbalancer-v1, /var/log/wbalancer-v1, /var/lock/wbalancer-v1-*.lock).
legacy_state_message() detects state left in the old place, so a pending
migration journal or maintenance checkpoint is never silently ignored.

Standard library only.
"""
import os

CONFIG_FILE = "/etc/wbalancer/wbalancer.conf"
# rc12: environment variable that replaces CONFIG_FILE as the default config
# file of both CLIs (wbalancer --config, rolling-maint --wlb-config); an
# explicit option still wins. For a non-root run or a test that must not read
# the host's root-only /etc file.
CONFIG_FILE_ENV = "WLB_CONFIG_PATH"

STATE_DIR = "/var/lib/wbalancer"
COOLDOWN_FILE = STATE_DIR + "/vm-cooldown.json"
JOURNAL_FILE = STATE_DIR + "/operations/migration-journal.json"   # next to the cooldown file
MAINT_STATE_DIR = STATE_DIR + "/maintenance"

LOG_DIR = "/var/log/wbalancer"
EVENTS_FILE = LOG_DIR + "/migrations.jsonl"
MAINT_LOG_DIR = LOG_DIR + "/maintenance"

COORDINATOR_LOCK = "/var/lock/wbalancer-maint.lock"
BALANCER_LOCK = "/var/lock/wbalancer-balancer.lock"

LEGACY_STATE_DIR = "/var/lib/wbalancer-v1"


def default_config_file():
    """The config file used when no --config / --wlb-config is given:
    $WLB_CONFIG_PATH when set and not empty, else /etc/wbalancer/wbalancer.conf."""
    return os.environ.get(CONFIG_FILE_ENV) or CONFIG_FILE


def _has_state(base):
    """True when ``base`` holds a cooldown file, a journal or a checkpoint."""
    if os.path.isfile(os.path.join(base, "vm-cooldown.json")):
        return True
    if os.path.isfile(os.path.join(base, "operations", "migration-journal.json")):
        return True
    try:
        return any(entry.is_file() for entry in os.scandir(os.path.join(base, "maintenance")))
    except OSError:
        return False


def legacy_state_message(state_dir=None, legacy_dir=None):
    """A message when state of an earlier version sits in ``legacy_dir`` while
    ``state_dir`` has none; None otherwise. Ignoring it would lose pending
    migration journals, resumable maintenance checkpoints and VM cooldowns."""
    state_dir = STATE_DIR if state_dir is None else state_dir
    legacy_dir = LEGACY_STATE_DIR if legacy_dir is None else legacy_dir
    if _has_state(state_dir) or not _has_state(legacy_dir):
        return None
    return (f"state from an earlier version was found in {legacy_dir} but {state_dir} has none; running now would "
            "ignore pending migrations, maintenance checkpoints and VM cooldowns. Move it first: "
            f"mkdir -p {state_dir} && mv {legacy_dir}/* {state_dir}/  (or keep the old place with explicit "
            "--cooldown-file / --state-dir options)")
