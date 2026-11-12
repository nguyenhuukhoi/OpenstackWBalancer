"""Compatibility imports for source checkout users."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from compute_maintenance.adapters.persistence import HostState, StateError, SCHEMA_VERSION, digest, file_digest
from compute_maintenance.adapters.locking import coordinator_lock
