import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


@pytest.fixture(autouse=True)
def hermetic_default_config(monkeypatch, tmp_path_factory):
    """No test reads the host's /etc/wbalancer/wbalancer.conf (it may exist and
    be root-only): the CLIs' default config file is a path in the test's temp
    directory that does not exist, like a host without any config file. Tests
    that need a config pass --config / --wlb-config explicitly."""
    monkeypatch.setenv("WLB_CONFIG_PATH", str(tmp_path_factory.getbasetemp() / "absent-wbalancer.conf"))
