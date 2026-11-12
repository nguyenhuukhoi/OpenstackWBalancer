"""OpenStack client libraries are installed by the operator, not by this package.

``openstacksdk`` (used in-process by wbalancer) and ``python-openstackclient``
(the ``openstack`` CLI used by rolling-maint) are deliberately not declared as
dependencies, so an operator can match them to their cloud. The tested baseline
is the OpenStack 2026.1 upper-constraints set; older versions only warn.

Standard library only: this module runs before anything imports openstacksdk.
"""
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys

BASELINE_RELEASE = "2026.1"
CONSTRAINTS_URL = "https://releases.openstack.org/constraints/upper/2026.1"
BASELINE_VERSIONS = {
    "openstacksdk": "4.10.0",
    "python-openstackclient": "9.0.1",
}
INSTALL_HINT = (
    "Install the OpenStack clients yourself (tested baseline: OpenStack {release}):\n"
    "  {python} -m pip install -c {url} openstacksdk python-openstackclient"
)
CLI_VERSION_TIMEOUT = 60


def install_hint():
    return INSTALL_HINT.format(release=BASELINE_RELEASE, python=sys.executable, url=CONSTRAINTS_URL)


def installed_version(dist):
    """Version of distribution ``dist`` in this interpreter, or None."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def release_tuple(version):
    """Leading numeric release of a version string: '4.10.0rc1' -> (4, 10, 0)."""
    parts = []
    for piece in str(version).strip().split("."):
        match = re.match(r"\d+", piece)
        if not match:
            break
        parts.append(int(match.group()))
        if match.end() != len(piece):
            break
    return tuple(parts)


def older_than_baseline(dist, version):
    found, baseline = release_tuple(version), release_tuple(BASELINE_VERSIONS[dist])
    # Compare equal-length tuples: (4, 10) is the same release as (4, 10, 0).
    width = max(len(found), len(baseline))
    pad = lambda t: t + (0,) * (width - len(t))
    return bool(found) and pad(found) < pad(baseline)


def baseline_warning(dist, version):
    return (f"{dist} {version} is older than the OpenStack {BASELINE_RELEASE} baseline "
            f"{BASELINE_VERSIONS[dist]}; this combination is untested.")


def check_sdk():
    """(errors, warnings) for the in-process openstacksdk."""
    version = installed_version("openstacksdk")
    if version is None:
        return [f"openstacksdk is not installed in this Python ({sys.executable})."], []
    return [], ([baseline_warning("openstacksdk", version)]
                if older_than_baseline("openstacksdk", version) else [])


def openstack_cli_path():
    """``openstack`` next to this interpreter (its venv, activated or not), else PATH.

    sudo, cron and systemd reset PATH, so a bare ``openstack`` can silently
    resolve to an unrelated system package (seen on the lab: /usr/bin/openstack
    6.6.0, whose ``server list --locked`` returns every server).
    """
    names = ("openstack.exe", "openstack") if os.name == "nt" else ("openstack",)
    for name in names:
        local = os.path.join(os.path.dirname(sys.executable), name)
        if os.path.isfile(local) and os.access(local, os.X_OK):
            return local
    return shutil.which("openstack")


def cli_version(path, run=subprocess.run):
    """Version printed by ``openstack --version`` (stdout or stderr), or None."""
    try:
        proc = run([path, "--version"], capture_output=True, text=True, timeout=CLI_VERSION_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\b(\d+(?:\.\d+)+\S*)", (proc.stdout or "") + " " + (proc.stderr or ""))
    return match.group(1) if match else None


def check_cli(run=subprocess.run):
    """(path, version, errors, warnings) for the ``openstack`` CLI."""
    path = openstack_cli_path()
    if path is None:
        missing = (f"the 'openstack' CLI (python-openstackclient) was not found next to "
                   f"{sys.executable} or on PATH.")
        return None, None, [missing], []
    version = cli_version(path, run=run)
    if version is None:
        return path, None, [f"cannot read the version of {path} ('--version' failed)."], []
    warnings = ([baseline_warning("python-openstackclient", version)]
                if older_than_baseline("python-openstackclient", version) else [])
    return path, version, [], warnings
