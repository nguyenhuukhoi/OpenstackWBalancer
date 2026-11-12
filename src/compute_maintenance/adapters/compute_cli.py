"""sources.compute — extracted domain responsibilities."""
import json
import os
import re

import compute_maintenance.adapters.commands as _maint_remote
import compute_maintenance.dependencies as _deps
import compute_maintenance.domain.maintenance_errors as _maint_errors
import compute_maintenance.profiles.defaults as _maint_settings

OSC_ENV_KEEP = ("OS_CLIENT_CONFIG_FILE", "OS_CACERT", "OS_CERT", "OS_KEY")


def clouds_yaml_files():
    """Files openstacksdk searches for clouds, in its order."""
    names = ("clouds.yaml", "clouds.yml", "clouds.json")
    dirs = [os.getcwd(),
            os.path.join(os.path.expanduser("~"), ".config", "openstack"),
            "/etc/openstack"]
    files = [os.environ["OS_CLIENT_CONFIG_FILE"]] if os.environ.get("OS_CLIENT_CONFIG_FILE") else []
    return files + [os.path.join(d, n) for d in dirs for n in names]


def cloud_in_clouds_yaml(name):
    """Path of the first clouds file defining cloud ``name``, else None."""
    for path in clouds_yaml_files():
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            continue
        try:
            if path.endswith(".json"):
                data = json.loads(text)
            else:
                import yaml
                data = yaml.safe_load(text)
            if name in ((data or {}).get("clouds") or {}):
                return path
        except Exception:  # noqa: BLE001 - unparsable / no PyYAML: fall back to a plain match
            if re.search(rf"^\s+{re.escape(name)}:\s*$", text, re.MULTILINE):
                return path
    return None


def osc_env(os_cloud):
    """Environment for ``openstack --os-cloud`` that authenticates exactly like
    wbalancer's ``openstack.connect(cloud=...)`` (openstacksdk):

    * cloud defined in clouds.yaml -> the SDK uses that file and ignores OS_*
      variables, but the CLI would MERGE them in. With an application
      credential cloud plus a sourced admin-openrc (OS_PROJECT_NAME, ...)
      Keystone answers 401 "Application credentials cannot request a scope".
      So OS_* are removed (TLS files and the config path are kept).
    * cloud not in any clouds.yaml -> both tools build it from the OS_*
      variables (openrc + OS_CLOUD_NAME=<cloud>): keep the environment as is.
    """
    source = cloud_in_clouds_yaml(os_cloud)
    if not source:
        _maint_settings.log.info("openstack auth: cloud '%s' not in any clouds.yaml, using OS_* from the environment", os_cloud)
        return dict(os.environ)
    dropped = sorted(k for k in os.environ if k.startswith("OS_") and k not in OSC_ENV_KEEP)
    _maint_settings.log.info("openstack auth: cloud '%s' from %s (same as wbalancer)", os_cloud, source)
    if dropped:
        _maint_settings.log.warning("  ignoring %s from the shell: clouds.yaml wins, as for wbalancer", ", ".join(dropped))
    return {k: v for k, v in os.environ.items() if not (k.startswith("OS_") and k not in OSC_ENV_KEEP)}


class Cloud:
    def __init__(self, os_cloud, cli=None):
        self.os_cloud = os_cloud
        self.env = None  # built on first call, so the warning lands in the log file
        # Operator-installed CLI; prefer the one in this interpreter's venv
        # over whatever PATH (reset by sudo/cron/systemd) points to.
        self.cli = cli or _deps.openstack_cli_path() or "openstack"

    def _run(self, args, timeout):
        if self.env is None:
            self.env = osc_env(self.os_cloud)
        return _maint_remote.run([self.cli, "--os-cloud", self.os_cloud, *args], timeout=timeout, env=self.env)

    def osc(self, *args, timeout=120):
        out = self._run([*args, "-f", "json"], timeout).stdout.strip()
        return json.loads(out) if out else []

    def osc_noout(self, *args, timeout=120):
        self._run(list(args), timeout)

    def nova_services(self):
        return self.osc("compute", "service", "list", "--service", "nova-compute", "--long")

    def aggregate_map(self):
        """{aggregate name: [hosts]} for every aggregate."""
        out = {}
        for row in self.osc("aggregate", "list", "--long"):
            hosts = row.get("Hosts") or []
            if isinstance(hosts, str):
                hosts = [h.strip() for h in hosts.split(",") if h.strip()]
            out[row.get("Name") or str(row.get("ID"))] = list(hosts)
        return out

    def aggregate_hosts(self, name):
        hosts = self.osc("aggregate", "show", name).get("hosts") or []
        if isinstance(hosts, str):  # very old clients print "a, b"
            hosts = [h.strip() for h in hosts.split(",") if h.strip()]
        return list(hosts)

    def nova_service(self, host):
        rows = self.osc(
            "compute", "service", "list", "--service", "nova-compute", "--host", host, "--long"
        )
        if len(rows) != 1:
            raise _maint_errors.StepFailed(f"expected 1 nova-compute service on {host}, got {len(rows)}")
        return rows[0]

    def set_nova_enabled(self, host, enabled, reason=None):
        if enabled:
            self.osc_noout("compute", "service", "set", "--enable", host, "nova-compute")
        else:
            self.osc_noout(
                "compute", "service", "set", "--disable", "--disable-reason", reason, host, "nova-compute"
            )

    def agents_dead(self, host):
        dead = []
        for a in self.osc("network", "agent", "list", "--host", host):
            if a.get("Alive") not in (True, ":-)", "True", "true"):
                dead.append("{}({})".format(a.get("Agent Type"), a.get("Alive")))
        return dead

    def agent_snapshot(self, host):
        return {str(a["ID"]): a.get("Alive") in (True, ":-)", "True", "true")
                for a in self.osc("network", "agent", "list", "--host", host)}

    def cinder_snapshot(self, host):
        return {str(s.get("Host")) + "/" + str(s.get("Binary")): s.get("State") == "up"
                for s in self.osc("volume", "service", "list")
                if (s.get("Host") == host or str(s.get("Host", "")).startswith(host + "@"))
                and s.get("Status") == "enabled"}

    def cinder_down(self, host):
        """Enabled cinder services on ``host`` or ``host@backend`` that are not up."""
        down = []
        for s in self.osc("volume", "service", "list"):
            h = s.get("Host") or ""
            if (h == host or h.startswith(host + "@")) and s.get("Status") == "enabled" and s.get("State") != "up":
                down.append("{} {}".format(s.get("Binary"), h))
        return down

    def servers_on(self, host):
        return self.osc("server", "list", "--all-projects", "--host", host)

    def server_show(self, server_id):
        return self.osc("server", "show", server_id)

    def locked_servers_on(self, host):
        """Candidate locked servers on ``host`` (Nova filter, microversion 2.73).

        Only a hint: python-openstackclient 6.6.0 accepts --locked but returns
        every server (seen on the lab). Callers confirm with server_locked().
        """
        return self.osc("--os-compute-api-version", "2.73", "server", "list", "--all-projects",
                        "--host", host, "--locked")

    def server_locked(self, server_id):
        """True/False from Nova's ``locked`` field (microversion 2.9+)."""
        value = self.osc("--os-compute-api-version", "2.9", "server", "show", server_id).get("locked")
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("true", "false"):
            return str(value).lower() == "true"
        raise _maint_errors.StepFailed(f"cannot tell whether server {server_id} is locked (locked={value!r})")

    def migrating(self):
        return self.osc("server", "list", "--all-projects", "--status", "MIGRATING")
