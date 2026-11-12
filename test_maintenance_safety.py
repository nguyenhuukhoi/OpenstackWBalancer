"""Regression tests; fake cloud never submits OpenStack mutations.

Also runs on the Linux coordinator with stdlib unittest (no pytest install).
"""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tests.legacy_api import maint as r
from tests.legacy_api import API
ms = API({**{k: "adapters.persistence" for k in ("HostState", "StateError", "SCHEMA_VERSION", "digest", "file_digest")}, "coordinator_lock": "adapters.locking"})


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = self.root / "config"
        self.cfg.write_text("PROMETHEUS_URL=http://example.invalid")
        self.args = r.parse_args(["--profile", "unused", "--state-dir", str(self.root),
                                  "--wlb-config", str(self.cfg), "--notify", "off", "-y", "h1"])
        self.profile = r.validate_profile({"name": "test", "reboot": "never"})
        r.merge_settings(self.args, self.profile)
        self.service = {"ID": "svc-1", "Status": "enabled", "State": "up"}
        self.servers = []
        self.cloud = SimpleNamespace(nova_service=lambda _: dict(self.service),
            aggregate_map=lambda: {"agg": ["h1", "h2"]}, servers_on=lambda _: self.servers)
        self.node = SimpleNamespace(name="h1", address="127.0.0.1", user="root", key=None,
                                    boot_id=lambda: "boot-1")
        self.runner = r.Runner(self.args, self.profile, self.cloud, lambda _: self.node)
        self.runner._lock_held = True
        self.runner.preflight = lambda *a, **k: None

    def checkpoint(self, stage="maintenance"):
        state = r.HostState(str(self.root), "h1")
        state.save(schema_version=ms.SCHEMA_VERSION, identity=self.runner.checkpoint_identity(self.node),
                   run_id="token", runner=socket.gethostname(), pid=0, profile="test",
                   result="stopped", next_stage=stage, last_stage=stage,
                   completed=r.STAGES[:r.STAGES.index(stage)],
                   baseline={"containers": [], "failed_units": []}, baseline_boot_id="boot-1",
                   healthy_boot_id="boot-1", left_on_host=[])
        return state

    def owned(self):
        self.service.update(Status="disabled", **{"Disabled Reason": "rolling-maint v1:token"})

    def test_resume_on_enabled_host_rejected_without_state_write(self):
        state = self.checkpoint()
        before = Path(state.path).read_bytes()
        with self.assertRaisesRegex(r.StepFailed, "exact run"):
            self.runner.process_host("h1", "maintenance")
        self.assertEqual(before, Path(state.path).read_bytes())

    def test_profile_same_name_changed_body_rejected(self):
        self.checkpoint()
        self.runner.profile = copy.deepcopy(self.profile)
        self.runner.profile["description"] = "changed"
        with self.assertRaisesRegex(ms.StateError, "profile_sha256"):
            self.runner.process_host("h1", "maintenance")

    def test_config_changed_rejected(self):
        self.checkpoint()
        self.cfg.write_text("changed")
        with self.assertRaisesRegex(ms.StateError, "config_sha256"):
            self.runner.process_host("h1", "maintenance")

    def test_skip_to_restore_rejected(self):
        self.checkpoint()
        with self.assertRaisesRegex(r.StepFailed, "pending stage"):
            self.runner.process_host("h1", "restore")

    def test_foreign_reason_rejected(self):
        state = self.checkpoint()
        self.service.update(Status="disabled", **{"Disabled Reason": "rolling-maint other"})
        with self.assertRaises(r.StepFailed):
            self.runner.guard_stage("maintenance", self.node, state)

    def test_running_vm_even_saved_as_left_rejected(self):
        state = self.checkpoint()
        self.owned()
        self.servers = [{"ID": "vm", "Status": "ACTIVE", "Name": "vm"}]
        state.save(left_on_host=["vm"])
        with self.assertRaisesRegex(r.StepFailed, "RUNNING"):
            self.runner.guard_stage("maintenance", self.node, state)

    def test_changed_boot_rejects_maintenance_and_restore(self):
        state = self.checkpoint()
        self.owned()
        self.node.boot_id = lambda: "boot-2"
        for stage in ("maintenance", "restore"):
            with self.subTest(stage=stage), self.assertRaisesRegex(r.StepFailed, "boot changed"):
                self.runner.guard_stage(stage, self.node, state)

    def test_restore_rechecks_health(self):
        state = self.checkpoint("restore")
        self.runner.health_once = lambda *_: (False, "container unhealthy")
        with self.assertRaisesRegex(r.StepFailed, "health recheck"):
            self.runner.guard_stage("restore", self.node, state)

    def test_legacy_checkpoint_preserved(self):
        state = r.HostState(str(self.root), "h1")
        state.save(profile="test", result="stopped")
        before = Path(state.path).read_bytes()
        with self.assertRaisesRegex(ms.StateError, "legacy"):
            self.runner._run_all()
        self.assertEqual(before, Path(state.path).read_bytes())

    def test_stopped_uses_live_status(self):
        self.checkpoint()
        for status in ("enabled", "disabled"):
            self.service["Status"] = status
            with self.assertRaises(ms.StateError) as caught:
                self.runner._run_all()
            self.assertEqual(caught.exception.observed_status, status)

    def test_stopped_unknown_when_query_fails(self):
        self.cloud.nova_service = lambda _: (_ for _ in ()).throw(r.StepFailed("offline"))
        with self.assertRaises(r.StepFailed) as caught:
            self.runner._run_all()
        self.assertEqual(caught.exception.observed_status, "UNKNOWN")

    def test_no_direct_execution_without_lock(self):
        self.runner._lock_held = False
        with self.assertRaisesRegex(r.StepFailed, "lock required"):
            self.runner.process_host("h1")

    def test_preflight_rejects_foreign_disabled_even_with_explicit_resume(self):
        self.args.resume_from = "maintenance"
        self.checkpoint()
        self.owned()
        self.cloud.nova_services = lambda: [dict(self.service, Host="h1")]
        self.cloud.agents_dead = lambda _: []
        self.cloud.migrating = lambda: []
        r.Runner.preflight(self.runner, ["h1"])
        self.service["Disabled Reason"] = "rolling-maint another run"
        with self.assertRaisesRegex(r.StepFailed, "exact run"):
            r.Runner.preflight(self.runner, ["h1"])

    def test_successful_workflow_and_completed_resume_boundary(self):
        seen = []

        def stage_fn(stage, node, state, resumed):
            def execute():
                seen.append(stage)
                if stage == "disable":
                    self.service.update(Status="disabled", **{
                        "Disabled Reason": "rolling-maint v1:" + state.data["run_id"]})
                elif stage == "baseline":
                    state.save(baseline={"containers": [], "failed_units": []}, baseline_boot_id="boot-1")
                elif stage == "health":
                    state.save(healthy_boot_id="boot-1")
                elif stage == "restore":
                    self.service["Status"] = "enabled"
            return execute

        self.runner.stage_fn = stage_fn
        self.runner.health_once = lambda *_: (True, "ok")
        self.runner.process_host("h1")
        state = r.HostState(str(self.root), "h1")
        self.assertEqual(state.load()["result"], "done")
        self.assertEqual(seen, r.STAGES[1:])
        self.assertEqual(self.service["Status"], "enabled")

    def test_resume_runs_only_pending_and_later_stages(self):
        state = self.checkpoint("post_restore")
        seen = []
        self.runner.stage_fn = lambda stage, *_: lambda: seen.append(stage)
        self.runner.process_host("h1")
        self.assertEqual(seen, ["post_restore"])
        self.assertEqual(state.load()["result"], "done")

    def test_profile_shell_error_contracts(self):
        bash = "bash" if os.name == "posix" else r"C:\Program Files\Git\bin\bash.exe"
        profile_dir = Path(__file__).parent / "profiles"
        apt = r.load_profile(str(profile_dir / "kernel-update.yaml"))["hooks"]["pre_drain"][0]["cmd"]
        for implementation, expected in (("return 100", 1), ("return 0", 3),
                                         ("printf 'Inst example\\n'", 0)):
            fresh_cache = ('test() { if [ "$1" = -f ]; then return 0; fi; command test "$@"; }; '
                           'date() { printf "1800000000\\n"; }; stat() { printf "1800000000\\n"; }; ')
            command = fresh_cache + "apt-get() { " + implementation + "; }; " + apt
            result = subprocess.run([bash, "-c", command], capture_output=True)
            self.assertEqual(result.returncode, expected, result.stderr)
        profile = r.load_profile(str(profile_dir / "reboot-only.yaml"))
        audit = next(line.strip() for line in profile["hooks"]["pre_drain"][1]["cmd"].splitlines()
                     if line.strip().startswith("audit="))
        result = subprocess.run([bash, "-c", "dpkg() { return 1; }; " + audit], capture_output=True)
        self.assertEqual(result.returncode, 1)

    def test_reboot_probe_errors_are_not_false(self):
        self.profile["reboot"] = "if_required"
        for rc in (2, 127, 255):
            self.node.ssh = lambda *a, **k: SimpleNamespace(returncode=rc)
            with self.subTest(rc=rc), self.assertRaisesRegex(r.StepFailed, "UNKNOWN"):
                self.runner.reboot_needed(self.node)
        for rc in (0, 1):
            self.node.ssh = lambda *a, **k: SimpleNamespace(returncode=rc)
            self.assertEqual(self.runner.reboot_needed(self.node)[0], rc == 0)

    def test_failed_units_checks_command_status(self):
        node = r.Node("h1", "127.0.0.1", "root", None, "docker")
        with patch.object(node, "ssh", side_effect=r.StepFailed("ssh 255")) as ssh:
            with self.assertRaises(r.StepFailed):
                node.failed_units()
        self.assertNotEqual(ssh.call_args.kwargs.get("check"), False)

    def test_bad_settings_rejected(self):
        for settings in ({"settle": -1}, {"boot_timeout": True}, {"health_timeout": 0},
                         {"ignore_units": [1]}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                r.validate_profile({"name": "test", "settings": settings})

    def test_any_cloud_name_accepted_but_not_empty(self):
        # rc7: the cloud name is no longer fixed to "openstack".
        self.assertEqual(r.parse_args(["--profile", "unused", "--os-cloud", " other "]).os_cloud, "other")
        with self.assertRaises(SystemExit):
            r.parse_args(["--profile", "unused", "--os-cloud", "  "])

    def test_corrupt_checkpoint_fails_closed(self):
        (self.root / "h1.json").write_text("{")
        with self.assertRaises(ms.StateError):
            r.HostState(str(self.root), "h1").load()

    def test_atomic_save_preserves_old_on_replace_failure(self):
        state = self.checkpoint()
        before = Path(state.path).read_bytes()
        with patch.object(ms.os, "replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                state.save(result="done")
        self.assertEqual(before, Path(state.path).read_bytes())
        self.assertEqual(list(self.root.glob(".checkpoint-*")), [])

    @unittest.skipUnless(ms.fcntl, "POSIX flock integration runs on Linux lab")
    def test_lock_blocks_other_process_and_releases(self):
        path = str(self.root / "coordinator.lock")
        script = "from maintenance_state import coordinator_lock; import sys\nwith coordinator_lock(sys.argv[1]): pass"
        with ms.coordinator_lock(path):
            child = subprocess.run([sys.executable, "-B", "-c", script, path],
                                   cwd=Path(__file__).parent, capture_output=True, text=True)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn("another maintenance run", child.stderr)
        with ms.coordinator_lock(path):
            pass

    def test_no_flock_fails_closed(self):
        with patch.object(ms, "fcntl", None), self.assertRaises(ms.StateError):
            with ms.coordinator_lock(str(self.root / "lock")):
                self.fail("must not enter")


if __name__ == "__main__":
    unittest.main(verbosity=2)
