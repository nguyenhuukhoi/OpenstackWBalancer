"""maint.profile — extracted domain responsibilities."""
import json
import os
import shlex
from pathlib import Path


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate profile key: " + str(key))
        result[key] = value
    return result
import compute_maintenance.profiles.defaults as _maint_settings

HOOK_STAGES = ("pre_drain", "maintenance", "post_boot", "post_restore")


REBOOT_POLICIES = ("always", "if_required", "never")


SKIP_EXIT_CODE = 3


SETTING_KEYS = {
    "boot_timeout": int,
    "health_timeout": int,
    "settle": int,
    "drain_timeout": int,
    "required_containers": list,
    "ignore_containers": list,
    "ignore_units": list,
    "drain_allow_hotspot": bool,
    "wlb_extra": str,
}


def load_profile(path):
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:
            raise SystemExit(f"PyYAML is required for {path} ({e})")
        class UniqueLoader(yaml.SafeLoader):
            pass
        def mapping(loader, node):
            loader.flatten_mapping(node)
            return unique_pairs((loader.construct_object(k), loader.construct_object(v)) for k, v in node.value)
        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        data = yaml.load(text, Loader=UniqueLoader)
    else:
        data = json.loads(text, object_pairs_hook=unique_pairs)
    # Resolve executable files once; normalized profile hashes the exact bytes
    # that will execute. Commands are sent over SSH, no persistent remote copy.
    if isinstance(data, dict):
        base = Path(path).resolve().parent
        hooks = data.get("hooks", {})
        if not isinstance(hooks, dict):
            raise ValueError("hooks must be a mapping")  # noqa: TRY004 - profile errors are ValueError (CLI: "invalid profile")
        for steps in hooks.values():
            if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
                raise ValueError("hooks must contain lists of step mappings")
            for step in steps:
                for source_key, target_key in (("script", "cmd"), ("check", "check_cmd")):
                    if source_key not in step:
                        continue
                    if target_key in step:
                        raise ValueError(f"{source_key} and {target_key} are mutually exclusive")
                    relative = step.pop(source_key)
                    if not isinstance(relative, str) or not relative.strip():
                        raise ValueError("profile script path must be a non-empty string")
                    file = (base / relative).resolve()
                    if not file.is_relative_to(base):
                        raise ValueError("profile script must remain within its profile directory")
                    step[target_key] = file.read_text(encoding="utf-8")
    return validate_profile(data, path)


def validate_profile(data, origin="profile"):
    """Normalise a profile dict; raise ValueError on anything unexpected."""
    if not isinstance(data, dict):
        raise ValueError(f"{origin}: top level must be a mapping")  # noqa: TRY004 - profile errors are ValueError (CLI: "invalid profile")
    unknown = set(data) - {"schema_version", "name", "description", "reboot", "reboot_required_cmd", "hooks", "settings"}
    if unknown:
        raise ValueError(f"{origin}: unknown key(s) {sorted(unknown)}")

    profile = {
        "schema_version": data.get("schema_version", 1),
        "name": str(data.get("name") or os.path.splitext(os.path.basename(origin))[0]),
        "description": str(data.get("description") or ""),
        "reboot": data.get("reboot", "always"),
        "reboot_required_cmd": data.get("reboot_required_cmd", "test -f /var/run/reboot-required"),
        "hooks": {stage: [] for stage in HOOK_STAGES},
        "settings": {},
    }
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        raise ValueError("unsupported profile schema version")
    if profile["reboot"] not in REBOOT_POLICIES:
        raise ValueError(f"{origin}: reboot must be one of {REBOOT_POLICIES}")
    if not isinstance(profile["reboot_required_cmd"], str) or not profile["reboot_required_cmd"].strip():
        raise ValueError("reboot_required_cmd must be a non-empty string")

    hooks = data.get("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError(f"{origin}: hooks must be a mapping")  # noqa: TRY004 - profile errors are ValueError (CLI: "invalid profile")
    for stage, steps in hooks.items():
        if stage not in HOOK_STAGES:
            raise ValueError(f"{origin}: unknown hook stage '{stage}' (allowed {HOOK_STAGES})")
        if not isinstance(steps, list):
            raise ValueError(f"{origin}: hooks.{stage} must be a list")  # noqa: TRY004 - profile errors are ValueError (CLI: "invalid profile")
        profile["hooks"][stage] = [
            validate_step(step, f"{origin}: hooks.{stage}[{i}]") for i, step in enumerate(steps)
        ]
        ids = [step["id"] for step in profile["hooks"][stage]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate hook ID in " + stage)

    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("settings must be a mapping")  # noqa: TRY004 - profile errors are ValueError (CLI: "invalid profile")
    for key, value in settings.items():
        if key not in SETTING_KEYS:
            raise ValueError(f"{origin}: unknown setting '{key}'")
        if not isinstance(value, SETTING_KEYS[key]):
            raise ValueError(f"{origin}: setting {key} must be {SETTING_KEYS[key].__name__}")  # noqa: TRY004 - profile errors are ValueError (CLI: "invalid profile")
        validate_setting(key, value)
        profile["settings"][key] = value
    return profile


def validate_step(step, where):
    if not isinstance(step, dict):
        raise ValueError(f"{where}: step must be a mapping")  # noqa: TRY004 - profile errors are ValueError (CLI: "invalid profile")
    unknown = set(step) - {"id", "name", "run", "cmd", "check_cmd", "idempotency", "capture", "timeout", "retries", "retry_delay"}
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}")
    if step.get("run", "ssh") not in ("ssh", "local"):
        raise ValueError(f"{where}: run must be ssh or local")
    if not isinstance(step.get("cmd"), str) or not step["cmd"].strip():
        raise ValueError(f"{where}: cmd must be a non-empty string")
    identity = step.get("id", where.rsplit("hooks.", 1)[-1])
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("hook id must be a non-empty string")
    idempotency = step.get("idempotency", "never")
    if idempotency not in ("never", "idempotent", "reconcile"):
        raise ValueError("idempotency must be never, idempotent or reconcile")
    check = step.get("check_cmd")
    if check is not None and (not isinstance(check, str) or not check.strip()):
        raise ValueError("check command must be a non-empty string")
    if idempotency == "reconcile" and check is None:
        raise ValueError("reconcile hook requires a check script/command")
    capture = step.get("capture")
    if capture not in (None, "expected_kernel", "package_plan"):
        raise ValueError("unsupported hook capture field")
    for key, default in (("timeout", 600), ("retries", 0), ("retry_delay", 10)):
        value = step.get(key, default)
        if type(value) is not int or value < (1 if key == "timeout" else 0):
            raise ValueError(f"{where}: invalid {key}")
    return {
        "id": identity,
        "idempotency": idempotency,
        "check_cmd": check,
        "capture": capture,
        "name": str(step.get("name") or step["cmd"][:60]),
        "run": step.get("run", "ssh"),
        "cmd": str(step["cmd"]),
        "timeout": int(step.get("timeout", 600)),
        "retries": int(step.get("retries", 0)),
        "retry_delay": int(step.get("retry_delay", 10)),
    }


def render(template, node, profile):
    """Substitute {host} {address} {profile}; other braces stay literal."""
    out = template
    for key, value in (("host", node.name), ("address", node.address), ("profile", profile["name"])):
        out = out.replace("{" + key + "}", shlex.quote(value))
    return out


def validate_setting(key, value):
    expected = SETTING_KEYS[key]
    if type(value) is not expected:
        raise ValueError(f"{key} must be {expected.__name__}")
    if expected is int and value < (0 if key == "settle" else 1):
        raise ValueError(f"{key} is outside its allowed range")
    if expected is list and any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"{key} must contain non-empty strings")


DEFAULTS = {
    "drain_timeout": _maint_settings.DRAIN_TIMEOUT_SECONDS,
    "boot_timeout": _maint_settings.BOOT_TIMEOUT_SECONDS,
    "health_timeout": _maint_settings.HEALTH_TIMEOUT_SECONDS,
    "settle": _maint_settings.SETTLE_SECONDS,
    "required_containers": _maint_settings.DEFAULT_REQUIRED_CONTAINERS,
    "ignore_containers": [],
    "ignore_units": [],
    "drain_allow_hotspot": False,
    "wlb_extra": "",
}


def merge_settings(args, profile):
    """CLI value > profile setting > built-in default."""
    for key, default in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, profile["settings"].get(key, default))
        validate_setting(key, getattr(args, key))
    if args.wlb_extra:
        raise ValueError("v1 does not accept --wlb-extra; use explicit supported options")
    return args


def read_env_file(path):
    from compute_maintenance.adapters.settings import read_settings
    return read_settings(path)
