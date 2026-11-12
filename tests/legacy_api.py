"""Test-only adapter for the pre-refactor regression suite. Not installed."""
import importlib
import json
from pathlib import Path


class API:
    def __init__(self, owners):
        object.__setattr__(self, "_owners", owners)
    def __getattr__(self, name):
        if name in ("VM_LAST_MOVED_AT", "VM_LAST_FAILED_MOVE_AT"):
            store = importlib.import_module("compute_maintenance.adapters.cooldown").default_store
            return store.moved if name == "VM_LAST_MOVED_AT" else store.failed
        if name in self._owners:
            return getattr(importlib.import_module("compute_maintenance." + self._owners[name]), name)
        if name in ("time", "os", "subprocess", "socket", "json", "logging", "smtplib", "openstack", "requests", "dotenv"):
            return importlib.import_module(name)
        if name == "fcntl":
            return importlib.import_module("compute_maintenance.adapters.locking").fcntl
        raise AttributeError(name)
    @property
    def __dict__(self):
        return {**{name: getattr(self, name) for name in self._owners}, "fcntl": self.fcntl}

    def __setattr__(self, name, value):
        if name in ("VM_LAST_MOVED_AT", "VM_LAST_FAILED_MOVE_AT"):
            setattr(importlib.import_module("compute_maintenance.adapters.cooldown").default_store, "moved" if name == "VM_LAST_MOVED_AT" else "failed", value)
        elif name == "fcntl":
            setattr(importlib.import_module("compute_maintenance.adapters.locking"), name, value)
        else:
            setattr(importlib.import_module("compute_maintenance." + self._owners[name]), name, value)

owners = json.loads(Path(__file__).with_name("api_owners.json").read_text())
balance = API(owners["balance"])
maint = API(owners["maint"])
