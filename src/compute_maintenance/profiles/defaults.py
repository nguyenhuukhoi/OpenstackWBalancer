"""maint.settings — extracted domain responsibilities."""
import logging

STAGES = [
    "preflight",
    "pre_drain",
    "disable",
    "drain",
    "verify",
    "baseline",
    "maintenance",
    "reboot",
    "health",
    "post_boot",
    "restore",
    "post_restore",
]


REBOOT_SAFE_LEFT_VM_STATUSES = frozenset(("SHUTOFF", "SHELVED"))


REBOOT_PRE_DRAIN_VM_STATUSES = REBOOT_SAFE_LEFT_VM_STATUSES | frozenset(("ACTIVE",))


INVENTORY_GROUP = "compute"


SETTLE_SECONDS = 300


DRAIN_TIMEOUT_SECONDS = 7200


BOOT_TIMEOUT_SECONDS = 3600


HEALTH_TIMEOUT_SECONDS = 1200


DEFAULT_REQUIRED_CONTAINERS = [
    "nova_compute",
    "nova_libvirt",
    "nova_ssh",
    "ovn_controller",
    "openvswitch_db",
    "openvswitch_vswitchd",
    "neutron_ovn_metadata_agent",
]


DEFAULT_SSH_KEY = "/root/.ssh/kolla_ansible"


DEFAULT_WLB_CONFIG = "/etc/wbalancer/wbalancer.conf"   # paths.CONFIG_FILE; $WLB_CONFIG_PATH overrides


log = logging.getLogger("rolling-maint")
