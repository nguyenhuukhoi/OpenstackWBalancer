import compute_maintenance.domain.models as _models
import compute_maintenance.domain.policies as _planning_constraints

"""sources.openstack — extracted domain responsibilities."""
import logging

import openstack

import compute_maintenance.adapters.prometheus as _sources_prometheus
import compute_maintenance.domain.errors as _errors
import compute_maintenance.domain.server as _execution_migrate


def retry_read(description, fn, attempts=None, delay=None, sleeper=None):
    """Run an idempotent OpenStack read with bounded retries.

    Observed on the lab: GET /servers/detail occasionally stalls past the
    30 s request timeout on both SDK attempts. A read is safe to repeat, and
    one stall must not abort a drain that is otherwise healthy.
    """
    import time as _time

    import compute_maintenance.config as _config
    attempts = attempts or _config.OPENSTACK_READ_ATTEMPTS
    delay = _config.OPENSTACK_READ_RETRY_DELAY if delay is None else delay
    sleeper = sleeper or _time.sleep
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # re-raised after the last attempt
            if attempt >= attempts:
                raise
            logging.warning("%s failed (attempt %d/%d): %s; retrying in %.0fs",
                            description, attempt, attempts, e, delay)
            sleeper(delay)


def get_openstack_connection(cloud=None):
    """Connection to ``cloud`` (a clouds.yaml entry, or the openrc environment
    under its OS_CLOUD_NAME, "envvars" by default); default ``openstack``."""
    import compute_maintenance.config as _config
    # SDK defaults can wait forever on an unresponsive API. Every blocking
    # request must return control to the migration journal/timeout handling.
    conn = openstack.connect(cloud=cloud or _config.DEFAULT_OS_CLOUD, api_timeout=30)
    return conn




def find_nova_compute_service(conn, host):
    try:
        services = list(conn.compute.services(host=host, binary="nova-compute"))
    except TypeError:
        try:
            services = list(conn.compute.services())
        except Exception as e:  # noqa: BLE001 - any SDK/HTTP error = service state unknown (callers fail closed)
            logging.warning("Failed to query nova-compute service for host %s: %s", host, e)
            return None
    except Exception as e:  # noqa: BLE001 - any SDK/HTTP error = service state unknown (callers fail closed)
        logging.warning("Failed to query nova-compute service for host %s: %s", host, e)
        return None

    for service in services:
        service_host = _models.get_resource_field(service, "host", "Host")
        binary = _models.get_resource_field(service, "binary", "Binary")
        if service_host == host and (binary in (None, "", "nova-compute")):
            return service

    return None


def is_nova_compute_service_ready(conn, host):
    service = find_nova_compute_service(conn, host)
    if not service:
        return False, f"nova-compute service not found for host {host}"

    status = str(_models.get_resource_field(service, "status", "Status") or "UNKNOWN").lower()
    state = str(_models.get_resource_field(service, "state", "State") or "UNKNOWN").lower()
    forced_down_raw = _models.get_resource_field(
        service, "is_forced_down", "forced_down", "Forced Down"
    )
    if isinstance(forced_down_raw, str):
        forced_down = forced_down_raw.strip().lower() not in ("", "0", "false", "no", "off")
    else:
        forced_down = bool(forced_down_raw or False)
    disabled_reason = _models.get_resource_field(service, "disabled_reason", "Disabled Reason")

    if status != "enabled":
        reason = f"nova-compute on {host} is not enabled (status={status})"
        if disabled_reason:
            reason = f"{reason}, disabled_reason={disabled_reason}"
        return False, reason

    if state != "up":
        return False, f"nova-compute on {host} is not up (state={state})"

    if forced_down:
        return False, f"nova-compute on {host} is forced down"

    return True, f"nova-compute on {host} is enabled and up"


def get_nova_service_disabled_state(conn, host):
    """Return (found, is_disabled, disabled_reason) for a host's nova-compute."""
    service = find_nova_compute_service(conn, host)
    if not service:
        return False, None, None
    status = str(_models.get_resource_field(service, "status", "Status") or "").strip().lower()
    reason = _models.get_resource_field(service, "disabled_reason", "Disabled Reason")
    return True, status == "disabled", reason


def set_nova_compute_service_enabled(conn, host, enabled, disabled_reason=None):
    """Enable or disable a host's nova-compute service.

    Returns (ok, detail). Used by drain mode so the Nova scheduler stops
    placing new instances on a host that is about to go into maintenance.
    """
    service = find_nova_compute_service(conn, host)
    if not service:
        return False, f"nova-compute service not found for host {host}"

    try:
        if enabled:
            conn.compute.enable_service(service)
            return True, f"nova-compute on {host} enabled"
        conn.compute.disable_service(service, disabled_reason=disabled_reason)
        return True, f"nova-compute on {host} disabled (reason={disabled_reason})"
    except Exception as e:  # noqa: BLE001 - reported to the caller as a failed toggle
        action = "enable" if enabled else "disable"
        return False, f"failed to {action} nova-compute on {host}: {e}"


def fetch_server_groups(conn):
    """Fetch all server groups once, failing closed on lookup errors."""
    try:
        return retry_read("list server groups",
                          lambda: list(conn.compute.server_groups(all_projects=True)))
    except Exception as e:  # noqa: BLE001 - any SDK/HTTP error -> OpenStackDataError (fail closed)
        raise _errors.OpenStackDataError(
            "Failed to fetch server groups; refusing to plan migrations without "
            f"affinity/anti-affinity data: {e}"
        )






def get_fresh_server_group_context(conn, vm_id, known_hosts=None, host_to_hypervisor=None, known_servers=None):
    """Fetch current group membership and member placement for one VM.

    A caller that read some of this in the same validation pass passes it in
    instead of having it fetched again: ``host_to_hypervisor`` (the map of
    ``known_hosts``) and ``known_servers`` ({server id: record just read}).
    """
    server_groups = fetch_server_groups(conn)
    vm_to_groups = _planning_constraints.build_server_group_index({vm_id}, server_groups)
    groups = vm_to_groups.get(vm_id) or []
    member_ids = {member for group in groups for member in (group.get("members") or [])}

    if host_to_hypervisor is None:
        host_to_hypervisor = build_nova_host_to_hypervisor_map(conn, known_hosts or [])
    hypervisor_to_host = {str(v): k for k, v in host_to_hypervisor.items()}
    known_servers = known_servers or {}
    vm_host_map = {}
    for member_id in member_ids:
        if member_id in known_servers:
            server = known_servers[member_id]
        else:
            try:
                server = conn.compute.get_server(member_id)
            except Exception as e:  # noqa: BLE001 - any SDK/HTTP error -> OpenStackDataError (fail closed)
                raise _errors.OpenStackDataError(
                    f"Failed to refresh server-group member {member_id}: {e}"
                )
        if not server:
            raise _errors.OpenStackDataError(
                f"Server-group member {member_id} could not be resolved"
            )
        host = _execution_migrate.get_server_current_host(server)
        host = hypervisor_to_host.get(str(host), host)
        if not host:
            raise _errors.OpenStackDataError(
                f"Server-group member {member_id} has no current compute host"
            )
        vm_host_map[member_id] = host

    return vm_host_map, vm_to_groups


def build_server_cache(conn, hosts_in_agg):
    server_cache = {}

    try:
        servers = retry_read("list all servers",
                             lambda: list(conn.compute.servers(all_projects=True, details=True)))
        for server in servers:
            server_cache[server.id] = server
        logging.info(
            "Built all-project Nova cache for %d servers; %d selected host(s).",
            len(server_cache),
            len(hosts_in_agg),
        )
    except Exception as e:  # noqa: BLE001 - any SDK/HTTP error -> OpenStackDataError (fail closed)
        raise _errors.OpenStackDataError(
            f"Failed to build the authoritative Nova server cache: {e}"
        )

    return server_cache


def get_server_project_id(server):
    if not server:
        return None

    project_id = _models.get_resource_field(server, "project_id", "tenant_id")
    if isinstance(project_id, dict):
        return project_id.get("id") or project_id.get("project_id") or project_id.get("tenant_id")

    return project_id


def resolve_project_name(conn, project_id, project_cache=None):
    if not project_id:
        return None

    if project_cache is not None and project_id in project_cache:
        return project_cache[project_id]

    project_name = None
    try:
        project = conn.identity.get_project(project_id)
        project_name = _models.get_resource_field(project, "name", "Name")
    except Exception as e:  # noqa: BLE001 - the project name is cosmetic; the id is kept
        logging.debug("Failed to resolve project name for project %s: %s", project_id, e)

    if not project_name:
        project_name = project_id

    if project_cache is not None:
        project_cache[project_id] = project_name

    return project_name


def unsupported_flavor_resources(flavor):
    """Return a short description of flavor needs the planner cannot model.

    Placement capacity is modelled for VCPU and MEMORY_MB only. Pinned CPUs
    allocate PCPU, PCI/vGPU devices and resources:* overrides allocate other
    resource classes: planning those with VCPU arithmetic picks hosts Nova
    then rejects, so such VMs are never moved automatically.
    """
    import compute_maintenance.config as _config
    specs = _models.get_resource_field(flavor, "extra_specs") or {}
    if not isinstance(specs, dict):
        return None
    problems = []
    policy = str(specs.get("hw:cpu_policy") or "").strip().lower()
    if policy in _config.VM_UNSUPPORTED_CPU_POLICIES:
        problems.append("hw:cpu_policy=" + policy)
    for key in sorted(specs):
        if str(key).startswith(_config.VM_UNSUPPORTED_EXTRA_SPEC_PREFIXES):
            problems.append(str(key))
    return ", ".join(problems) or None


def get_default_vm_metadata(vm_id):
    """Return non-migratable metadata when Nova cannot resolve the server."""
    return {
        "name": vm_id,
        "vcpus": None,
        "ram_gb": None,
        "project_id": None,
        "project_name": None,
        # Unknown status when the server cannot be resolved; treated as non-active
        # so only confirmed-ACTIVE VMs are considered for migration.
        "status": None,
        "metadata_complete": False,
        "locked": None,
        "task_state": None,
        "availability_zone": None,
        "unsupported_resources": None,
    }


def get_vm_metadata(conn, vm_id, server_cache=None, project_cache=None):
    """Resolve VM metadata while preserving the existing fallback behavior."""
    try:
        server = server_cache.get(vm_id) if server_cache else conn.compute.get_server(vm_id)
        if not server:
            return get_default_vm_metadata(vm_id)

        flavor = getattr(server, "flavor", {}) or {}
        ram_mb = _models.get_resource_field(flavor, "ram")
        name = getattr(server, "name", vm_id)
        vcpus = _models.get_resource_field(flavor, "vcpus")
        flavor_id = _models.get_resource_field(flavor, "id")
        if (ram_mb is None or vcpus is None) and flavor_id:
            try:
                full_flavor = conn.compute.get_flavor(flavor_id)
                if full_flavor:
                    ram_mb = ram_mb if ram_mb is not None else _models.get_resource_field(full_flavor, "ram")
                    vcpus = vcpus if vcpus is not None else _models.get_resource_field(full_flavor, "vcpus")
            except Exception as e:  # noqa: BLE001 - unresolved flavor leaves metadata_complete=False (never planned)
                logging.warning(
                    "Failed to resolve flavor %s for VM %s: %s",
                    flavor_id,
                    vm_id,
                    e,
                )
        project_id = get_server_project_id(server)
        project_name = resolve_project_name(
            conn,
            project_id,
            project_cache=project_cache,
        )

        status = getattr(server, "status", None)
        status = str(status).upper() if status else None
        locked = _models.get_resource_field(server, "is_locked", "locked")
        task_state = _models.get_resource_field(server, "task_state", "OS-EXT-STS:task_state")
        if task_state in ("", "None"):
            task_state = None
        availability_zone = _models.get_resource_field(
            server, "availability_zone", "OS-EXT-AZ:availability_zone"
        )

        metadata_complete = vcpus is not None and ram_mb is not None
        if not metadata_complete:
            logging.warning(
                "VM %s (%s) has incomplete flavor metadata; it will not be migrated.",
                name,
                vm_id,
            )
        return {
            "name": name,
            "vcpus": int(vcpus) if vcpus is not None else None,
            "ram_gb": float(ram_mb) / 1024.0 if ram_mb is not None else None,
            "project_id": project_id,
            "project_name": project_name,
            "status": status,
            "metadata_complete": metadata_complete,
            "locked": bool(locked) and str(locked).lower() not in ("false", "0"),
            "task_state": task_state,
            "availability_zone": availability_zone,
            "unsupported_resources": unsupported_flavor_resources(flavor),
        }
    except Exception as e:  # noqa: BLE001 - default metadata has metadata_complete=False (never planned)
        logging.debug(
            "Failed to resolve metadata for VM %s; using default values: %s",
            vm_id,
            e,
        )

    return get_default_vm_metadata(vm_id)


def calculate_vm_scores(conn, host, server_cache=None, vm_metrics=None, project_cache=None):
    """Build VM score inputs for a single host using already-fetched metrics."""
    if vm_metrics is None:
        raise ValueError("vm_metrics is required")

    vm_names = {}
    vm_details = {}

    for vm_id, metric in vm_metrics.items():
        if metric["host"] != host:
            continue

        metadata = get_vm_metadata(
            conn,
            vm_id,
            server_cache=server_cache,
            project_cache=project_cache,
        )

        cpu_pct = metric.get("cpu_pct", 0.0)
        ram_pct = metric.get("ram_pct", 0.0)
        ram_rss_gb = metric.get("ram_rss_gb")
        if ram_pct is None:
            # No guest memory stats: RSS share of the flavor RAM; unknown
            # flavor RAM -> assume full (never under-estimates the VM).
            ram_gb = metadata.get("ram_gb")
            ram_pct = (
                min(max(float(ram_rss_gb) / ram_gb * 100.0, 0.0), 100.0)
                if ram_gb and ram_rss_gb is not None
                else 100.0
            )

        vm_names[vm_id] = metadata["name"]
        vm_details[vm_id] = {
            "cpu_pct": cpu_pct,
            "ram_pct": ram_pct,
            "ram_rss_gb": ram_rss_gb,
            "vcpus": metadata["vcpus"],
            "ram_gb": metadata["ram_gb"],
            "host": host,
            "project_id": metadata["project_id"],
            "project_name": metadata["project_name"],
            "status": metadata.get("status"),
            "metadata_complete": metadata.get("metadata_complete", False),
            "metrics_complete": metric.get("metrics_complete", False),
            "locked": metadata.get("locked"),
            "task_state": metadata.get("task_state"),
            "availability_zone": metadata.get("availability_zone"),
            "unsupported_resources": metadata.get("unsupported_resources"),
        }

    return vm_names, vm_details


def get_placement_proxy(conn):
    placement = getattr(conn, "placement", None)
    if placement is None:
        raise _errors.PlacementError(
            "Placement endpoint is not available through openstacksdk connection. "
            "Check service catalog and openstacksdk Placement support."
        )
    return placement


def fetch_placement_host_resource(conn, cfg, host, provider_name=None):
    """Fetch authoritative VCPU/MEMORY_MB inventory and usage for one compute RP."""
    placement = get_placement_proxy(conn)

    candidates = []
    for name in (host, provider_name):
        if name and name not in candidates:
            candidates.append(name)

    rp = None
    last_error = None
    for name in candidates:
        try:
            rp = placement.find_resource_provider(name, ignore_missing=True)
        except Exception as e:  # noqa: BLE001 - try the next provider name; the last error is reported
            last_error = e
            continue
        if rp is not None:
            break

    if rp is None:
        detail = f"; last error: {last_error}" if last_error else ""
        raise _errors.PlacementError(
            f"No Placement resource provider found for Nova host {host!r} "
            f"(tried: {', '.join(repr(x) for x in candidates)}). "
            "This balancer intentionally does not fall back to Nova hypervisor capacity"
            f"{detail}."
        )

    rp_uuid = str(_models.get_resource_field(rp, "id", "uuid") or "")
    if not rp_uuid:
        raise _errors.PlacementError(f"Placement resource provider {host!r} has no UUID.")

    try:
        # Do not depend on openstacksdk inventory/usage convenience helpers here.
        # Older SDKs used with OpenStack 2023.1 expose Placement resource-provider
        # lookup but not resource_provider_inventories(). The REST endpoints are
        # stable and work across old/new SDK versions because Proxy inherits the
        # authenticated Adapter get() method.
        inv_response = placement.get(f"/resource_providers/{rp_uuid}/inventories")
        inv_payload = inv_response.json() or {}
        inv_by_rc = inv_payload.get("inventories") or {}

        usage_response = placement.get(f"/resource_providers/{rp_uuid}/usages")
        usage_payload = usage_response.json() or {}
        usages = usage_payload.get("usages") or {}
    except Exception as e:  # noqa: BLE001 - any SDK/HTTP error -> PlacementError (fail closed)
        raise _errors.PlacementError(f"Failed to fetch Placement inventory/usage for host {host}: {e}")

    if not isinstance(inv_by_rc, dict):
        raise _errors.PlacementError(
            f"Placement inventories response for host {host} ({rp_uuid}) is malformed: "
            f"expected object, got {type(inv_by_rc).__name__}"
        )

    if not isinstance(usages, dict):
        raise _errors.PlacementError(
            f"Placement usages response for host {host} ({rp_uuid}) is malformed: "
            f"expected object, got {type(usages).__name__}"
        )

    missing = [rc for rc in ("VCPU", "MEMORY_MB") if rc not in inv_by_rc]
    if missing:
        raise _errors.PlacementError(
            f"Placement provider {host} ({rp_uuid}) is missing required inventory: {', '.join(missing)}"
        )

    vcpu = _planning_constraints.build_placement_inventory_record(
        inv_by_rc["VCPU"],
        usages.get("VCPU", 0),
        cfg.placement_vcpu_headroom_percent,
    )
    memory = _planning_constraints.build_placement_inventory_record(
        inv_by_rc["MEMORY_MB"],
        usages.get("MEMORY_MB", 0),
        cfg.placement_ram_headroom_percent,
    )

    return {
        "resource_provider_uuid": rp_uuid,
        "resource_provider_name": str(_models.get_resource_field(rp, "name") or host),
        "VCPU": vcpu,
        "MEMORY_MB": memory,
    }


def build_nova_host_to_hypervisor_map(conn, hosts):
    """Map nova-compute service host -> hypervisor hostname without using capacity fields.

    Placement resource-provider names follow the virt driver's hypervisor hostname,
    while host aggregates contain nova-compute service host names. They are often
    identical, but this mapping keeps the balancer correct when they differ.
    """
    wanted = set(hosts)
    mapping = {h: h for h in hosts}
    try:
        for hypervisor in conn.compute.hypervisors():
            try:
                full = conn.compute.get_hypervisor(hypervisor.id)
            except Exception:  # noqa: BLE001 - the detail view is optional; the list entry is used
                full = hypervisor
            hv_name = (
                _models.get_resource_field(full, "name", "hypervisor_hostname")
                or _models.get_resource_field(hypervisor, "name", "hypervisor_hostname")
            )
            service = _models.get_resource_field(full, "service")
            service_host = None
            if isinstance(service, dict):
                service_host = service.get("host")
            elif service is not None:
                service_host = _models.get_resource_field(service, "host")
            if service_host in wanted and hv_name:
                mapping[service_host] = str(hv_name)
    except Exception as e:  # noqa: BLE001 - best-effort map; names are assumed equal (logged)
        logging.warning(
            "Could not build Nova service-host -> hypervisor-name map; assuming names match: %s",
            e,
        )
    return mapping


def count_instances_by_host(server_cache, hosts, host_to_hypervisor=None):
    """Count all non-deleted Nova servers assigned to each aggregate host."""
    counts = {h: 0 for h in hosts}
    host_to_hypervisor = host_to_hypervisor or {h: h for h in hosts}
    hypervisor_to_host = {str(v): k for k, v in host_to_hypervisor.items()}
    host_set = set(hosts)
    for server in (server_cache or {}).values():
        hv = _models.get_resource_field(server, "hypervisor_hostname")
        current_host = _execution_migrate.get_server_current_host(server)
        host = hypervisor_to_host.get(str(hv), None)
        if host is None:
            host = hypervisor_to_host.get(str(current_host), current_host)
        if host in host_set:
            counts[host] += 1
    return counts


def fetch_host_availability_zones(conn, hosts):
    """Return {nova-compute host: availability zone} for ``hosts``.

    Hosts that belong to no AZ aggregate are reported by Nova under the
    default zone, so every enabled compute host gets a zone here.
    """
    wanted = set(hosts)
    mapping = {}
    zones = retry_read("list availability zones",
                       lambda: list(conn.compute.availability_zones(details=True)))
    for zone in zones:
        name = _models.get_resource_field(zone, "name", "zoneName")
        for host in (_models.get_resource_field(zone, "hosts") or {}):
            if host in wanted:
                mapping[host] = name
    return mapping


def server_migration_rows(conn, server_id):
    """All /os-migrations rows of one server, terminal rows included.

    GET /servers/{id}/migrations lists only *in-progress* live migrations, so
    it can never prove that a migration failed. /os-migrations keeps them.
    """
    return retry_read("list migrations of " + str(server_id),
                      lambda: list(conn.compute.migrations(server_id=server_id)))


def max_migration_id(rows):
    ids = []
    for row in rows:
        try:
            ids.append(int(_models.get_resource_field(row, "id")))
        except (TypeError, ValueError):
            continue
    return max(ids) if ids else 0


def get_host_resources_map(conn, cfg, hosts, server_cache=None, errors=None, hypervisor_map=None):
    """Build the dual resource model used by the balancer.

    Runtime capacity (logical CPU + physical RAM) comes from Prometheus and is
    used only for projecting observed load impact. Scheduler capacity comes
    directly from Placement inventory/usage and is a hard migration constraint.
    ``hypervisor_map`` (a dict) receives the service-host -> hypervisor-name
    map read here, so the caller can reuse it within the same pass.
    """
    runtime = _sources_prometheus.get_runtime_capacity_map(cfg, hosts, missing=errors)
    host_to_hypervisor = build_nova_host_to_hypervisor_map(conn, hosts)
    if hypervisor_map is not None:
        hypervisor_map.update(host_to_hypervisor)
    running_counts = count_instances_by_host(
        server_cache or {}, hosts, host_to_hypervisor=host_to_hypervisor
    )
    host_resources = {}

    for host in hosts:
        if host not in runtime:
            continue  # recorded in ``errors``; only aggregates holding it fail
        try:
            placement = fetch_placement_host_resource(
                conn, cfg, host, provider_name=host_to_hypervisor.get(host)
            )
        except _errors.PlacementError as e:
            if errors is None:
                raise
            errors[host] = str(e)
            continue
        host_resources[host] = {
            "total_cores": round(runtime[host]["logical_cpus"]),
            "total_ram_gb": float(runtime[host]["ram_gb"]),
            # Existing names are retained so the rest of the scorer changes as
            # little as possible. They now mean Placement *policy-safe* free
            # VCPU/MEMORY_MB, not real CPU/RAM headroom.
            "free_cores": float(placement["VCPU"]["free"]),
            "free_ram_gb": float(placement["MEMORY_MB"]["free"]) / 1024.0,
            "running_vms": int(running_counts.get(host, 0)),
            "placement": placement,
        }

    return host_resources


def placement_allocation_candidate_sanity_check(conn, placement, vcpus, ram_mb):
    """Optional nominal Placement candidate check scoped to the destination tree.

    This intentionally requests only VCPU/MEMORY_MB. It is a sanity check, not a
    replacement for Nova scheduler validation of traits, NUMA, PCI, networking,
    or any other constraints.
    """
    proxy = get_placement_proxy(conn)
    params = {
        "resources": f"VCPU:{int(vcpus)},MEMORY_MB:{int(ram_mb)}",
        "in_tree": placement["resource_provider_uuid"],
    }
    try:
        response = proxy.get("/allocation_candidates", params=params, microversion="1.31")
        payload = response.json()
    except Exception as e:  # noqa: BLE001 - a failed sanity query rejects the candidate
        return False, f"allocation_candidates query failed: {e}"

    if payload.get("allocation_requests"):
        return True, None
    return False, "Placement returned no allocation candidate in destination provider tree"


def get_aggregate_map(conn, aggregate_names=None, show_available_on_missing=False):
    """Return {aggregate_name: [host, ...]} for the selected aggregates.

    Preserve the original fast path (hosts from ``conn.compute.aggregates()``)
    because that is known to work on the existing deployments.  If a LIST
    response contains no hosts, fetch that aggregate's detail as a defensive
    fallback.  This avoids silently turning a valid selection into
    ``all_hosts=[]`` while keeping the original behavior when LIST already
    contains host membership.
    """
    aggregate_map = {}
    requested = set(aggregate_names or [])
    discovered_names = set()

    logging.debug(
        "Requested aggregate filter: %s",
        sorted(requested) if requested else "(ALL)",
    )

    for agg in conn.compute.aggregates():
        agg_id = getattr(agg, "id", None)
        agg_name = getattr(agg, "name", None) or str(agg_id if agg_id is not None else id(agg))
        discovered_names.add(agg_name)

        list_hosts = list(getattr(agg, "hosts", None) or [])
        logging.debug(
            "Aggregate LIST result: id=%s name=%r hosts=%s",
            agg_id,
            agg_name,
            list_hosts,
        )

        if requested and agg_name not in requested:
            logging.debug(
                "Skipping aggregate %r because it does not match requested filter %s",
                agg_name,
                sorted(requested),
            )
            continue

        hosts = list_hosts

        # Keep the original LIST behavior as the primary path.  Only use the
        # aggregate-detail call when LIST did not expose membership.
        if not hosts and agg_id is not None:
            try:
                full_agg = conn.compute.get_aggregate(agg_id)
                hosts = list(getattr(full_agg, "hosts", None) or [])
                logging.debug(
                    "Aggregate DETAIL result: id=%s name=%r hosts=%s",
                    agg_id,
                    agg_name,
                    hosts,
                )
            except Exception as e:  # noqa: BLE001 - one unreadable aggregate is skipped (logged)
                logging.warning(
                    "Failed to fetch aggregate detail id=%s name=%r: %s",
                    agg_id,
                    agg_name,
                    e,
                )

        aggregate_map[agg_name] = hosts

    if requested:
        missing = requested - set(aggregate_map)
        if missing:
            message = (
                "Requested aggregate(s) not found: "
                + ", ".join(sorted(missing))
            )
            if show_available_on_missing:
                message += (
                    ". Available aggregates: "
                    + (", ".join(sorted(discovered_names)) if discovered_names else "(none)")
                )
            raise _errors.ConfigError(message)

    logging.info(
        "Selected %d aggregate(s): %s",
        len(aggregate_map),
        ", ".join(
            f"{name}({len(hosts)} hosts)"
            for name, hosts in aggregate_map.items()
        ) if aggregate_map else "(none)",
    )

    for name, hosts in aggregate_map.items():
        logging.debug(
            "Aggregate %s selected hosts: %s",
            name,
            ", ".join(hosts) if hosts else "(none)",
        )

    return aggregate_map


def build_selected_host_list(aggregate_map):
    """Flatten aggregate membership and fail fast when no host is selectable."""
    all_hosts = sorted({
        host
        for hosts in aggregate_map.values()
        for host in hosts
    })

    logging.info(
        "Selected %d unique compute host(s) across aggregates.",
        len(all_hosts),
    )
    logging.debug(
        "Selected compute hosts: %s",
        ", ".join(all_hosts) if all_hosts else "(none)",
    )

    if not all_hosts:
        empty = [name for name, hosts in aggregate_map.items() if not hosts]
        detail = (
            "; aggregate(s) with no hosts: " + ", ".join(empty)
            if empty else ""
        )
        raise _errors.ConfigError(
            "No compute hosts found from selected aggregate(s)" + detail
        )

    return all_hosts
