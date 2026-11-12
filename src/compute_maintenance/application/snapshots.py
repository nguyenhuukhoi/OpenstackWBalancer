"""snapshot — extracted domain responsibilities."""
import logging
import time

import compute_maintenance.adapters.cooldown as _execution_cooldown
import compute_maintenance.adapters.openstack as _sources_openstack
import compute_maintenance.adapters.prometheus as _sources_prometheus
import compute_maintenance.adapters.selection as _selection
import compute_maintenance.application.migrations as _execution_migrate
import compute_maintenance.config as _config
import compute_maintenance.domain.balance as _planning_balance
import compute_maintenance.domain.errors as _errors
import compute_maintenance.domain.models as _models
import compute_maintenance.domain.plan_context as _plan_context
import compute_maintenance.domain.policies as _planning_constraints
import compute_maintenance.domain.severity as _severity
import compute_maintenance.presentation.reports as _report


def build_host_resource_views(host_resources, hosts_in_agg):
    """Build separate runtime and Placement views for aggregate hosts."""
    missing = [h for h in hosts_in_agg if h not in host_resources]
    if missing:
        raise _errors.PlacementError(
            "Missing authoritative host resource data for: " + ", ".join(missing)
        )

    return (
        {h: host_resources[h]["total_cores"] for h in hosts_in_agg},
        {h: host_resources[h]["free_cores"] for h in hosts_in_agg},
        {h: host_resources[h]["total_ram_gb"] for h in hosts_in_agg},
        {h: host_resources[h]["free_ram_gb"] for h in hosts_in_agg},
        {h: host_resources[h]["running_vms"] for h in hosts_in_agg},
        {h: host_resources[h]["placement"] for h in hosts_in_agg},
    )


def build_effective_host_metrics(hosts_in_agg, cpu_metrics, mem_metrics, agg_name):
    """Return host CPU/RAM maps for hosts with both metrics available."""
    host_cpu_pct = {}
    host_ram_pct = {}

    for host in hosts_in_agg:
        cpu_entry = cpu_metrics.get(host)
        mem_entry = mem_metrics.get(host)
        if not cpu_entry or not mem_entry:
            logging.warning("Host %s in aggregate %s missing CPU or RAM metrics, skipping.", host, agg_name)
            continue
        cpu_value = float(cpu_entry["value"])
        ram_value = float(mem_entry["value"])
        if not (
            -_config.PERCENT_METRIC_TOLERANCE <= cpu_value <= 100.0 + _config.PERCENT_METRIC_TOLERANCE
        ) or not (
            -_config.PERCENT_METRIC_TOLERANCE <= ram_value <= 100.0 + _config.PERCENT_METRIC_TOLERANCE
        ):
            logging.warning(
                "Host %s in aggregate %s has out-of-range metrics CPU=%s RAM=%s; skipping.",
                host,
                agg_name,
                cpu_value,
                ram_value,
            )
            continue
        host_cpu_pct[host] = min(max(cpu_value, 0.0), 100.0)
        host_ram_pct[host] = min(max(ram_value, 0.0), 100.0)

    return host_cpu_pct, host_ram_pct


def get_destination_readiness(conn, effective_hosts, agg_name):
    """Return destination-ready hosts and exclusion reasons."""
    dest_ready_hosts = set()
    dest_not_ready_reasons = {}

    for host in effective_hosts:
        ready, reason = _sources_openstack.is_nova_compute_service_ready(conn, host)
        if ready:
            dest_ready_hosts.add(host)
        else:
            dest_not_ready_reasons[host] = reason
            logging.warning(
                "Host %s excluded as migration destination for aggregate %s: %s",
                host,
                agg_name,
                reason,
            )

    return dest_ready_hosts, dest_not_ready_reasons


def build_aggregate_vm_inventory(conn, effective_hosts, server_cache, vm_metrics, project_cache,
                                 host_to_hypervisor=None):
    """Build VM inventory with Nova authoritative for placement.

    Nova servers without complete Prometheus metrics remain in ``vm_host_map``
    so server-group checks still see them, but they never become candidates.
    ``host_to_hypervisor`` is a service-host -> hypervisor-name map read in the
    same pass (by the cluster cache); without one covering every host here,
    the map is fetched.
    """
    all_vm_names = {}
    all_vm_details = {}
    vm_host_map = {}
    effective_host_set = set(effective_hosts)
    if host_to_hypervisor is not None and all(h in host_to_hypervisor for h in effective_hosts):
        # Per host the map does not depend on which other hosts were asked for,
        # so restricting it gives exactly what a fetch for effective_hosts returns.
        host_to_hypervisor = {h: host_to_hypervisor[h] for h in effective_hosts}
    else:
        host_to_hypervisor = _sources_openstack.build_nova_host_to_hypervisor_map(conn, effective_hosts)
    hypervisor_to_host = {str(v): k for k, v in host_to_hypervisor.items()}

    for vm_id, server in (server_cache or {}).items():
        current = _execution_migrate.get_server_current_host(server)
        current = hypervisor_to_host.get(str(current), current)
        if not current:
            continue
        vm_host_map[vm_id] = current
        if current in effective_host_set:
            all_vm_names[vm_id] = getattr(server, "name", None) or vm_id

    vm_metrics_by_host = _sources_prometheus.group_vm_metrics_by_host(vm_metrics)

    for src_host in effective_hosts:
        host_vm_metrics = vm_metrics_by_host.get(src_host, {})
        vm_names, vm_details = _sources_openstack.calculate_vm_scores(
            conn,
            src_host,
            server_cache=server_cache,
            vm_metrics=host_vm_metrics,
            project_cache=project_cache,
        )
        for vm_id, details in vm_details.items():
            nova_host = vm_host_map.get(vm_id)
            if nova_host is None:
                logging.warning(
                    "Prometheus reports VM %s on %s but Nova has no matching server "
                    "on the aggregate; ignoring the stale/unknown metric series.",
                    vm_id,
                    src_host,
                )
                continue
            if nova_host != src_host:
                logging.warning(
                    "VM %s host mismatch (Nova=%s, Prometheus=%s); keeping it in "
                    "server-group topology but skipping it as a migration candidate.",
                    vm_id,
                    nova_host,
                    src_host,
                )
                continue
            all_vm_names[vm_id] = vm_names[vm_id]
            all_vm_details[vm_id] = details

    return all_vm_names, all_vm_details, vm_host_map


def ram_metric_window_seconds(cfg):
    """Look-back of the host RAM query: how long a moved VM stays under-counted."""
    from compute_maintenance.application.recovery import promql_lookback_seconds
    window = promql_lookback_seconds(getattr(cfg, "prometheus_query_mem_used", "") or "")
    return window if window > 0 else _config.COMPENSATION_DEFAULT_RAM_WINDOW_SECONDS


def cpu_compensation_window_seconds(cfg):
    """How long a just-arrived VM is still missing from the host CPU reading.

    Host CPU is irate() over node_cpu_seconds_total, i.e. the last two scrapes,
    so the reading catches up within 2 x scrape interval (30 s at 15 s
    scrapes, 60 s at 30 s scrapes). The interval is read from Prometheus; an
    explicit COMPENSATION_CPU_WINDOW_SECONDS wins; unknown -> conservative
    default.
    """
    configured = getattr(cfg, "compensation_cpu_window_seconds", None)
    if configured:
        return float(configured)
    interval = _sources_prometheus.get_scrape_interval_seconds(
        getattr(cfg, "prometheus_query_url", None), getattr(cfg, "prometheus_node_job", None))
    if interval:
        return 2.0 * interval
    logging.warning("Scrape interval unknown; using a %.0fs CPU compensation window.",
                    _config.COMPENSATION_CPU_WINDOW_SECONDS)
    return _config.COMPENSATION_CPU_WINDOW_SECONDS


def vm_runtime_load(details, server, sample=None):
    """(cores, ram_gb) a VM puts on its host.

    Order of evidence: planner details; else ``sample`` -- the VM's CPU just
    before its migration started (``pre_cpu_pct``), or its sample taken while
    it was visible on two hosts; else its full flavor (never under-estimates).
    """
    if details:
        vcpus = float(details.get("vcpus") or 1)
        cores = vcpus * float(details.get("cpu_pct") or 0.0) / 100.0
        rss = details.get("ram_rss_gb")
        if rss is not None:
            ram = float(rss)
        else:
            ram = float(details.get("ram_gb") or 0.0) * float(details.get("ram_pct") or 0.0) / 100.0
        return cores, ram
    flavor = getattr(server, "flavor", None) or {}
    vcpus = float(_models.get_resource_field(flavor, "vcpus") or 0)
    ram_gb = float(_models.get_resource_field(flavor, "ram") or 0) / 1024.0
    if sample:
        cpu_pct = sample.get("pre_cpu_pct", sample.get("cpu_pct"))
        cores = vcpus * float(cpu_pct) / 100.0 if cpu_pct is not None else vcpus
        if sample.get("ram_rss_gb") is not None:
            ram = float(sample["ram_rss_gb"])
        elif "ram_pct" in sample:
            ram = ram_gb * float(sample["ram_pct"]) / 100.0
        else:
            ram = ram_gb
        return cores, ram
    return vcpus, ram_gb


def apply_arrival_compensation(arrivals, host_cpu_pct, host_ram_pct, host_cores, host_ram,
                               vm_host_map, all_vm_details, server_cache, ram_window,
                               ambiguous_samples=None, cpu_window=None, now=None,
                               ram_instant=None, ram_sample_ts=None, vm_rss_gb=None):
    """Add the load of recently arrived VMs that the planner's metrics miss.

    RAM (host query = ``ram_window`` average), per receiving host:
      * "measured": the latest raw sample, when it is newer than every
        arrival (the caller waits a few scrapes for that). Lab canaries
        (33 arrivals on two 504 GiB hosts): worst over +0.03, worst under
        -0.02 percentage points against the next raw sample;
      * "measured+rss": raw sample not newer yet -> max(average, raw sample)
        plus the host-side RSS of VMs completed after it (over-estimates by
        at most those VMs, the safe side);
      * "model": no raw sample at all -> RSS x (1 - age/ram_window).
    The guest-RAM model of the first rc5 build was always under (worst
    -1.15; lab-uyuni reports 5.95 GiB used but holds 31.77 GiB RSS).
    CPU (irate) catches up within ``cpu_window``; the VM's CPU is added with a
    linear decay over that window. The source host is never reduced.
    Returns {host: {"cpu": +pct, "ram": +pct, "vms": n, "method": ...}}.
    """
    now = time.time() if now is None else now
    ram_instant = ram_instant or {}
    ram_sample_ts = ram_sample_ts or {}
    vm_rss_gb = vm_rss_gb or {}
    cpu_window = cpu_window or _config.COMPENSATION_CPU_WINDOW_SECONDS
    per_host = {}
    for arrival in arrivals:
        vm_id, dst = arrival["vm_id"], arrival["dst_host"]
        if vm_host_map.get(vm_id) != dst or dst not in host_ram_pct:
            continue  # moved again since, or not a host of this aggregate
        per_host.setdefault(dst, []).append(arrival)

    added = {}
    for dst, items in sorted(per_host.items()):
        cpu_add = 0.0
        late_ram_gb = 0.0
        model_ram_gb = 0.0
        sample_ts = ram_sample_ts.get(dst)
        for arrival in items:
            vm_id = arrival["vm_id"]
            cores, ram_gb = vm_runtime_load(all_vm_details.get(vm_id), (server_cache or {}).get(vm_id),
                                            (ambiguous_samples or {}).get(vm_id))
            ram_gb = vm_rss_gb.get(vm_id, ram_gb)  # host RSS is what the host loses
            done_at = arrival.get("done_at", now - arrival["age"])
            if sample_ts is not None and done_at > sample_ts:
                late_ram_gb += ram_gb
            model_ram_gb += ram_gb * max(0.0, 1.0 - arrival["age"] / ram_window)
            if host_cores.get(dst):
                cpu_add += cores / host_cores[dst] * 100.0 * max(0.0, 1.0 - arrival["age"] / cpu_window)
        before = host_ram_pct[dst]
        if dst in ram_instant and sample_ts is not None and host_ram.get(dst):
            if late_ram_gb == 0 and all(a.get("done_at", now - a["age"]) <= sample_ts for a in items):
                method = "measured"
                ram_new = float(ram_instant[dst])
            else:
                method = "measured+rss"
                ram_new = max(before, float(ram_instant[dst])) + late_ram_gb / host_ram[dst] * 100.0
        else:
            method = "model"
            ram_new = before + (model_ram_gb / host_ram[dst] * 100.0 if host_ram.get(dst) else 0.0)
        ram_new = min(100.0, ram_new)
        host_ram_pct[dst] = ram_new
        host_cpu_pct[dst] = min(100.0, host_cpu_pct[dst] + cpu_add)
        added[dst] = {"cpu": cpu_add, "ram": ram_new - before, "vms": len(items), "method": method}
    for host, entry in sorted(added.items()):
        logging.info(
            "Host %s: +%.1f%% CPU / +%.1f%% RAM (%s) for %d recently arrived VM(s) "
            "not yet fully visible in the metric window.",
            host, entry["cpu"], entry["ram"], entry["method"], entry["vms"],
        )
    return added


def wait_for_fresh_ram_samples(cfg, arrivals, hosts, sleeper=time.sleep, clock=time.time):
    """Wait (bounded) until each receiving host has a raw RAM sample newer
    than its last arrival, so that sample measures the arrived VMs exactly.

    Bounded by max(90 s, 3 x scrape interval); returns False on timeout (the
    compensation then adds RSS for the late VMs instead).
    """
    wanted = set(hosts)
    last_done = {}
    now = clock()
    for arrival in arrivals:
        if arrival["dst_host"] in wanted:
            done_at = arrival.get("done_at", now - arrival["age"])
            last_done[arrival["dst_host"]] = max(last_done.get(arrival["dst_host"], 0.0), done_at)
    if not last_done:
        return True
    interval = _sources_prometheus.get_scrape_interval_seconds(
        getattr(cfg, "prometheus_query_url", None), getattr(cfg, "prometheus_node_job", None)) or 30.0
    deadline = clock() + max(90.0, 3.0 * interval)
    while True:
        try:
            stamps = _sources_prometheus.get_metrics(
                cfg.prometheus_query_url, cfg.prometheus_query_mem_sample_timestamp, sorted(last_done))
        except (_errors.PrometheusError, AttributeError) as e:
            logging.warning("Cannot read raw RAM sample times (%s).", e)
            return False
        stale = sorted(h for h, done in last_done.items()
                       if (stamps.get(h) or {}).get("value", 0.0) < done)
        if not stale:
            return True
        if clock() >= deadline:
            logging.warning("No fresh raw RAM sample for %s yet; adding the arrived VMs' RSS instead.",
                            ", ".join(stale))
            return False
        logging.info("Waiting for a raw RAM sample newer than the last arrival on %s.", ", ".join(stale))
        sleeper(5.0)


def pre_migration_cpu(cfg, arrivals, all_vm_details):
    """CPU % of arrived VMs sampled 15 s before their migration started.

    Only for VMs the planner has no details for (their new series has no
    irate() yet). Failures leave the VM out; the flavor is used instead.
    """
    result = {}
    for arrival in arrivals:
        vm_id, begin = arrival["vm_id"], arrival.get("begin_at")
        if vm_id in all_vm_details or begin is None:
            continue
        try:
            value = _sources_prometheus.get_vm_cpu_pct_at(cfg, vm_id, begin - 15.0)
        except (_errors.PrometheusError, AttributeError) as e:
            logging.debug("No pre-migration CPU for %s: %s", vm_id, e)
            continue
        if value is not None:
            result[vm_id] = value
    return result


def measure_receiving_hosts(cfg, arrivals, hosts):
    """Raw RAM sample, its timestamp and arriving VMs' RSS for compensation.

    Any failure returns empty maps; compensation then falls back to the model.
    """
    receiving = sorted({a["dst_host"] for a in arrivals if a["dst_host"] in set(hosts)})
    if not receiving:
        return {}, {}, {}
    try:
        instant = {h: v["value"] for h, v in _sources_prometheus.get_metrics(
            cfg.prometheus_query_url, cfg.prometheus_query_mem_used_instant, receiving).items()}
        stamps = {h: v["value"] for h, v in _sources_prometheus.get_metrics(
            cfg.prometheus_query_url, cfg.prometheus_query_mem_sample_timestamp, receiving).items()}
        rss = _sources_prometheus.get_vm_rss_gb(cfg, [a["vm_id"] for a in arrivals])
        return instant, stamps, rss
    except (_errors.PrometheusError, AttributeError) as e:
        logging.warning("Cannot measure receiving hosts (%s); using the RSS model.", e)
        return {}, {}, {}


def run_project_names(cfg):
    """Keystone project names resolved so far in this run ({project id: name}).

    Kept on the run's config object (one per wbalancer process), so every
    snapshot of the run reuses them instead of asking Keystone again. Names
    are display-only (events, alert mails).
    """
    names = getattr(cfg, "project_names", None)
    if names is None:
        names = {}
        try:
            cfg.project_names = names
        except AttributeError:  # a read-only config object: names resolve per snapshot, as before
            pass
    return names


def remember_project_names(cfg, project_cache):
    """Keep the names a snapshot resolved for the rest of the run.

    A project Keystone could not name is shown by its id; it is not remembered,
    so the next snapshot asks again, exactly as a fresh per-snapshot cache did.
    """
    names = run_project_names(cfg)
    for project_id, name in (project_cache or {}).items():
        if name != project_id:
            names[project_id] = name


def build_cluster_cache(conn, cfg, hosts, enforce_server_groups=True):
    """Fetch cluster-wide data once so it can be reused across all aggregates.

    These lookups (Placement inventory/usage, runtime host capacity, the full
    server list, and server groups) do not depend on which aggregate is
    evaluated, so fetching them once avoids repeating expensive calls per
    aggregate.
    """
    server_cache = _sources_openstack.build_server_cache(conn, hosts)
    host_errors = {}
    host_to_hypervisor = {}
    try:
        host_az = _sources_openstack.fetch_host_availability_zones(conn, hosts)
    except Exception as e:  # fail closed on the AZ topology (re-raised as OpenStackDataError)
        raise _errors.OpenStackDataError(f"Failed to read availability zones: {e}") from e
    return {
        "host_resources": _sources_openstack.get_host_resources_map(
            conn, cfg, hosts, server_cache=server_cache, errors=host_errors,
            hypervisor_map=host_to_hypervisor,
        ),
        "host_resource_errors": host_errors,
        # The service-host -> hypervisor map read for the Placement lookups;
        # evaluate_aggregate_state reuses it only for a cache built in its own pass.
        "host_to_hypervisor": host_to_hypervisor,
        "host_az": host_az,
        "server_cache": server_cache,
        "server_groups": _sources_openstack.fetch_server_groups(conn) if enforce_server_groups else None,
        # Starts from the names this run already resolved (run_project_names),
        # so Keystone is asked once per project per run, not once per snapshot;
        # shared across aggregates when the cache is.
        "project_cache": dict(run_project_names(cfg)),
    }


def prefetched_cluster_cache_is_used(cfg):
    """True when evaluate_aggregate_state reads a cluster cache its caller passes in.

    Mirrors the settle branches below: the compensate settle mode (drain)
    always collects its own cache, and with a migration journal the settle
    barrier discards any cache collected before it. Only a journal-less
    barrier caller (dry-run, monitor, library use) reads a prefetched cache,
    so building one for any other caller is collected and thrown away.
    """
    return (getattr(cfg, "migration_journal", None) is None
            and getattr(cfg, "settle_mode", _config.SETTLE_MODE_BARRIER) != _config.SETTLE_MODE_COMPENSATE)


def evaluate_aggregate_state(
    cfg,
    conn,
    agg_name,
    hosts_in_agg,
    enforce_server_groups=True,
    cluster_cache=None,
    exclude_vm_patterns=None,
    exclude_unready_hosts=False,
    plan_balance=True,
):
    """Evaluate one aggregate using runtime load + authoritative Placement capacity.

    ``exclude_unready_hosts`` removes hosts that cannot receive a VM from the
    algorithm entirely. Balance and monitor mode set it; drain must not, since
    drain disables its own target host and still has to plan around it.
    """
    journal = getattr(cfg, "migration_journal", None)
    settle_mode = getattr(cfg, "settle_mode", _config.SETTLE_MODE_BARRIER)
    arrivals = []
    if settle_mode == _config.SETTLE_MODE_COMPENSATE:
        # Drain: no metric barrier between VMs. Running migrations are still
        # waited for, and recent arrivals are added to their destination.
        window = ram_metric_window_seconds(cfg)
        if journal is not None:
            arrivals = journal.settle_for_compensation(conn, hosts_in_agg, window)
        else:
            from compute_maintenance.application.recovery import (
                recent_arrivals,
                scan_host_migrations,
            )
            _active, _latest, terminal = scan_host_migrations(conn, hosts_in_agg)
            if _active:
                # Dry-run does not wait like apply does: say so instead of
                # silently planning on numbers that are about to change.
                logging.warning(
                    "Aggregate %s: live migration(s) still in progress; this dry-run snapshot "
                    "does not include them yet (a real drain waits for them first).", agg_name)
            arrivals = recent_arrivals(terminal, time.time(), window)
        cluster_cache = None
    elif journal is not None:
        journal.wait_settled(conn, hosts=hosts_in_agg)
        # A cache collected before the settle barrier is no longer authoritative.
        cluster_cache = None
    collected_at = time.time()
    _execution_cooldown.store_for(cfg).prune()
    pass_hypervisor_map = None
    if cluster_cache is None:
        cluster_cache = build_cluster_cache(
            conn, cfg, hosts_in_agg, enforce_server_groups=enforce_server_groups
        )
        # Read seconds ago in this same pass (after the settle step above): the
        # VM inventory below translates this cache's servers with it. A cache
        # passed in was read earlier, so the inventory still fetches its own.
        pass_hypervisor_map = cluster_cache.get("host_to_hypervisor")

    host_resources = cluster_cache["host_resources"]
    resource_errors = cluster_cache.get("host_resource_errors") or {}
    broken = sorted(h for h in hosts_in_agg if h not in host_resources)
    if broken:
        return _plan_context.PlanContext(
            aggregate=agg_name,
            error="missing_host_resources:" + "; ".join(
                resource_errors.get(h, h + ": no Placement/runtime capacity") for h in broken),
            hosts_in_agg=[h for h in hosts_in_agg if h in host_resources],
        )
    (
        host_cores,
        host_free_cores,
        host_ram,
        host_free_ram,
        host_running_vms,
        host_placement,
    ) = build_host_resource_views(host_resources, hosts_in_agg)

    cpu_metrics = _sources_prometheus.get_metrics(
        cfg.prometheus_query_url, cfg.prometheus_query_cpu_used, hosts_in_agg)
    mem_metrics = _sources_prometheus.get_metrics(
        cfg.prometheus_query_url, cfg.prometheus_query_mem_used, hosts_in_agg)
    host_cpu_pct, host_ram_pct = build_effective_host_metrics(
        hosts_in_agg,
        cpu_metrics,
        mem_metrics,
        agg_name,
    )

    effective_hosts = list(host_cpu_pct.keys())
    missing_metric_hosts = sorted(set(hosts_in_agg) - set(effective_hosts))
    if missing_metric_hosts:
        return _plan_context.PlanContext(
            aggregate=agg_name,
            error="missing_or_invalid_host_metrics:" + ",".join(missing_metric_hosts),
            hosts_in_agg=effective_hosts,
        )
    if len(effective_hosts) < 2:
        return _plan_context.PlanContext(
            aggregate=agg_name,
            error="less_than_2_hosts_with_metrics",
            hosts_in_agg=effective_hosts,
        )

    # Restrict all resource maps to the same effective host set.
    host_cores = {h: host_cores[h] for h in effective_hosts}
    host_free_cores = {h: host_free_cores[h] for h in effective_hosts}
    host_ram = {h: host_ram[h] for h in effective_hosts}
    host_free_ram = {h: host_free_ram[h] for h in effective_hosts}
    host_running_vms = {h: host_running_vms[h] for h in effective_hosts}
    host_placement = {h: host_placement[h] for h in effective_hosts}

    dest_ready_hosts, dest_not_ready_reasons = get_destination_readiness(
        conn,
        effective_hosts,
        agg_name,
    )

    # Balance and monitor mode take hosts that cannot receive a VM out of the
    # algorithm completely: their load stops feeding the mean, the MAD, the
    # hotspot mode, the peak and the severity. Drain passes False here, because
    # it disables its own target and still has to see it.
    #
    # The VM inventory below is deliberately still built over every host in the
    # aggregate. Dropping those VMs would hide affinity/anti-affinity members
    # that live on a disabled host, and the server-group checks would then
    # approve moves that split a group.
    inventory_hosts = list(effective_hosts)
    excluded_hosts = {}
    if exclude_unready_hosts:
        excluded_hosts = {
            host: dest_not_ready_reasons.get(host, "not destination-ready")
            for host in effective_hosts
            if host not in dest_ready_hosts
        }
    if excluded_hosts:
        effective_hosts = [h for h in effective_hosts if h in dest_ready_hosts]
        for host, reason in sorted(excluded_hosts.items()):
            logging.warning(
                "Host %s is excluded from the balancing algorithm for aggregate %s; "
                "its load, spread, hotspot state and severity are all ignored: %s",
                host, agg_name, reason,
            )
        if len(effective_hosts) < 2:
            return _plan_context.PlanContext(
                aggregate=agg_name,
                error=(
                    "less_than_2_destination_ready_hosts:"
                    + ",".join(sorted(excluded_hosts))
                ),
                hosts_in_agg=effective_hosts,
                excluded_hosts=excluded_hosts,
            )
        host_cpu_pct = {h: host_cpu_pct[h] for h in effective_hosts}
        host_ram_pct = {h: host_ram_pct[h] for h in effective_hosts}
        host_cores = {h: host_cores[h] for h in effective_hosts}
        host_free_cores = {h: host_free_cores[h] for h in effective_hosts}
        host_ram = {h: host_ram[h] for h in effective_hosts}
        host_free_ram = {h: host_free_ram[h] for h in effective_hosts}
        host_running_vms = {h: host_running_vms[h] for h in effective_hosts}
        host_placement = {h: host_placement[h] for h in effective_hosts}
    server_cache = cluster_cache["server_cache"]
    project_cache = cluster_cache["project_cache"]
    ambiguous_samples = {}
    vm_metrics = _sources_prometheus.get_all_vm_metrics(
        cfg, inventory_hosts, server_cache=server_cache, ambiguous=ambiguous_samples)
    all_vm_names, all_vm_details, vm_host_map = build_aggregate_vm_inventory(
        conn,
        inventory_hosts,
        server_cache,
        vm_metrics,
        project_cache,
        host_to_hypervisor=pass_hypervisor_map,
    )
    remember_project_names(cfg, project_cache)
    freshness = {}
    if journal is not None:
        from compute_maintenance.adapters.freshness import validate_runtime_freshness
        freshness = validate_runtime_freshness(cfg, effective_hosts,
            {vm: vm_host_map[vm] for vm in all_vm_details if vm in vm_host_map})
    vm_status_map = {
        vm_id: _execution_migrate.get_server_status(server_cache[vm_id])
        for vm_id in vm_host_map
        if vm_id in server_cache
    }
    compensation = {}
    if arrivals:
        wait_for_fresh_ram_samples(cfg, arrivals, effective_hosts)
        ram_instant, ram_sample_ts, vm_rss_gb = measure_receiving_hosts(cfg, arrivals, effective_hosts)
        for vm_id, pct in pre_migration_cpu(cfg, arrivals, all_vm_details).items():
            ambiguous_samples.setdefault(vm_id, {})["pre_cpu_pct"] = pct
        compensation = apply_arrival_compensation(
            arrivals, host_cpu_pct, host_ram_pct, host_cores, host_ram,
            vm_host_map, all_vm_details, server_cache, ram_metric_window_seconds(cfg),
            ambiguous_samples=ambiguous_samples, cpu_window=cpu_compensation_window_seconds(cfg),
            ram_instant=ram_instant, ram_sample_ts=ram_sample_ts, vm_rss_gb=vm_rss_gb,
        )
    host_az = {h: z for h, z in (cluster_cache.get("host_az") or {}).items()
               if h in set(inventory_hosts)} if cluster_cache.get("host_az") is not None else None

    excluded_vm_map = _selection.build_excluded_vm_map(all_vm_names, exclude_vm_patterns or [])
    excluded_vm_ids = set(excluded_vm_map)
    excluded_vms = _selection.build_excluded_vm_report(excluded_vm_map, all_vm_names)
    if excluded_vms:
        for pattern, names in _selection.group_excluded_vms_by_pattern(excluded_vms).items():
            logging.info(
                "Aggregate %s: exclude entry %r pins %d VM(s): %s",
                agg_name, pattern, len(names), ", ".join(names),
            )

    unmatched_patterns = [
        pattern for pattern in (exclude_vm_patterns or [])
        if pattern not in excluded_vm_map.values()
    ]
    if unmatched_patterns:
        logging.debug(
            "Aggregate %s: exclude entr(ies) matching no VM here: %s",
            agg_name, ", ".join(unmatched_patterns),
        )

    vm_ids = set(vm_host_map.keys())
    if enforce_server_groups:
        vm_to_groups = _planning_constraints.build_server_group_index(vm_ids, cluster_cache["server_groups"])
    else:
        vm_to_groups = {vm_id: [] for vm_id in vm_ids}

    baseline = _planning_balance.compute_baseline_metrics(host_cpu_pct, host_ram_pct)
    cpu_mean = baseline["cpu_mean"]
    ram_mean = baseline["ram_mean"]
    baseline_cpu_mad = baseline["baseline_cpu_mad"]
    baseline_ram_mad = baseline["baseline_ram_mad"]
    mode = baseline["mode"]
    max_cpu = baseline["max_cpu"]
    max_ram = baseline["max_ram"]
    cpu_weight = baseline["cpu_weight"]
    ram_weight = baseline["ram_weight"]
    cpu_signal = baseline["cpu_signal"]
    ram_signal = baseline["ram_signal"]
    baseline_weighted_mad = baseline["baseline_weighted_mad"]
    direct_improvement_threshold = _planning_balance.determine_improvement_threshold()

    host_max_instances = {
        h: cfg.max_instances_per_host.get(h, cfg.max_instances_per_host_default)
        for h in effective_hosts
    }
    reject_stats = {}
    dest_reject_stats = {}
    skip_stats = {}
    ctx = _models.BalanceContext(
        effective_hosts=effective_hosts,
        host_cpu_pct=host_cpu_pct,
        host_ram_pct=host_ram_pct,
        host_cores=host_cores,
        host_free_cores=host_free_cores,
        host_ram=host_ram,
        host_free_ram=host_free_ram,
        host_placement=host_placement,
        vm_host_map=vm_host_map,
        all_vm_details=all_vm_details,
        vm_to_groups=vm_to_groups,
        enforce_server_groups=enforce_server_groups,
        dest_ready_hosts=dest_ready_hosts,
        host_running_vms=host_running_vms,
        host_max_instances=host_max_instances,
        excluded_vm_ids=excluded_vm_ids,
        reject_stats=reject_stats,
        dest_reject_stats=dest_reject_stats,
        skip_stats=skip_stats,
        vm_status_map=vm_status_map,
        cooldown_remaining={vm: _execution_cooldown.store_for(cfg).remaining(vm) for vm in vm_ids},
        search_budget=getattr(cfg, "planner_candidate_budget", 100000),
        host_az=host_az,
    )

    from compute_maintenance.domain.snapshot import (
        ClusterSnapshot,
        freeze,
        freeze_context_inputs,
    )
    freeze_context_inputs(ctx)
    best_move, direct_improvement, direct_scope, hotspot_source_host = (_planning_balance.search_best_move_for_state(
        ctx,
        mode,
        baseline_weighted_mad,
        cpu_weight,
        ram_weight,
        direct_improvement_threshold,
    ) if plan_balance else (None, 0.0, "not_requested", None))
    direct_plan = _planning_balance.direct_move_to_plan(best_move, baseline_weighted_mad, direct_scope)

    # Optional nominal allocation-candidate sanity check for the final direct
    # candidate. This is deliberately *not* called for every VM x host pair.
    if (
        direct_plan
        and direct_plan.improvement >= direct_improvement_threshold
        and cfg.placement_final_allocation_candidate_check
    ):
        step = direct_plan.steps[0]
        ok, why = _sources_openstack.placement_allocation_candidate_sanity_check(
            conn,
            host_placement[step.dst_host],
            step.vcpus,
            round(step.vm_ram_gb * 1024.0),
        )
        if not ok:
            logging.warning(
                "Direct candidate %s %s->%s failed allocation-candidate sanity check: %s",
                step.vm_id, step.src_host, step.dst_host, why,
            )
            _planning_balance.record_rejection(ctx, "allocation_candidate_sanity", step.dst_host)
            direct_plan = None
            best_move = None
            direct_improvement = 0.0

    make_room_plan = None
    # Direct valid moves always win. Only invoke the more expensive two-step
    # planner when direct balancing cannot meet the improvement threshold.
    if plan_balance and not (
        direct_plan
        and direct_plan.improvement >= direct_improvement_threshold
    ):
        make_room_plan = _planning_balance.search_best_make_room_for_state(
            ctx,
            cfg,
            mode,
            baseline_weighted_mad,
            cpu_weight,
            ram_weight,
        )

    candidates = [p for p in (direct_plan, make_room_plan) if p is not None]
    best_plan_candidate = max(candidates, key=lambda p: p.improvement) if candidates else None
    improvement_threshold = _planning_balance.determine_plan_improvement_threshold(
        best_plan_candidate, cfg, direct_improvement_threshold
    )
    has_candidate = (
        best_plan_candidate is not None
        or best_move is not None
        or ctx.technical_candidate_count > 0
    )
    has_valid_move = bool(
        best_plan_candidate and best_plan_candidate.improvement >= improvement_threshold
    )
    best_plan = best_plan_candidate if has_valid_move else None
    best_improvement = (
        best_plan_candidate.improvement
        if best_plan_candidate is not None
        else direct_improvement
    )
    search_scope = (
        best_plan_candidate.search_scope
        if best_plan_candidate is not None
        else direct_scope
    )

    detail_status = _planning_balance.classify_cluster_detail_status(
        mode=mode,
        max_cpu=max_cpu,
        max_ram=max_ram,
        baseline_cpu_mad=baseline_cpu_mad,
        baseline_ram_mad=baseline_ram_mad,
        has_valid_move=has_valid_move,
    )
    reason = _planning_balance.derive_reason(has_candidate, has_valid_move, mode)
    severity = _severity.map_detail_status_to_severity(detail_status, reason)
    exit_code = _severity.map_severity_to_exit_code(severity)

    # rc11 ladder rung "soft_aa_heal": after the hotspots, before MAD
    # balancing. A feasible heal move replaces the balancing plan (still one
    # move per aggregate per run); the load classification above is kept as
    # load_status and combined with the heal WARNING in the monitor's
    # severity order, so a CRITICAL load state is never masked.
    soft_aa = (
        _planning_balance.plan_soft_aa_heal(ctx, mode, baseline_weighted_mad, cpu_weight, ram_weight)
        if plan_balance else None
    )
    load_status = None
    if soft_aa is not None and soft_aa["plan"] is not None:
        load_status = {"detail_status": detail_status, "severity": severity,
                       "exit_code": exit_code, "reason": reason}
        best_plan = best_plan_candidate = soft_aa["plan"]
        has_valid_move = has_candidate = True
        best_improvement = best_plan.improvement
        search_scope = best_plan.search_scope
        improvement_threshold = _planning_balance.determine_plan_improvement_threshold(
            best_plan, cfg, direct_improvement_threshold)
        detail_status = _severity.SOFT_AA_HEAL_DETAIL_STATUS
        reason = None
        severity, exit_code = _severity.heal_verdict(exit_code)

    placement_status_by_host = {
        h: _planning_constraints.placement_capacity_status(host_placement[h]) for h in effective_hosts
    }
    placement_summary = {
        "vcpu_pressure_hosts": sum(
            1 for h in effective_hosts
            if host_placement[h]["VCPU"]["allocation_pct"] >= _config.runtime().placement_pressure_threshold
        ),
        "memory_pressure_hosts": sum(
            1 for h in effective_hosts
            if host_placement[h]["MEMORY_MB"]["allocation_pct"] >= _config.runtime().placement_pressure_threshold
        ),
        "vcpu_full_hosts": sum(
            1 for h in effective_hosts if host_placement[h]["VCPU"]["free"] <= 0
        ),
        "memory_full_hosts": sum(
            1 for h in effective_hosts if host_placement[h]["MEMORY_MB"]["free"] <= 0
        ),
    }

    state = _plan_context.PlanContext(
        aggregate=agg_name,
        hosts_in_agg=effective_hosts,
        host_cpu_pct=host_cpu_pct,
        host_ram_pct=host_ram_pct,
        host_cores=host_cores,
        host_free_cores=host_free_cores,
        host_ram=host_ram,
        host_free_ram=host_free_ram,
        host_running_vms=host_running_vms,
        host_max_instances=host_max_instances,
        host_placement=host_placement,
        placement_status_by_host=placement_status_by_host,
        placement_summary=placement_summary,
        dest_ready_hosts=sorted(dest_ready_hosts),
        dest_not_ready_reasons=dest_not_ready_reasons,
        excluded_hosts=excluded_hosts,
        all_vm_names=all_vm_names,
        all_vm_details=all_vm_details,
        vm_host_map=vm_host_map,
        vm_status_map=vm_status_map,
        vm_to_groups=vm_to_groups,
        excluded_vm_ids=sorted(excluded_vm_ids),
        excluded_vms=excluded_vms,
        unmatched_exclude_patterns=unmatched_patterns,
        cpu_mean=cpu_mean,
        ram_mean=ram_mean,
        baseline_cpu_mad=baseline_cpu_mad,
        baseline_ram_mad=baseline_ram_mad,
        baseline_weighted_mad=baseline_weighted_mad,
        improvement_threshold=improvement_threshold,
        # Individual executable moves are revalidated against the direct-move
        # threshold. The higher plan threshold above is paid once by the whole
        # two-step chain, not again by its final step after room has been made.
        step_improvement_threshold=direct_improvement_threshold,
        make_room_improvement_threshold=max(
            direct_improvement_threshold,
            getattr(
                cfg,
                "make_room_min_improvement",
                _config.DEFAULT_MAKE_ROOM_MIN_IMPROVEMENT,
            ),
        ),
        mode=mode,
        max_cpu=max_cpu,
        max_ram=max_ram,
        cpu_weight=cpu_weight,
        ram_weight=ram_weight,
        cpu_signal=cpu_signal,
        ram_signal=ram_signal,
        # best_move remains for event/backward compatibility; best_plan is the
        # authoritative action selected by the new engine.
        best_move=best_move,
        best_plan_candidate=best_plan_candidate,
        best_plan=best_plan,
        best_improvement=best_improvement,
        has_candidate=has_candidate,
        has_valid_move=has_valid_move,
        search_scope=search_scope,
        hotspot_source_host=hotspot_source_host,
        detail_status=detail_status,
        reason=reason,
        severity=severity,
        exit_code=exit_code,
        reject_stats=reject_stats,
        dest_reject_stats=dest_reject_stats,
        skip_stats=skip_stats,
        soft_aa_heal=soft_aa,
        load_status=load_status,
        collected_at=collected_at,
        # Planning context: every refresh of this state (revalidation before
        # submit, MAKE_ROOM step 2) must rebuild the baseline with the same rule.
        exclude_unready_hosts=exclude_unready_hosts,
        host_az=host_az,
        settle_mode=settle_mode,
        arrival_compensation=compensation,
        planner_candidate_budget=ctx.search_budget,
        balance_candidates_examined=ctx.candidates_examined,
        raw_metric_max_age_seconds=freshness,
        snapshot=ClusterSnapshot(agg_name, collected_at, freeze(host_cpu_pct),
                                 freeze(host_ram_pct), freeze(vm_host_map), freeze(freshness)),
    )
    state.collection_seconds = time.time() - collected_at
    state.message = _report.build_monitor_message(state)
    return state
