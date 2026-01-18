#!/usr/bin/env python3
"""
OpenStack Load Leveller — Automated VM migration to minimize MAD/spread.
Balances CPU and RAM load across aggregates, repeatedly migrating VMs for optimal balance.
"""
import argparse
import time
from datetime import datetime
import os
import logging
import requests
import dotenv
import click
import openstack
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
DEFAULT_BEST_IMPROVEMENT = 0.05
PROMETHEUS_NODE_JOB = 'Openstack-Node-Exporter'
PROMETHEUS_LIBVIRT_JOB = 'Openstack-LibVirt-Exporter'

DEFAULT_PROMETHEUS_CPU_USED = (
    f'sort_desc(100 - avg(irate(node_cpu_seconds_total{{job="{PROMETHEUS_NODE_JOB}",mode="idle"}}[5m]) * 100) by (alias))'
)
DEFAULT_PROMETHEUS_MEM_USED = (
    f'sort_desc(100 - ((avg_over_time(node_memory_MemAvailable_bytes{{job="{PROMETHEUS_NODE_JOB}"}}[5m]) * 100) / avg_over_time(node_memory_MemTotal_bytes{{job="{PROMETHEUS_NODE_JOB}"}}[5m])))'
)

TARGET_AVG_MAD = 1


PROMETHEUS_QUERY_URL = "http://kprometheus.com:9090/api/v1/query"

# === MIGRATION MONITOR CONSTANTS ============================================
# These control migration status polling and interpretation.
# Integrated for detailed migration tracking during live-migrate actions.

POLL_INTERVAL = 2
SUCCESS_STATUSES = {'success', 'completed', 'done', 'finished', 'succeeded'}
FAILED_STATUSES = {'error', 'failed'}
PROGRESS_STATUSES = {'running', 'migrating', 'pre-migrating', 'queued'}


class PrometheusError(Exception):
    pass


class ConfigError(Exception):
    pass


class LoadbalancerConfig:
    def __init__(
        self,
        prometheus_query_url=PROMETHEUS_QUERY_URL,
        prometheus_query_mem_used=DEFAULT_PROMETHEUS_MEM_USED,
        prometheus_query_cpu_used=DEFAULT_PROMETHEUS_CPU_USED,
    ):
        self.prometheus_query_url = prometheus_query_url
        self.prometheus_query_mem_used = prometheus_query_mem_used
        self.prometheus_query_cpu_used = prometheus_query_cpu_used

    @classmethod
    def load_config(cls, filename: str):
        dotenv.load_dotenv(dotenv_path=filename)
        return cls(
            prometheus_query_url=os.getenv("PROMETHEUS_QUERY_URL", PROMETHEUS_QUERY_URL),
            prometheus_query_mem_used=os.getenv("PROMETHEUS_MEM_USED", DEFAULT_PROMETHEUS_MEM_USED),
            prometheus_query_cpu_used=os.getenv("PROMETHEUS_CPU_USED", DEFAULT_PROMETHEUS_CPU_USED),
        )


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s: %(message)s'
    )


def do_query(query_url: str, query: str):
    try:
        params = {'query': query}
        response = requests.get(query_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('status') != 'success':
            raise PrometheusError(f"Prometheus query error: {data.get('error', 'Unknown error')}")
        return data
    except requests.RequestException as e:
        raise PrometheusError(f"Prometheus HTTP error: {e}")


def get_metrics(query_url: str, query: str):
    data = do_query(query_url, query)
    results = {}
    try:
        for r in data["data"]["result"]:
            host = r["metric"]["alias"]
            value = float(r["value"][1])
            results[host] = {"host": host, "value": value}
        return results
    except (KeyError, IndexError, TypeError) as e:
        raise PrometheusError(f"Prometheus returned malformed results: {e}")


def get_openstack_connection():
    return openstack.connect()


def mean_absolute_deviation(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum(abs(x - mean) for x in values) / len(values)


# === SERVER GROUP HANDLING ==================================================


def build_server_group_index(conn, vm_ids):
    """
    Build mapping: vm_id -> list of server groups it belongs to.

    Each entry: {
        "id": <group_id>,
        "name": <group_name>,
        "policy": "affinity" | "anti-affinity" | None,
        "members": [server_id, ...]
    }

    If anything fails, we log a warning and return an empty mapping so
    the rest of the algorithm can continue without server-group constraints.
    """
    vm_ids = set(vm_ids)
    vm_to_groups = {vm_id: [] for vm_id in vm_ids}

    try:
        for sg in conn.compute.server_groups(all_projects=True):
            members = list(getattr(sg, "member_ids", []) or [])      
            if not members:
                continue

            # Only care if any member is in our aggregate
            if not any(m in vm_ids for m in members):
                continue

            # Policy can be "policies" (list) or "policy" (string)
            policy = None
            policies = getattr(sg, "policies", None)
            if policies and isinstance(policies, (list, tuple)) and policies:
                policy = policies[0]
            else:
                policy = getattr(sg, "policy", None)

            info = {
                "id": getattr(sg, "id", None),
                "name": getattr(sg, "name", None),
                "policy": policy,
                "members": members,
            }

            for m in members:
                if m in vm_ids:
                    vm_to_groups[m].append(info)

        logging.info(
            "Built server group index for %d VMs (some may have 0 groups).",
            len(vm_to_groups),
        )
    except Exception as e:
        logging.warning(
            "Failed to build server group index, continuing WITHOUT server-group constraints: %s",
            e,
        )
        # Return all empty lists; migration will ignore server-group rules
        return {vm_id: [] for vm_id in vm_ids}

    return vm_to_groups


def is_server_group_move_compliant(vm_id, src_host, dst_host, vm_host_map, vm_to_groups):
    """
    Check if moving vm_id from src_host to dst_host would keep all
    of its server groups compliant.

    - affinity: all members must end up on the same host
    - anti-affinity: no two members may end up on the same host

    vm_host_map: current mapping vm_id -> host (for members we know about)
    vm_to_groups: vm_id -> list of group dicts (from build_server_group_index)
    """
    groups = vm_to_groups.get(vm_id) or []
    if not groups:
        return True  # no server groups, always OK

    for g in groups:
        policy = g.get("policy")
        members = g.get("members") or []
        if not policy or not members:
            continue

        # Simulate new host placement for all members in this group.
        if policy == "affinity":
            ref_host = None
            for member in members:
                host = dst_host if member == vm_id else vm_host_map.get(member)
                if host is None:
                    continue
                if ref_host is None:
                    ref_host = host
                elif host != ref_host:
                    logging.debug(
                        "Reject move of %s %s→%s: affinity group %s/%s "
                        "would span hosts (%s vs %s).",
                        vm_id,
                        src_host,
                        dst_host,
                        g.get("name"),
                        g.get("id"),
                        ref_host,
                        host,
                    )
                    return False

        elif policy == "anti-affinity":
            seen_hosts = set()
            for member in members:
                host = dst_host if member == vm_id else vm_host_map.get(member)
                if host is None:
                    continue
                if host in seen_hosts:
                    logging.debug(
                        "Reject move of %s %s→%s: anti-affinity group %s/%s "
                        "would have two members on host %s.",
                        vm_id,
                        src_host,
                        dst_host,
                        g.get("name"),
                        g.get("id"),
                        host,
                    )
                    return False
                seen_hosts.add(host)

    return True


# === VM SCORE CALCULATION ===================================================


def calculate_vm_scores(host, cfg):
    # Per-VM RAM usage (%): 100 - usable/available * 100
    mem_query = (
        f'(100 - (avg_over_time(libvirt_domain_stat_memory_usable_bytes{{job="{PROMETHEUS_LIBVIRT_JOB}", host="{host}"}}[5m]) '
        f'/ avg_over_time(libvirt_domain_stat_memory_available_bytes{{job="{PROMETHEUS_LIBVIRT_JOB}", host="{host}"}}[5m])) * 100)'
    )
    mem_data = do_query(cfg.prometheus_query_url, mem_query)
    mem_by_vm = {
        entry['metric']['instanceId']: float(entry['value'][1])
        for entry in mem_data['data']['result']
    }

    # Per-VM CPU usage per vCPU (%)
    cpu_query = (
        f'(irate(libvirt_domain_info_cpu_time_seconds_total{{job="{PROMETHEUS_LIBVIRT_JOB}", host="{host}"}}[5m]) '
        f'/ libvirt_domain_info_virtual_cpus{{job="{PROMETHEUS_LIBVIRT_JOB}", host="{host}"}}) * 100'
    )
    cpu_data = do_query(cfg.prometheus_query_url, cpu_query)
    cpu_by_vm = {
        entry['metric']['instanceId']: float(entry['value'][1])
        for entry in cpu_data['data']['result']
    }

    conn = get_openstack_connection()
    vm_names = {}
    vm_details = {}

    for vm_id in set(mem_by_vm.keys()) | set(cpu_by_vm.keys()):
        try:
            server = conn.compute.get_server(vm_id)   
            vcpus = 1
            ram_mb = 2048             
            vcpus = server.flavor.get('vcpus')
            ram_mb = server.flavor.get('ram')
            ram_gb = float(ram_mb) / 1024.0            
            name = getattr(server, "name", vm_id)            
        except Exception as e:
        #   print(f"[DEBUG] Failed to get full server info for {vm_id}: {e}")
            # Fallback if Nova lookup fails
            vcpus = 1
            ram_gb = 2.0
            name = vm_id

        cpu_pct = cpu_by_vm.get(vm_id, 0.0)
        ram_pct = mem_by_vm.get(vm_id, 0.0)

        vm_names[vm_id] = name
        vm_details[vm_id] = {
            "cpu_pct": cpu_pct,
            "ram_pct": ram_pct,
            "vcpus": vcpus,
            "ram_gb": ram_gb,
            "host": host,
        }
    return vm_names, vm_details


def monitor_migration(conn, server_id, vm_name, timeout=600, verbose=True):
    waited = 0
    migration = None
    orig_host = None
    dest_host = None

    while waited < timeout:
        server_migrations = list(conn.compute.server_migrations(server_id))
        if server_migrations:
            migration = server_migrations[-1]
            orig_host = getattr(migration, 'source_node', None)
            dest_host = getattr(migration, 'dest_node', None)
            break
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
        if verbose:
            print(f"Waiting for migration object to appear for VM {vm_name}... ({waited})")
    if not migration:
        if verbose:
            print(f"No migration record found for VM {vm_name} after waiting. Exiting monitoring loop.")
        return False

    migration_id = getattr(migration, "id", None) or getattr(migration, "uuid", None)
    if verbose:
        print(f"Migration ID: {migration_id}")
        print(f"Status: {getattr(migration, 'status', 'n/a')}")
        print(f"Source Server: {orig_host}")
        print(f"Destination Server: {dest_host}\n")

    start_time = time.time()
    last_state = None

    while True:
        try:
            migrations = list(conn.compute.server_migrations(server_id))
            mig = None
            for m in migrations:
                mid = getattr(m, "id", None) or getattr(m, "uuid", None)
                if mid == migration_id:
                    mig = m
                    break
            if mig is None and migrations:
                mig = migrations[-1]
            if not mig:
                # Fallback: check global migrations and server status
                global_found = False
                for m in conn.compute.migrations():
                    mid = getattr(m, "id", None) or getattr(m, "uuid", None)
                    if str(mid) == str(migration_id):
                        status = getattr(m, "status", None)
                        if verbose:
                            print(f"\nMigration {migration_id} found in global migrations with status: {status}")
                        if status and status.lower() in SUCCESS_STATUSES:
                            print(f"Migration {migration_id} completed successfully (global record).")
                            return True
                        elif status and status.lower() in FAILED_STATUSES:
                            print(f"Migration {migration_id} failed (global record).")
                            return False
                        else:
                            print(f"Migration {migration_id} is in state '{status}' (global record).")
                        global_found = True
                        break
                if not global_found:
                    server = conn.compute.get_server(server_id)
                    current_host = getattr(server, "OS-EXT-SRV-ATTR:host", None)
                    status = getattr(server, "status", None)
                    if dest_host and current_host == dest_host and status == "ACTIVE":
                        print(f"\nMigration {migration_id} completed: VM is now ACTIVE on {dest_host}.")
                        return True
                    else:
                        print(f"\nMigration {migration_id} vanished before completion. VM is on {current_host} with status {status}. Please check logs.")
                        return False
                break

            percent = None
            server = conn.compute.get_server(server_id)
            if hasattr(server, 'progress') and server.progress is not None:
                percent = server.progress
            elif hasattr(mig, 'progress') and mig.progress is not None:
                percent = mig.progress
            elif isinstance(mig, dict) and 'progress' in mig and mig['progress'] is not None:
                percent = mig['progress']
            progress_str = f"{percent}%" if percent is not None else "?"

            status = getattr(mig, "status", None) or (mig['status'] if isinstance(mig, dict) and 'status' in mig else 'n/a')
            state = "UNKNOWN"
            if status and status.lower() in PROGRESS_STATUSES:
                state = "IN PROGRESS"
            elif status and status.lower() in SUCCESS_STATUSES:
                state = "SUCCESS"
            elif status and status.lower() in FAILED_STATUSES:
                state = "FAILED"
            else:
                state = f"UNKNOWN ({status})"

            if state != last_state and verbose:
                print()
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Migration {migration_id} | VM: {vm_name} | "
                f"State: {state} | Progress: {progress_str}   ",
                end="\r", flush=True
            )
            last_state = state

            if state == "SUCCESS":
                print(f"\nMigration {migration_id} completed successfully!")
                return True
            elif state == "FAILED":
                print(f"\nMigration {migration_id} failed!")
                return False

            if (time.time() - start_time) > timeout:
                print(f"\nMigration {migration_id} monitoring timed out after {timeout} seconds.")
                return False

            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"\nMonitoring error or migration disappeared (will retry): {e}")
            time.sleep(POLL_INTERVAL)


def get_host_resources_map(conn):
    host_resources = {}

    for hypervisor in conn.compute.hypervisors():
        full = conn.compute.get_hypervisor(hypervisor.id)

        name = getattr(full, "name", None)
        total_cores = getattr(full, "vcpus", 0) or 0
        used_cores = getattr(full, "vcpus_used", 0) or 0

        mem_mb = getattr(full, "memory_size", 0) or 0
        used_mem_mb = getattr(full, "memory_used", 0) or 0

        if not name:
            continue

        # Overcommit: your formula ~ "each core ×7 capacity"
        free_cores = (total_cores * 7) - used_cores
        free_cores = max(free_cores, 0)

        # Your RAM formula including 20% overhead and 24GB reserved
        total_ram_gb = ((float(mem_mb) * 1.2) - 24576) / 1024.0
        free_ram_gb = ((float(mem_mb) * 1.2) - 24576 - float(used_mem_mb)) / 1024.0

        total_ram_gb = max(total_ram_gb, 0.0)
        free_ram_gb = max(free_ram_gb, 0.0)

        host_resources[name] = {
            "total_cores": int(total_cores),
            "free_cores": float(free_cores),
            "total_ram_gb": float(total_ram_gb),
            "free_ram_gb": float(free_ram_gb),
        }

    return host_resources


def auto_balance_aggregate(cfg, conn, agg_name, hosts_in_agg, dry_run=False, enforce_server_groups=True, assume_yes=False):
    print(f"\n[Aggregate: {agg_name}] Auto-migrating all VMs to improve balance.\n")

    if enforce_server_groups:
        logging.info("Server group checks ENABLED (default behavior).")
    else:
        logging.warning("Server group checks DISABLED (--no-server-groups). Affinity/anti-affinity will be ignored!")

    # Gather host resources (total cores/RAM) from hypervisors
    host_resources = get_host_resources_map(conn)
    host_cores = {
    h: host_resources.get(h, {}).get("total_cores", 64)
        for h in hosts_in_agg
    }

    host_free_cores = {
        h: host_resources.get(h, {}).get("free_cores", 64)
        for h in hosts_in_agg
    }

    host_ram = {
        h: host_resources.get(h, {}).get("total_ram_gb", 256.0)
        for h in hosts_in_agg
    }

    host_free_ram = {
        h: host_resources.get(h, {}).get("free_ram_gb", 256.0)
        for h in hosts_in_agg
    }
      
    # Per-host usage from Prometheus
    cpu_metrics = get_metrics(cfg.prometheus_query_url, cfg.prometheus_query_cpu_used)
    mem_metrics = get_metrics(cfg.prometheus_query_url, cfg.prometheus_query_mem_used)

    host_cpu_pct = {}
    host_ram_pct = {}

    # Only keep hosts that actually have both CPU & RAM metrics
    for h in hosts_in_agg:
        cpu_entry = cpu_metrics.get(h)
        mem_entry = mem_metrics.get(h)
        if not cpu_entry or not mem_entry:
            logging.warning(f"Host {h} in aggregate {agg_name} missing CPU or RAM metrics, skipping.")
            continue
        host_cpu_pct[h] = cpu_entry['value']
        host_ram_pct[h] = mem_entry['value']

    hosts_in_agg = list(host_cpu_pct.keys())
    if len(hosts_in_agg) < 2:
        logging.info(f"Aggregate {agg_name}: less than 2 hosts with Prometheus data after filtering, skipping.")
        return

    # Gather all VMs and track mapping VM->host
    all_vm_names = {}
    all_vm_details = {}
    vm_host_map = {}
    for src_host in hosts_in_agg:
        vm_names, vm_details = calculate_vm_scores(src_host, cfg)
        for vm_id, det in vm_details.items():
            all_vm_names[vm_id] = vm_names[vm_id]
            all_vm_details[vm_id] = det
            vm_host_map[vm_id] = src_host

    # === Build server group index for compliance checks
    vm_ids = set(vm_host_map.keys())
    if enforce_server_groups:
        vm_to_groups = build_server_group_index(conn, vm_ids)
    else:
        vm_to_groups = {vm_id: [] for vm_id in vm_ids}

    print("Baseline host CPU and RAM usage before migration:")
    print(f"{'Host':<18} {'CPU %':>7} {'RAM %':>8} {'Cores':>7} {'RAM (GB)':>10} {'Avail Cores':>13} {'Avail RAM (GB)':>15}")
    print("-" * 80)
    for h in hosts_in_agg:
        cpu_pct = host_cpu_pct[h]
        ram_pct = host_ram_pct[h]
        cores = host_cores[h]
        ram_gb = host_ram[h]
        avail_cores = host_free_cores[h]
        #avail_cores = cores * (1 - cpu_pct / 100)
        avail_ram = host_free_ram[h]     
        #avail_ram = ram_gb * (1 - ram_pct / 100)
        print(f"{h:<18} {cpu_pct:7.2f} {ram_pct:8.2f} {cores:7d} {ram_gb:10.2f} {avail_cores:13.2f} {avail_ram:15.2f}")
    print()

    # Baseline MADs (for info)
    cpu_mean = sum(host_cpu_pct.values()) / len(host_cpu_pct)
    ram_mean = sum(host_ram_pct.values()) / len(host_ram_pct)
    baseline_cpu_mad = mean_absolute_deviation(list(host_cpu_pct.values()))
    baseline_ram_mad = mean_absolute_deviation(list(host_ram_pct.values()))
    baseline_avg_mad = (baseline_cpu_mad + baseline_ram_mad) / 2

    print(f"Baseline Means: CPU={cpu_mean:.2f}%  RAM={ram_mean:.2f}%")
    print(f"Baseline MADs:  CPU={baseline_cpu_mad:.2f}  RAM={baseline_ram_mad:.2f}  Avg MAD={baseline_avg_mad:.2f}\n")

    migration_round = 1
    prev_avg_mad = baseline_avg_mad  
    # Hill-climbing: keep applying the best single migration that reduces MAD
    while True:
        best_move = None
        best_improvement = 0.0

        # Baseline for this round (current state)
        baseline_cpu_mad = mean_absolute_deviation(list(host_cpu_pct.values()))
        baseline_ram_mad = mean_absolute_deviation(list(host_ram_pct.values()))
        baseline_avg_mad = (baseline_cpu_mad + baseline_ram_mad) / 2

        # Simulate all possible VM moves
        for vm_id, src_host in list(vm_host_map.items()):
            details = all_vm_details[vm_id]
            vcpus = details.get("vcpus", 1)
            vm_cpu_pct = details.get("cpu_pct", 0.0)
            vm_ram_gb = details.get("ram_gb", 2.0)
            vm_ram_pct = details.get("ram_pct", 0.0)

            # "Real" cores consumed by this VM on its current host
            real_vm_cores = vcpus * (vm_cpu_pct / 100.0)
            real_vm_ram = vm_ram_gb * (vm_ram_pct / 100.0)
            for dst_host in hosts_in_agg:
                if dst_host == src_host:
                    continue

                # Check dst host has enough spare capacity
                host_cpu_avail = host_free_cores[dst_host]
                #host_cpu_avail = host_cores[dst_host] * (1 - host_cpu_pct[dst_host] / 100)
                host_ram_avail = host_free_ram[dst_host]                    
                #host_ram_avail = host_ram[dst_host] * (1 - host_ram_pct[dst_host] / 100)
                if host_cpu_avail < vcpus or host_ram_avail < vm_ram_gb:
                    continue

                # === Server group compliance check ===
                if enforce_server_groups:
                    if not is_server_group_move_compliant(vm_id, src_host, dst_host, vm_host_map, vm_to_groups):
                        # Move would violate affinity/anti-affinity, skip
                        continue
                
              #  === Server group compliance check ===
                # if enforce_server_groups:
                #     if not is_server_group_move_compliant(vm_id, src_host, dst_host, vm_host_map, vm_to_groups):
                #         vm_name = all_vm_names.get(vm_id, vm_id[:8])
                #         violating_groups = [
                #             f"{g['name'] or g['id'][:8]} ({g['policy']})"
                #             for g in vm_to_groups.get(vm_id, [])
                #             if g.get("policy") in ("affinity", "anti-affinity")
                #         ]
                #         print(f"  REJECTED MOVE (server-group violation): "
                #               f"{vm_name} {src_host} → {dst_host} "
                #               f"blocked by group(s): {', '.join(violating_groups) or 'unknown'}")
                #         logging.debug(
                #             "Server-group violation: VM %s (%s) cannot move %s→%s due to %s rules",
                #             vm_name, vm_id[:8], src_host, dst_host, violating_groups
                #         )
                #         continue
                # else:
                #     # Optional: show when moves are allowed only because checks are disabled
                #     if not enforce_server_groups and vm_to_groups.get(vm_id):
                #         vm_name = all_vm_names.get(vm_id, vm_id[:8])
                #         groups = [f"{g['name'] or g['id'][:8]} ({g['policy']})" 
                #                 for g in vm_to_groups.get(vm_id, []) 
                #                 if g.get("policy")]
                #         if groups:
                #             print(f"  ALLOWED (but --no-server-groups used): "
                #                   f"{vm_name} {src_host} → {dst_host} "
                #                   f"would violate: {', '.join(groups)}")

                # Impact on src host when we remove this VM
                cpu_impact = (real_vm_cores / host_cores[src_host]) * 100
                ram_impact = (real_vm_ram / host_ram[src_host]) * 100

                # Simulate move
                new_host_cpu = host_cpu_pct.copy()
                new_host_ram = host_ram_pct.copy()
                new_host_cpu[src_host] -= cpu_impact
                new_host_cpu[dst_host] += cpu_impact
                new_host_ram[src_host] -= ram_impact
                new_host_ram[dst_host] += ram_impact

                mad_cpu = mean_absolute_deviation(list(new_host_cpu.values()))
                mad_ram = mean_absolute_deviation(list(new_host_ram.values()))
                avg_mad = (mad_cpu + mad_ram) / 2
                mad_impr = baseline_avg_mad - avg_mad

                if mad_impr > best_improvement:
                    best_improvement = mad_impr
                    best_move = (
                        vm_id, src_host, dst_host,
                        cpu_impact, ram_impact,
                        new_host_cpu, new_host_ram,
                        real_vm_cores, vcpus, real_vm_ram, vm_ram_gb,
                        host_cpu_avail, host_ram_avail,
                        avg_mad, mad_impr, mad_cpu, mad_ram
                    )

        # Apply best move if it improves MAD
        if best_move and best_improvement > DEFAULT_BEST_IMPROVEMENT:  # threshold to avoid float noise
            (
                vm_id, src_host, dst_host,
                cpu_impact, ram_impact,
                new_host_cpu, new_host_ram,
                real_vm_cores, vcpus, real_vm_ram, vm_ram_gb,
                dest_cpu_avail, dest_ram_avail,
                new_avg_mad, mad_impr, new_cpu_mad, new_ram_mad
            ) = best_move

            vm_name = all_vm_names.get(vm_id, vm_id)
            new_cpu_mean = sum(new_host_cpu.values()) / len(new_host_cpu)
            new_ram_mean = sum(new_host_ram.values()) / len(new_host_ram)

            print(f"[Round {migration_round}] Migrate {vm_name} from {src_host} to {dst_host}")
            print(f"   {vm_name} ({src_host} → {dst_host}): Need {real_vm_cores:.2f} real cores(flavor {vcpus}), {real_vm_ram} real GB(flavor {vm_ram_gb:.2f} GB) — "
                  f"Available {dest_cpu_avail:.2f} real cores, {dest_ram_avail:.2f} GB.")
            print(f"   CPU impact: {cpu_impact:.2f}%  | RAM impact: {ram_impact:.2f}%")
            print(f"   New host CPU: {[f'{v:.2f}' for v in new_host_cpu.values()]}")
            print(f"   New host RAM: {[f'{v:.2f}' for v in new_host_ram.values()]}")
            print(f"   Means after migration: CPU={new_cpu_mean:.2f}%  RAM={new_ram_mean:.2f}%")
            print(f"   MADs after migration:  CPU={new_cpu_mad:.2f}  RAM={new_ram_mad:.2f}  Avg MAD={new_avg_mad:.2f}")
            print(f"   MAD improvement this round: baseline_avg_mad ({baseline_avg_mad:.3f}) - "
                  f"new_avg_mad ({new_avg_mad:.3f}) = {mad_impr:.3f}\n")

            if not dry_run:
                print(f'vm_id {vm_id} host {dst_host}')              
                # Simple yes/no question
                if not assume_yes:
                    answer = input(f"Do you want to migrate {vm_name} ({vm_id}) from {src_host} to {dst_host}? (y/n): ").strip().lower()
                    if answer != "y":
                        print("Migration cancelled.")
                        return
                conn.compute.live_migrate_server(server=vm_id, host=dst_host, block_migration=False)
                monitor_migration(conn, vm_id, vm_name)
                return

            # Update in-memory state
            host_cpu_pct = new_host_cpu
            host_ram_pct = new_host_ram
            vm_host_map[vm_id] = dst_host
            prev_avg_mad = new_avg_mad
            migration_round += 1
        else:
            print("No further improvement possible in this aggregate.\n")
            break


def balance_by_aggregate(cfg: LoadbalancerConfig, dry_run=False, aggregate_names=None, enforce_server_groups=True, assume_yes=False):
    conn = get_openstack_connection()
    aggregate_map = {}

    for agg in conn.compute.aggregates():
        agg_name = getattr(agg, 'name', None) or getattr(agg, 'id', str(id(agg)))
        if aggregate_names and agg_name not in aggregate_names:
            continue
        aggregate_map[agg_name] = list(agg.hosts)

    for agg_name, hosts in aggregate_map.items():
        hosts_in_agg = hosts
        if len(hosts_in_agg) < 2:
            logging.info(f"Aggregate {agg_name}: less than 2 hosts, skipping.")
            continue
        auto_balance_aggregate(cfg, conn, agg_name, hosts_in_agg, dry_run=dry_run, enforce_server_groups=enforce_server_groups, assume_yes=assume_yes)


@click.command()
@click.option('--config', default="/etc/loadleveller-secrets.conf", help="Path to config file")
@click.option('--verbose', is_flag=True, help="Enable verbose logging")
@click.option('--dry-run', is_flag=True, help="Only show what would be migrated, do not migrate")
@click.option('--aggregate', '-a', multiple=True, help="Only operate on these aggregate(s)")
@click.option(
    '--no-server-groups',
    is_flag=True,
    help="Disable server group (affinity/anti-affinity) checks. WARNING: may violate policies."
)
@click.option('-y', '--yes', 'assume_yes', is_flag=True, help="Skip y/n prompt and auto-approve migrations")
def main(config, verbose, dry_run, aggregate, no_server_groups, assume_yes):
    setup_logging(verbose)
    enforce_server_groups = not no_server_groups
    try:
        cfg = LoadbalancerConfig.load_config(config)
        aggregate_names = list(aggregate) if aggregate else None
        balance_by_aggregate(
            cfg,
            dry_run=dry_run,
            aggregate_names=aggregate_names,
            enforce_server_groups=enforce_server_groups,
            assume_yes=assume_yes,
        )
    except PrometheusError as e:
        logging.error(f"Prometheus query failed: {e}")
    except ConfigError as e:
        logging.error(f"Configuration error: {e}")
    except KeyError as e:
        logging.error(f"Missing required environment variable: {e}")
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
