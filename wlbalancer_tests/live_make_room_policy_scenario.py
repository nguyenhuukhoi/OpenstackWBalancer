#!/usr/bin/env python3
"""Opt-in controller utility for the mutating MAKE_ROOM live scenario.

This file is deliberately not collected by pytest.  It provides small,
checkpointed operations so an interrupted test can inspect and restore the two
scenario VMs instead of hiding several large live migrations in one test case.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.legacy_api import balance as wlb  # noqa: E402


def emit(payload):
    print(json.dumps(payload, sort_keys=True))


def get_servers(conn, vm_ids):
    servers = {}
    for vm_id in vm_ids:
        server = conn.compute.get_server(vm_id)
        if not server:
            raise RuntimeError(f"VM not found: {vm_id}")
        servers[vm_id] = server
    return servers


def status(conn, vm_ids):
    servers = get_servers(conn, vm_ids)
    result = {}
    for vm_id, server in servers.items():
        migrations = list(conn.compute.server_migrations(server))
        active = [
            {
                "id": str(wlb.get_migration_identifier(item)),
                "status": str(wlb.get_migration_status(item)),
            }
            for item in migrations
            if str(wlb.get_migration_status(item)).lower() in wlb.PROGRESS_STATUSES
        ]
        result[vm_id] = {
            "name": wlb.get_resource_field(server, "name", "Name") or vm_id,
            "host": wlb.get_server_current_host(server),
            "status": wlb.get_server_status(server),
            "active_migrations": active,
        }
    emit(result)


def migrate(conn, vm_id, destination, timeout):
    server = get_servers(conn, [vm_id])[vm_id]
    vm_name = wlb.get_resource_field(server, "name", "Name") or vm_id
    source = wlb.get_server_current_host(server)
    if source == destination:
        emit(
            {
                "vm_id": vm_id,
                "name": vm_name,
                "source": source,
                "destination": destination,
                "result": "already_there",
            }
        )
        return

    if wlb.get_server_status(server) != wlb.ACTIVE_VM_STATUS:
        raise RuntimeError(
            f"refusing migration: {vm_name} status={wlb.get_server_status(server)}"
        )

    ready, reason = wlb.is_nova_compute_service_ready(conn, destination)
    if not ready:
        raise RuntimeError(f"destination {destination} is not ready: {reason}")

    preexisting_ids = wlb.get_server_migration_ids(conn, vm_id)
    conn.compute.live_migrate_server(
        server=vm_id,
        host=destination,
        block_migration=False,
    )
    result = wlb.monitor_migration(
        conn,
        vm_id,
        vm_name,
        src_host=source,
        dst_host=destination,
        timeout=timeout,
        verbose=False,
        preexisting_migration_ids=preexisting_ids,
    )
    refreshed = conn.compute.get_server(vm_id)
    final_host = wlb.get_server_current_host(refreshed)
    emit(
        {
            "vm_id": vm_id,
            "name": vm_name,
            "source": source,
            "destination": destination,
            "final_host": final_host,
            "status": wlb.get_server_status(refreshed),
            "result": result,
        }
    )
    if result != wlb.MIGRATION_RESULT_SUCCESS or final_host != destination:
        raise RuntimeError(
            f"migration did not converge: result={result}, final_host={final_host}"
        )


def evaluate_policy(conn, config_path, aggregate, vm_ids, timeout, interval):
    cfg = wlb.LoadbalancerConfig.load_config(config_path)
    aggregate_map = wlb.get_aggregate_map(conn, [aggregate], True)
    hosts = aggregate_map[aggregate]
    deadline = time.monotonic() + timeout
    attempt = 0

    while True:
        attempt += 1
        cache = wlb.build_cluster_cache(conn, cfg, hosts, enforce_server_groups=True)
        required = get_servers(conn, vm_ids)
        excluded_names = [
            wlb.get_resource_field(server, "name", "Name") or server.id
            for server in cache["server_cache"].values()
            if server.id not in set(vm_ids)
        ]
        state = wlb.evaluate_aggregate_state(
            cfg,
            conn,
            aggregate,
            hosts,
            enforce_server_groups=True,
            cluster_cache=cache,
            exclude_vm_patterns=excluded_names,
        )
        snapshot_ready = all(
            vm_id in state.get("all_vm_details", {})
            and state.get("vm_host_map", {}).get(vm_id)
            == wlb.get_server_current_host(required[vm_id])
            for vm_id in vm_ids
        )
        plan = state.get("best_plan_candidate")
        payload = {
            "attempt": attempt,
            "snapshot_ready": snapshot_ready,
            "plan_type": plan.plan_type if plan else None,
            "improvement": plan.improvement if plan else state.get("best_improvement"),
            "threshold": state.get("improvement_threshold"),
            "make_room_threshold": state.get("make_room_improvement_threshold"),
            "has_valid_move": state.get("has_valid_move"),
            "reason": state.get("reason"),
        }
        emit(payload)

        if snapshot_ready and plan and plan.plan_type == "make_room":
            if state["improvement_threshold"] != cfg.make_room_min_improvement:
                raise RuntimeError("MAKE_ROOM candidate did not use configured threshold")
            if plan.improvement < state["improvement_threshold"]:
                if state["has_valid_move"] or state["best_plan"] is not None:
                    raise RuntimeError("below-threshold MAKE_ROOM plan was marked executable")
                return
            raise RuntimeError(
                "live MAKE_ROOM candidate is above the new threshold; "
                "this topology cannot prove the intended rejection boundary"
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("timed out waiting for converged MAKE_ROOM candidate")
        time.sleep(min(interval, remaining))


def wait_snapshot(conn, config_path, aggregate, vm_ids, timeout, interval):
    cfg = wlb.LoadbalancerConfig.load_config(config_path)
    aggregate_map = wlb.get_aggregate_map(conn, [aggregate], True)
    hosts = aggregate_map[aggregate]
    deadline = time.monotonic() + timeout
    attempt = 0

    while True:
        attempt += 1
        cache = wlb.build_cluster_cache(conn, cfg, hosts, enforce_server_groups=True)
        required = get_servers(conn, vm_ids)
        state = wlb.evaluate_aggregate_state(
            cfg,
            conn,
            aggregate,
            hosts,
            enforce_server_groups=True,
            cluster_cache=cache,
            exclude_vm_patterns=[],
        )
        failures = {}
        for vm_id in vm_ids:
            metric_host = state.get("vm_host_map", {}).get(vm_id)
            nova_host = wlb.get_server_current_host(required[vm_id])
            if vm_id not in state.get("all_vm_details", {}):
                failures[vm_id] = "missing single-host metrics"
            elif metric_host != nova_host:
                failures[vm_id] = f"Nova={nova_host}, Prometheus={metric_host}"

        emit({"attempt": attempt, "failures": failures})
        if not failures:
            return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("timed out waiting for the restored metric snapshot")
        time.sleep(min(interval, remaining))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("status", "migrate", "evaluate-policy", "wait-snapshot")
    )
    parser.add_argument("--vm-id", action="append", required=True)
    parser.add_argument("--destination")
    parser.add_argument("--config", default="/opt/loadleveller-test/loadleveller.conf")
    parser.add_argument("--aggregate", default="loadtest-agg")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--interval", type=float, default=20.0)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    conn = wlb.get_openstack_connection()
    if args.action == "status":
        status(conn, args.vm_id)
    elif args.action == "migrate":
        if len(args.vm_id) != 1 or not args.destination:
            raise RuntimeError("migrate requires one --vm-id and --destination")
        migrate(conn, args.vm_id[0], args.destination, args.timeout)
    elif args.action == "evaluate-policy":
        if len(args.vm_id) != 2:
            raise RuntimeError("evaluate-policy requires exactly two --vm-id values")
        evaluate_policy(
            conn,
            args.config,
            args.aggregate,
            args.vm_id,
            args.timeout,
            args.interval,
        )
    else:
        wait_snapshot(
            conn,
            args.config,
            args.aggregate,
            args.vm_id,
            args.timeout,
            args.interval,
        )


if __name__ == "__main__":
    main()
