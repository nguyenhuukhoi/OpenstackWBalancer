# OpenStack Workload Balancer

**Automated VM live migration tool to balance CPU and RAM load across OpenStack compute aggregates**

This script repeatedly finds and performs (or simulates) the single VM live migration  
that would most improve the **combined Mean Absolute Deviation (MAD)** of CPU and RAM usage  
across all hypervisors in one or more host aggregates.

It tries to minimize load imbalance (spread / MAD) while respecting:

- Server group **affinity** and **anti-affinity** rules (by default)
- Basic capacity checks (vCPUs and RAM)
- Real measured load from Prometheus (not just flavor sizes)

 ## **I JUST TESTED with Openstack 2025.1. THIS IS UNDER DEVELOPMENT**

## Main features

- Balances **CPU %** + **RAM %** together (50/50 weighted average MAD)
- Uses **Prometheus** node_exporter + libvirt_exporter metrics
- Respects OpenStack server group policies (affinity / anti-affinity)
- Conservative hill-climbing: migrates **one VM at a time**
- Detailed live migration progress monitoring
- `--dry-run` mode (very recommended first)
- Can target specific aggregates with `-a / --aggregate`
- `--yes` / `-y` to auto-approve migrations
- `--no-server-groups` to ignore affinity rules (use with caution!)


## What this script does

- Reads current **host CPU%** and **host RAM%** from Prometheus [node_exporter](https://github.com/prometheus/node_exporter)
- Collects **per-VM CPU** and **per-VM RAM usage** from Prometheus [libvirt_exporter](https://github.com/zhangjianweibj/prometheus-libvirt-exporter)-Calculates baseline **MAD (Mean Absolute Deviation)** for CPU and RAM
- For every VM and every possible destination host, it:
  - Checks capacity (basic: **flavor vCPU/RAM** vs hypervisor “available” capacity with your overcommit model)
  - Checks **server group compliance** (affinity / anti-affinity simulation)
  - Simulates the move and computes the new MAD

- Selects the move that gives the **largest reduction** in **average MAD**
- If improvement is above a small threshold (~0.05)-**YOU CAN AJUST IT**, it performs a **live migration** (or prints the plan in `--dry-run`)
- Repeats until no meaningful improvement is found  
  > Note: in the current script, a real run performs **one migration per execution** (it returns after the first live migration).



## Requirements

- Python >= 3.10.12
- python-openstackclient >= 2025.1
- OpenStack SDK (`openstacksdk >= 4.0.1`)
- OS_COMPUTE_API_VERSION=2.86
- [node_exporter](https://github.com/prometheus/node_exporter)
- [libvirt_exporter](https://github.com/zhangjianweibj/prometheus-libvirt-exporter)
- Prometheus endpoint exposing:
  - `node_exporter` (job = `Openstack-Node-Exporter`). We MUST set host alias = Openstack compute name.
  - `libvirt_exporter` (job = `Openstack-LibVirt-Exporter`)
- Working OpenStack authentication (clouds.yaml or environment variables)

```bash
pip install openstacksdk python-dotenv click requests python-openstackclient

PROMETHEUS_QUERY_URL=http://prometheus.example.com:9090/api/v1/query
Change this to your Prometheus Endpoint.

# Optional overrides (usually not needed)
# PROMETHEUS_CPU_USED=sort_desc(100 - avg(irate(node_cpu_seconds_total{job="..."}[5m])*100) by (alias))
# PROMETHEUS_MEM_USED=sort_desc(100 - ((avg_over_time(node_memory_MemAvailable_bytes{job="..."}[5m])*100)/avg_over_time(node_memory_MemTotal_bytes{job="..."}[5m])))

. /path/to/openstack/admin-openrc.sh

export OS_COMPUTE_API_VERSION=2.86

Quick start
1. Run only on specific aggregates
  python3 wbalancer.py -a ur host aggregate 
2. Skip y/n prompt
  python3 wblancer.py -a ur host aggregate -y
  I think it is ok if we run this type via crobtab.
3. For help
  python3 wbalancer.py --help

Some Options may not work properly. I will try fix in the future
```

## Important warnings

- Live migration is not “invisible” — VMs may pause briefly (especially under heavy memory dirtying)
- Default behavior respects server groups — disabling them can break application HA design
- Capacity checks are simplistic (does not consider huge pages, NUMA, I/O, pinned CPUs, etc.)
- This script does not drain hosts / evacuate — it only balances load
- Ensure the Prometheus `alias` label matches your hypervisor names exactly
- Very large-memory VMs or VMs with heavy write load may fail live migration

---

## Typical Prometheus jobs (hardcoded — adjust if needed)

```python
PROMETHEUS_NODE_JOB = 'Openstack-Node-Exporter'
PROMETHEUS_LIBVIRT_JOB = 'Openstack-LibVirt-Exporter'
