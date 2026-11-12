"""Mode ladder: MAD, adaptive CPU/RAM weights, hotspot/pressure mode choice,
the baseline of an aggregate and the improvement thresholds a plan must meet."""
import compute_maintenance.config as _config


def mean_absolute_deviation(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum(abs(x - mean) for x in values) / len(values)


def calculate_adaptive_weights(cpu_signal, ram_signal, min_cpu_weight=_config.MIN_CPU_WEIGHT, min_ram_weight=_config.MIN_RAM_WEIGHT):
    if min_cpu_weight < 0 or min_ram_weight < 0 or min_cpu_weight + min_ram_weight > 1:
        raise ValueError("minimum CPU/RAM weights must be non-negative and sum to <= 1")
    total = cpu_signal + ram_signal
    if total <= 0:
        return 0.5, 0.5

    cpu_share = cpu_signal / total
    # Clamp one side and derive the other so the configured minima remain true
    # minima after normalization.  The old max-then-renormalize algorithm could
    # turn a 0.20 floor into 0.1818 for a 90/10 signal split.
    cpu_weight = min(max(cpu_share, min_cpu_weight), 1.0 - min_ram_weight)
    ram_weight = 1.0 - cpu_weight

    return cpu_weight, ram_weight


def calculate_pressure_signals(cpu_mad, ram_mad, cpu_mean, ram_mean):
    cpu_signal = (cpu_mad * _config.PRESSURE_MAD_WEIGHT) + (cpu_mean * _config.PRESSURE_MEAN_WEIGHT)
    ram_signal = (ram_mad * _config.PRESSURE_MAD_WEIGHT) + (ram_mean * _config.PRESSURE_MEAN_WEIGHT)
    return cpu_signal, ram_signal


def determine_adaptive_mode(cpu_mean, ram_mean, threshold=None):
    # Read at call time: the threshold is configurable (config.apply_thresholds).
    threshold = _config.runtime().auto_pressure_threshold if threshold is None else threshold
    if cpu_mean > threshold or ram_mean > threshold:
        return _config.MODE_PRESSURE
    return _config.MODE_MAD


def determine_hotspot_mode(host_cpu_pct, host_ram_pct):
    max_cpu = max(host_cpu_pct.values()) if host_cpu_pct else 0.0
    max_ram = max(host_ram_pct.values()) if host_ram_pct else 0.0

    if max_ram > _config.runtime().ram_critical_threshold:
        return _config.MODE_RAM_CRITICAL_HOTSPOT, max_cpu, max_ram
    if max_ram > _config.runtime().ram_hotspot_threshold:
        return _config.MODE_RAM_HOTSPOT, max_cpu, max_ram
    if max_cpu > _config.runtime().cpu_hotspot_threshold:
        return _config.MODE_CPU_HOTSPOT, max_cpu, max_ram
    return _config.MODE_NORMAL, max_cpu, max_ram


def choose_effective_mode(cpu_mean, ram_mean, host_cpu_pct, host_ram_pct):
    hotspot_mode, max_cpu, max_ram = determine_hotspot_mode(host_cpu_pct, host_ram_pct)
    if hotspot_mode != _config.MODE_NORMAL:
        return hotspot_mode, max_cpu, max_ram
    return determine_adaptive_mode(cpu_mean, ram_mean), max_cpu, max_ram


def get_hotspot_source_host(mode, host_cpu_pct, host_ram_pct):
    if mode in _config.RAM_HOTSPOT_MODES and host_ram_pct:
        return max(host_ram_pct, key=host_ram_pct.get)
    if mode == _config.MODE_CPU_HOTSPOT and host_cpu_pct:
        return max(host_cpu_pct, key=host_cpu_pct.get)
    return None


def calculate_mode_weights(cpu_mad, ram_mad, cpu_mean, ram_mean, mode):
    if mode == _config.MODE_PRESSURE:
        cpu_signal, ram_signal = calculate_pressure_signals(cpu_mad, ram_mad, cpu_mean, ram_mean)
        cpu_weight, ram_weight = calculate_adaptive_weights(cpu_signal, ram_signal)
        return cpu_weight, ram_weight, cpu_signal, ram_signal

    if mode == _config.MODE_RAM_HOTSPOT:
        cpu_signal, ram_signal = cpu_mad, ram_mad
        return 0.2, 0.8, cpu_signal, ram_signal

    if mode == _config.MODE_RAM_CRITICAL_HOTSPOT:
        cpu_signal, ram_signal = cpu_mad, ram_mad
        return 0.1, 0.9, cpu_signal, ram_signal

    if mode == _config.MODE_CPU_HOTSPOT:
        cpu_signal, ram_signal = cpu_mad, ram_mad
        return 0.8, 0.2, cpu_signal, ram_signal

    cpu_signal, ram_signal = cpu_mad, ram_mad
    cpu_weight, ram_weight = calculate_adaptive_weights(cpu_signal, ram_signal)
    return cpu_weight, ram_weight, cpu_signal, ram_signal


def compute_baseline_metrics(host_cpu_pct, host_ram_pct):
    """Compute cluster baseline means, MADs, and adaptive weights."""
    cpu_mean = sum(host_cpu_pct.values()) / len(host_cpu_pct)
    ram_mean = sum(host_ram_pct.values()) / len(host_ram_pct)
    baseline_cpu_mad = mean_absolute_deviation(list(host_cpu_pct.values()))
    baseline_ram_mad = mean_absolute_deviation(list(host_ram_pct.values()))
    mode, max_cpu, max_ram = choose_effective_mode(cpu_mean, ram_mean, host_cpu_pct, host_ram_pct)
    cpu_weight, ram_weight, cpu_signal, ram_signal = calculate_mode_weights(
        baseline_cpu_mad, baseline_ram_mad, cpu_mean, ram_mean, mode
    )
    baseline_weighted_mad = (baseline_cpu_mad * cpu_weight) + (baseline_ram_mad * ram_weight)

    return {
        "cpu_mean": cpu_mean,
        "ram_mean": ram_mean,
        "baseline_cpu_mad": baseline_cpu_mad,
        "baseline_ram_mad": baseline_ram_mad,
        "mode": mode,
        "max_cpu": max_cpu,
        "max_ram": max_ram,
        "cpu_weight": cpu_weight,
        "ram_weight": ram_weight,
        "cpu_signal": cpu_signal,
        "ram_signal": ram_signal,
        "baseline_weighted_mad": baseline_weighted_mad,
    }


def determine_improvement_threshold():
    """Return the fixed improvement threshold used to accept a move."""
    return _config.runtime().min_improvement_threshold


def determine_plan_improvement_threshold(plan, cfg, direct_threshold=None):
    """Return the minimum total benefit required for the candidate plan.

    A MAKE_ROOM plan has two migrations and therefore uses its own, higher
    churn-aware threshold. Direct plans keep the original threshold. ``max``
    is intentional defense in depth for programmatically constructed configs;
    normal config loading already rejects a MAKE_ROOM threshold below direct.
    """
    direct_threshold = (
        determine_improvement_threshold()
        if direct_threshold is None
        else direct_threshold
    )
    if plan is not None and plan.plan_type == "make_room":
        configured = getattr(
            cfg, "make_room_min_improvement", _config.DEFAULT_MAKE_ROOM_MIN_IMPROVEMENT
        )
        return max(direct_threshold, configured)
    return direct_threshold
