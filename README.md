# OpenStack Workload Balancer

English | [Vietnamese](README.vi.md)

An operator-focused Python script that evaluates OpenStack compute aggregates and performs a single live migration when that migration is expected to improve cluster balance safely.

The script is designed for production environments where predictability matters more than aggressive rebalancing. It uses Prometheus metrics, OpenStack inventory data, server-group checks, destination-host readiness checks, and cooldown windows to reduce migration churn.

## What This Script Does

The balancer works at the aggregate level.

For each selected aggregate, it:

1. Collects host CPU and RAM usage from Prometheus.
2. Collects host capacity and allocation ratios from OpenStack and Prometheus.
3. Builds an inventory of VMs running on the aggregate.
4. Chooses an operating mode:
   - RAM critical hotspot
   - RAM hotspot
   - CPU hotspot
   - Pressure
   - Normal MAD-based balancing
5. Searches for the best single VM move that improves balance while respecting safety rules.
6. Either:
   - reports the result in monitoring mode,
   - shows the proposed migration in dry-run mode,
   - or performs the migration and monitors its outcome.

This script performs at most one migration per aggregate per run.

## Quick Start

Use these commands first if you want to understand or operate the script quickly.

Check health only, without changing anything:

```bash
python wloadbalancer.py --monitor-only
```

Preview the proposed migration without executing it:

```bash
python wloadbalancer.py --dry-run --aggregate my-compute-aggregate
```

Run one migration interactively:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate
```

## How to Read This Document

This README is intentionally detailed. The best reading path depends on what you need.

If you are an operator:

- start with `What This Script Does`
- then read `Quick Start`
- then read `Reading a Decision as an Operator`
- then read `Monitoring and Execution Modes`

If you are reviewing or maintaining the algorithm:

- read `Core Decision Logic`
- then `How the Algorithm Thinks`
- then `Math and Scoring Reference`
- then `Simulated Moves in the Three Main Cases`

If you are deploying the script:

- read `Requirements`
- then `OpenStack Requirements`
- then `Prometheus Requirements`
- then `Configuration`
- then `Command-Line Usage`

## Core Decision Logic

The script uses both spread and pressure signals.

### 1. Hotspot detection takes priority

If any host crosses a hotspot threshold, the script switches from normal balancing to hotspot handling.

Default hotspot thresholds:

- RAM hotspot: `75%`
- RAM critical hotspot: `85%`
- CPU hotspot: `85%`

When a hotspot exists, the balancer first looks for moves from the hottest source host before falling back to a full aggregate search.

### 2. Pressure mode comes next

If there is no hotspot but the cluster average CPU or RAM exceeds the pressure threshold, the script enters `pressure` mode.

Default pressure threshold:

- Cluster average CPU or RAM above `80%`

### 3. Otherwise, use MAD-based balancing

If there is no hotspot and no pressure, the script uses a weighted Mean Absolute Deviation (MAD) model to measure imbalance across hosts.

Default acceptable host deviation:

- Target MAD: `3`

### 4. A candidate move must be safe and useful

A VM move is considered only if all of the following are true:

- The VM is not in cooldown after a recent successful migration.
- The VM is not in cooldown after a recent failed move.
- The destination host is a ready `nova-compute` service.
- The destination has enough allocatable CPU and RAM for the VM flavor.
- The move does not violate server-group affinity or anti-affinity rules.
- In hotspot mode, the hotspot actually improves and the destination must not become a hotspot.

### 5. A candidate move must clear the improvement threshold

The script does not migrate just because a move is technically possible.

It requires a minimum improvement:

- In hotspot modes: a fixed minimum improvement floor is used.
- In other modes: the threshold is proportional to the current weighted MAD.

Default minimum improvement floor:

- `0.05`

## How the Algorithm Thinks

This section explains the decision process in plain English.

### Step 1. Read the current cluster state

The script first reads the current aggregate state:

- host CPU percentage
- host RAM percentage
- per-VM CPU percentage
- per-VM RAM percentage
- host allocatable CPU and RAM capacity

At this point, the script has two views of the cluster:

- host-level pressure and spread
- VM-level workload that could potentially be moved

### Step 2. Choose the operating mode

The script does not always optimize the cluster the same way.

It chooses one mode based on urgency:

1. Hotspot mode
   - used when one host is already too hot
   - priority is to cool that host down
2. Pressure mode
   - used when the whole aggregate is running hot on average
   - priority is to reduce overall pressure
3. MAD mode
   - used when there is no hotspot and no pressure
   - priority is to reduce imbalance between hosts

In short:

- hotspot mode asks: "Can I reduce the hottest host safely?"
- pressure mode asks: "Can I reduce aggregate stress?"
- MAD mode asks: "Can I make the hosts more even?"

### Step 3. Convert VM usage into host impact

For each VM, the script estimates how much host load would move with that VM.

It uses:

- VM flavor vCPUs
- VM flavor RAM
- observed VM CPU usage percentage
- observed VM RAM usage percentage

Conceptually:

- real CPU moved = `vcpus * vm_cpu_pct`
- real RAM moved = `vm_ram_gb * vm_ram_pct`

Those values are then converted into host-level percentage impact for:

- the source host
- the destination host

This is important because the script does not score a migration based only on VM flavor size. It scores the move using the VM's observed load.

### Step 4. Simulate each possible destination

For every VM on allowed source hosts, the script simulates moving that VM to every other host in the aggregate.

For each simulated move, it checks:

- cooldown rules
- destination host readiness
- destination allocatable capacity
- server-group policy compliance
- hotspot guardrails if the aggregate is in hotspot mode

If any safety check fails, that candidate is rejected immediately.

### Step 5. Recalculate the cluster after the simulated move

If a candidate passes the safety checks, the script recomputes the cluster as if the move already happened.

It then recomputes:

- post-move CPU MAD
- post-move RAM MAD
- post-move weighted MAD

Improvement is defined as:

```text
improvement = current_weighted_mad - new_weighted_mad
```

So:

- positive improvement means the move helps
- zero or negative improvement means the move does not help

### Step 6. Pick the best valid move

The script keeps only the move with the highest improvement.

Then it applies the improvement threshold:

- if the best move is below threshold, no migration is performed
- if the best move meets or exceeds threshold, that move becomes the proposed migration

This is how the script stays conservative:

- technical feasibility alone is not enough
- the move must be meaningful enough to justify migration

## Math and Scoring Reference

This section explains the formulas behind the scoring model and walks through them by hand.

### Example A. Mean and MAD

Imagine three hosts with CPU usage:

- Host A: `70%`
- Host B: `40%`
- Host C: `35%`

First, calculate the mean CPU usage:

```text
mean_cpu = (70 + 40 + 35) / 3 = 48.33
```

Then calculate the absolute distance from the mean:

```text
|70 - 48.33| = 21.67
|40 - 48.33| = 8.33
|35 - 48.33| = 13.33
```

Now take the average of those distances:

```text
cpu_mad = (21.67 + 8.33 + 13.33) / 3 = 14.44
```

Do the same for RAM. If RAM usage is:

- Host A: `68%`
- Host B: `42%`
- Host C: `38%`

Then:

```text
mean_ram = (68 + 42 + 38) / 3 = 49.33

|68 - 49.33| = 18.67
|42 - 49.33| = 7.33
|38 - 49.33| = 11.33

ram_mad = (18.67 + 7.33 + 11.33) / 3 = 12.44
```

Interpretation:

- higher MAD means hosts are more uneven
- lower MAD means hosts are more balanced

### Example B. Adaptive weights in MAD mode

In normal MAD mode, the script uses CPU MAD and RAM MAD as the two signals.

Using the previous values:

```text
cpu_signal = 14.44
ram_signal = 12.44
total = 26.88
```

Raw shares:

```text
cpu_share = 14.44 / 26.88 = 0.537
ram_share = 12.44 / 26.88 = 0.463
```

### Signal -> Share -> Weight

These three terms are related, but they are not the same thing.

`Signal`

- A signal is the raw priority score before any normalization.
- In normal MAD mode:

```text
cpu_signal = cpu_mad
ram_signal = ram_mad
```

So in this example:

```text
cpu_signal = 14.44
ram_signal = 12.44
```

At this stage, the script is only saying:

- CPU imbalance matters `14.44` units
- RAM imbalance matters `12.44` units

`Share`

- A share is the fraction of the total signal that belongs to CPU or RAM.
- The script converts raw signals into relative percentages:

```text
cpu_share = cpu_signal / (cpu_signal + ram_signal)
ram_share = ram_signal / (cpu_signal + ram_signal)
```

So here:

```text
cpu_share = 14.44 / (14.44 + 12.44) = 0.537
ram_share = 12.44 / (14.44 + 12.44) = 0.463
```

That means:

- CPU contributes `53.7%` of the total priority
- RAM contributes `46.3%` of the total priority

`Weight`

- A weight is the final value actually used in scoring.
- Weights start from the shares, but they may be adjusted by the minimum-weight policy:

```text
cpu_weight = max(cpu_share, MIN_CPU_WEIGHT)
ram_weight = max(ram_share, MIN_RAM_WEIGHT)
```

Then they are normalized again so the total is exactly `1.0`.

In this example, both shares are already above the minimum `0.2`, so:

```text
cpu_weight = 0.537
ram_weight = 0.463
```

In other words:

- `signal` answers: "How much raw priority does CPU or RAM have?"
- `share` answers: "What fraction of the total priority belongs to CPU or RAM?"
- `weight` answers: "What final proportion will the scoring formula actually use?"

The script also enforces minimum weights:

- minimum CPU weight: `0.2`
- minimum RAM weight: `0.2`

In this case both raw shares are already above the minimum, so the final weights stay almost the same:

```text
cpu_weight = 0.537
ram_weight = 0.463
```

Weighted MAD becomes:

```text
weighted_mad = (cpu_mad * cpu_weight) + (ram_mad * ram_weight)
weighted_mad = (14.44 * 0.537) + (12.44 * 0.463)
weighted_mad = 13.52
```

This is the score the script tries to reduce.

### Why this formula exists

The script needs one final score to rank migration candidates fairly.

That is necessary because a move can improve CPU but hurt RAM, or improve RAM but hurt CPU.
If the script scored candidates using CPU only, it could accidentally choose a move that makes RAM much worse.
If it scored candidates using RAM only, it could accidentally choose a move that makes CPU much worse.

`weighted_mad` solves that problem by combining both dimensions into one comparable score.

What this means in practice:

- `cpu_mad` says how uneven CPU usage is across hosts
- `ram_mad` says how uneven RAM usage is across hosts
- `weighted_mad` combines both into one score so the script can compare candidate moves with a single number

You can think of it as:

```text
overall imbalance score = CPU imbalance contribution + RAM imbalance contribution
```

In this example:

```text
CPU contribution = 14.44 * 0.537 = 7.75
RAM contribution = 12.44 * 0.463 = 5.76
Total weighted_mad = 7.75 + 5.76 = 13.52
```

So the script is effectively saying:

- CPU matters slightly more than RAM right now
- both CPU and RAM imbalance still matter
- the current overall cluster imbalance score is `13.52`

This is important because the script does not choose a move just because it improves CPU or just because it improves RAM.
It chooses the move that lowers the combined score the most.

For example:

```text
Candidate 1:
new_cpu_mad = 11.00
new_ram_mad = 12.00
new_weighted_mad = (11.00 * 0.537) + (12.00 * 0.463) = 11.46
improvement = 13.52 - 11.46 = 2.06

Candidate 2:
new_cpu_mad = 9.50
new_ram_mad = 14.50
new_weighted_mad = (9.50 * 0.537) + (14.50 * 0.463) = 11.82
improvement = 13.52 - 11.82 = 1.70
```

Even though Candidate 2 improves CPU more aggressively, it hurts RAM enough that the overall result is worse.
So Candidate 1 is the better move.

This is the main reason `weighted_mad` exists:

- without it, the script would need separate "best CPU move" and "best RAM move" logic
- with it, the script can rank all candidates consistently with one final score
- lower `weighted_mad` is always better than higher `weighted_mad`

### Example C. Adaptive weights in pressure mode

In pressure mode, the script gives weight based on both:

- spread (`MAD`)
- average utilization (`mean`)

The formula is:

```text
cpu_signal = (cpu_mad * 0.7) + (cpu_mean * 0.3)
ram_signal = (ram_mad * 0.7) + (ram_mean * 0.3)
```

Suppose:

- `cpu_mean = 85`
- `ram_mean = 60`
- `cpu_mad = 7`
- `ram_mad = 5`

Then:

```text
cpu_signal = (7 * 0.7) + (85 * 0.3) = 4.9 + 25.5 = 30.4
ram_signal = (5 * 0.7) + (60 * 0.3) = 3.5 + 18.0 = 21.5
```

Now convert signals into weights:

```text
total = 30.4 + 21.5 = 51.9

cpu_weight = 30.4 / 51.9 = 0.586
ram_weight = 21.5 / 51.9 = 0.414
```

### Pressure mode: Signal -> Share -> Weight

The same three-step idea also applies in pressure mode, but the inputs are different.

`Signal`

- In pressure mode, the script does not use MAD alone.
- Instead it combines spread and overall resource pressure:

```text
cpu_signal = (cpu_mad * 0.7) + (cpu_mean * 0.3)
ram_signal = (ram_mad * 0.7) + (ram_mean * 0.3)
```

So in this example:

```text
cpu_signal = 30.4
ram_signal = 21.5
```

At this stage, the script is saying:

- CPU deserves more attention because it is hotter overall and still uneven enough to matter
- RAM still matters, but less than CPU in this specific state

`Share`

- Once the signals exist, the script turns them into relative shares:

```text
cpu_share = cpu_signal / (cpu_signal + ram_signal)
ram_share = ram_signal / (cpu_signal + ram_signal)
```

So here:

```text
cpu_share = 30.4 / (30.4 + 21.5) = 0.586
ram_share = 21.5 / (30.4 + 21.5) = 0.414
```

That means:

- CPU contributes `58.6%` of the total pressure priority
- RAM contributes `41.4%` of the total pressure priority

`Weight`

- In this example, both shares are above the minimum weight floor.
- So the final weights remain the same as the shares:

```text
cpu_weight = 0.586
ram_weight = 0.414
```

In other words:

- `signal` answers: "After combining spread and pressure, how much raw priority does CPU or RAM have?"
- `share` answers: "What fraction of the total priority belongs to CPU or RAM?"
- `weight` answers: "What final proportion will the weighted-MAD formula actually use?"

Interpretation:

- CPU receives more weight because the aggregate is more CPU-stressed than RAM-stressed
- this makes the script prefer moves that reduce CPU pressure more strongly

### Why pressure mode uses `cpu_signal` and `ram_signal`

Pressure mode exists for a specific situation:
the aggregate is not just uneven, it is already running hot on average.

That means the script cannot look only at `MAD`.
If it used only `MAD`, it might ignore the fact that one resource is already dangerously busy across the whole aggregate.

But the script also cannot look only at `mean`.
If it used only average utilization, it might ignore the fact that one resource is distributed much more unevenly and is therefore more likely to create local overload on individual hosts.

So pressure mode combines both:

- `mean` answers: "How hot is this resource across the aggregate?"
- `MAD` answers: "How unevenly is this resource distributed across hosts?"
- `signal` answers: "Which resource should matter more right now when scoring migrations?"

This is why the script uses:

```text
cpu_signal = (cpu_mad * 0.7) + (cpu_mean * 0.3)
ram_signal = (ram_mad * 0.7) + (ram_mean * 0.3)
```

This is a heuristic, not a universal mathematical law.
The design choice here is:

- keep balancing as the main objective, so `MAD` still has the larger share (`0.7`)
- but include overall pressure in the decision, so `mean` still contributes (`0.3`)

In other words, pressure mode means:

```text
still rebalance the cluster,
but give more importance to the resource that is under more real pressure.
```

#### Why not use only MAD?

Suppose:

- `cpu_mean = 90`
- `ram_mean = 60`
- `cpu_mad = 4`
- `ram_mad = 8`

If you used only MAD:

```text
CPU priority = 4
RAM priority = 8
```

That would tell the script to focus on RAM.
But operationally, CPU is the more urgent problem because the aggregate is already very hot on CPU overall.

With the pressure formula:

```text
cpu_signal = (4 * 0.7) + (90 * 0.3) = 2.8 + 27.0 = 29.8
ram_signal = (8 * 0.7) + (60 * 0.3) = 5.6 + 18.0 = 23.6
```

Now CPU gets the higher priority, which better reflects the real risk.

#### Why not use only mean?

Suppose:

- `cpu_mean = 82`
- `ram_mean = 78`
- `cpu_mad = 3`
- `ram_mad = 14`

If you used only mean:

```text
CPU priority = 82
RAM priority = 78
```

That would tell the script to focus on CPU.
But RAM is much more unevenly distributed, so RAM is more likely to create a local hotspot on one or two hosts.

With the pressure formula:

```text
cpu_signal = (3 * 0.7) + (82 * 0.3) = 2.1 + 24.6 = 26.7
ram_signal = (14 * 0.7) + (78 * 0.3) = 9.8 + 23.4 = 33.2
```

Now RAM gets the higher priority, which better reflects the distribution problem.

#### Short version

In pressure mode:

- `mean` tells the script which resource is hotter overall
- `MAD` tells the script which resource is more uneven
- `signal` combines both so the script can choose better weights

Then those signals are converted into:

```text
cpu_weight = cpu_signal / (cpu_signal + ram_signal)
ram_weight = ram_signal / (cpu_signal + ram_signal)
```

Those weights are then used inside `weighted_mad` to rank migration candidates.

### Weight calculation in every mode

The script does not calculate weights the same way in every mode.
The table below shows the exact behavior.

| Mode | How weights are chosen | Result |
| --- | --- | --- |
| Normal MAD mode | Use `cpu_mad` and `ram_mad` as signals, then normalize | Adaptive |
| Pressure mode | Use `(mad * 0.7) + (mean * 0.3)` as signals, then normalize | Adaptive |
| RAM hotspot | Fixed values | `cpu_weight = 0.2`, `ram_weight = 0.8` |
| RAM critical hotspot | Fixed values | `cpu_weight = 0.1`, `ram_weight = 0.9` |
| CPU hotspot | Fixed values | `cpu_weight = 0.8`, `ram_weight = 0.2` |
| Zero-signal fallback | If both signals are `0` | `cpu_weight = 0.5`, `ram_weight = 0.5` |

### Example D. Normal mode with minimum-weight protection

In normal mode the script starts from:

```text
cpu_signal = cpu_mad
ram_signal = ram_mad
```

Suppose:

- `cpu_mad = 1`
- `ram_mad = 9`

Raw shares:

```text
cpu_share = 1 / (1 + 9) = 0.10
ram_share = 9 / (1 + 9) = 0.90
```

But the script enforces minimum weights:

- `MIN_CPU_WEIGHT = 0.2`
- `MIN_RAM_WEIGHT = 0.2`

So before normalization:

```text
cpu_weight = max(0.10, 0.2) = 0.2
ram_weight = max(0.90, 0.2) = 0.9
```

Now normalize them so the total becomes `1`:

```text
weight_sum = 0.2 + 0.9 = 1.1

cpu_weight = 0.2 / 1.1 = 0.182
ram_weight = 0.9 / 1.1 = 0.818
```

Interpretation:

- RAM still dominates because RAM imbalance is much larger
- CPU is not allowed to fall all the way to zero importance
- this keeps the score from becoming completely one-dimensional

### Example E. RAM hotspot mode

Suppose one host is above the RAM hotspot threshold.
In this mode, the script does not derive weights from the current MAD values.
It uses fixed hotspot weights:

```text
cpu_weight = 0.2
ram_weight = 0.8
```

If a simulated move produces:

```text
new_cpu_mad = 8
new_ram_mad = 6
```

Then:

```text
new_weighted_mad = (8 * 0.2) + (6 * 0.8)
                 = 1.6 + 4.8
                 = 6.4
```

Interpretation:

- RAM is the main emergency
- CPU still matters, but much less
- a move that reduces RAM imbalance is preferred unless it creates a bad CPU side effect

### Example F. RAM critical hotspot mode

If one host is above the critical RAM threshold, RAM gets even more priority:

```text
cpu_weight = 0.1
ram_weight = 0.9
```

For example:

```text
new_cpu_mad = 10
new_ram_mad = 5

new_weighted_mad = (10 * 0.1) + (5 * 0.9)
                 = 1.0 + 4.5
                 = 5.5
```

Interpretation:

- in a critical RAM hotspot, the algorithm is intentionally RAM-first
- CPU imbalance is still counted, but only lightly

### Example G. CPU hotspot mode

If one host is above the CPU hotspot threshold, the fixed weights are reversed:

```text
cpu_weight = 0.8
ram_weight = 0.2
```

For example:

```text
new_cpu_mad = 4
new_ram_mad = 9

new_weighted_mad = (4 * 0.8) + (9 * 0.2)
                 = 3.2 + 1.8
                 = 5.0
```

Interpretation:

- CPU is the main priority
- RAM still contributes to the score
- the script still avoids moves that fix CPU only by creating an obviously bad RAM tradeoff

### Example H. Zero-signal fallback

If both signals are `0`, the script falls back to:

```text
cpu_weight = 0.5
ram_weight = 0.5
```

This is mostly a defensive edge case.
It means CPU and RAM are treated equally if there is no signal telling the script to prefer one over the other.

### Example I. Why a move can improve one host but still lose overall

Suppose the current weighted MAD is:

```text
current_weighted_mad = 13.52
```

After simulating one VM move, the new host percentages produce:

```text
new_cpu_mad = 10.20
new_ram_mad = 11.40
cpu_weight = 0.537
ram_weight = 0.463
```

Then:

```text
new_weighted_mad = (10.20 * 0.537) + (11.40 * 0.463)
new_weighted_mad = 10.76
```

Improvement:

```text
improvement = 13.52 - 10.76 = 2.76
```

That is a strong candidate.

But if another move gives:

```text
new_weighted_mad = 13.49
improvement = 0.03
```

Then the script may still reject it if the threshold is `0.05`.

So the script is not asking:

- "Did one host get better?"

It is asking:

- "Did the aggregate get better enough to justify a migration?"

## Simulated Moves in the Three Main Cases

The examples below walk through one simulated move in each major decision family:

- normal MAD balancing
- pressure mode
- hotspot mode

These are simplified by design, but they match the script's logic closely enough to explain how a candidate is evaluated.

### Simulation 1. Normal MAD mode

Assume the aggregate has three hosts:

- Host A: CPU `70%`, RAM `68%`
- Host B: CPU `40%`, RAM `42%`
- Host C: CPU `35%`, RAM `38%`

There is no hotspot and no pressure, so the script stays in normal MAD mode.

From the earlier math:

- `cpu_mad = 14.44`
- `ram_mad = 12.44`
- `cpu_weight = 0.537`
- `ram_weight = 0.463`
- `baseline_weighted_mad = 13.52`

Baseline formulas:

```text
cpu_mean = (70 + 40 + 35) / 3 = 48.33
ram_mean = (68 + 42 + 38) / 3 = 49.33

cpu_mad = (|70 - 48.33| + |40 - 48.33| + |35 - 48.33|) / 3
        = (21.67 + 8.33 + 13.33) / 3
        = 14.44

ram_mad = (|68 - 49.33| + |42 - 49.33| + |38 - 49.33|) / 3
        = (18.67 + 7.33 + 11.33) / 3
        = 12.44
```

Now simulate moving one VM from Host A to Host B.
Assume that VM contributes:

- CPU impact: `12%`
- RAM impact: `12%`

After the simulated move:

```text
CPU: [70, 40, 35] -> [58, 52, 35]
RAM: [68, 42, 38] -> [56, 54, 38]
```

Recalculate MAD:

```text
new_cpu_mean = (58 + 52 + 35) / 3 = 48.33
new_ram_mean = (56 + 54 + 38) / 3 = 49.33

new_cpu_mad = (|58 - 48.33| + |52 - 48.33| + |35 - 48.33|) / 3
            = (9.67 + 3.67 + 13.33) / 3
            = 8.89

new_ram_mad = (|56 - 49.33| + |54 - 49.33| + |38 - 49.33|) / 3
            = (6.67 + 4.67 + 11.33) / 3
            = 7.56
```

Recalculate weighted MAD:

```text
new_cpu_mad = 8.89
new_ram_mad = 7.56
new_weighted_mad = (8.89 * 0.537) + (7.56 * 0.463)
                 = 4.77 + 3.50
                 = 8.27
```

Improvement:

```text
improvement = 13.52 - 8.27 = 5.25
```

Interpretation:

- the hosts become much more even
- CPU and RAM both improve
- this is exactly the kind of move normal MAD mode wants

### Simulation 2. Pressure mode

Assume the aggregate has three hosts:

- Host A: CPU `84%`, RAM `74%`
- Host B: CPU `82%`, RAM `70%`
- Host C: CPU `80%`, RAM `68%`

There is no hotspot:

- max CPU is `84%` which is below the CPU hotspot threshold
- max RAM is `74%` which is below the RAM hotspot threshold

But the average CPU is high:

```text
cpu_mean = (84 + 82 + 80) / 3 = 82.00
```

So the script enters `pressure` mode.

Baseline spread:

```text
cpu_mean = (84 + 82 + 80) / 3 = 82.00
ram_mean = (74 + 70 + 68) / 3 = 70.67

cpu_mad = (|84 - 82.00| + |82 - 82.00| + |80 - 82.00|) / 3
        = (2 + 0 + 2) / 3
        = 1.33

ram_mad = (|74 - 70.67| + |70 - 70.67| + |68 - 70.67|) / 3
        = (3.33 + 0.67 + 2.67) / 3
        = 2.22
```

Pressure signals:

```text
cpu_mad = 1.33
ram_mad = 2.22
cpu_signal = (1.33 * 0.7) + (82.00 * 0.3) = 25.53
ram_signal = (2.22 * 0.7) + (70.67 * 0.3) = 22.76
```

Weights:

```text
cpu_weight = 25.53 / (25.53 + 22.76) = 0.529
ram_weight = 22.76 / (25.53 + 22.76) = 0.471
```

Baseline weighted MAD:

```text
baseline_weighted_mad = (1.33 * 0.529) + (2.22 * 0.471)
                      = 0.70 + 1.05
                      = 1.75
```

Now simulate moving one VM from Host A to Host C.
Assume the move changes host percentages like this:

```text
CPU: [84, 82, 80] -> [82, 82, 82]
RAM: [74, 70, 68] -> [71, 70, 71]
```

Recalculate MAD:

```text
new_cpu_mean = (82 + 82 + 82) / 3 = 82.00
new_ram_mean = (71 + 70 + 71) / 3 = 70.67

new_cpu_mad = (|82 - 82.00| + |82 - 82.00| + |82 - 82.00|) / 3
            = 0.00

new_ram_mad = (|71 - 70.67| + |70 - 70.67| + |71 - 70.67|) / 3
            = (0.33 + 0.67 + 0.33) / 3
            = 0.44
```

New weighted MAD:

```text
new_cpu_mad = 0.00
new_ram_mad = 0.44
new_weighted_mad = (0.00 * 0.529) + (0.44 * 0.471)
                 = 0.00 + 0.21
                 = 0.21
```

Improvement:

```text
improvement = 1.75 - 0.21 = 1.54
```

Interpretation:

- the cluster was already hot on average, so pressure mode was correct
- the move still gets ranked by improvement in weighted MAD
- CPU gets slightly more importance than RAM because CPU is the hotter resource overall

### Simulation 3. Hotspot mode

Assume the aggregate has three hosts:

- Host A: CPU `55%`, RAM `88%`
- Host B: CPU `50%`, RAM `52%`
- Host C: CPU `48%`, RAM `49%`

This immediately triggers `ram_critical_hotspot`.

In this mode the weights are fixed:

```text
cpu_weight = 0.1
ram_weight = 0.9
```

Baseline MAD:

```text
cpu_mean = (55 + 50 + 48) / 3 = 51.00
ram_mean = (88 + 52 + 49) / 3 = 63.00

cpu_mad = (|55 - 51| + |50 - 51| + |48 - 51|) / 3
        = (4 + 1 + 3) / 3
        = 2.67

ram_mad = (|88 - 63| + |52 - 63| + |49 - 63|) / 3
        = (25 + 11 + 14) / 3
        = 16.67

baseline_weighted_mad = (2.67 * 0.1) + (16.67 * 0.9)
                      = 0.27 + 15.00
                      = 15.27
```

Now simulate moving one VM from Host A to Host B.
Assume that VM changes host percentages like this:

```text
cpu_mad = 2.67
ram_mad = 16.67
CPU: [55, 50, 48] -> [51, 54, 48]
RAM: [88, 52, 49] -> [70, 70, 49]
```

First, check the hotspot guardrail:

- source hotspot improves: `88% -> 70%`
- destination does not become a RAM hotspot: `70% < 75%`

So the move is allowed to continue.

Recalculate MAD:

```text
new_cpu_mean = (51 + 54 + 48) / 3 = 51.00
new_ram_mean = (70 + 70 + 49) / 3 = 63.00

new_cpu_mad = (|51 - 51| + |54 - 51| + |48 - 51|) / 3
            = (0 + 3 + 3) / 3
            = 2.00

new_ram_mad = (|70 - 63| + |70 - 63| + |49 - 63|) / 3
            = (7 + 7 + 14) / 3
            = 9.33
```

New weighted MAD:

```text
new_cpu_mad = 2.00
new_ram_mad = 9.33
new_weighted_mad = (2.00 * 0.1) + (9.33 * 0.9)
                 = 0.20 + 8.40
                 = 8.60
```

Improvement:

```text
improvement = 15.27 - 8.60 = 6.67
```

Interpretation:

- the hottest host is cooled down sharply
- RAM dominates the score because this is a RAM emergency
- the move is valid because it reduces the hotspot without creating a new hotspot elsewhere

## Operational Scenarios

These scenarios are still simplified, but they are written from the operator's point of view rather than as pure formula walkthroughs.

### Example 1. Simple MAD balancing

Imagine an aggregate with three hosts:

- Host A: CPU `70%`, RAM `68%`
- Host B: CPU `40%`, RAM `42%`
- Host C: CPU `35%`, RAM `38%`

There is no hotspot and no pressure, so the script uses MAD mode.

Now suppose a VM on Host A currently consumes roughly:

- `2.0` real CPU cores
- `6.0` real GB RAM

The script simulates moving that VM from Host A to Host B.

After the simulated move:

- Host A becomes less loaded
- Host B becomes more loaded
- the host percentages become more even overall

If the new weighted MAD is lower than before by enough margin, the move becomes a strong candidate.

If the same VM were moved from Host A to Host C instead and that produced an even lower weighted MAD, Host C would become the better destination.

### Example 2. Hotspot handling

Imagine this aggregate:

- Host A RAM: `88%`
- Host B RAM: `52%`
- Host C RAM: `49%`

This immediately triggers `ram_critical_hotspot`.

The script will first focus on Host A as the source host.

Now consider two candidate VMs on Host A:

- VM 1 reduces Host A from `88%` to `82%`, but pushes Host B to `77%`
- VM 2 reduces Host A from `88%` to `84%`, and keeps Host B at `70%`

Even if VM 1 looks larger, it is rejected because the destination would become a RAM hotspot.

VM 2 is safer because:

- the hotspot host improves
- the destination does not cross the hotspot threshold

That is exactly what the hotspot guardrail is meant to enforce.

### Example 3. Technically possible but still rejected

Imagine a move that is allowed by:

- capacity
- server-group policy
- destination readiness

But after simulation the improvement is tiny:

- current weighted MAD: `6.20`
- post-move weighted MAD: `6.17`
- improvement: `0.03`

If the required threshold is `0.05`, the script rejects the move.

So the result is:

- a candidate exists
- but no valid move exists above threshold

This is why the monitor output may say that the cluster is unbalanced while still refusing to migrate.

### Example 4. Why a VM can be skipped even if it looks useful

A VM can look like a perfect balancing candidate and still be skipped because:

- it was migrated recently and is still in cooldown
- its previous migration failed and failed-move cooldown is still active
- the target host is not a healthy `nova-compute`
- the destination has insufficient allocatable flavor capacity
- affinity or anti-affinity rules would be violated

This is intentional. The script prioritizes safe and predictable behavior over aggressive balancing.

## Reading a Decision as an Operator

When you look at the output, read it in this order:

1. Mode
   - tells you whether the script is reacting to hotspot, pressure, or spread
2. Status / severity
   - tells you whether this is healthy, warning-level, or critical
3. Candidate line
   - tells you whether a move exists and whether it cleared the threshold
4. Reason
   - explains why no move was accepted if the cluster still looks unhealthy
5. Search scope
   - tells you whether the script stayed on the hotspot host first or used full aggregate search

This makes it easier to distinguish:

- "the cluster is fine"
- "the cluster is not ideal but no safe move exists"
- "the cluster is unhealthy and a safe corrective move is available"

## Safety Features

This script is intentionally conservative.

It includes:

- VM cooldown after successful migration: `1800` seconds
- VM cooldown after failed migration: `3600` seconds
- Destination-host readiness checks using `nova-compute`
- Server-group policy checks by default
- Migration outcome monitoring with timeout detection
- Optional reset from `ERROR` back to `ACTIVE` after failed migration attempts
- Persistent cooldown state stored on disk
- Migration event logging to JSONL

## Monitoring and Execution Modes

### `--monitor-only`

Use this mode when you want an Icinga/Nagios-compatible health result without changing anything.

It prints a human-readable summary and exits with:

- `0` for `OK`
- `1` for `WARNING`
- `2` for `CRITICAL`
- `3` for `UNKNOWN`

### `--dry-run`

Use this mode to see what the script would migrate without performing the migration.

It still evaluates the aggregate fully and records a `dry_run` event if a valid move exists.

### Default execution mode

Without `--monitor-only` or `--dry-run`, the script:

- evaluates the aggregate,
- shows the proposed move,
- asks for confirmation,
- performs one live migration,
- monitors the result,
- updates cooldown state,
- optionally sends email alerts.

Use `-y` or `--yes` to skip the confirmation prompt.

## Requirements

You need:

- Python 3
- Access to OpenStack via `openstacksdk`
- A configured OpenStack cloud profile named `openstack`
- Prometheus endpoints and metrics reachable from the host where the script runs

Python packages used by the script:

- `openstacksdk`
- `requests`
- `python-dotenv`
- `click`
- `urllib3`

Install them with your preferred package manager, for example:

```bash
pip install openstacksdk requests python-dotenv click urllib3
```

## OpenStack Requirements

The script connects using:

- cloud name: `openstack`
- compute microversion: `2.87`

Make sure your environment has a valid `clouds.yaml` or equivalent OpenStack configuration that defines a cloud named `openstack`.

## Prometheus Requirements

The script expects Prometheus data for:

- host CPU usage
- host RAM usage
- OpenStack placement allocation ratios
- per-VM CPU usage
- per-VM RAM usage

The default queries assume the presence of metrics from:

- a node exporter job
- a libvirt exporter job
- an OpenStack exporter job

The script also expects specific labels to exist in Prometheus results:

- `alias` for host-level usage queries
- `hostname` for allocation ratio queries
- `instanceId` and `host` for per-VM queries

If your metric names or labels differ, override the Prometheus queries in the config file.

## Configuration

The script loads configuration from a dotenv-style file.

Default config file path:

```bash
/etc/loadleveller-secrets.conf
```

Supported environment variables:

- `PROMETHEUS_QUERY_URL`
- `PROMETHEUS_MEM_USED`
- `PROMETHEUS_CPU_USED`
- `PROMETHEUS_CPU_RATIO`
- `PROMETHEUS_MEM_RATIO`
- `ALERT_EMAIL_TO`
- `EMAIL_TO`
- `ALERT_EMAIL_FROM`
- `EMAIL_FROM`
- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASSWORD`
- `SMTP_STARTTLS`

Example:

```dotenv
PROMETHEUS_QUERY_URL=http://kprometheus.com:9090/api/v1/query

ALERT_EMAIL_TO=ops@example.com,cloud@example.com
ALERT_EMAIL_FROM=loadleveller@example.com
SMTP_SERVER=smtp.example.com
SMTP_PORT=25
SMTP_USER=
SMTP_PASSWORD=
SMTP_STARTTLS=false
```

## Files Written by the Script

By default, the script writes:

- cooldown state: `/var/log/loadleveller_vm_cooldown.json`
- migration events: `/var/log/loadleveller_migration_events.jsonl`

You can override both at runtime:

- `--cooldown-file`
- `--events-file`

## Command-Line Usage

Show all options:

```bash
python wloadbalancer.py --help
```

Monitor all aggregates:

```bash
python wloadbalancer.py --monitor-only
```

Monitor one aggregate:

```bash
python wloadbalancer.py --monitor-only --aggregate my-compute-aggregate
```

Preview the best move without changing anything:

```bash
python wloadbalancer.py --dry-run --aggregate my-compute-aggregate
```

Run interactively and ask before migrating:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate
```

Run non-interactively:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --yes
```

Send email alerts for failed migrations only:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --send-error
```

Send email alerts for both successful and failed migrations:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --send-all
```

Disable server-group checks only if you fully understand the risk:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --no-server-groups
```

## Recommended Operator Workflow

For production use, the safest workflow is:

1. Start with `--monitor-only` to understand the current cluster state.
2. Run `--dry-run` to inspect the proposed candidate and expected improvement.
3. Run interactively without `-y` for the first execution.
4. Enable `--send-error` or `--send-all` if you want email notifications.
5. Use custom `--cooldown-file` and `--events-file` paths if your environment requires different storage locations.

## Understanding the Output

The monitoring summary includes:

- the selected decision mode
- cluster peaks and averages
- CPU and RAM MAD
- adaptive weights
- whether a valid move exists
- the best candidate move
- the reason no move was selected, if applicable

When a migration is proposed, the runtime output also shows:

- source and destination hosts
- estimated CPU and RAM impact
- post-migration host percentages
- post-migration weighted MAD
- improvement for that migration round

## Event Log Format

The JSONL events file is append-only and records operational outcomes such as:

- `dry_run`
- `cancelled`
- `skipped_destination_not_ready`
- `submit_or_monitor_failed`
- `migration_success`
- `migration_timeout`
- `migration_failed`
- `stuck_active`

Each event may contain:

- aggregate name
- result
- detail status
- severity
- duration
- timestamps
- improvement values
- VM metadata
- source and destination hosts

## Operational Notes

- The script does not continuously rebalance until the cluster is perfect.
- It chooses one best move per aggregate per invocation.
- It favors predictable improvement over aggressive churn.
- It is designed to be run repeatedly by an operator, scheduler, or monitoring system.

## Troubleshooting

### The script says no aggregates matched selection

Check the aggregate name passed with `--aggregate` and verify it exists in OpenStack.

### The script skips a destination host

The most common reasons are:

- `nova-compute` is not enabled
- `nova-compute` is not up
- the host is forced down
- the destination lacks allocatable capacity

### The script finds no valid move

Common reasons:

- all candidates are below the improvement threshold
- server-group rules block the move
- cooldown blocks the VM
- destination hosts are not ready
- no technically feasible migration exists

### The migration appears to start but times out

The script monitors live migration status for up to `600` seconds by default. A timeout means the VM did not reach a terminal success or failure condition within that window.

## Summary

Use this script when you want a cautious, explainable, single-step balancing tool for OpenStack aggregates.

It is best suited for environments where:

- live migration is allowed,
- Prometheus metrics are trusted,
- server-group policies matter,
- and operators prefer controlled, observable balancing over rapid automation.
