# Hướng dẫn vận hành

[English](operations.md) | Tiếng Việt

Dành cho người đã chạy tool: tool giữ trạng thái ở đâu, chờ và timeout thế nào, planner từ chối những gì, và cách recovery khi một lượt chạy dừng. Cài đặt và cấu hình xem [hướng dẫn sử dụng](user-guide.vi.md#3-cài-đặt); logic ra quyết định xem [algorithm.vi.md](algorithm.vi.md).

Phạm vi: một coordinator Linux Python 3.12, một cloud (mặc định `openstack`, xem `OS_CLOUD` / `--os-cloud`), Nova/Placement/libvirt và Prometheus. Profile package/kernel dành cho Ubuntu/Debian. Không có lock phân tán.

## 1. Trước lần chạy đầu

- Cài v1 vào venv riêng, không cài vào venv của script legacy ([hướng dẫn sử dụng mục 3](user-guide.vi.md#3-cài-đặt)).
- Giữ `wbalancer.conf`, `clouds.yaml` và SSH key ở chỗ bạn đang quản lý; không đưa credential vào thư mục cài đặt.
- Host key SSH theo cấu hình OpenSSH của chính coordinator (`~/.ssh/config` của user chạy `rolling-maint`, rồi `/etc/ssh/ssh_config`); tool không tự đặt `StrictHostKeyChecking` (từ 1.0.0rc10, bản này cũng bỏ `--ssh-strict-host-key`). Tool chạy ssh với `BatchMode=yes`, nên với chính sách mặc định của OpenSSH, host key lạ hoặc bị đổi sẽ làm lệnh lỗi (exit 255) và dừng run. Trước khi rollout, hoặc nạp sẵn `known_hosts` bằng key đã xác minh của các compute, ví dụ `ssh-keyscan -t ed25519 compute-03 >> ~/.ssh/known_hosts` rồi so fingerprint với host, hoặc đặt chính sách mong muốn trong `ssh_config`, ví dụ `Host compute-*` / `StrictHostKeyChecking accept-new`. Compute cài lại thì xóa key cũ (`ssh-keygen -R compute-03`) rồi thêm key mới.
- Luôn đặt `--prom-url` cho `rolling-maint`. Tham số này bật gate đòi mọi target Prometheus của host phải up; bỏ đi thì gate này không chạy.
- Bắt đầu bằng `--dry-run`. Xem kết quả xong thì bỏ `--dry-run` và thêm `-y` cho lần thực thi đã duyệt.
- **Đặt timeout ngay từ đầu.** Máy vật lý có thể boot rất lâu (firmware, kiểm tra RAM, controller lưu trữ). Đo tay một lần reboot rồi đặt `--boot-timeout` (mặc định 3600 s) có dư: host cần 50 phút mới lên thì đặt `--boot-timeout 5400`. Timeout là một phần định danh của run nên không đổi được khi run còn dang dở. Hết boot timeout thì run dừng, host vẫn disabled; kiểm tra console (BMC/IPMI). Tool không bao giờ tự power-cycle host.

## 2. File trạng thái và lock

| Dữ liệu | Mặc định |
|---|---|
| Trạng thái bảo trì (checkpoint) | `/var/lib/wbalancer/maintenance` (`rolling-maint --state-dir`) |
| Migration journal | `/var/lib/wbalancer/operations/migration-journal.json` (cạnh file cooldown) |
| Cooldown | `/var/lib/wbalancer/vm-cooldown.json` |
| Log / kết quả | `/var/log/wbalancer/maintenance` (`rolling-maint --log-dir`) |
| Audit event | `/var/log/wbalancer/migrations.jsonl` |
| Coordinator lock | `/var/lock/wbalancer-maint.lock` (`rolling-maint --lock-file`) |
| Balancer lock | `/var/lock/wbalancer-balancer.lock` |
| Config (chỉ đọc) | `/etc/wbalancer/wbalancer.conf` (`wbalancer --config`, `rolling-maint --wlb-config`; biến môi trường `WLB_CONFIG_PATH` thay mặc định này) |

**Chỉ một coordinator.** Lock là file `flock` local, coordinator thứ hai sẽ không thấy.

| Process | Cách giữ lock |
|---|---|
| `rolling-maint` (apply) | Giữ coordinator lock suốt lượt chạy |
| `wbalancer --drain-host` do `rolling-maint` gọi | Dùng chung lock của process cha (descriptor kế thừa) |
| `wbalancer` apply chạy riêng (cron, chạy tay) | Lấy cùng coordinator lock, rồi tới balancer lock. Lock nào đang bị giữ thì in `Not running: …` hoặc "Another … apply run holds the lock" và exit 1 |
| `wbalancer --monitor-only`, `--dry-run` | Không lấy lock |
| Legacy `wloadbalancer.py` | Không tham gia lock này: không bao giờ chạy apply legacy cùng lúc với v1 |

- Đổi `--state-dir` không cho phép chạy song song: đường dẫn lock vẫn là một.
- Giữ đường dẫn lock mặc định. Nếu đổi `rolling-maint --lock-file` thì export cùng đường dẫn vào `WLB_COORDINATOR_LOCK` cho các lượt `wbalancer` chạy riêng, nếu không hai bên không còn loại trừ nhau.
- Không xóa hay sửa checkpoint/journal khi còn run dang dở.
- `WLB_CONFIG_PATH=/duong/dan/wbalancer.conf` (rc12) đặt file đó làm config mặc định cho cả hai lệnh; `--config` / `--wlb-config` ghi rõ vẫn được ưu tiên, giá trị rỗng coi như không đặt. Dùng khi chạy thử không phải root; bộ test đặt biến này về một đường dẫn tạm nên không bao giờ đọc file `/etc/wbalancer/wbalancer.conf` chỉ root đọc được của máy.

**Nâng cấp từ 1.0.0rc8 trở về trước:** các bản đó dùng cùng bố cục nhưng có hậu tố `-v1` (`/var/lib/wbalancer-v1`, `/var/log/wbalancer-v1`, `/var/lock/wbalancer-v1-*.lock`). Nếu `/var/lib/wbalancer-v1` còn file cooldown, migration journal hoặc checkpoint bảo trì mà `/var/lib/wbalancer` chưa có gì, mọi lượt chạy thật sẽ dừng trước khi lấy lock và in hướng dẫn; `--dry-run` và `--monitor-only` chỉ cảnh báo. Chuyển trạng thái khi không có gì đang chạy:

```bash
mkdir -p /var/lib/wbalancer && mv /var/lib/wbalancer-v1/* /var/lib/wbalancer/
mkdir -p /var/log/wbalancer && mv /var/log/wbalancer-v1/* /var/log/wbalancer/   # tùy chọn: giữ lịch sử
```

## 3. Chờ sau mỗi migration (settle)

| Chế độ | Sau mỗi migration | Lý do |
|---|---|---|
| Balance apply (cron) | Chờ đủ cửa sổ metric, mặc định **360 giây** (`RUNTIME_SETTLE_SECONDS`), kể cả sau migration chạy ngoài tool | Cron 10 phút nên gần như không còn gì phải chờ |
| Drain (`--drain-host`, rolling-maint) | **Không chờ.** Vẫn chờ migration đang chạy kết thúc, rồi cộng tải dự phóng của các VM vừa tới vào host đích | Drain 28 VM có chờ đủ cửa sổ mất 3 giờ 27 phút |

Host nguồn không bị trừ tải. Report drain in dòng "Load added for VMs that arrived within the metric window …" kèm `method` của từng host.

### 3.1. RAM: đo trực tiếp

Host RAM là `avg_over_time(...[5m])`, nên VM vừa tới chỉ hiện dần. Với mỗi host nhận VM:

1. Tool chờ, tối đa `max(90 s, 3 × scrape interval)`, cho tới khi **mẫu RAM thô** (không lấy trung bình) của host mới hơn thời điểm VM cuối cùng tới xong.
2. Có mẫu mới: RAM = mẫu thô (`method=measured`). Sai số đo trên 2 canary lab (33 lượt VM tới, host 504 GiB): tối đa +0,03 / −0,02 điểm %.
3. Hết hạn mà chưa có mẫu mới: RAM = `max(TB, mẫu thô) + RSS trên host` của các VM tới sau mẫu đó (`measured+rss`, thiên về dư). Ví dụ: host 504 GiB, TB 50,0%, mẫu thô 51,0%, 2 VM RSS 16 GiB tới sau mẫu → 51,0 + 32/504 × 100 = 57,3%.
4. Không đọc được mẫu thô: dùng mô hình `RSS × (1 − t/300)`, 300 s là cửa sổ query RAM, t là số giây kể từ lúc VM tới (`method=model`).
5. Luôn dùng **RSS trên host**, không dùng RAM guest báo. Ví dụ: một VM báo trong guest dùng 5,95 GiB nhưng trên host giữ 31,77 GiB RSS.

`PROMETHEUS_MEM_USED_INSTANT` và `PROMETHEUS_MEM_SAMPLE_TIMESTAMP` (query đọc mẫu thô) mặc định dựng từ node job.

### 3.2. CPU: mô hình ngắn hạn

Host CPU là `irate`, bắt kịp sau 2 lần scrape. Phần CPU cộng thêm giảm tuyến tính trong cửa sổ = 2 × scrape interval của node job, đọc từ `/api/v1/targets` của Prometheus (scrape 30 s → cửa sổ 60 s). Không đọc được interval thì dùng 60 s; ép giá trị bằng `COMPENSATION_CPU_WINDOW_SECONDS`.

Ví dụ: VM 4 vCPU chạy 50% (2 core) tới host 64 core → +3,1 điểm lúc vừa tới, +1,6 điểm sau 30 s, về 0 sau 60 s.

Tải CPU của VM lấy từ metric của chính VM; nếu VM đang lộ trên 2 host thì dùng mẫu đo trước migration; không có gì thì dùng nguyên flavor.

## 4. Timeout của migration

- Ngừng chờ khi Nova **không báo tiến độ trong `MIGRATION_TIMEOUT` = 600 s**, tính từ lúc bắt đầu hoặc từ lần có tiến độ gần nhất. Tiến độ là thay đổi của bộ đếm bộ nhớ/đĩa, trạng thái hoặc `updated_at` của chính migration, hoặc của migration đang chạy trước nó trên cùng host nguồn (vì `max_concurrent_live_migrations` làm VM sau phải `queued`).
- Trần tuyệt đối `MIGRATION_MAX_WAIT` = 1800 s (30 phút). VM lớn thật sự cần lâu hơn thì tăng `MIGRATION_MAX_WAIT` (ví dụ VM 256 GiB chép ở tốc độ thực 150 MB/s cần khoảng 30 phút).
- Cả hai là khóa config (`wbalancer.conf`), được kiểm lúc khởi động (`MIGRATION_MAX_WAIT` ≥ `MIGRATION_TIMEOUT`). Khóa `MIGRATION_STALL_TIMEOUT` của rc8 đã lỗi thời: tool bỏ qua và in cảnh báo.
- Cùng quy tắc cho monitor live migration, cho việc chờ migration journal và chờ migration chạy ngoài tool.

Ví dụ: đứng im ngay từ đầu → dừng ở phút 10; có tiến độ tới phút 12 rồi đứng im → dừng ở phút 22; tới phút 30 vẫn còn tiến độ → dừng ở trần. Trên lab kiểm định, Nova cập nhật tiến độ khoảng mỗi 6 s khi chép bộ nhớ; khoảng im lặng dài nhất đo được là 34 s. Nova tự abort migration sau `live_migration_completion_timeout` (mặc định 800 s/GiB).

**Khi timeout:** tool **không** hủy migration trong Nova. Tool ghi event `migration_timeout`, đặt cooldown cho VM, giữ journal ở trạng thái pending và không submit gì thêm. Lần chạy sau tự reconcile khi migration kết thúc (mục 10).

## 5. Độ tươi của metric và cửa sổ settle

- Mỗi snapshot đều kiểm tra lại tuổi mẫu thô (tối đa 120 s) và validate lại đúng move đã duyệt.
- `RUNTIME_SETTLE_SECONDS` là biến môi trường, không phải khóa config. Giá trị phải phủ cửa sổ PromQL lớn nhất của mọi query đang cấu hình (tính cả `offset` và duration ghép như `10m30s`) cộng ít nhất 60 s; thiếu thì apply từ chối chạy. Ví dụ: query tùy biến dùng `[5m] offset 1m` có cửa sổ 360 s, nên `RUNTIME_SETTLE_SECONDS` phải ≥ 420.
- Query tùy biến vẫn phải dùng các raw exporter series mà freshness guard kiểm tra: `node_cpu_seconds_total`, `node_memory_MemAvailable_bytes`, `node_memory_MemTotal_bytes`, `libvirt_domain_info_cpu_time_seconds_total`, `libvirt_domain_info_virtual_cpus`, `libvirt_domain_stat_memory_rss_bytes` (và `libvirt_domain_stat_memory_usable_bytes` / `_available_bytes` nếu có).
- `PROMETHEUS_QUERY_URL` là bắt buộc: thiếu thì báo lỗi cấu hình, không có giá trị mặc định trỏ tới host public.
- Giá trị số trong config phải hữu hạn: `nan` và `inf` bị từ chối.

## 6. Đặt timeout cho drain

`--drain-timeout` mặc định 7200 s đủ cho khoảng 100 VM ở nhịp drain (không chờ giữa các migration). Nhịp đo trên lab: 1364 s / 19 VM và 1335 s / 19 VM, tức ~70 s/VM.

Host lớn hơn thì tăng timeout **trước khi** chạy, vì settings thuộc định danh checkpoint. Ví dụ: 150 VM × 70 s = 10 500 s; dư 20% thì đặt `--drain-timeout 12600`.

## 7. Ràng buộc của planner

- VM bị **lock**, đang có **task** Nova, hoặc flavor cần tài nguyên planner không mô hình được (`hw:cpu_policy=dedicated/mixed`, `pci_passthrough:*`, `resources:*`) không bao giờ được chọn để balance. Khi drain, các VM này được báo `unmovable` với lý do `locked`, `task_in_progress` (tạm thời, drain tự chờ) hoặc `unsupported_resources`.
- `rolling-maint` **dừng trước khi disable host** nếu host còn VM ACTIVE bị lock, giống cách xử lý VM ERROR. Danh sách `--locked` của CLI chỉ là gợi ý (python-openstackclient 6.6.0 bỏ qua filter và trả về mọi VM), nên từng VM được xác nhận lại bằng `server show` microversion 2.9.
- Server group dùng đúng ngữ nghĩa filter của Nova: chỉ xét host đích. `anti-affinity` tôn trọng `max_server_per_host`. Group `affinity` đã bị tách sẵn thì được vào bất kỳ host nào đang có member.
- `soft-anti-affinity` / `soft-affinity`: balance không bao giờ làm tệ hơn. Từ rc11, group `soft-anti-affinity` đã vỡ sẵn còn được tách lại, mỗi lần chạy một VM (7.1). Drain ưu tiên đích ít vi phạm nhất rồi mới tới MAD (rc11: rồi tới đích ít thành viên soft-anti-affinity nhất); vi phạm không tránh được thì in vào mục "Soft (anti-)affinity cannot be fully kept".
- VM chỉ di chuyển trong availability zone của nó (lấy theo `os-availability-zone` detail). Hệ quả: **mọi move sang AZ khác đều bị chặn, kể cả VM lúc tạo không chỉ định AZ**, vì Nova báo zone VM đang chạy cho mọi VM. Cách này bảo thủ hơn chính Nova, vốn cho scheduler đặt VM không pin AZ vào zone bất kỳ. Cần chuyển VM như vậy sang zone khác thì làm tay.
- Host đích không được vượt ngưỡng hotspot CPU/RAM sau move, kể cả khi đã vượt từ trước. MAKE_ROOM không đặt VM đệm lên chính host nguồn đang nóng.
- Một host thiếu metric/Placement, hoặc một target Prometheus thiếu label `alias`, chỉ làm lỗi aggregate chứa nó. Các aggregate khác vẫn chạy, exit cuối vẫn khác 0.

Planner là heuristic: không tìm thấy plan không chứng minh không tồn tại cách sắp xếp khác. Phần dự phóng runtime không bảo đảm guest không tăng tải sau khi submit. Placement vẫn là nguồn capacity của scheduler. Không hạ ngưỡng hay thời gian settle chỉ để vượt guard; giữ kiểm tra server group khi apply.

`PLANNER_CANDIDATE_BUDGET` (trong config, mặc định 100.000 lượt xét candidate cho mỗi lần tìm) giới hạn việc tìm kiếm. Vượt budget thì trả `budget_exhausted`, không báo "infeasible", và không apply plan chưa được đánh giá đầy đủ. Drain không chạy thêm planner direct/make-room. Benchmark trên lab 3 host / 47 VM (1.0.0rc13, đếm mọi HTTP request): một lần snapshot từ trạng thái nguội (`--monitor-only`) gọi 31 GET tới OpenStack, một POST xác thực và 7 truy vấn Prometheus; bản thân planner không gọi API. Đây là số đo, không phải cam kết độ trễ cho mọi cluster.

### 7.1. Heal soft-anti-affinity (rc11)

Khi một group `soft-anti-affinity` có từ 2 thành viên trên cùng một host của aggregate và không host nào là hotspot, lần chạy balance chuyển một thành viên sang host chưa có thành viên, thay cho move cân bằng thông thường ([thuật toán, mục 8](algorithm.vi.md#8-heal-soft-anti-affinity-rc11)). Ví dụ dry-run (ví dụ mẫu trong tài liệu thuật toán):

```text
WARNING - soft-anti-affinity spread broken (1 violation(s) in 1 group(s)) and can be healed automatically (next: group db)
...
Proposed plan: SOFT_AA_HEAL | db2 (A -> B)
Load status: UNBALANCED (WARNING)
Soft-anti-affinity: 1 broken group(s), 1 violation(s)
  - heal db: db2 (A -> B), group violations 1 -> 0, weighted MAD 10.13 -> 8.57

INFO: Aggregate prod-agg: soft_aa_heal: group db (1 violation(s)) -> move db2 A -> B (weighted MAD 10.133 -> 8.572)
[Round 1] [SOFT_AA_HEAL] Plan=SOFT_AA_HEAL steps=1
   Step 1: db2 (A -> B) reason=soft_aa_heal
   ...
   Soft-anti-affinity heal: group db violations 1 -> 0
   Heal MAD gate: CPU 0.22 <= 3.00, RAM 10.66 <= 12.27 (per resource: TARGET_AVG_MAD 3, or before + 0.05 when already above it)
```

Group không bao giờ sửa được trong aggregate này (ở đây 4 thành viên trên 3 host) mang nhãn `unresolvable`; nó không làm đổi exit code:

```text
OK - cluster balanced
...
Soft-anti-affinity: 1 broken group(s), 1 violation(s)
  - etcd (1 violation(s)): unresolvable: every host of the aggregate already holds a member (3 host(s), 4 member(s))

INFO: Aggregate prod-agg: soft_aa_heal: group etcd (1 violation(s)) unresolvable: every host of the aggregate already holds a member (3 host(s), 4 member(s))
```

Group chỉ chưa heal được trong lượt này mang nhãn `deferred`, kèm số đếm nguyên nhân. Ví dụ ngay sau một lần move tay, khi metric của VM vừa move còn nằm trên hai host (khoảng 5 phút) và thành viên kia đang cooldown: `deferred: no movable member on the crowded host(s) (cooldown=1, missing_vm_details=1)`. Các lý do deferred khác: `no member-free destination passes the filters (would_create_new_hotspot=1, mad_tolerance=1)`, `planner candidate budget exhausted`. `mad_tolerance` nghĩa là move sẽ đẩy MAD CPU hoặc MAD RAM vượt `TARGET_AVG_MAD` (làm aggregate thành `UNBALANCED`), hoặc làm một MAD vốn đã vượt ngưỡng tăng quá `SOFT_AA_HEAL_MAD_TOLERANCE`. Ở aggregate đã cân bằng, chuyện này chỉ xảy ra với thành viên đủ lớn để làm mất cân bằng (ví dụ VM 32 GiB trên host 256 GiB: MAD RAM 0 → 8,33 > 3).

Event audit (`/var/log/wbalancer/migrations.jsonl`):

| `result` của event | Khi nào | Trường của heal |
|---|---|---|
| `dry_run`, `migration_success`, `migration_failed`, `migration_timeout`, `step_revalidation_failed`, … | Move heal đi đúng đường thực thi thông thường (journal, kiểm lại, cooldown, email) | `"plan_type": "soft_aa_heal"`, `"mode": "soft_aa_heal"`, `"step_reason": "soft_aa_heal"`, `"soft_aa_heal": {"group_name": "db", "group_violations_before": 1, "group_violations_after": 0, "violation_reduction": 1, …}` |
| `soft_aa_heal_unresolvable` | Một lần chạy balance (apply hoặc dry-run) gặp group vỡ không bao giờ sửa được trong aggregate này (mọi host đều có thành viên) | `"mode": "soft_aa_heal"`, `"detail": "etcd: unresolvable: …"`, `"soft_aa_unresolvable": [...]`, `"soft_aa_violations": 1` |
| `soft_aa_heal_deferred` | Một lần chạy balance gặp group vỡ mà lượt này chưa heal được (cooldown, lock, metric sau move, bộ lọc, cổng MAD, ngân sách) | `"mode": "soft_aa_heal"`, `"detail": "db: deferred: …"`, `"soft_aa_deferred": [{…, "blocking": {"cooldown": 1, "missing_vm_details": 1}}]` |

- Mọi event mang nhãn của snapshot mà kế hoạch được **lập** từ đó. `migration_success` của một heal ghi `"mode": "soft_aa_heal"`, `"detail_status": "SOFT_AA_HEAL"`, `"severity": "WARNING"`, giống event `dry_run` của nó; move trực tiếp mang mode tải của nó (`"mode": "mad"`, …). Event drain mang `"mode": "drain"`, `"detail_status": "DRAIN_FEASIBLE"` / `"DRAIN_INFEASIBLE"` và `"severity": null`: drain không phải phép kiểm cân bằng, kết quả của nó nằm ở `result`. (Tới rc10, mọi `migration_success` và event drain đều ghi `CRITICAL` / `UNBALANCED_NO_VALID_MOVE`, chép từ lần làm mới không lập kế hoạch trước khi submit.)
- Bước heal bị từ chối ghi rõ lý do, ví dụ `soft_aa_heal_no_longer_heals` (group đã được sửa trong lúc chờ) hoặc `refreshed RAM MAD 3.120 above heal limit 3.000`, và nhận cooldown "bị từ chối" 30 phút như thường.
- `soft_aa_heal_unresolvable` / `soft_aa_heal_deferred` được ghi mỗi aggregate mỗi lần chạy balance chừng nào tình trạng còn (cron 10 phút thì tới 144 dòng một ngày với group cứ mãi unresolvable; group deferred thường hết sau vài lượt). `--monitor-only` không ghi event.
- Lọc mọi event heal: `jq -c 'select(.mode == "soft_aa_heal")' /var/log/wbalancer/migrations.jsonl`.

Muốn tắt thì đặt `HEAL_SOFT_ANTI_AFFINITY=false` trong `wbalancer.conf` (hoặc trong biến môi trường của cron/unit); lần chạy sau hành xử đúng như rc10, kể cả tie-break, và vẫn liệt kê group vỡ kèm `heal disabled`. Muốn giữ riêng một VM khỏi heal (và khỏi balance) thì đưa nó vào file loại trừ (`EXCLUDE_VMS_FILE`); heal sẽ thử thành viên kế tiếp. Tắt tính năng không hủy migration heal đang pending: journal vẫn đối chiếu như thường.

## 8. Profile `kernel-update`

- Chạy `dist-upgrade` **toàn bộ package plan đã duyệt**, không chỉ kernel.
- Refresh APT metadata trước lượt chạy: profile đòi success stamp của APT update không cũ hơn 24 giờ.
- Simulation có gỡ gói thì bị từ chối.
- Plan được capture ở `pre_drain`, trước khi disable host.
- Trước khi upgrade, phần việc còn lại phải là **tập con theo tên gói** của plan đã duyệt. Chấp nhận: apt-daily nâng version, hoặc unattended-upgrades đã cài một phần plan. Từ chối: tên gói mới, hoặc bất kỳ gói nào bị gỡ. Ví dụ: plan `{linux-image-…, openssl, libssl3}`; unattended-upgrades đã cài `openssl` → còn `{linux-image-…, libssl3}` thì chấp nhận; xuất hiện `curl` thì từ chối.
- Plan đã cài hết thì `dist-upgrade.sh` trả 0.
- Reboot khi có `/var/run/reboot-required` **hoặc** khi `/boot` có kernel mới hơn kernel đang chạy (máy không có hook update-notifier).
- Sau đó kiểm tra dpkg/initrd/modules/GRUB, lưu expected kernel, xác minh kernel thực sau boot. GRUB có saved/next entry tùy chỉnh thì bị từ chối.
- Chỉ khi không có update/kernel mới mới SKIP; lỗi apt/dpkg phải fail.

## 9. Hook: schema và exit code

Cách viết profile: [profiles.vi.md](profiles.vi.md). Các khóa của hook:

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `id` | `<stage>[index]` | ID ổn định; trạng thái resume lưu theo `<stage>/<id>` |
| `name` | 60 ký tự đầu của lệnh | Nhãn trong log |
| `script` hoặc `cmd` | bắt buộc | File script trong thư mục profile, hoặc lệnh viết thẳng (chỉ một trong hai) |
| `check` hoặc `check_cmd` | không có | Check recovery; bắt buộc với `idempotency: reconcile` |
| `run` | `ssh` | `ssh` = chạy trên compute, `local` = chạy trên coordinator (bash) |
| `timeout` | 600 s | Hết hạn thì kết quả là UNKNOWN |
| `idempotency` | `never` | `never`, `idempotent` hoặc `reconcile` |
| `capture` | không có | Chỉ `expected_kernel` hoặc `package_plan` |
| `retries` / `retry_delay` | 0 / 10 s | Thử lại khi exit khác 0, chỉ với hook `idempotent` |

Exit code:

| Lệnh | rc 0 | rc 1 | rc 3 | rc 10 | Khác |
|---|---|---|---|---|---|
| Hook | Thành công | Lỗi | Lỗi; riêng `pre_drain`: không cần bảo trì, skip host | Lỗi | Lỗi |
| Check recovery | Đã hoàn tất và kiểm chứng: đánh dấu done, không chạy lại | UNKNOWN | UNKNOWN | Chứng minh an toàn để chạy lại: hook được chạy lại | UNKNOWN |
| `reboot_required_cmd` | Cần reboot | Không cần | UNKNOWN | UNKNOWN | UNKNOWN |

UNKNOWN làm run dừng.

Hook khi resume:

| Trạng thái hook từ lần chạy trước | Khi resume |
|---|---|
| `done` | Bỏ qua |
| Pending hoặc failed, có `check` | Check quyết định (bảng trên) |
| Failed, không có check, `idempotency: idempotent` | Chạy lại |
| Mọi trường hợp khác không có check (kể cả timeout hoặc SSH exit 255) | STOPPED: kết quả UNKNOWN, phải đối chiếu tay |

Không gắn `idempotent` tùy tiện. Placeholder `{host}`, `{address}`, `{profile}` được quote như shell token: không bọc thêm nháy quanh placeholder.

## 10. Khi bị STOPPED

1. Đọc state, log, result và trạng thái Nova thật. Giữ host disabled khi chưa chứng minh an toàn.
2. Giữ nguyên phiên bản tool, profile và config, **chạy lại đúng lệnh cũ**: tool tự tiếp tục ở stage đang dở. `--resume-from` không dùng để nhảy qua guard được.
3. **Migration pending:** migration journal tự reconcile ở lần chạy sau bằng `/os-migrations?instance_uuid=`, chỉ xét các bản ghi có id lớn hơn id lớn nhất ghi lại lúc submit.

   | Bằng chứng trong Nova | Kết quả journal |
   |---|---|
   | Bản ghi `error`/`failed`/`cancelled` và VM ở nguồn | `failed` |
   | VM ACTIVE ở đích | `success` |
   | Không có bản ghi, không có `task_state`, VM ACTIVE ở nguồn sau 120 giây kể từ submit | `not_started`. Nova API set `task_state=migrating` trước khi gửi sang conductor, nên điều kiện này chứng minh migration không bao giờ chạy |
   | VM đã bị xóa | `vm_deleted` |
   | Migration còn đang chạy | Chờ theo quy tắc ở mục 4 |
   | Không chứng minh được (ví dụ VM ở host thứ ba) | Giữ UNKNOWN: không submit migration mới |

   Event, email và cooldown của migration lỗi được ghi **trước** khi reconcile. VM bị để lại ở trạng thái ERROR trên nguồn sau migration lỗi thì journal đóng là `failed`; sau đó rolling-maint dừng ở bước kiểm VM ERROR, kèm hướng dẫn.

   Với kết quả UNKNOWN, tự đối chiếu bản ghi migration, instance action, host và task của server trong Nova; không reset ERROR hay submit lại mù. Trước mọi thao tác reset tay: lưu checkpoint và journal; xác nhận đúng VM, operation ID, nguồn/đích, các bản ghi migration tạo sau thời điểm submit và trạng thái terminal của chúng, không còn task hay job, domain ở nguồn thực sự đang chạy và không có domain ở đích, guest còn kết nối được. Chỉ reset trạng thái VM (`openstack server set --state active`) sau khi chứng minh guest vẫn chạy, và bảo đảm trong lúc đó không có lượt wbalancer/rolling-maint nào khởi động. Trước khi thử lại một migration lỗi, đối chiếu network info cache của Nova, port Neutron và XML domain libvirt ở nguồn và đích; không reboot guest hay sửa port để che một lỗi chưa giải thích được.
4. **Đã có reboot intent nhưng boot ID chưa đổi:** kiểm tra reboot job tạm (systemd unit `compute-maintenance-reboot`) và host; không xóa intent để ép reboot lần nữa. Resume chỉ chờ boot ID mới.
5. **Identity mismatch hoặc checkpoint legacy:** đối chiếu thủ công; không sửa hash trong state. Hoàn tất run bằng đúng phiên bản tool đã tạo checkpoint rồi mới nâng phiên bản.

Các trường hợp khác:

- Hook timeout hoặc SSH exit 255 giữ hook ở pending, kết quả UNKNOWN. Hook đã xong không replay; hook chưa rõ kết quả cần check để xác minh.
- Lỗi `post_restore` đưa host về disabled dưới quyền sở hữu của run (quarantine); resume bắt đầu lại từ health gate, kiểm tra inventory và không tự drain/reboot workload đã được schedule trong lúc host enabled.
- Health gate phải PASS liên tục, cùng boot ID, đủ các container, network agent, service Cinder và target Prometheus như mong đợi.

## 11. Rollback

- Rollback tool không hoàn tác migration, package upgrade hay reboot.
- Chỉ đổi phiên bản đã cài khi không còn run dang dở; giữ phiên bản cũ và state của nó cho recovery.
- Không bao giờ enable host bằng tay để vượt ownership guard: từ `drain` tới `restore`, host phải ở trạng thái disabled với reason `rolling-maint v1:<run_id>` của chính run đó.

## 12. Phạm vi recovery khi upgrade package

- Check upgrade chỉ trả rc 10 (an toàn để thử lại) khi dpkg sạch, không gỡ gói nào, và mọi tên gói còn lại đều nằm trong plan đã duyệt; trả rc 0 khi không còn gì để cài. Tên gói mới thì giữ UNKNOWN.
- Canary drain/reboot không kiểm định live package upgrade.
- Trước khi reboot, rolling-maint còn từ chối nếu host vẫn chạy process QEMU guest dù Nova không liệt kê VM nào đang chạy.
- Trước một cửa sổ upgrade đã duyệt, ghi lại và tạm dừng apt timer/unattended-upgrades theo runbook của host, kiểm tra không còn process apt/dpkg nào khác, và khôi phục lịch cũ sau khi xong.
- Không bao giờ sửa `package_plan` trong checkpoint để vượt guard.

Kết quả kiểm định từng bản: [CHANGELOG](../CHANGELOG.md).
