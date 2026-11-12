# Cách wbalancer ra quyết định

[English](algorithm.md) | Tiếng Việt

Tài liệu này giải thích `wbalancer` chọn live migration như thế nào, lập kế hoạch drain ra sao, và vì sao có lúc nó từ chối không chuyển gì. Mọi ngưỡng và công thức dưới đây đều lấy từ mã nguồn v1; link file trỏ tới nơi đặt từng quy tắc. Mỗi ý đi kèm một ví dụ số ngắn.

Không có ở đây: cài đặt và lệnh ([hướng dẫn sử dụng](user-guide.vi.md)), file trạng thái, lock và recovery ([vận hành](operations.vi.md)), chuyển từ script cũ ([hướng dẫn chuyển đổi](migrating-from-wloadbalancer.vi.md)).

## Khác gì so với `wloadbalancer.py`

Ý tưởng cốt lõi giữ nguyên: hotspot xét trước, rồi pressure, rồi cân bằng theo MAD có trọng số; trọng số CPU/RAM tự thích nghi theo tình huống; planner chọn đúng một move tốt nhất và chỉ thực hiện khi nó cải thiện đủ nhiều. Chi tiết thì đã đổi:

| Mục | `wloadbalancer.py` | v1 |
|---|---|---|
| RAM hotspot | > 75% | > 80% |
| RAM critical hotspot | > 85% | > 95% |
| CPU hotspot | > 85% | > 70% |
| Cải thiện tối thiểu, move trực tiếp | 0,05 | 0,15 |
| Kế hoạch 2 bước "dọn chỗ" | không có | MAKE_ROOM, cần ≥ 0,30 |
| Sàn trọng số 0,2 khi tín hiệu chia 1:9 | 0,182 / 0,818 (lấy sàn rồi chuẩn hóa lại) | 0,20 / 0,80 (sàn thật) |
| Capacity cho scheduler | Số liệu hypervisor của Nova × allocation ratio lấy từ openstack-exporter (mặc định dự phòng 7,0 / 1,2, chừa cứng 24 GiB RAM) | Inventory và usage của Placement, trừ 5% headroom |
| Mẫu số runtime (một VM chiếm bao nhiêu %) | Trường `vcpus` của hypervisor trong Nova; RAM = RAM sau overcommit − 24 GiB | Số CPU logic và `MemTotal` thật từ node_exporter |
| "Host đích không được thành hotspot" | chỉ ở chế độ hotspot | mọi chế độ, cả MAKE_ROOM và drain |
| Server group | xét cả nhóm; đọc lỗi → chạy tiếp như không có group | chỉ xét host đích (như Nova), `max_server_per_host`, policy soft; đọc lỗi → dừng |
| Host thiếu metric | bỏ host ra, chỉ in cảnh báo | từ chối cả aggregate (monitor: UNKNOWN) |
| Host có nova-compute chưa sẵn sàng | không làm đích, nhưng vẫn tính vào MAD | bỏ hẳn khỏi phép tính |
| VM bị bỏ qua | chỉ do cooldown | thêm: danh sách loại trừ, không ACTIVE, bị lock, đang có task, CPU pinning / PCI / `resources:*`, thiếu metric hoặc flavor |
| Availability zone, số VM tối đa mỗi host | không kiểm | có kiểm |
| Cooldown sau thành công / thất bại | 4 giờ / 4 giờ ở bản script cuối (README cũ vẫn ghi 30 / 60 phút) | 4 giờ / 4 giờ, thêm 30 phút khi move bị từ chối lúc kiểm lại |
| Timeout migration | cố định 600 giây | dừng sau 600 giây không có tiến độ (tính từ đầu hoặc từ lần có tiến độ gần nhất), tối đa 30 phút (cả hai cấu hình được) |
| Drain một host | không có | `--drain-host`, fail-closed |
| Group soft-anti-affinity bị vỡ (2+ thành viên trên một host) | không bao giờ được sửa | tự tách lại, mỗi lần chạy một move, khi không có host hotspot (rc11, mục 8) |
| Đổi ngưỡng | sửa script | khóa config (rc7), ví dụ `RAM_HOTSPOT_THRESHOLD=75`, `RAM_CRITICAL_THRESHOLD=85`, `CPU_HOTSPOT_THRESHOLD=85`, `MIN_IMPROVEMENT_THRESHOLD=0.05` cho lại đúng giá trị cũ |

## 1. Mỗi lần chạy, mỗi aggregate

```text
snapshot (Nova + Placement + Prometheus)
  -> chế độ (hotspot > pressure > MAD) -> trọng số CPU/RAM
  -> move trực tiếp tốt nhất (ưu tiên host hotspot, sau đó mọi host)
  -> nếu không move trực tiếp nào đạt 0.15: kế hoạch MAKE_ROOM tốt nhất
  -> nhận nếu cải thiện >= ngưỡng -> trạng thái + mức Icinga
  -> không có hotspot + group soft-anti-affinity bị vỡ: một move heal thay cho kế hoạch (mục 8)
  -> thực hiện: kiểm lại trên dữ liệu mới, migrate, ghi cooldown
```

- Mỗi aggregate có **tối đa một kế hoạch mỗi lần chạy**: một migration trực tiếp, một kế hoạch MAKE_ROOM 2 bước, hoặc một move heal soft-anti-affinity ([application/balancing.py](../src/compute_maintenance/application/balancing.py)).
- Sau một migration, tool lấy lại dữ liệu cluster trước khi xét aggregate tiếp theo, vì các aggregate có thể dùng chung host.
- Lỗi dữ liệu ở một aggregate (thiếu metric, thiếu resource provider trong Placement) chỉ làm bỏ qua aggregate đó; cả lần chạy vẫn exit 1.
- Planner xét tối đa `PLANNER_CANDIDATE_BUDGET` (mặc định 100 000) cặp VM × host cho mỗi aggregate. Hết ngân sách thì aggregate báo lỗi "feasibility unknown", không trả về kết quả tìm dở.

## 2. Planner nhìn thấy gì

### 2.1 Tải của host

Dựng trong [application/snapshots.py](../src/compute_maintenance/application/snapshots.py) từ các query trong [config.py](../src/compute_maintenance/config.py):

| Giá trị | Nguồn |
|---|---|
| CPU % của host | `1 − idle / tổng các mode` của `node_cpu_seconds_total`, dùng `irate` (2 lần scrape gần nhất) |
| RAM % của host | `100 − MemAvailable / MemTotal`, `avg_over_time` 5 phút |
| Số CPU logic, dung lượng RAM | đếm số series CPU idle, `MemTotal` (mẫu số runtime) |

Các quy tắc fail-closed:

- Chỉ cần một host trong aggregate thiếu metric CPU hoặc RAM, hoặc có giá trị ngoài khoảng −1…101%, là **cả aggregate** bị từ chối. Script cũ chỉ in cảnh báo rồi bỏ host đó, nên MAD được tính trên một bức tranh thiếu.
- Ít hơn 2 host có metric: bỏ qua aggregate.
- Balance và monitor loại các host có nova-compute không ở trạng thái enabled + up + không forced down. Tải của chúng không còn được tính vào trung bình, MAD, chế độ, đỉnh hay mức cảnh báo. VM trên các host này vẫn nằm trong topology, để kiểm server group vẫn thấy chúng.
- Ở chế độ apply, mọi sample thô (CPU host, MemAvailable, MemTotal, CPU VM, RSS VM, số vCPU) phải mới không quá 120 giây ([adapters/freshness.py](../src/compute_maintenance/adapters/freshness.py)).

### 2.2 Tải của VM

Lấy từ libvirt exporter ([adapters/prometheus.py](../src/compute_maintenance/adapters/prometheus.py)):

| Giá trị | Công thức |
|---|---|
| CPU % của VM | `irate(cpu_time) / virtual_cpus × 100`, chặn ở 100 (chấp nhận giá trị thô tới 1000% vì cgroup tính cả thread emulator và vhost) |
| RAM % của VM | `100 − usable / available` từ balloon driver trong guest, trung bình 5 phút |
| Số core thật mang đi | `vcpus × cpu% / 100` |
| RAM thật mang đi | `flavor_ram × ram% / 100`, hoặc RSS phía host |

RSS (`libvirt_domain_stat_memory_rss_bytes`) thay cho số liệu trong guest ở hai trường hợp: mọi VM khi chạy với `--use-vm-ram-rss`, và tự động cho guest không có balloon stats (appliance FreeBSD, Windows thiếu balloon service). RSS luôn lớn hơn hoặc bằng phần guest thật dùng, nên ước lượng nghiêng về phía an toàn.

VM không được làm ứng viên khi series của nó xuất hiện trên hai host cùng lúc (ngay sau một live migration), hoặc khi Nova và Prometheus báo hai host khác nhau.

### 2.3 Capacity cho scheduler lấy từ Placement

[adapters/openstack.py](../src/compute_maintenance/adapters/openstack.py) đọc inventory và usage `VCPU`, `MEMORY_MB` của resource provider từng host; [domain/policies.py](../src/compute_maintenance/domain/policies.py) đổi chúng thành capacity:

```text
nominal_capacity = (total - reserved) × allocation_ratio
policy_capacity  = nominal_capacity × (1 - headroom / 100)        headroom mặc định 5%
free             = policy_capacity - used
vừa              = flavor_vcpus <= free_VCPU  và  flavor_ram_mb <= free_MEMORY_MB
```

Ví dụ `VCPU`: 64 pCPU, ratio 4,0, đã dùng 236.

```text
nominal = 64 × 4.0 = 256      policy = 256 × 0.95 = 243.2      free = 243.2 - 236 = 7.2
flavor 4 vCPU vừa, flavor 8 vCPU không vừa (lý do từ chối placement_vcpu)
```

- Kiểm có vừa hay không thì dùng flavor; chấm điểm thì dùng mức dùng thực tế. VM chỉ dùng 10% của 32 GiB vẫn cần 32 GiB trống trong Placement.
- Tôn trọng `min_unit`, `max_unit` và `step_size` của inventory.
- Không cần openstack-exporter nữa. Các khóa cũ `PROMETHEUS_CPU_RATIO`, `PROMETHEUS_MEM_RATIO`, `PROMETHEUS_OPENSTACK_EXPORTER_JOB` bị bỏ qua kèm cảnh báo.
- Host đã cấp phát ≥ 90% capacity nominal được đếm ở dòng `Placement:` trong output. Chỉ để tham khảo.
- `PLACEMENT_FINAL_ALLOCATION_CANDIDATE_CHECK=true` (mặc định tắt) hỏi thêm `GET /allocation_candidates` cho ứng viên trực tiếp cuối cùng.

### 2.4 Dự phóng một move

[domain/balance/filters.py](../src/compute_maintenance/domain/balance/filters.py) `project_runtime_move`:

```text
cpu_impact = real_cores / logical_cpus(host) × 100
ram_impact = real_ram_gb / MemTotal_gb(host) × 100
nguồn -= impact(nguồn)        đích += impact(đích)
```

Ví dụ: VM 8 vCPU dùng 50% (4 core), 32 GiB dùng 50% (16 GiB). Host nguồn 64 CPU / 512 GiB, host đích 32 CPU / 256 GiB.

| | CPU | RAM |
|---|---|---|
| Nguồn | −4/64 = −6,25 điểm | −16/512 = −3,13 điểm |
| Đích | +4/32 = +12,50 điểm | +16/256 = +6,25 điểm |

Script cũ chia RAM cho capacity sau overcommit trừ 24 GiB: 512 × 1,2 − 24 = 590,4 GiB, nên cùng VM đó chỉ được tính 2,71 điểm thay vì 3,13.

## 3. Chọn chế độ

[domain/balance/modes.py](../src/compute_maintenance/domain/balance/modes.py) `choose_effective_mode`, xét theo thứ tự:

| Thứ tự | Chế độ | Điều kiện | Trọng số CPU / RAM |
|---|---|---|---|
| 1 | `ram_critical_hotspot` | có host RAM > 95% | 0,1 / 0,9 cố định |
| 2 | `ram_hotspot` | có host RAM > 80% | 0,2 / 0,8 cố định |
| 3 | `cpu_hotspot` | có host CPU > 70% | 0,8 / 0,2 cố định |
| 4 | `pressure` | trung bình CPU hoặc RAM của aggregate > 80% | thích nghi, từ MAD và trung bình |
| 5 | `mad` | còn lại | thích nghi, từ MAD |

- Chế độ critical chỉ đổi trọng số; tên trạng thái giống hệt `ram_hotspot`.
- **Với ngưỡng mặc định của v1, chế độ pressure không bao giờ xảy ra.** Trung bình trên 80% nghĩa là ít nhất một host trên 80%, mà như vậy đã là RAM hotspot (> 80%) hoặc CPU hotspot (> 70%). Ví dụ CPU 84/82/80 (trung bình 82) là pressure ở script cũ; ở v1 thành `cpu_hotspot`. Từ rc7 các ngưỡng là khóa config: đặt `PRESSURE_THRESHOLD` thấp hơn `RAM_HOTSPOT_THRESHOLD` hoặc `CPU_HOTSPOT_THRESHOLD` thì pressure mới dùng được. Ví dụ `PRESSURE_THRESHOLD=65` với CPU 68/69/67 (trung bình 68, không host nào vượt 70) cho ra `pressure`. Khi pressure không thể kích hoạt, lượt chạy ghi một dòng log INFO.
- **Nấc heal (rc11):** giữa `cpu_hotspot` và `pressure`/`mad` có nấc heal soft-anti-affinity (mục 8). Nó không phải một chế độ: chỉ chạy khi không có host hotspot, giữ nguyên trọng số của `pressure`/`mad`, và nhường lại cho chúng khi không có gì sửa được.
- **Tìm ở host hotspot trước:** ở chế độ hotspot, planner thử trước các VM trên host nóng nhất (RAM cao nhất với chế độ RAM, CPU cao nhất với `cpu_hotspot`). Nếu không move nào từ đó đạt ngưỡng, nó tìm trên mọi host rồi giữ kết quả nào cải thiện nhiều hơn. Output ghi là `HOTSPOT-FIRST (host)`, `FALLBACK-FULL (after host)` hoặc `FULL`.

## 4. Chấm điểm: MAD và trọng số thích nghi

### Ví dụ A. Độ lệch tuyệt đối trung bình (MAD)

Ba host, CPU 60/40/35%, RAM 68/42/38%.

```text
cpu_mean = (60 + 40 + 35) / 3 = 45.00
cpu_mad  = (|60-45| + |40-45| + |35-45|) / 3 = (15 + 5 + 10) / 3 = 10.00
ram_mean = (68 + 42 + 38) / 3 = 49.33
ram_mad  = (18.67 + 7.33 + 11.33) / 3 = 12.44
```

MAD càng cao thì các host càng lệch nhau. Không host nào vượt 70% CPU hay 80% RAM, nên đây là chế độ `mad`.

### Ví dụ B. Trọng số thích nghi với sàn thật

Ở chế độ `mad`, tín hiệu chính là MAD; ở chế độ `pressure`, tín hiệu là `0,7 × MAD + 0,3 × trung bình`. Tỷ phần của mỗi tín hiệu thành trọng số, rồi bị kẹp để không bên nào xuống dưới 0,2:

```text
cpu_weight = min(max(cpu_signal / (cpu_signal + ram_signal), 0.2), 1 - 0.2)
ram_weight = 1 - cpu_weight                (cả hai tín hiệu bằng 0 -> 0.5 / 0.5)
```

Từ ví dụ A:

```text
cpu_weight = 10.00 / (10.00 + 12.44) = 0.446      ram_weight = 0.554
weighted_mad = 10.00 × 0.446 + 12.44 × 0.554 = 11.36
```

Ví dụ về sàn, `cpu_mad = 1`, `ram_mad = 9`: tỷ phần CPU là 0,10, v1 cho đúng **0,20 / 0,80**. Script cũ lấy `max(tỷ phần, 0,2)` rồi chuẩn hóa lại, khiến sàn chỉ còn 0,182 / 0,818.

Ví dụ pressure (chỉ có nghĩa khi sửa ngưỡng): `cpu_mean 90`, `ram_mean 60`, `cpu_mad 4`, `ram_mad 8`:

```text
cpu_signal = 4 × 0.7 + 90 × 0.3 = 29.8      ram_signal = 8 × 0.7 + 60 × 0.3 = 23.6
cpu_weight = 29.8 / 53.4 = 0.558             ram_weight = 0.442
```

Nếu chỉ nhìn MAD thì RAM được ưu tiên (8 > 4); cộng thêm trung bình thì tài nguyên nóng hơn thắng. Tỷ lệ 0,7 / 0,3 giữ cân bằng là mục tiêu chính.

### Ví dụ C. Mức cải thiện của một move

```text
improvement = baseline_weighted_mad - weighted_mad_sau_move       (giữ nguyên trọng số)
```

Chuyển một VM tương đương 12 điểm CPU và 12 điểm RAM từ host A sang host B (các host giống nhau):

```text
CPU 60/40/35 -> 48/52/35      cpu_mad 10.00 -> 6.67
RAM 68/42/38 -> 56/54/38      ram_mad 12.44 -> 7.56
sau = 6.67 × 0.446 + 7.56 × 0.554 = 7.16
improvement = 11.36 - 7.16 = 4.20
```

Planner mô phỏng chuyển từng VM đi được sang từng host còn lại, giữ mức cải thiện dương lớn nhất, và chỉ nhận khi nó **≥ 0,15** (`MIN_IMPROVEMENT_THRESHOLD`). Gộp thành một điểm duy nhất giúp so sánh được move giúp CPU nhiều nhưng làm hại RAM với move giúp cả hai một chút.

MAD mục tiêu 3 chỉ dùng để gắn nhãn trạng thái (`UNBALANCED` hay `BALANCED`), không chặn việc thực hiện: aggregate có MAD ≤ 3 mà có move đạt 0,15 (`BALANCED_OPTIMIZABLE`) thì move đó vẫn được chạy.

## 5. Bộ lọc: khi nào move bị cấm

Mỗi lần từ chối được đếm theo lý do và in ra ở dòng `Rejected candidate pairs: …`.

### 5.1 Lọc VM

| Lý do bỏ qua | Nghĩa | Khi drain |
|---|---|---|
| `excluded` | Khớp `EXCLUDE_VMS_FILE` / `--exclude-file` | bỏ qua quy tắc này |
| `cooldown` | Đã chuyển hoặc chuyển lỗi trong 4 giờ qua, hoặc bị từ chối lúc kiểm lại trong 30 phút qua | bỏ qua quy tắc này |
| `missing_vm_details` | Không có metric dùng được, series nằm trên hai host, hoặc Nova và Prometheus lệch host | unmovable (tạm thời) |
| `not_active` | Trạng thái Nova khác `ACTIVE` | unmovable |
| `incomplete_metadata` / `incomplete_metrics` | Không rõ vCPU/RAM của flavor, hoặc thiếu metric CPU/RAM | unmovable (tạm thời) |
| `locked` | VM bị lock; Nova từ chối migrate | unmovable |
| `task_in_progress` | Nova đang có `task_state` (rebuild, snapshot, resize…) | unmovable (tạm thời) |
| `unsupported_resources` | Flavor có `hw:cpu_policy=dedicated` hoặc `mixed`, `pci_passthrough:*` hay `resources:*`; các loại này cấp phát PCPU / PCI / resource class riêng mà mô hình VCPU/MEMORY_MB không kiểm được | unmovable |

### 5.2 Lọc host đích, theo thứ tự

| Lý do từ chối | Quy tắc |
|---|---|
| `destination_not_ready` | nova-compute phải enabled, up, không forced down |
| `max_instances` | `số server trên host + 1 ≤ giới hạn` lấy từ `MAX_INSTANCES_PER_HOST` (JSON theo host) hoặc `MAX_INSTANCES_PER_HOST_DEFAULT`; 0 = không giới hạn; mọi server Nova trên host đều được đếm |
| `server_group` | affinity / anti-affinity cứng (5.3) |
| `soft_server_group` | move làm group soft tệ hơn (5.3) |
| `availability_zone` | zone của VM (Nova `OS-EXT-AZ`, không có thì lấy zone của host nguồn) phải trùng zone host đích; host đích không rõ zone → từ chối. Hệ quả: **không move nào được đi sang AZ khác**, kể cả VM lúc tạo không chỉ định AZ. Nova báo `OS-EXT-AZ` (zone VM đang chạy) cho mọi VM, nên VM như vậy vẫn bị giữ ở zone hiện tại. Cách này bảo thủ hơn Nova, vốn cho scheduler đặt VM không pin AZ vào zone bất kỳ |
| `placement_vcpu`, `placement_memory`, `placement_*_min_unit`… | flavor không vừa Placement (2.3) |
| `runtime_capacity_missing` | thiếu số CPU logic / MemTotal của host |
| `would_create_new_hotspot` | host đích sẽ vượt CPU 70% hoặc RAM 80% (5.4) |
| `hotspot_guardrail` | chỉ ở chế độ hotspot (5.4) |

### 5.3 Server group theo cách Nova áp dụng

v1 làm giống `ServerGroupAffinityFilter` / `ServerGroupAntiAffinityFilter` của Nova: chỉ xét **host đích** ([domain/policies.py](../src/compute_maintenance/domain/policies.py)).

| Policy | Được chuyển khi |
|---|---|
| `affinity` | host đích đã có một thành viên khác, hoặc chưa thành viên nào khác được đặt |
| `anti-affinity` | số thành viên khác trên host đích < `rules.max_server_per_host` (mặc định 1) |
| `soft-anti-affinity` | điểm phạt = số thành viên khác trên host; balance từ chối move làm điểm phạt tăng (gác cổng). Từ rc11, group đã vỡ còn được chủ động tách lại (mục 8) |
| `soft-affinity` | điểm phạt = 1 khi có thành viên khác nhưng không ai ở host đó; quy tắc như trên (chỉ gác cổng, không bao giờ gom lại) |

Ví dụ: group anti-affinity có vm1 trên A, còn vm2 và vm3 cùng trên B (vi phạm có sẵn).

- vm1 → C: C không có thành viên nào khác → **được phép**. Script cũ xét cả nhóm, thấy B có hai thành viên nên chặn mọi move của mọi thành viên.
- vm1 → B: B đã có 2 thành viên khác ≥ 1 → từ chối. Nếu `max_server_per_host=3` thì được (2 < 3).

Policy soft quan trọng vì Nova không áp nó khi live migration chỉ định thẳng host đích. Drain không từ chối vì policy soft; nó ưu tiên host đích có điểm phạt thấp nhất và liệt kê vi phạm dưới dạng cảnh báo (mục 9). Server group được đọc trên mọi project; đọc lỗi thì lần chạy dừng (script cũ chạy tiếp như không có group). `--no-server-groups` tắt cả kiểm cứng lẫn soft.

Từ rc11, soft-anti-affinity được xử lý theo hai cách:

- **Gác cổng** (giữ nguyên): không move balance hay MAKE_ROOM nào được làm tăng điểm phạt của group soft, nên balance không bao giờ làm group vỡ thêm.
- **Sửa** (mới): group đã vỡ sẵn, ví dụ do drain hoặc failover không còn chỗ khác, được nấc heal tách lại, mỗi lần chạy một move (mục 8). Tới rc10 group như vậy cứ vỡ mãi, trừ khi tình cờ có move MAD tách nó ra.

### 5.4 Quy tắc hotspot

Hai lớp kiểm trong [domain/balance/filters.py](../src/compute_maintenance/domain/balance/filters.py) (`destination_hot_after`, `is_hotspot_guardrail_satisfied`):

| Kiểm | Áp dụng ở | Quy tắc |
|---|---|---|
| Không làm nóng host đích | mọi chế độ, MAKE_ROOM, kiểm lại trước khi balance | host đích sau move phải ≤ 70% CPU và ≤ 80% RAM; host đích đã vượt sẵn cũng bị từ chối |
| Hotspot guardrail | chế độ hotspot | đỉnh của tài nguyên đang nóng trong aggregate phải giảm hẳn, và host đích phải nằm hẳn dưới ngưỡng |

Ví dụ D bên dưới minh họa cả hai.

## 6. Nhận một move: ví dụ tính tay

### Ví dụ D. RAM hotspot

Host (CPU / RAM): A 55 / 86, B 50 / 52, C 48 / 49. RAM 86 > 80 → `ram_hotspot`, trọng số 0,2 / 0,8. (Script cũ coi 86% là critical vì ngưỡng critical của nó là 85.)

```text
cpu_mad = 2.67   ram_mad = 15.78   baseline = 2.67 × 0.2 + 15.78 × 0.8 = 13.16
```

Tìm ở host hotspot trước: chỉ xét VM trên A.

| Ứng viên | Kết quả | Quyết định |
|---|---|---|
| VM1 (4 điểm CPU, 16 điểm RAM) A → B | A 51 / 70, B 54 / 68; đỉnh RAM 86 → 70; B 68 < 80 | sau = 2,00 × 0,2 + 8,89 × 0,8 = 7,51, cải thiện **5,64 ≥ 0,15**, nhận |
| VM2 (32 điểm RAM) A → C | C sẽ lên 49 + 32 = 81% | từ chối: `would_create_new_hotspot` |
| VM2 A → B | B sẽ lên 84% | từ chối |

VM2 giải phóng nhiều RAM hơn cho A, nhưng chỉ dời hotspot sang chỗ khác. Trạng thái `RAM_HOTSPOT_OPTIMIZABLE`, mức WARNING.

### Ví dụ E. CPU hotspot

CPU 74 / 45 / 40, RAM 50 / 48 / 46 trên các host 64 CPU, 512 GiB. CPU 74 > 70 → `cpu_hotspot` (script cũ vẫn ở chế độ MAD cho tới 85). Trọng số 0,8 / 0,2, baseline 14,00 × 0,8 + 1,33 × 0,2 = 11,47. Một VM 8 vCPU dùng 80% (10 điểm CPU) và dùng 8 GiB RAM (1,56 điểm RAM) rời A:

| Host đích | CPU sau move | Weighted MAD sau | Cải thiện |
|---|---|---|---|
| B | 64 / 55 / 40 | 7,20 | 4,27 |
| C | 64 / 45 / 50 | 5,93 | **5,54**, được chọn |

### Ví dụ F. Move hợp lệ mà vẫn bị từ chối

RAM 56 / 50 / 44, CPU 30 / 30 / 30. RAM MAD 4,00 > 3 → lệch. CPU MAD bằng 0 nên sàn cho 0,2 / 0,8, baseline 3,20. Giả sử mọi VM lớn đang cooldown và VM duy nhất chuyển được chỉ tương đương 0,2 điểm RAM và 0,1 điểm CPU (khoảng 1 GiB và 0,06 core trên host 512 GiB, 64 CPU):

```text
RAM 56/50/44 -> 55.8/50/44.2    ram_mad 3.87      CPU MAD 0.07
sau = 0.07 × 0.2 + 3.87 × 0.8 = 3.11      improvement = 0.09 < 0.15
```

Move qua được mọi bộ lọc, nhưng live migration chỉ để được 0,09 là xáo trộn vô ích. Không migrate gì. Trạng thái `UNBALANCED_NO_VALID_MOVE`, lý do `no_candidate_above_improvement_threshold`, mức WARNING. Với ngưỡng cũ (0,05) thì script cũ đã migrate.

## 7. MAKE_ROOM: kế hoạch 2 bước

Khi không move trực tiếp nào đạt 0,15, planner tìm kế hoạch 2 bước ([domain/balance/makeroom.py](../src/compute_maintenance/domain/balance/makeroom.py) `find_best_make_room_plan`): host B rất hợp cho VM V nhưng bị chặn **chỉ vì capacity Placement**, nên trước tiên chuyển một VM nhỏ R từ B sang host thứ ba C, rồi chuyển V từ A sang B. Tie-break soft-anti-affinity của rc11 (8.5) chỉ áp cho ứng viên move trực tiếp ở các mode cân bằng (hotspot, MAD lẫn pressure); việc chọn kế hoạch MAKE_ROOM cố ý không dùng nó.

| Quy tắc | Giá trị |
|---|---|
| Bật | `ENABLE_MAKE_ROOM=true` (mặc định) |
| Độ sâu | đúng 2 bước (`MAKE_ROOM_MAX_DEPTH` bắt buộc là 2) |
| Host đích B | V không vừa B **chỉ** vì `placement_vcpu` / `placement_memory` |
| VM dọn chỗ R | qua được bộ lọc VM; CPU của chính nó ≤ 10% (`MAKE_ROOM_MAX_VM_CPU_PERCENT`) và RAM trong guest ≤ 50% (`MAKE_ROOM_MAX_VM_RAM_PERCENT`) |
| Host đệm C | host thứ ba (không phải A hay B); qua các bộ lọc như host đích của move trực tiếp, kể cả không thành hotspot |
| Kiểm bước 2 | sau khi mô phỏng R nhả tài nguyên, V phải vừa B; server group và số instance được kiểm lại với R đã ở C |
| Trạng thái cuối | không host nào thành hotspot mới; B và C giữ ≤ 70% CPU / 80% RAM; có hotspot guardrail ở chế độ hotspot |
| Ngưỡng | tổng cải thiện ≥ `MAKE_ROOM_MIN_IMPROVEMENT`, mặc định 0,30 (= 0,15 × 2 migration; cấu hình dưới 0,15 bị từ chối) |

### Ví dụ G. MAKE_ROOM

Các host 64 CPU / 512 GiB. A 60 / 78, B 30 / 45, C 45 / 60 (CPU / RAM). Chế độ MAD, trọng số 0,469 / 0,531, baseline 10,71.

- V trên A: 8 vCPU dùng 80%, 32 GiB dùng 80% → 10,00 điểm CPU, 5,00 điểm RAM.
- B còn trống 7,2 VCPU trong Placement (ví dụ ở 2.3) → V không vừa (`placement_vcpu`).
- V → C bị từ chối: V nằm trong group anti-affinity với một VM trên C. Không move trực tiếp nào khác đạt 0,15.
- R trên B: 4 vCPU dùng 5% CPU, 8 GiB dùng 30% RAM → đạt điều kiện (5 ≤ 10, 30 ≤ 50).

```text
Bước 1  R: B -> C     B nhả 4 VCPU -> trống 11.2      C 45.31 / 60.47 (không thành hotspot)
Bước 2  V: A -> B     giờ V vừa (8 <= 11.2)
Cuối    CPU 50 / 39.69 / 45.31   RAM 73 / 49.53 / 60.47
        cpu_mad 3.54   ram_mad 8.00   sau = 5.91   improvement = 4.80 >= 0.30
```

Cách thực hiện ([application/balancing.py](../src/compute_maintenance/application/balancing.py)):

1. Bước 1 được kiểm lại và không được làm C thành hotspot, nhưng tự nó không cần cải thiện MAD.
2. Trước bước 2, tool chờ cửa sổ metric ổn định (10.2), lấy lại toàn bộ dữ liệu, và chỉ thử lại khi Placement vẫn còn báo usage cũ (`PLACEMENT_SYNC_TIMEOUT` 30 giây, 2 giây/lần).
3. Bước 2 phải tự cải thiện trạng thái **mới** ít nhất 0,15 (4,91 nếu metric khớp dự phóng). Nếu không còn an toàn hay đáng làm thì bỏ bước 2; R cứ nằm lại ở C và event log ghi lý do.

Khi có tìm MAKE_ROOM, planner giữ ứng viên nào (trực tiếp hay MAKE_ROOM) cải thiện nhiều hơn, và ứng viên đó phải vượt ngưỡng của chính nó. Ví dụ: trực tiếp 0,10, MAKE_ROOM 0,25 → giữ MAKE_ROOM, cần 0,30, nên không chạy gì.

## 8. Heal soft-anti-affinity (rc11)

Nova chỉ kiểm group soft-anti-affinity lúc đặt VM. Một lần drain hoặc failover không còn chỗ khác có thể để hai thành viên của group nằm chung một host: cứu workload trước, tách lại sau, đúng đánh đổi mà VMware HA chọn mặc định. Tới rc10 không có gì sửa lại. Từ rc11 balancer tự sửa, theo cách sửa vi phạm ràng buộc (constraint correction) của VMware DRS: sửa vi phạm ở lần chạy kế tiếp thay vì trông chờ một move cân tải tình cờ sửa giúp ([domain/balance/heal.py](../src/compute_maintenance/domain/balance/heal.py) `plan_soft_aa_heal`). Mặc định bật; `HEAL_SOFT_ANTI_AFFINITY=false` cho lại đúng hành vi rc10, khi đó group vỡ chỉ được báo cáo.

### 8.1 Phát hiện

Với mỗi group `soft-anti-affinity`, theo từng aggregate:

```text
vi phạm = tổng theo host của max(0, số thành viên trên host - 1)
```

Chỉ tính các VM có trong snapshot và đang nằm trên host của aggregate (host bị loại vì chưa sẵn sàng, 2.1, không tính). Group có ít nhất 1 vi phạm là **vỡ**. Ví dụ: thành viên ở A, A, A, B → (3 − 1) + (1 − 1) = 2 vi phạm. Group `anti-affinity` cứng và `soft-affinity` không được heal.

### 8.2 Vị trí trong thang ưu tiên

```text
ram_critical_hotspot > ram_hotspot > cpu_hotspot > soft_aa_heal > cân bằng MAD (pressure / mad)
```

- **Sau các hotspot.** Còn hotspot thì heal được hoãn (`heal deferred while a hotspot is handled`). Host 95% RAM là vấn đề an toàn tài nguyên ngay lúc này; hai thành viên của một group chung host là vấn đề dự phòng, chờ được vài lượt cron. VMware DRS xếp ngược lại: nó sửa vi phạm ràng buộc trước khi cân tải, vì với DRS mất cân bằng chỉ là tối ưu. Ở wbalancer hotspot là ngưỡng an toàn, nên an toàn tài nguyên đứng trên việc sửa dự phòng.
- **Trước cân bằng MAD.** Không có hotspot thì move heal khả thi sẽ thay cho kế hoạch cân bằng của lượt đó. Tìm kiếm MAD bình thường vẫn chạy; kết luận của nó hiện ở dòng `Load status:`.
- **Không bao giờ chặn cân bằng.** Heal bị tắt, không có group vỡ, hoặc không có move heal khả thi: lượt chạy tiếp tục cân bằng MAD y như trước.
- Vẫn **một move mỗi aggregate mỗi lần chạy**.

### 8.3 Chọn move heal

1. **Group:** group có host tệ nhất chứa nhiều thành viên nhất được xét trước. Không sửa được thì thử group vỡ kế tiếp.
2. **Nguồn:** host chứa nhiều thành viên của group nhất (sau đó tới các host đông khác).
3. **VM:** thành viên có RAM flavor nhỏ nhất qua được bộ lọc VM (5.1: không bị loại trừ, không trong cooldown, không bị lock, không có task, không CPU pinning / PCI / `resources:*`). Nếu nó không có host đích hợp lệ thì thử thành viên nhỏ kế tiếp.
4. **Đích:** chỉ host **không có** thành viên nào của group, và phải qua **toàn bộ** bộ lọc host đích ở 5.2: sẵn sàng, giới hạn instance, server group cứng, group soft không tệ hơn, AZ, Placement với headroom 5%, capacity runtime, không tạo hotspot mới.
5. **Lợi ích ròng:** move phải giảm ít nhất một vi phạm tính tổng, đếm mọi group soft-anti-affinity của VM.
6. **Cổng MAD, theo từng tài nguyên:** với CPU và RAM riêng rẽ, `MAD sau ≤ giới hạn`, trong đó `giới hạn = TARGET_AVG_MAD` khi `MAD trước ≤ TARGET_AVG_MAD`, và `giới hạn = MAD trước + SOFT_AA_HEAL_MAD_TOLERANCE` khi `MAD trước > TARGET_AVG_MAD` (mặc định 3 và 0,05; `soft_aa_heal_mad_check`). Nhãn `UNBALANCED` dùng đúng phép thử này (MAD CPU hoặc MAD RAM vượt `TARGET_AVG_MAD`, mục 11), nên heal không bao giờ đẩy MAD của một tài nguyên vượt ngưỡng nếu trước đó nó chưa vượt, và không bao giờ tạo thêm việc cho nấc MAD. Tài nguyên đã vượt ngưỡng chỉ được tăng tối đa bằng dung sai. Heal không được mua dự phòng bằng một aggregate mất cân bằng.
7. Trong các đích còn lại, chọn đích có **headroom** lớn nhất: `min(70 − CPU sau, 80 − RAM sau)` (hai ngưỡng hotspot), rồi tới weighted MAD thấp hơn.

### Ví dụ I. Heal

Host 32 CPU / 256 GiB. CPU / RAM: A 30 / 70, B 25 / 40, C 28 / 45. Không hotspot, chế độ `mad`: CPU MAD 1,78, RAM MAD 12,22, trọng số 0,20 / 0,80 (sàn), baseline 10,13. Group `db` (soft-anti-affinity) gồm db1 (8 vCPU, 16 GiB, CPU 50%, RAM 90% = 14,4 GiB) và db2 (4 vCPU, 8 GiB, CPU 20%, RAM 75% = 6 GiB), cả hai trên A: 1 vi phạm.

- Nguồn A (2 thành viên). Flavor nhỏ nhất: db2 (8 GiB), tương đương 0,8 core = 2,50 điểm CPU và 6 GiB = 2,34 điểm RAM.
- Đích: B và C (không có thành viên `db`); cả hai qua mọi bộ lọc.

| Đích | CPU / RAM sau | Headroom | MAD CPU / MAD RAM sau (giới hạn 3,00 / 12,27) | Weighted MAD sau |
|---|---|---|---|---|
| B | 27,50 / 42,34 | min(70 − 27,50, 80 − 42,34) = **37,66** | 0,22 / 10,66, qua cổng | 8,57 |
| C | 30,50 / 47,34 | min(39,50, 32,66) = 32,66 | 1,89 / 10,66, qua cổng | 8,91 |

Giới hạn tính theo từng tài nguyên: MAD CPU 1,78 ≤ 3 nên giới hạn là 3,00; MAD RAM 12,22 > 3 nên giới hạn là 12,22 + 0,05 = 12,27. db2 A → B: MAD CPU 1,78 → 0,22, MAD RAM 12,22 → 10,66 (weighted 10,13 → 8,57), vi phạm 1 → 0. Trạng thái `SOFT_AA_HEAL`, mức WARNING, `Load status: UNBALANCED (WARNING)`. Nếu chỉ có nấc MAD thì nó sẽ chuyển db1 (cải thiện 2,00); nấc heal chọn thành viên nhỏ hơn, migration rẻ hơn.

### 8.4 Group chưa heal được, thực thi, tốc độ hồi phục

- **Lượt này không có move heal:** cân bằng MAD vẫn chạy, còn output, log và một event ghi lý do. Mức monitor **không** bị nâng: giống fault "Could not fix anti-affinity rule violation" của DRS, có báo nhưng không spam cảnh báo. Có hai nhãn:
  - `unresolvable`, chỉ dành cho trường hợp do cấu trúc: mọi host của aggregate đều đã có thành viên. Ví dụ: group 4 thành viên trong aggregate 3 host luôn còn ít nhất 1 vi phạm, và lượt chạy nào cũng ghi như vậy (event `soft_aa_heal_unresolvable`).
  - `deferred`, cho nguyên nhân có thể tự hết: thành viên đang cooldown, bị lock, có task, hoặc metric còn nằm trên hai host ngay sau một lần move (`missing_vm_details`, khoảng 5 phút); mọi đích không có thành viên đều bị một bộ lọc hoặc cổng MAD từ chối; hết ngân sách tìm. Kèm số đếm nguyên nhân, ví dụ `deferred: no movable member on the crowded host(s) (cooldown=1, missing_vm_details=1)` (event `soft_aa_heal_deferred` với `"blocking": {"cooldown": 1, "missing_vm_details": 1}`).
- **Cổng trên aggregate đã cân bằng** (trường hợp chính: group vỡ trong khi tải vẫn ổn). Cùng host và group như Ví dụ I, nhưng CPU 30 / 30 / 30 và RAM 50 / 50 / 50: MAD CPU 0 và MAD RAM 0, đều không vượt ngưỡng, nên cả hai giới hạn đều là **3**. db2 (flavor 8 GiB, dùng 6 GiB: 2,50 điểm CPU và 2,34 điểm RAM) đi A → B cho CPU 27,5 / 32,5 / 30 và RAM 47,66 / 52,34 / 50: **MAD CPU 1,67 ≤ 3 và MAD RAM 1,56 ≤ 3, được phép**; aggregate vẫn `BALANCED`. Phản ví dụ: nếu cả hai thành viên là VM 8 vCPU / 32 GiB dùng hết RAM và nửa CPU (mỗi VM 12,5 điểm CPU và 12,5 điểm RAM), mọi move heal đều kết thúc ở MAD CPU 8,33 và MAD RAM 8,33 > 3 và vẫn bị từ chối (`mad_tolerance`); group như vậy phải chuyển tay hoặc thêm capacity.
- **Một tài nguyên vượt là đủ để từ chối.** Trọng số 0,2 / 0,8 (trước đó MAD CPU 0, MAD RAM 2,0): heal kết thúc ở MAD CPU 0,5 / MAD RAM 3,6 bị từ chối vì RAM 3,6 > 3 (MAD RAM 2,0 trước đó chưa vượt ngưỡng nên giới hạn là ngưỡng), dù weighted MAD (2,98) dưới 3; cùng heal đó với thành viên dùng ít RAM hơn (MAD RAM 2,4) thì qua. **Sát dưới ngưỡng thì ngưỡng vẫn quyết định:** MAD RAM trước 2,98 → giới hạn 3,00 (không phải 3,03), nên heal kết thúc ở 3,00 thì qua, ở 3,02 thì không. **Khi đã vượt ngưỡng thì dung sai quyết định:** MAD RAM trước 3,50 → giới hạn RAM 3,55, nên heal kết thúc ở MAD RAM 3,51 thì qua, ở 3,60 thì không.
- **Thực thi** đi đúng đường của move trực tiếp: journal, freshness, kiểm lại, cooldown, event. Bước kiểm lại (10.1) đọc lại server group và kiểm move **vẫn còn heal** (nếu không: `soft_aa_heal_no_longer_heals`) và MAD CPU, MAD RAM trên dữ liệu mới vẫn trong cùng giới hạn theo từng tài nguyên, tính từ MAD CPU/RAM của chính snapshot mới bằng cùng hàm với planner; ngưỡng cải thiện 0,15 không áp dụng. Loại kế hoạch, lý do bước và mode của event đều là `soft_aa_heal`.
- **Tốc độ hồi phục:** mỗi aggregate một move mỗi lần chạy, cộng settle barrier (10.2) trước mỗi snapshot balance. N vi phạm cần khoảng N lần chạy, tức khoảng **N × 10 phút** với cron 10 phút. Ví dụ: drain để lại 3 vi phạm thì khoảng 30 phút được sửa xong, lâu hơn nếu phải xử lý hotspot trước hoặc có thành viên đang trong cooldown.

### 8.5 Phân định khi ngang nhau (tie-break)

Cả hai đều tắt khi `HEAL_SOFT_ANTI_AFFINITY=false`.

- **Cân bằng (mọi mode cân bằng — hotspot, MAD, pressure):** trong các move trực tiếp vượt 0,15 và nằm trong khoảng `SOFT_AA_TIEBREAK_MARGIN` (mặc định 0,15, tương đối) so với mức cải thiện tốt nhất, move giảm được nhiều vi phạm soft-anti-affinity nhất được chọn; giảm bằng nhau thì giữ move cải thiện tốt hơn. Ví dụ: move tốt nhất 10,25 (không thuộc group), một move heal được group 9,00: 9,00 ≥ 10,25 × 0,85 = 8,71 nên chọn move heal; một move heal chỉ 4,50 thì không. Move dưới 0,15 không bao giờ được nhận.
- **Drain:** khóa xếp `(điểm phạt soft, weighted MAD)` (9.1) có thêm khóa thứ ba là số thành viên thuộc các group soft-anti-affinity của VM trên host đích. Nó chỉ quyết định khi hai khóa đầu bằng nhau, ví dụ VM thuộc một group soft-anti-affinity và một group soft-affinity, đứng trước hai host giống hệt nhau, trong đó một host chứa thành viên của cả hai group. Nó không bao giờ đổi việc drain nào khả thi.

## 9. Lập kế hoạch drain

`--drain-host H` làm trống một host mà vẫn giữ các host còn lại cân bằng ([domain/drain.py](../src/compute_maintenance/domain/drain.py)). Mục tiêu là làm trống host chứ không phải cải thiện MAD, nên ngưỡng cải thiện và hotspot guardrail được tắt. Placement, server group cứng, AZ, giới hạn instance và trần runtime vẫn giữ. Danh sách loại trừ và cooldown bị bỏ qua.

### 9.1 Các bước

1. **Phân loại** từng VM trên H là chuyển được hay unmovable (lý do ở 5.1). `locked`, `not_active:SHUTOFF`, `unsupported_resources` và tương tự cần người vận hành xử lý. Khi chỉ còn lại lý do tạm thời (`task_in_progress`, `incomplete_metrics`, `incomplete_metadata`, `missing_vm_details`, ví dụ VM còn bị che bởi cửa sổ metric của migration vừa xong), drain lập lại kế hoạch 15 giây/lần trong tối đa 360 giây rồi mới kết luận.
2. **Ngân sách capacity:** tổng flavor của các VM chuyển được so với tổng `free` Placement trên các host còn lại đang sẵn sàng. Thiếu là kết luận không khả thi ngay, kèm số cụ thể (`not enough VCPU: need 14, only 10 safe-free on 2 host(s) (short 4)`).
3. **Xếp tham lam (greedy)**, thử theo ba thứ tự (RAM flavor giảm dần, vCPU flavor giảm dần, RAM thực dùng giảm dần). Với mỗi VM, từng host còn lại được kiểm: sẵn sàng, giới hạn instance, server group cứng, AZ, Placement, rồi trần runtime. Trong các host qua được, chọn host có `(điểm phạt group soft, weighted MAD của các host còn lại)` nhỏ nhất; từ rc11, khi hai khóa này bằng nhau hoàn toàn thì host có ít thành viên thuộc các group soft-anti-affinity của VM hơn được chọn (8.5). Sau mỗi lần xếp, usage Placement, tải, số instance và topology group đều được cập nhật.
4. **Chọn thứ tự tốt nhất:** ít VM bị chặn nhất, rồi ít cảnh báo group soft nhất, rồi weighted MAD cuối thấp nhất.
5. **Fail-closed:** chỉ khả thi khi không VM nào bị chặn, không thiếu capacity và không host nào dự phóng vượt 95% RAM. Ngược lại thì không migrate gì, không đụng tới service nova-compute; exit 2.

Trần runtime trên host nhận ([config.py](../src/compute_maintenance/config.py)):

| | Trần CPU | Trần RAM |
|---|---|---|
| mặc định | 70% | 80% |
| `--drain-allow-hotspot` | không giới hạn (100%) | 95%, trần tuyệt đối, không bao giờ nới |

Trọng số lấy từ snapshot hiện tại của aggregate, tính cả host đang drain; điểm chỉ tính trên các host còn lại. Group affinity có từ 2 thành viên trên H không thể rời đi từng VM một; báo cáo nêu tên group và gợi ý chuyển tay hoặc dùng `--no-server-groups`.

### Ví dụ H. Drain

Drain H1; còn lại H2 và H3 (mỗi host 64 CPU / 512 GiB). CPU / RAM: H1 30 / 60, H2 40 / 72, H3 35 / 55. Chế độ MAD, trọng số 0,341 / 0,659.

| VM trên H1 | Flavor | Mức dùng | Điểm CPU / RAM |
|---|---|---|---|
| vm1 | 8 vCPU, 64 GiB | CPU 50%, RAM 75% (48 GiB) | 6,25 / 9,38 |
| vm2 | 4 vCPU, 32 GiB | CPU 25%, RAM 50% | 1,56 / 3,13 |
| vm3 | 2 vCPU, 16 GiB | CPU 50%, RAM 50% | 1,56 / 1,56 |

Thứ tự theo RAM flavor: vm1, vm2, vm3. Placement còn chỗ trên cả hai host.

| VM | → H2 | → H3 | Chọn |
|---|---|---|---|
| vm1 | RAM 81,4% > 80 → `ram_cap_exceeded` | RAM 64,4%, điểm 2,73 | H3 |
| vm2 | RAM 75,1%, điểm 3,60 | RAM 67,5%, điểm 1,96 | H3 |
| vm3 | RAM 73,6%, điểm 2,21 | RAM 69,1%, điểm 1,71 | H3 |

Cuối cùng H2 40 / 72, H3 44,4 / 69,1: khả thi.

Cùng aggregate đó, nhưng H3 chỉ còn **48 GiB** RAM trống an toàn trong Placement. vm1 (64 GiB) không còn vừa H3 (`placement_memory`) và sẽ đẩy H2 lên 81,4%. vm1 bị chặn nên **cả kế hoạch drain không khả thi**: không VM nào được chuyển, exit 2, báo cáo liệt kê vm1 kèm cả hai lý do. Thêm `--drain-allow-hotspot` thì trần RAM thành 95%: vm1 → H2 (81,4%), vm2 và vm3 → H3, khả thi.

### 9.2 Vòng thực hiện

[application/balancing.py](../src/compute_maintenance/application/balancing.py) `drain_host_workload`:

- nova-compute trên H bị disable trước (trừ khi có `--drain-no-disable`); chỉ được enable lại nếu drain bị hủy và chính lần chạy này đã disable nó.
- Mỗi vòng lấy dữ liệu Nova, Placement, Prometheus mới, lập lại kế hoạch cho toàn bộ phần còn lại và chỉ migrate **bước đầu tiên**. Không migration nào được gửi từ một kế hoạch cũ.
- Trước khi gửi, bước đó được kiểm lại. Bước bị từ chối (chưa gửi gì) được lập lại sau 30 giây, tối đa 3 lần liên tiếp. Migration lỗi, hoặc kế hoạch lập lại không còn khả thi, thì hủy drain (exit 1).
- Tối đa `2 × số bước + 3` vòng. Drain kết thúc khi kế hoạch không còn bước nào; exit 0 chỉ khi H không còn VM (còn VM unmovable → exit 2).

### 9.3 Bù tải: vì sao drain không phải chờ 6 phút mỗi VM

RAM host là trung bình 5 phút, CPU host dùng `irate`. VM vừa tới được t giây thì số đo của host đích vẫn thiếu `(1 − t/300)` phần RAM của nó. Không bù thì vòng sau thấy host đích trống hơn thực tế và tiếp tục dồn VM vào. Balance tránh chuyện này bằng cách chờ đủ cửa sổ (10.2). Drain mà cũng chờ thì mất 6 phút mỗi VM, nên thay vào đó nó cộng tải của các VM vừa tới vào host đích ([application/snapshots.py](../src/compute_maintenance/application/snapshots.py) `apply_arrival_compensation`):

| Cách | Khi nào | RAM của host nhận |
|---|---|---|
| `measured` | có sample `MemAvailable` thô mới hơn mọi lần VM tới (tool chờ tối đa max(90 giây, 3 × chu kỳ scrape) để có sample này) | lấy đúng sample thô đó, không ước lượng |
| `measured+rss` | có sample thô nhưng một số VM tới sau sample đó | max(trung bình, sample thô) + RSS phía host của các VM tới sau |
| `model` | không có sample thô | trung bình + RSS × (1 − tuổi / cửa sổ RAM) |

CPU được cộng `core của VM / số CPU host × 100 × (1 − tuổi / cpu_window)`, với `cpu_window = 2 × chu kỳ scrape` (dự phòng 60 giây, hoặc `COMPENSATION_CPU_WINDOW_SECONDS`). Host nguồn không bao giờ bị trừ (đếm dư là phía an toàn). Tải của VM lấy từ dữ liệu planner; VM vừa tới mà chưa có dữ liệu thì lấy CPU của nó 15 giây trước khi migration bắt đầu; không còn nguồn nào khác thì lấy nguyên flavor.

Ví dụ: vm1 (RSS 48 GiB, 4 core) tới H3 (512 GiB, 64 CPU) được 60 giây; chu kỳ scrape 30 giây.

```text
RAM trung bình còn thiếu 48 × (1 - 60/300) = 38.4 GiB = 7.5 điểm   -> model cộng +7.5
CPU: 4/64 × 100 × (1 - 60/60) = 0        (irate đã bắt kịp)
     nếu VM mới tới 15 giây:  6.25 × (1 - 15/60) = +4.69 điểm
```

Đo trên lab 3 compute: RAM sau khi bù lệch không quá ±0,03 điểm so với sample thô kế tiếp, trong khi không bù thì sai khoảng 2,6 điểm.

## 10. An toàn khi thực hiện

### 10.1 Kiểm lại trước khi gửi

[application/migrations.py](../src/compute_maintenance/application/migrations.py) `validate_migration_step` đọc lại Nova và Placement ngay trước migration:

- VM còn tồn tại, `ACTIVE`, không có task, không bị lock, vẫn ở host nguồn như kế hoạch.
- Host đích sẵn sàng, cùng AZ, flavor vừa Placement **mới đọc**, chưa chạm giới hạn instance.
- Server group đọc lại từ Nova (cả group soft, trừ khi drain).
- Dự phóng mới: host đích không vượt 70% CPU / 80% RAM (drain có `--drain-allow-hotspot`: không vượt 95% RAM), và không bao giờ vượt 95% RAM.
- Move của balance (trực tiếp và bước 2 của MAKE_ROOM): hotspot guardrail, và cải thiện trên dữ liệu mới ≥ 0,15.
- Move heal soft-anti-affinity (rc11): thay cho mức cải thiện 0,15, move phải còn giảm được vi phạm trên topology server group mới đọc, và MAD CPU, MAD RAM trên dữ liệu mới đều phải không vượt `TARGET_AVG_MAD`, hoặc chỉ tăng tối đa `SOFT_AA_HEAL_MAD_TOLERANCE` nếu đã vượt từ trước: đúng cổng của planner (8.3).
- Snapshot mới được dựng theo đúng quy tắc host của planner: balance bỏ host không nhận được VM ra khỏi baseline, drain thì giữ. Vì vậy host chuyển sang unready sau lúc lập kế hoạch không làm lệch mean/MAD dùng để đo mức cải thiện; nếu đó là host nguồn hoặc đích của bước thì bước bị từ chối.
- Từ lúc lấy snapshot mới tới lúc gửi quá 120 giây → từ chối, lập kế hoạch lại ở lần chạy sau.

Bước bị từ chối không gửi gì cả và đưa VM vào cooldown "bị từ chối" 30 phút, để cron 10 phút không lập lại đúng move đó ở mỗi lần chạy.

### 10.2 Chờ metric ổn định (balance)

Ở chế độ apply, snapshot của balance chờ tới khi không còn migration nào chạy trên các host của aggregate (kể cả migration không do tool này tạo) và đã qua `RUNTIME_SETTLE_SECONDS` (mặc định 360) kể từ migration cuối cùng kết thúc. 360 = cửa sổ query 5 phút + 60 giây dư cho scrape; đặt giá trị nhỏ hơn khoảng PromQL dài nhất + 60 giây thì bị từ chối.

### 10.3 Cooldown

| Sự kiện | Cooldown | Lưu ở |
|---|---|---|
| Migration thành công | 4 giờ (`VM_COOLDOWN_SECONDS`) | `moved` |
| Migration lỗi / timeout | 4 giờ (`VM_FAILED_MOVE_COOLDOWN_SECONDS`) | `failed` |
| Bị từ chối khi kiểm lại | 30 phút (`VM_REJECTED_MOVE_COOLDOWN_SECONDS`) | `rejected` |

Mục nào còn lâu nhất thì tính mục đó; thành công thì xóa hai mục kia. Drain bỏ qua cả ba.

### 10.4 Timeout migration theo tiến độ

[domain/errors.py](../src/compute_maintenance/domain/errors.py) `ProgressTimer(600, 600, 1800)`: một cửa sổ "không có tiến độ" (khóa config `MIGRATION_TIMEOUT`, mặc định 600 giây) và một trần (`MIGRATION_MAX_WAIT`, mặc định 1800 giây):

- ngừng chờ khi 600 giây liền không có gì thay đổi, tính từ lúc bắt đầu hoặc từ lần thay đổi gần nhất: trạng thái migration, bộ nhớ/đĩa đã chuyển và còn lại, `updated_at` của bản ghi, hoặc migration xếp hàng phía trước trên cùng host nguồn (đọc 15 giây/lần);
- không bao giờ chờ quá 30 phút (1800 giây); bản thân migration vẫn chạy tiếp trong Nova và lần chạy sau sẽ đối chiếu.

| Migration | Script cũ | v1 |
|---|---|---|
| Có tiến độ suốt, xong ở phút 25 | timeout ở phút 10 | thành công ở phút 25 |
| Đứng im ngay từ đầu | timeout ở phút 10 | dừng ở phút 10 |
| Có tiến độ tới phút 2 rồi đứng im | timeout ở phút 10 | dừng ở phút 12 |
| Có tiến độ tới phút 12 rồi đứng im | timeout ở phút 10 | dừng ở phút 22 |
| Tới phút 30 vẫn còn tiến độ | timeout ở phút 10 | dừng ở phút 30 (`max_wait`); tăng `MIGRATION_MAX_WAIT` cho các VM như vậy |

VM vẫn `ACTIVE` trên host nguồn mà sau 120 giây chưa có bản ghi migration thì báo `stuck_active`. Timeout không được coi là bằng chứng thất bại: migration journal đối chiếu kết quả thật với Nova ở lần chạy sau ([vận hành](operations.vi.md)). Khác script cũ, chế độ apply không reset VM `ERROR` về `ACTIVE`.

## 11. Trạng thái monitor và mức Icinga

[domain/balance/classify.py](../src/compute_maintenance/domain/balance/classify.py) `classify_cluster_detail_status` và [domain/severity.py](../src/compute_maintenance/domain/severity.py) `map_detail_status_to_severity`. "Move hợp lệ" là kế hoạch trực tiếp hoặc MAKE_ROOM vượt được ngưỡng của nó.

| Chế độ / tình trạng | Move hợp lệ | Trạng thái | Mức | Exit |
|---|---|---|---|---|
| RAM hotspot hoặc critical | có | `RAM_HOTSPOT_OPTIMIZABLE` | WARNING | 1 |
| RAM hotspot hoặc critical | không | `RAM_HOTSPOT_NO_VALID_MOVE` | CRITICAL | 2 |
| CPU hotspot | có / không | `CPU_HOTSPOT_OPTIMIZABLE` / `CPU_HOTSPOT_NO_VALID_MOVE` | WARNING / CRITICAL | 1 / 2 |
| Pressure | có / không | `RAM_PRESSURE` hoặc `CPU_PRESSURE` / `PRESSURE_NO_VALID_MOVE` | WARNING / CRITICAL | 1 / 2 |
| MAD > 3 (CPU hoặc RAM) | có | `UNBALANCED` | WARNING | 1 |
| MAD > 3 | không, nhưng có move qua được bộ lọc | `UNBALANCED_NO_VALID_MOVE` | WARNING | 1 |
| MAD > 3 | không, không move nào qua được bộ lọc (`no_technical_candidate`) | `UNBALANCED_NO_VALID_MOVE` | CRITICAL | 2 |
| MAD ≤ 3 | có / không | `BALANCED_OPTIMIZABLE` / `BALANCED` | OK | 0 |
| Không hotspot, group soft-anti-affinity vỡ, có move heal khả thi (rc11) | move heal | `SOFT_AA_HEAL` | WARNING, hoặc mức của hàng tải nếu mức đó tệ hơn | 1 (2) |
| Group vỡ nhưng heal không làm được / bị tắt / bị hoãn vì hotspot | — | hàng tải giữ nguyên + ghi chú `Soft-anti-affinity:` | giữ nguyên | giữ nguyên |
| Lỗi dữ liệu, < 2 host, hết ngân sách tìm | — | `UNKNOWN` | UNKNOWN | 3 |

- WARNING của heal được gộp với kết quả riêng của nấc MAD (hiện ở dòng `Load status:`) theo cùng thứ tự CRITICAL > WARNING > UNKNOWN > OK như ý kế tiếp, nên không bao giờ che một CRITICAL. Ví dụ: chế độ pressure hết move hợp lệ (CRITICAL) cộng một move heal khả thi vẫn là CRITICAL; aggregate cân bằng (OK) có move heal khả thi thành WARNING.

- `--monitor-only` in mỗi aggregate một dòng và exit bằng kết quả nghiêm trọng nhất, xếp CRITICAL > WARNING > UNKNOWN > OK (không phải số lớn nhất). Lỗi dữ liệu ở một aggregate không bao giờ che cảnh báo ở aggregate khác: CRITICAL + UNKNOWN ra 2, WARNING + UNKNOWN ra 1, OK + UNKNOWN ra 3. Dòng riêng của từng aggregate vẫn ghi UNKNOWN của nó.
- Ví dụ: một host RAM 96% mà còn move hợp lệ thì chỉ WARNING; cũng host đó mà hết move hợp lệ thì CRITICAL.
- Khác script cũ: ở đó, move qua bộ lọc nhưng cải thiện bằng 0 hoặc âm vẫn bị tính là "no technical candidate" (CRITICAL). v1 tính nó là ứng viên dưới ngưỡng (WARNING).

## 12. Đọc một quyết định khi vận hành

Đọc output từ trên xuống:

| Dòng | Cho biết |
|---|---|
| Dòng đầu | Mức, trạng thái, cải thiện tốt nhất so với ngưỡng, host nóng nhất |
| `Mode:` / `Peak:` | Quy tắc nào đang chi phối và CPU/RAM cao nhất |
| `Spread:` / `Priority:` | Các MAD, weighted MAD và trọng số CPU/RAM đang dùng |
| `Placement:` | Số host cấp phát ≥ 90% hoặc hết chỗ an toàn |
| `Reason:` | Vì sao không nhận move nào (`no_technical_candidate`, `…_above_improvement_threshold`) |
| `Load status:` | Chỉ có khi đề xuất move heal: nấc MAD tự kết luận gì (rc11) |
| `Soft-anti-affinity:` | Các group vỡ và số vi phạm, move heal, hoặc cho từng group `unresolvable: <lý do>` (do cấu trúc) / `deferred: <lý do>` (tạm thời); `heal disabled` / `heal deferred` (rc11) |
| `Rejected candidate pairs:` | Các bộ lọc chặn nhiều nhất; host hay bị Placement chặn nhất được liệt kê ngay dưới |
| `Search scope:` | `HOTSPOT-FIRST`, `FALLBACK-FULL`, `FULL`, `MAKE_ROOM …`, `SOFT_AA_HEAL (group)` hoặc `DRAIN host` |
| `Step n:` | VM, nguồn → đích, flavor cần, tải thực tế, chỗ trống Placement trước bước đó, ảnh hưởng lên hai host |

Nhờ vậy phân biệt được "aggregate ổn", "chưa lý tưởng nhưng không có move an toàn hoặc đáng làm" và "có move sửa được và an toàn".

## 13. Hằng số

Tất cả nằm trong [config.py](../src/compute_maintenance/config.py). "Cấu hình được" nghĩa là đặt được trong `wbalancer.conf` hoặc biến môi trường.

| Tên | Giá trị | Cấu hình được |
|---|---|---|
| `RAM_HOTSPOT_THRESHOLD` / `RAM_CRITICAL_THRESHOLD` / `CPU_HOTSPOT_THRESHOLD` | 80 / 95 / 70 % | có |
| `PRESSURE_THRESHOLD` (trong code là `AUTO_PRESSURE_THRESHOLD`) | 80 % | có |
| `TARGET_AVG_MAD` | 3 | có |
| `MIN_IMPROVEMENT_THRESHOLD` | 0,15 | có |
| `PLACEMENT_PRESSURE_THRESHOLD` (chỉ để báo cáo) | 90 % | có |
| `MIN_CPU_WEIGHT` / `MIN_RAM_WEIGHT` | 0,2 / 0,2 | không |
| `PRESSURE_MAD_WEIGHT` / `PRESSURE_MEAN_WEIGHT` | 0,7 / 0,3 | không |
| `PLACEMENT_VCPU_HEADROOM_PERCENT` / `PLACEMENT_RAM_HEADROOM_PERCENT` | 5 / 5 % | có |
| `ENABLE_MAKE_ROOM`, `MAKE_ROOM_MAX_DEPTH` | true, 2 (chỉ cho phép 2) | có |
| `MAKE_ROOM_MIN_IMPROVEMENT` | 2 × `MIN_IMPROVEMENT_THRESHOLD` = 0,30 (phải ≥ giá trị đó) | có |
| `MAKE_ROOM_MAX_VM_CPU_PERCENT` / `MAKE_ROOM_MAX_VM_RAM_PERCENT` | 10 / 50 % | có |
| `PLACEMENT_SYNC_TIMEOUT` / `PLACEMENT_SYNC_INTERVAL` | 30 / 2 giây | có |
| `MAX_INSTANCES_PER_HOST_DEFAULT`, `MAX_INSTANCES_PER_HOST` | 0 (không giới hạn), map JSON | có |
| `PLANNER_CANDIDATE_BUDGET` | 100 000 | có |
| `HEAL_SOFT_ANTI_AFFINITY` | true | có |
| `SOFT_AA_HEAL_MAD_TOLERANCE` | 0,05 điểm MAD (≥ 0); cổng heal theo từng tài nguyên: `TARGET_AVG_MAD`, hoặc MAD trước + 0,05 nếu đã vượt ngưỡng | có |
| `SOFT_AA_TIEBREAK_MARGIN` | 0,15, tương đối (0–1) | có |
| Trần drain: CPU / RAM / RAM tuyệt đối | 70 / 80 / 95 % | theo `CPU_HOTSPOT_THRESHOLD` / `RAM_HOTSPOT_THRESHOLD` / `RAM_CRITICAL_THRESHOLD` |
| `DRAIN_METRIC_SETTLE_TIMEOUT` / chu kỳ | 360 / 15 giây | không |
| `VM_COOLDOWN_SECONDS` / `VM_FAILED_MOVE_COOLDOWN_SECONDS` | 14 400 / 14 400 giây | không |
| `VM_REJECTED_MOVE_COOLDOWN_SECONDS` | 1 800 giây | không |
| `MIGRATION_TIMEOUT` (cửa sổ không tiến độ) / `MIGRATION_MAX_WAIT` | 600 / 1 800 giây | có |
| `RUNTIME_SETTLE_SECONDS` | 360 giây | chỉ qua biến môi trường |
| `COMPENSATION_CPU_WINDOW_SECONDS` | 2 × chu kỳ scrape (dự phòng 60 giây) | có |

## Tài liệu liên quan

- [Hướng dẫn sử dụng](user-guide.vi.md): cài đặt, cấu hình, lệnh, exit code.
- [Vận hành](operations.vi.md): file trạng thái, lock, journal và recovery.
- [Chuyển từ `wloadbalancer.py`](migrating-from-wloadbalancer.vi.md): khác biệt về config và ngưỡng, các bước chuyển.
