# Kiến trúc

[English](architecture.md) | Tiếng Việt

Tài liệu này mô tả kiến trúc cuối của package `compute-maintenance` (hai lệnh `wbalancer` và `rolling-maint`) tính đến 1.0.0rc13. Viết cho developer và AI agent bảo trì hoặc mở rộng codebase: code đặt ở đâu, những bất biến không được phá, và cách thêm tính năng theo đúng lối. Thuật toán *quyết định* gì được đặc tả ở [algorithm.vi.md](algorithm.vi.md); tài liệu này nói về cách code được *tổ chức*.

## 1. Mục đích và tiền đề vận hành

Một host điều phối Linux chạy hai CLI tool lên một cloud OpenStack: `wbalancer` giữ các compute trong aggregate cân bằng và giữ các nhóm soft-anti-affinity tách host, mỗi lần chạy tối đa một live migration cho mỗi aggregate; `rolling-maint` đưa từng compute qua chu trình disable → drain → maintain → reboot → restore, mỗi lúc một host. Cả hai thiết kế cho cron + Icinga, không phải daemon.

Các ràng buộc có chủ đích, đừng "sửa" chúng:

- **Một coordinator duy nhất.** Mọi lock là `fcntl` cục bộ (`/var/lock/wbalancer-*.lock`). Không có và không cần điều phối phân tán.
- **Một move mỗi aggregate mỗi lần chạy.** Hội tụ trải theo nhịp cron; giới hạn bán kính ảnh hưởng và giữ mỗi run giải thích được.
- **Fail closed.** Dữ liệu đầu vào thiếu hoặc bất thường (series Prometheus, bản ghi Placement, đọc server group) làm hủy quyết định, không bao giờ âm thầm hạ chuẩn.
- **Tool không bao giờ làm policy xấu đi.** Policy cứng theo đúng ngữ nghĩa filter của Nova; policy mềm có gác cổng (move không được tăng điểm phạt của nhóm) và, từ rc11, được chủ động heal.

## 2. Phân tầng

```
cli/            parse tham số, wiring, exit code             (import mọi tầng dưới)
presentation/   văn bản báo cáo cho người đọc                (import application, domain)
application/    điều phối: run, round, state, journal        (import domain, adapters)
adapters/       toàn bộ I/O: OpenStack SDK/REST, Prometheus,
                file, lock, ssh, mail                        (import domain, config)
domain/         logic quyết định thuần, tuyệt đối không I/O  (chỉ import config)
config.py       ngưỡng, key, parse; paths.py: vị trí file
```

Các quy tắc giữ cấu trúc này khỏe:

1. `domain/` phải sạch import: không SDK, không requests, không file I/O, không đọc đồng hồ ngoài tham số truyền vào. Mọi thứ trong `domain/` unit-test được bằng dict thường.
2. `application/` sở hữu trình tự và state; không bao giờ tự nói HTTP — nó gọi `adapters/`.
3. `adapters/` không bao giờ quyết định; nó lấy dữ liệu, parse fail-closed, trả về cấu trúc trung lập.
4. Dữ liệu đi qua ranh giới tầng là dataclass: một aggregate đã đánh giá là `PlanContext` (`domain/plan_context.py`), đọc theo thuộc tính; code mới không được thêm state dạng dict khóa chuỗi (xem §9).
5. Chiều phụ thuộc một chiều, từ trên xuống theo danh sách trên. Nếu một import muốn đi ngược, thiết kế đang sai.

Bản đồ module (lớn trước): `application/migrations.py` thực thi plan, revalidation, theo dõi tiến độ, nối journal; `config.py` key, parse, `RuntimeSettings`; `application/snapshots.py` snapshot cụm, bù tải VM mới đến, dựng `PlanContext`; `adapters/openstack.py`; `presentation/reports.py`; `application/balancing.py` vòng balance/drain; `application/stages.py` hành động các stage rolling-maint; `domain/drain.py` xếp chỗ drain fail-closed; `adapters/prometheus.py`; `application/recovery.py` reconcile journal và settle; `application/maintenance.py` runner rolling-maint; package `domain/balance/` (rc12 tách từ `balance.py` cũ, mọi tên vẫn được re-export): `makeroom.py` MAKE_ROOM, `filters.py` điều kiện VM ứng viên + chuỗi lọc đích + chặn hotspot, `heal.py` heal soft-anti-affinity, `scoring.py` tìm move trực tiếp và tie-break, `modes.py` thang mode, baseline, ngưỡng, `classify.py` detail status và reason; `domain/plan_context.py` `PlanContext` / `PlanLabels`; `domain/severity.py` mapping severity và exit code (nguồn duy nhất).

### 2.1 Muốn sửa gì thì mở file nào

| Bạn muốn sửa… | Mở |
|---|---|
| Một ngưỡng, timeout hay config key | `config.py` (nhóm key, field `RuntimeSettings` nếu là giá trị sống; + `wbalancer.conf.example`, user-guide 4.2) |
| Thang mode, baseline, ngưỡng cải thiện | `domain/balance/modes.py` |
| Điều kiện VM ứng viên, chuỗi lọc đích, chặn hotspot | `domain/balance/filters.py` |
| Chấm điểm move trực tiếp, tie-break | `domain/balance/scoring.py` |
| Kế hoạch MAKE_ROOM hai bước | `domain/balance/makeroom.py` |
| Cách heal chọn và cổng MAD của nó | `domain/balance/heal.py` |
| Detail status và reason của một snapshot | `domain/balance/classify.py` |
| Severity, exit code Icinga, gộp exit code | `domain/severity.py` |
| Những gì một aggregate đã đánh giá mang qua các tầng | `domain/plan_context.py` (dựng trong `application/snapshots.py`) |
| Luật server group / AZ / vừa-sức-chứa | `domain/policies.py` |
| Xếp chỗ drain (thứ tự, chọn đích) | `domain/drain.py` |
| Thực thi migration, revalidation, timeout tiến độ | `application/migrations.py` |
| Nội dung snapshot, kiểm tra metric hợp lý, bù tải VM đến | `application/snapshots.py` (parse: `adapters/prometheus.py`) |
| Luồng vòng balance/drain, exit code mức run | `application/balancing.py` |
| Output `--monitor-only` | `application/monitoring.py` (luật gộp: `domain/severity.py`) |
| Reconcile journal, settle, xử lý UNKNOWN | `application/recovery.py` |
| Hành vi một stage rolling-maint | `application/stages.py` (trình tự/checkpoint: `application/maintenance.py`) |
| Một lời gọi API OpenStack | `adapters/openstack.py` |
| Lệnh ssh/scp tới compute | `adapters/commands.py` |
| Câu chữ báo cáo cho người đọc | `presentation/reports.py` |
| Field trong event JSONL | nơi phát event trong `application/balancing.py` / `migrations.py` |
| Một flag CLI | `cli/balance.py` hoặc `cli/maintain.py` |

## 3. Giải phẫu một lần chạy balance

`cli/balance.py` nối một run: lock coordinator/balancer (chỉ apply mode; `--dry-run` và `--monitor-only` không lấy lock), journal chỉ ở apply mode, rồi theo từng aggregate:

1. **Snapshot** (`application/snapshots.py`): server + hypervisor từ Nova, inventory/usage Placement qua REST thô, series node và libvirt từ Prometheus. Dựng `PlanContext`: `collected_at`, `exclude_unready_hosts`, CPU/RAM% từng host, MAD baseline theo từng tài nguyên, plan và kết luận. Thiếu metric của host nào ⇒ fail closed cho aggregate đó (một `PlanContext` chỉ có `error`).
2. **Thang mode** (`domain/balance/modes.py`, bậc heal `heal.py`, hằng `MODE_*` trong `config.py`): `ram_critical` (>95%) → `ram_hotspot` (>80%) → `cpu_hotspot` (>70%) → `soft_aa_heal` (nhóm soft-anti-affinity vỡ, rc11) → cân `mad` (MAD từng tài nguyên > `TARGET_AVG_MAD`), cộng `pressure` khi trung bình cụm cao. Heal không bao giờ chặn bậc dưới: hết nước heal ⇒ rơi xuống MAD.
3. **Plan**: một move `DIRECT`, một plan hai bước `MAKE_ROOM`, hoặc một move `SOFT_AA_HEAL`. Mọi ứng viên qua cùng một chuỗi lọc: sức chứa Placement kèm headroom, không tạo hotspot mới, availability zone, policy server group, cooldown từng VM, `max_server_per_host`.
4. **Revalidate** (`application/migrations.py`): snapshot mới với *đúng* cờ `exclude_unready_hosts` của plan; snapshot quá 120 s ⇒ từ chối, run sau plan lại; move heal qua lại cổng MAD từng tài nguyên.
5. **Thực thi**: `journal.begin()` được ghi bền **trước khi** submit `live_migrate_server` — thứ tự này là bất biến, có test riêng canh. Tiến độ theo dõi bằng cửa sổ gộp: dừng sau `MIGRATION_TIMEOUT` (600 s) không tiến triển, trần cứng `MIGRATION_MAX_WAIT` (1800 s).
6. **Settle**: balance chờ barrier 360 s; drain thì bù tải VM mới đến bằng mẫu RAM thô đo thật (fallback RSS, rồi mô hình suy giảm).
7. **Ghi nhận**: kết quả vào journal, một event JSONL vào `/var/log/wbalancer/migrations.jsonl` (nhãn của snapshot mà kế hoạch được lập từ đó; `mode`/`plan_type` nhận diện move heal; nhóm chưa heal được phát `soft_aa_heal_unresolvable` (do cấu trúc) hoặc `soft_aa_heal_deferred` (tạm thời)), cooldown cho VM vừa move, exit code Nagios.

Run bị ngắt giữa chừng được reconcile ở run kế từ journal ghi-trước (`application/recovery.py`): `/os-migrations` theo instance cộng sàn `global_max_id` phân loại từng mục treo thành success / failed / not_started / vm_deleted / in_progress; UNKNOWN chặn submit mới và không bao giờ tự retry.

## 4. Bất biến (tuyệt đối không phá)

1. **Journal trước submit.** Không live migration nào được yêu cầu trước khi mục journal của nó đã bền trên đĩa.
2. **Fail closed** với mọi metric, bản ghi Placement, hay lần đọc server group bị thiếu/bất thường.
3. **Ghi file atomic**: mọi file state ghi theo `mkstemp` → `fsync` → `os.replace` → fsync thư mục (`adapters/persistence.py`). Không bao giờ ghi state kiểu khác.
4. **Cổng độ tươi**: không submit từ snapshot quá 120 s. Một lần đọc chỉ được dùng lại bên trong một lượt snapshot hoặc một lượt revalidate (rc13: một map hypervisor, một lần đọc VM của step), không bao giờ qua settle barrier hay qua một lần refresh; cluster cache chỉ được prefetch cho caller thật sự đọc nó (`snapshots.prefetched_cluster_cache_is_used`). Tên project Keystone (chỉ để hiển thị) được resolve một lần mỗi run.
5. **Gác cổng**: không move balance/make-room/heal nào được tăng điểm phạt của bất kỳ nhóm mềm nào; policy cứng theo ngữ nghĩa filter Nova (xét theo host đích, `domain/policies.py`).
6. **Cổng MAD của heal** (riêng từng tài nguyên, CPU và RAM): MAD sau move ≤ `TARGET_AVG_MAD` nếu baseline đang ở mức đó trở xuống, ngược lại baseline + `SOFT_AA_HEAL_MAD_TOLERANCE`. Heal không bao giờ tự làm một aggregate mới mất cân bằng.
7. **Exit code là hợp đồng**: monitor 0/1/2/3 (Nagios) gộp bằng `worst_exit_code` xếp CRITICAL > WARNING > UNKNOWN > OK, tất cả định nghĩa một lần trong `domain/severity.py`; drain 0/1/2. Đổi chúng là phá Icinga của operator.
8. **Một move mỗi aggregate mỗi run**; cooldown từng VM áp cho mọi move đã chạy hoặc bị từ chối.
9. **Check AZ bảo thủ có chủ đích**: chặn mọi move xuyên AZ, kể cả VM không pin AZ.
10. **Identity resolve một lần duy nhất**: chọn cloud theo `--os-cloud` > env `OS_CLOUD` > `OS_CLOUD` trong config file > `openstack`, chỉ qua `config.resolve_os_cloud()`. Không bao giờ tự resolve chỗ khác; drain child luôn nhận `--os-cloud` tường minh.

## 5. State trên đĩa (`paths.py` là nguồn sự thật duy nhất)

| Đường dẫn | Nội dung |
|---|---|
| `/etc/wbalancer/wbalancer.conf` | config của operator (`KEY=value`); biến môi trường `WLB_CONFIG_PATH` thay mặc định này |
| `/var/lib/wbalancer/vm-cooldown.json` | cooldown từng VM |
| `/var/lib/wbalancer/operations/migration-journal.json` | journal migration ghi-trước |
| `/var/lib/wbalancer/maintenance/` | checkpoint rolling-maint |
| `/var/log/wbalancer/migrations.jsonl` | mỗi quyết định/migration một event JSON |
| `/var/log/wbalancer/maintenance/` | log và kết quả rolling-maint |
| `/var/lock/wbalancer-balancer.lock`, `-maint.lock` | lock coordinator cục bộ |

Thư mục cũ `/var/lib/wbalancer-v1` còn state trong khi thư mục mới trống thì apply bị từ chối (`legacy_state_message`) — để việc chuyển state là hành động có ý thức.

## 6. Hệ thống cấu hình

Key nằm trong `config.py` theo ba nhóm: `THRESHOLD_KEYS` (`RAM_HOTSPOT_THRESHOLD`, `RAM_CRITICAL_THRESHOLD`, `CPU_HOTSPOT_THRESHOLD`, `PRESSURE_THRESHOLD`, `TARGET_AVG_MAD`, `MIN_IMPROVEMENT_THRESHOLD`, `PLACEMENT_PRESSURE_THRESHOLD`), `SOFT_AA_HEAL_KEYS` (`HEAL_SOFT_ANTI_AFFINITY`, `SOFT_AA_HEAL_MAD_TOLERANCE`, `SOFT_AA_TIEBREAK_MARGIN`), `TIMEOUT_KEYS` (`MIGRATION_TIMEOUT`, `MIGRATION_MAX_WAIT`). Giá trị đến từ config file và/hoặc biến môi trường, được validate, rồi CLI áp cho cả tiến trình qua `config.apply_thresholds` / `apply_soft_aa_heal` / `apply_migration_timeouts` — các hàm này thay `RuntimeSettings` đang sống.

**Tầng chuẩn (rc12)**: giá trị sống là một object bất biến `RuntimeSettings`; code đọc nó tại thời điểm gọi, `_config.runtime().ram_hotspot_threshold`. Các hằng viết hoa (`RAM_HOTSPOT_THRESHOLD` …, xem `RUNTIME_CONSTANTS`) chỉ còn là lớp tương thích: mỗi `apply_*` ghi lại chúng, và gán một hằng (code cũ, test qua `monkeypatch`) được chuyển vào `runtime()`, nên hai bên không bao giờ lệch nhau. Cái bẫy còn lại: `from compute_maintenance.config import X` chép giá trị lúc import và âm thầm bỏ qua config của operator. Không bao giờ import giá trị config theo tên (`tests/test_rc12_refactor.py` kiểm điều này cho `src/`), và không thêm chỗ đọc mới từ các hằng.

`wbalancer.conf.example` phải liệt kê đủ mọi key thật và không thừa key nào — có test ép nó parse và validate được.

## 7. Máy trạng thái rolling-maint

`application/maintenance.py` chạy từng host qua `application/stages.py`: `preflight → pre_drain → disable → drain → verify → baseline → maintenance → reboot → health → post_boot → restore → post_restore`, với checkpoint bền (chạy lại đúng lệnh cũ là resume), kiểm định danh checkpoint, và quarantine (host để nguyên disabled) khi post-restore fail. Drain là tiến trình con `wbalancer --drain-host` thừa kế FD lock coordinator (`pass_fds`, kiểm bằng dev/ino) và luôn nhận `--os-cloud`. Chính sách host-key SSH theo đúng cấu hình ssh của OS trên coordinator — tool không truyền option `StrictHostKeyChecking` nào (rc10 đã bỏ flag).

## 8. Chuẩn kiểm thử ("xong" nghĩa là gì ở repo này)

- Bộ test offline và kín từ thiết kế (không cloud, không mạng, không bao giờ đọc file `/etc/wbalancer/wbalancer.conf` chỉ root đọc được của máy: `conftest.py` ở gốc trỏ `WLB_CONFIG_PATH` về một đường dẫn tạm) và phải luôn xanh: trên Linux chạy đủ; trên Windows các test POSIX-lock và apt skip. Kiểm trước release phải chạy suite trên Linux ít nhất một lần.
- Một fix hành vi đi kèm test fail trên hành vi cũ (thực hành rc10/rc11: mutation check — tạm đảo ngược chỗ sửa, xác nhận có test fail, khôi phục giống từng byte).
- Refactor rủi ro được kiểm bằng lấy mẫu vi phân so với bản trước (sinh N kịch bản ngẫu nhiên, so quyết định; rc11 dùng 1500 với 0 khác biệt ngoài ý muốn, rc12 dùng 1500 cho mỗi cấu hình cộng text báo cáo và event giống từng byte). `ruff check src tests` giữ ở 0.
- Docs đi theo cặp EN/VI cùng cấu trúc; thay đổi hành vi CLI phải cập nhật `user-guide`, `operations`, `algorithm` và `wbalancer.conf.example` trong cùng thay đổi.
- Ngữ nghĩa exit code/Icinga và thứ tự thang mode đổi là phải có test tường minh.

## 9. Cách mở rộng (làm theo các công thức này)

**Thêm một config key**: hằng số + mục trong đúng nhóm `*_KEYS` ở `config.py`, parse + validate + nối vào hàm apply (giá trị sống cho cả tiến trình thì thêm field `RuntimeSettings`, đọc qua `runtime()`), chú thích trong `wbalancer.conf.example`, thêm hàng ở `user-guide` 4.2 (EN+VI), test cho default/parse/validate/apply.

**Thêm một mục tiêu cân bằng**: viết logic thuần trong `domain/` (module mới trong `domain/balance/`, re-export qua `__init__` của nó; giữ mỗi module dưới ~800 dòng thật xa), chèn bậc của nó vào thang một cách tường minh kèm lý do vị trí được ghi lại, tái dùng chuỗi lọc đích sẵn có (`filters.check_direct_destination`) thay vì viết check mới, phát `mode` riêng trong event, map severity của nó trong `domain/severity.py`, và viết vào `algorithm.md` kèm ví dụ số. Hành vi mới bật mặc định phải có key tắt.

**Thêm một adapter**: module mới trong `adapters/`, parse fail-closed, không quyết định, retry qua helper sẵn có; inject từ `application/`, không bao giờ import từ `domain/`.

**Nợ còn lại sau rc12 (đừng nuôi nó)**: bên trong `PlanContext`, các kết quả lồng `soft_aa_heal`, `load_status`, `placement_summary` và kết quả của `compute_baseline_metrics` vẫn là dict khóa chuỗi; các hằng phản chiếu `RuntimeSettings` và cách truy cập kiểu dict của `PlanContext` chỉ để tương thích (code mới dùng thuộc tính và `runtime()`). rc12 đã bỏ state dict không type, mutation global của `config.py`, file `balance.py` 1 300 dòng và mapping severity trùng lặp.

## 10. Lịch sử tham chiếu

Các báo cáo review độc lập và fix (rc10: M1–M4; rc11: thiết kế heal, quá trình tiến hóa công thức cổng, canary lab) nằm ngoài repo trong workspace của operator; kết quả chắt lọc ở [CHANGELOG.md](../CHANGELOG.md). Tiền lệ thiết kế của heal là cơ chế sửa vi phạm ràng buộc của VMware DRS; ngữ nghĩa gác cổng phản chiếu scheduler filter của Nova.
