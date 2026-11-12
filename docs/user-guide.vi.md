# Hướng dẫn sử dụng

[English](user-guide.md) | Tiếng Việt

Dành cho người vận hành OpenStack lần đầu dùng tool. Đọc theo thứ tự: mục 1–4 để cài và cấu hình, mục 5 là lộ trình chạy an toàn, các mục sau dùng để tra cứu. Đang dùng script cũ `wloadbalancer.py` thì đọc thêm [hướng dẫn chuyển đổi](migrating-from-wloadbalancer.vi.md).

## 1. Tool làm gì

Có hai lệnh, cài chung một package (`compute-maintenance`):

| Lệnh | Việc | Ví dụ |
|---|---|---|
| `wbalancer` | **Cân bằng** CPU/RAM giữa các compute trong một aggregate bằng live migration | 3 host RAM 40/50/60% → chuyển 1 VM từ host 60% sang host 40% |
| `wbalancer --drain-host X` | **Chuyển hết VM khỏi host X** và rải đều sang các host còn lại | Host X có 19 VM → chia ra 2 host kia sao cho tải vẫn cân |
| `wbalancer --monitor-only` | **Chỉ đánh giá**, trả exit code theo chuẩn Icinga/Nagios | Dùng làm check cảnh báo |
| `rolling-maint` | **Bảo trì cuốn chiếu** từng host một: disable → drain → bảo trì → reboot → kiểm health → enable | Reboot cả 6 compute lần lượt, không VM nào bị tắt |

Tool chỉ dùng live migration, không tắt hay rebuild VM. Mọi lệnh đều có `--dry-run`.

## 2. Hệ thống của bạn có dùng được không

Kiểm hết bảng này trước khi cài. Thiếu mục nào thì tool sẽ báo lỗi cấu hình hoặc từ chối chạy.

| Yêu cầu | Cách kiểm |
|---|---|
| OpenStack có Nova + Placement, tài khoản **admin** | `openstack hypervisor list` và `openstack resource provider list` chạy được |
| Live migration chạy được giữa các compute, **storage dùng chung** (ví dụ Ceph RBD). Tool gọi migration với `block_migration=False` | `openstack server migrate --live-migration <vm>` thành công bằng tay |
| Compute nằm trong **host aggregate**, mỗi aggregate ít nhất 2 host | `openstack aggregate show <agg>` |
| Prometheus có **node_exporter** trên mỗi compute, label `alias` = tên host Nova | Query A bên dưới |
| Prometheus có **libvirt exporter** với các metric `libvirt_domain_*`, label `host` = tên host Nova và `instanceId` = UUID VM (ví dụ [prometheus-libvirt-exporter](https://github.com/zhangjianweibj/prometheus-libvirt-exporter), exporter mà bản gốc khuyến nghị) | Query B, C bên dưới |
| Một máy **coordinator** Linux, Python 3.12, tới được API OpenStack và Prometheus | `python3.12 --version` |
| Riêng `rolling-maint`: SSH được vào compute (mặc định `root`), host key của compute đã được cấu hình SSH của coordinator tin cậy (xem mục 7.2); profile package/kernel dành cho Ubuntu/Debian; container theo kiểu Kolla (`docker` hoặc `podman`) | `ssh -o BatchMode=yes root@<compute> uptime` |

Kiểm metric (thay `node`, `libvirt` bằng tên job thật của bạn):

```promql
# A. Mỗi compute 1 dòng, alias phải trùng cột Host của "openstack compute service list --service nova-compute"
count by (alias) (node_memory_MemTotal_bytes{job="node"})

# B. Mỗi compute đang có VM 1 dòng, giá trị = số VM, label host trùng tên host Nova
#    (host trống không có dòng nào là bình thường)
count by (host) (libvirt_domain_info_virtual_cpus{job="libvirt"})

# C. Bằng số VM đang chạy (mỗi series mang instanceId = UUID VM)
count(libvirt_domain_info_cpu_time_seconds_total{job="libvirt", instanceId!=""})
```

Tool cần thêm `node_cpu_seconds_total`, `node_memory_MemAvailable_bytes`, `libvirt_domain_stat_memory_usable_bytes`, `libvirt_domain_stat_memory_available_bytes` và `libvirt_domain_stat_memory_rss_bytes`.

Nếu thiếu label `alias` hoặc `host`, gắn nó ngay trong scrape config:

```yaml
- job_name: node
  static_configs:
    - targets: ["10.0.0.11:9100"]
      labels: {alias: compute-01}   # đúng tên host Nova
```

## 3. Cài đặt

Trên coordinator. Không cài chung venv với bản legacy.

**Tool không tự cài `openstacksdk` và `python-openstackclient`.** Bạn tự cài hai gói này cho hợp với cloud của mình. Chuẩn đã kiểm định là **OpenStack 2026.1**: `openstacksdk` 4.10.0 và `python-openstackclient` 9.0.1, theo upper-constraints chính thức. Repo cũng kèm bản offline của file đó ở `constraints/openstack-2026.1.txt` (dùng khi regenerate lock file).

```bash
git clone https://github.com/nguyenhuukhoi/OpenstackWBalancer.git && cd OpenstackWBalancer
python3 -m venv /opt/wbalancer/venv            # python3 phải là 3.12 (requires-python >=3.12,<3.13)
VENV=/opt/wbalancer/venv

# 1. Tool và các thư viện riêng của nó (đã khóa hash, cùng phiên bản với chuẩn 2026.1)
$VENV/bin/pip install --require-hashes -r requirements.lock.txt
$VENV/bin/pip install --no-deps .

# 2. Client OpenStack: bạn tự chọn và tự cài, cài vào CÙNG venv
$VENV/bin/pip install -c https://releases.openstack.org/constraints/upper/2026.1 openstacksdk python-openstackclient

$VENV/bin/pip check
$VENV/bin/wbalancer --help && $VENV/bin/rolling-maint --help
```

Profile có sẵn được cài vào `/opt/wbalancer/venv/share/compute-maintenance/profiles/`.

Cài theo chuẩn khác (ví dụ cloud 2025.2) thì thay file constraints bằng `https://releases.openstack.org/constraints/upper/2025.2`. Bản cũ hơn chuẩn 2026.1 vẫn chạy được nhưng tool sẽ in cảnh báo "older than the OpenStack 2026.1 baseline … untested"; hãy chạy lại lộ trình ở mục 5 trước khi dùng thật.

Khi khởi động, tool kiểm tra:

| Tình huống | Kết quả |
|---|---|
| Chưa cài `openstacksdk` | Dừng ngay, in lệnh cài; exit 1, riêng `--monitor-only` exit 3 (UNKNOWN) |
| `rolling-maint` không tìm thấy lệnh `openstack` | Dừng trước khi đụng host, exit 1 |
| Phiên bản cũ hơn chuẩn 2026.1 | Chạy tiếp, in cảnh báo |

`rolling-maint` dùng lệnh `openstack` **nằm trong cùng venv với tool** trước, chỉ khi không có mới tìm trong `PATH`. Vì vậy chạy qua `sudo`, cron hay systemd không bị lẫn sang bản hệ thống (ví dụ `/usr/bin/openstack` 6.6.0). Log mỗi lượt ghi rõ đường dẫn và phiên bản CLI đã dùng.

Muốn chạy test trước khi cài (không nối cloud): `uv sync --extra test --locked && uv run --extra test python -m pytest -q`. Nhóm `dev` của uv cài sẵn hai client theo chuẩn 2026.1, chỉ dùng cho test.

## 4. Cấu hình

### 4.1. Credential OpenStack

Mặc định tool kết nối tới cloud tên `openstack`. Muốn dùng entry khác trong `clouds.yaml` thì đặt `OS_CLOUD=<tên>` trong file config hoặc biến môi trường, hoặc truyền `--os-cloud <tên>` cho `wbalancer` / `rolling-maint`. Hai lệnh chọn cloud cùng một cách: `--os-cloud` > biến môi trường `OS_CLOUD` > `OS_CLOUD` trong file config (`--config`, với `rolling-maint` là `--wlb-config`) > `openstack`. `rolling-maint` luôn truyền cloud đã chọn cho bước drain qua `--os-cloud`, kể cả khi dùng child tùy biến `--wbalancer`. Khai báo cloud trong `/etc/openstack/clouds.yaml` hoặc `~/.config/openstack/clouds.yaml`, quyền `600`:

```yaml
clouds:
  openstack:
    auth:
      auth_url: https://keystone.example:5000/v3
      application_credential_id: "..."
      application_credential_secret: "..."
    auth_type: v3applicationcredential
    region_name: RegionOne
```

Credential phải có quyền admin: tool đọc Placement, `os-migrations`, chi tiết availability zone, và bật/tắt nova-compute.

Dùng file openrc thay cho `clouds.yaml`: source file đó rồi dùng `--os-cloud envvars`. Đó là tên openstacksdk đặt cho cloud dựng từ các biến `OS_*`; đổi tên này bằng `OS_CLOUD_NAME`.

### 4.2. File cấu hình `/etc/wbalancer/wbalancer.conf`

Định dạng `KEY=VALUE`. Biến môi trường cùng tên ghi đè giá trị trong file. Nên bắt đầu từ [`wbalancer.conf.example`](../wbalancer.conf.example): file này liệt kê mọi khóa kèm giá trị mặc định và giải thích ngắn.

```bash
install -m 600 wbalancer.conf.example /etc/wbalancer/wbalancer.conf
```

Tối thiểu cần 3 dòng:

```ini
PROMETHEUS_QUERY_URL=http://prometheus.example:9090/api/v1/query
PROMETHEUS_NODE_JOB=node
PROMETHEUS_LIBVIRT_JOB=libvirt
```

Các khóa tùy chọn hay dùng:

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `EXCLUDE_VMS_FILE` | trống | File liệt kê VM không bao giờ bị balance |
| `MAX_INSTANCES_PER_HOST_DEFAULT` | `0` (không giới hạn) | Số VM tối đa trên mỗi host |
| `MAX_INSTANCES_PER_HOST` | trống | JSON riêng cho từng host, ví dụ `{"compute-01": 40}` |
| `PLACEMENT_VCPU_HEADROOM_PERCENT` / `PLACEMENT_RAM_HEADROOM_PERCENT` | `5` / `5` | Chừa lại 5% capacity Placement, không xếp kín |
| `ENABLE_MAKE_ROOM` | `true` | Cho phép kế hoạch 2 bước: dời một VM nhỏ đi để lấy chỗ cho VM lớn |
| `OS_CLOUD` | `openstack` | Tên cloud trong `clouds.yaml`; biến môi trường `OS_CLOUD` ghi đè giá trị trong file, `--os-cloud` ghi đè cả hai |
| `RAM_HOTSPOT_THRESHOLD` / `RAM_CRITICAL_THRESHOLD` | `80` / `95` | Host có RAM vượt mức này (%) là hotspot / hotspot critical |
| `CPU_HOTSPOT_THRESHOLD` | `70` | Host có CPU vượt mức này (%) là hotspot |
| `PRESSURE_THRESHOLD` | `80` | Trung bình aggregate vượt mức này (%) là pressure; chỉ có tác dụng khi thấp hơn một ngưỡng hotspot |
| `TARGET_AVG_MAD` | `3` | Độ lệch (MAD, điểm %) vượt mức này thì aggregate bị coi là lệch |
| `MIN_IMPROVEMENT_THRESHOLD` | `0.15` | Mức cải thiện tối thiểu của move trực tiếp; `MAKE_ROOM_MIN_IMPROVEMENT` mặc định gấp đôi giá trị này |
| `PLACEMENT_PRESSURE_THRESHOLD` | `90` | Mức cấp phát Placement (%) mà từ đó host được báo là gần đầy (chỉ để báo cáo) |
| `MIGRATION_TIMEOUT` / `MIGRATION_MAX_WAIT` | `600` / `1800` | Số giây: ngừng chờ live migration khi không có tiến độ chừng này / tổng thời gian chờ không quá chừng này (vận hành, mục 4) |
| `HEAL_SOFT_ANTI_AFFINITY` | `true` | Tách lại server group soft-anti-affinity có từ 2 thành viên trên cùng một host, mỗi lần chạy một VM, khi không có host hotspot (rc11); đồng thời bật các tie-break soft-anti-affinity. `false` = đúng hành vi rc10, group vỡ chỉ được báo cáo |
| `SOFT_AA_HEAL_MAD_TOLERANCE` | `0.05` | Cổng heal tính bằng điểm MAD (tuyệt đối, ≥ 0), xét riêng CPU và RAM: MAD đang không vượt `TARGET_AVG_MAD` thì sau heal vẫn phải không vượt; MAD đã vượt thì chỉ được tăng tối đa giá trị này. Ví dụ: aggregate cân bằng → heal kết thúc ở MAD CPU 1,67 / MAD RAM 1,56 thì được (cả hai ≤ 3); CPU 0,5 / RAM 3,6 thì bị từ chối (RAM > 3); MAD RAM trước 2,98 → tối đa 3,00; MAD RAM trước 3,50 → heal kết thúc ở 3,51 thì được, ở 3,60 thì không |
| `SOFT_AA_TIEBREAK_MARGIN` | `0.15` | Khoảng tương đối (0–1): trong các move cân bằng cách mức cải thiện tốt nhất không quá 15%, move heal được một group soft-anti-affinity được chọn. Ví dụ: tốt nhất 2,00, một move heal 1,80 (thấp hơn 10%) được chọn |
| `ALERT_EMAIL_TO`, `ALERT_EMAIL_FROM`, `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_STARTTLS` | tắt | Gửi email cảnh báo. Có mật khẩu thì `chmod 600` file này |
| `PROMETHEUS_CPU_USED`, `PROMETHEUS_MEM_USED`, … | dựng từ tên job | Thay query host nếu label của bạn khác |

Các ngưỡng được kiểm lúc khởi động (trong khoảng 0–100, RAM critical phải cao hơn RAM hotspot); với `--verbose`, mỗi lượt chạy ghi log giá trị đang dùng. Muốn giữ hành vi của `wloadbalancer.py` cũ thì đặt `RAM_HOTSPOT_THRESHOLD=75`, `RAM_CRITICAL_THRESHOLD=85`, `CPU_HOTSPOT_THRESHOLD=85`, `MIN_IMPROVEMENT_THRESHOLD=0.05`.

`RUNTIME_SETTLE_SECONDS` (mặc định 360) là **biến môi trường**, không đặt trong file: sau mỗi migration, balance chờ đủ chừng ấy giây để metric phản ánh tải mới.

### 4.3. Danh sách VM loại trừ (tùy chọn)

Mỗi dòng là UUID, tên chính xác hoặc glob tên VM; dòng bắt đầu bằng `#` là chú thích.

```text
# DB cluster, không tự động di chuyển
3f2a9c1e-0b7d-4e7e-9a51-2c3d4e5f6a7b
prod-db-*
```

## 5. Lộ trình chạy lần đầu

Đi lần lượt; chỉ sang bước sau khi bước trước cho kết quả đúng như bạn mong đợi.

| Bước | Lệnh | Cần thấy | Rủi ro |
|---|---|---|---|
| 1. Đọc trạng thái | `wbalancer --monitor-only -a AGG; echo $?` | Exit 0–2, không có 3 (UNKNOWN = thiếu dữ liệu) | Không có |
| 2. Xem kế hoạch balance | `wbalancer --dry-run -a AGG` | Kế hoạch đề xuất VM, nguồn → đích, mức cải thiện | Không có |
| 3. Balance có người xác nhận | `wbalancer -a AGG` → trả lời `y` | 1–2 migration rồi dừng | Migrate VM thật |
| 4. Xem kế hoạch drain | `wbalancer --dry-run -a AGG --drain-host HOST` | Mọi VM có đích, hoặc danh sách `unmovable` kèm lý do | Không có |
| 5. Drain 1 host ít VM | `wbalancer -a AGG --drain-host HOST` | Exit 0, host trống và còn ở trạng thái disabled | Migrate thật, disable host |
| 6. Thử rolling-maint không đụng host | Profile `dummy` (mục 7) | `SKIPPED`, `Overall OK` | Không có |
| 7. Canary rolling-maint | Profile `drain-reboot` trên 1 host | Host qua đủ các stage, được enable lại | Reboot host thật |
| 8. Bật cron balance | Mục 6.1 | Mỗi lần chạy 0–2 migration | Tự động |

Nên chạy bước 1–2 hằng ngày trong khoảng một tuần trước khi sang bước 3, để thấy planner đề xuất có hợp lý không.

## 6. Cân bằng và drain (`wbalancer`)

### 6.1. Balance

```bash
wbalancer --dry-run -a AGG                 # chỉ in kế hoạch
wbalancer -a AGG                           # hỏi [y/N] trước khi migrate
wbalancer -y -a AGG --send-error           # tự duyệt, email khi lỗi (dùng cho cron)
```

Mỗi lần chạy, mỗi aggregate thực hiện **tối đa 1 kế hoạch**: 1 migration trực tiếp, 2 migration kiểu MAKE_ROOM, hoặc 1 migration heal soft-anti-affinity. VM vừa bị chuyển sẽ không bị chuyển lại trong 4 giờ.

Ví dụ cron chạy 10 phút/lần, đặt trong `/etc/cron.d/wbalancer`:

```cron
*/10 * * * * root /opt/wbalancer/venv/bin/wbalancer -y --send-error >> /var/log/wbalancer/cron.log 2>&1
```

Tối đa 6 lần/giờ × 2 migration = 12 migration/giờ cho mỗi aggregate; thực tế ít hơn nhiều vì chỉ chạy khi có cải thiện đủ lớn. Hai lần chạy không bao giờ chồng nhau vì có lock. Lần chạy rơi vào lúc `rolling-maint` đang chạy sẽ tự bỏ qua, in `Not running: …` và exit 1; đó là hành vi đúng.

Khi nào tool coi là cần cân bằng:

| Trạng thái | Điều kiện | Ví dụ |
|---|---|---|
| Lệch (MAD) | Độ lệch tuyệt đối trung bình của CPU% hoặc RAM% giữa các host > 3 điểm | RAM 40/50/60% → MAD 6,7 → cần cân bằng |
| Pressure | Trung bình CPU hoặc RAM của aggregate > 80% | Với ngưỡng mặc định, trạng thái này không bao giờ tự xảy ra: trung bình > 80% thì đã có host > 80%, và host đó được xử lý như hotspot trước. Đặt `PRESSURE_THRESHOLD` thấp hơn một ngưỡng hotspot thì mới dùng được ([chi tiết](algorithm.vi.md#3-chọn-chế-độ)) |
| Hotspot | Một host RAM > 80% (> 95% là critical) hoặc CPU > 70% | Ưu tiên dời VM khỏi host đó |
| Group soft-anti-affinity bị vỡ (rc11) | Không có hotspot, và từ 2 thành viên của một server group `soft-anti-affinity` nằm chung host (ví dụ sau drain) | db1 + db2 trên A (RAM 70%), B 40%, C 45% → db2 (VM nhỏ hơn) chuyển sang B; mỗi lần chạy một VM, nên 3 vi phạm mất khoảng 30 phút với cron 10 phút ([chi tiết](algorithm.vi.md#8-heal-soft-anti-affinity-rc11)) |

Các con số trong bảng là mặc định, đổi được trong file config (mục 4.2). Một move chỉ được chọn khi cải thiện đủ ngưỡng, không làm host đích vượt hotspot, và tuân thủ Placement, server group, availability zone.

### 6.2. Drain một host

```bash
wbalancer --dry-run -a AGG --drain-host compute-03   # xem kế hoạch
wbalancer -a AGG --drain-host compute-03             # thực hiện, hỏi xác nhận
```

- Tool **disable nova-compute** của host trước khi drain, để scheduler không đặt VM mới lên đó. Không muốn disable thì thêm `--drain-no-disable`.
- **Fail-closed:** nếu không lập được kế hoạch cho toàn bộ VM **live-migrate được** thì không migrate VM nào, in lý do và exit 2. VM hoàn toàn không live-migrate được (bị lock, SHUTOFF, `unsupported_resources`…, xem bảng dưới) không chặn drain: chúng được báo cáo và để lại trên host, các VM khác vẫn được migrate, và drain vẫn kết thúc bằng exit 2 vì host chưa trống.
- Drain bỏ qua cooldown và danh sách loại trừ đối với VM trên chính host đó.
- Tốc độ đo trên lab: khoảng 70 giây/VM, tức 19 VM mất khoảng 23 phút.
- Xong thì host **vẫn ở trạng thái disabled**. Bảo trì xong, bạn trả host lại:
  `openstack compute service set --enable compute-03 nova-compute`
- Nếu các host còn lại không đủ chỗ dưới ngưỡng hotspot, `--drain-allow-hotspot` cho phép vượt ngưỡng; trần RAM 95% vẫn luôn được giữ.

VM nằm trong danh sách `unmovable` thường thuộc một trong các trường hợp sau:

| Lý do | Nghĩa | Cách xử lý |
|---|---|---|
| `locked` | VM bị lock | `openstack server unlock` nếu được phép |
| `task_in_progress` | VM đang có tác vụ Nova | Chờ xong; drain tự chờ |
| `unsupported_resources` | Flavor có CPU pinning, PCI passthrough hoặc `resources:*` | Chuyển tay |
| Trạng thái ERROR | VM lỗi | Sửa VM trước |

### 6.3. Gắn vào Icinga/Nagios

```bash
wbalancer --monitor-only -a AGG
```

| Exit | Nghĩa |
|---|---|
| 0 OK | Đã cân bằng |
| 1 WARNING | Lệch, pressure hoặc hotspot nhưng còn move hợp lệ; hoặc lệch nhưng các move có thể làm đều cải thiện quá ít; hoặc (rc11) group soft-anti-affinity bị vỡ mà một move heal sửa được, trạng thái `SOFT_AA_HEAL` |
| 2 CRITICAL | Pressure hoặc hotspot mà **không còn move hợp lệ**, hoặc lệch mà không VM nào di chuyển được: cần thêm capacity hoặc người can thiệp |
| 3 UNKNOWN | Thiếu metric, Placement hoặc dữ liệu OpenStack |

Soft-anti-affinity (rc11): group vỡ chỉ nâng lên WARNING khi có move heal làm được; nó không bao giờ hạ hay che một CRITICAL, và dòng `Load status:` cho biết riêng phần kiểm cân bằng sẽ báo gì. Group vỡ **không** heal được, dù là không bao giờ (`unresolvable`, ví dụ 4 thành viên trong aggregate 3 host) hay chỉ lượt này (`deferred`, ví dụ có thành viên đang cooldown, kèm số đếm nguyên nhân), hoặc mọi group vỡ khi `HEAL_SOFT_ANTI_AFFINITY=false`, chỉ thêm một dòng `Soft-anti-affinity: … unresolvable: …` / `… deferred: …` / `… heal disabled`; exit code giữ đúng như khi không có group đó.

## 7. Bảo trì cuốn chiếu (`rolling-maint`)

Mỗi host đi qua các stage theo thứ tự: `preflight → pre_drain → disable → drain → verify → baseline → maintenance → reboot → health → post_boot → restore → post_restore`. Host trước xong hoàn toàn mới sang host sau. Bước drain gọi chính `wbalancer --drain-host`.

### 7.1. Profile có sẵn

| Profile | Làm gì | Reboot |
|---|---|---|
| `dummy` | Không đụng host, dừng ở `pre_drain`. Dùng để thử tích hợp | Không |
| `drain-reboot` | Drain, reboot, kiểm health, enable lại | Luôn |
| `reboot-only` | Chỉ reboot khi đã có kernel mới cài sẵn, sau boot xác minh đúng kernel | Luôn |
| `kernel-update` | Chạy dist-upgrade đúng gói trong plan đã duyệt, reboot khi cần, xác minh kernel | Khi cần |

`kernel-update` chưa được kiểm định bằng một lần upgrade thật; chạy thử trên lab trước.

### 7.2. Cách chạy

Chọn host theo **một trong hai cách**:

```bash
P=/opt/wbalancer/venv/share/compute-maintenance/profiles

# Cách 1: liệt kê host, tự khai địa chỉ SSH
rolling-maint --profile $P/drain-reboot.yaml -a AGG \
  --host-address compute-03=10.0.0.13 --ssh-key /root/.ssh/id_ed25519 \
  --prom-url http://prometheus.example:9090 --dry-run compute-03

# Cách 2: lấy host và SSH từ inventory Kolla-Ansible (cần ansible-inventory trong PATH)
rolling-maint --profile $P/drain-reboot.yaml -a AGG -i /etc/kolla/multinode --dry-run
```

- Với cách 2, nếu không chỉ định host thì tool chạy mọi host vừa thuộc aggregate vừa nằm trong nhóm `[compute]` của inventory.
- Dry-run ổn thì bỏ `--dry-run`. Mỗi host sẽ được hỏi xác nhận; thêm `-y` để tự duyệt.
- Nên luôn đặt `--prom-url`: tool sẽ đòi mọi target Prometheus của host phải `up` trước khi enable lại.
- Host key SSH: tool theo cấu hình OpenSSH của bạn (`~/.ssh/config`, `/etc/ssh/ssh_config`) và không tự đặt tùy chọn host key nào. `--ssh-strict-host-key` đã bị bỏ từ 1.0.0rc10; hãy xóa nó khỏi lệnh. ssh chạy không tương tác (`BatchMode=yes`), nên với chính sách mặc định của OpenSSH, compute có key chưa nằm trong `known_hosts` hoặc key bị đổi sẽ dừng run với exit 255. Nạp sẵn `known_hosts` bằng key đã xác minh (`ssh-keyscan -t ed25519 compute-03 >> ~/.ssh/known_hosts`, rồi so fingerprint), hoặc đặt chính sách trong `ssh_config` (ví dụ `StrictHostKeyChecking accept-new` dưới `Host compute-*`). Xem [vận hành](operations.vi.md#1-trước-lần-chạy-đầu).

Tham số hay chỉnh:

| Tham số | Mặc định | Khi nào đổi |
|---|---|---|
| `--drain-timeout` | 7200 s (~100 VM) | Host nhiều VM hơn |
| `--boot-timeout` | 3600 s | Máy vật lý boot chậm: đo một lần reboot thật rồi đặt dư ra |
| `--health-timeout` / `--settle` | 1200 s / 300 s | Phải healthy liên tục `settle` giây thì mới enable |
| `--required-containers` | `nova_compute nova_libvirt nova_ssh ovn_controller openvswitch_db openvswitch_vswitchd neutron_ovn_metadata_agent` | Dùng ML2/OVS thay cho OVN |
| `--container-engine` | `docker` | Dùng `podman` |
| `--notify` | `all` | `fail` chỉ gửi khi STOPPED, `summary` gửi thêm khi FINISHED, `off` tắt |

Đặt các timeout **ngay từ lần chạy đầu**, vì chúng là một phần định danh của lần chạy; đổi giữa chừng thì tool không cho resume.

### 7.3. Khi bị STOPPED

Tool dừng ngay khi có điều không chứng minh được là an toàn, và in sẵn lệnh để chạy tiếp:

```text
host compute-03 left at stage 'health' (nova-compute observed DISABLED).
  Fix the cause, then re-run (it continues there):  rolling-maint ... -y compute-03 compute-04
  Or give it back by hand:     openstack compute service set --enable compute-03 nova-compute
```

1. Đọc log trong `/var/log/wbalancer/maintenance/` và kiểm tra Nova thật.
2. Sửa nguyên nhân.
3. **Chạy lại đúng lệnh cũ**, không đổi profile, config hay phiên bản tool. Tool tự tiếp tục từ stage dang dở và không lặp lại việc đã xong (không reboot lần hai, không migrate lại VM đã chuyển).
4. Migration chưa rõ kết quả được tự đối chiếu với Nova ở lần chạy sau. Đừng reset trạng thái VM hay migrate tay trước khi chạy lại.

### 7.4. Viết profile riêng

Xem [profiles.vi.md](profiles.vi.md): copy template, viết script cho các hook `pre_drain`, `maintenance`, `post_boot`, `post_restore`, rồi kiểm offline bằng `/opt/wbalancer/venv/bin/python tools/profile_check.py <profile.yaml>`. Lệnh này không SSH, không gọi API.

## 8. Dữ liệu tool ghi ra

| Nội dung | Vị trí mặc định |
|---|---|
| Lịch sử migration (JSONL, mỗi dòng một sự kiện) | `/var/log/wbalancer/migrations.jsonl` |
| Log và kết quả rolling-maint | `/var/log/wbalancer/maintenance/` |
| Checkpoint bảo trì (để resume) | `/var/lib/wbalancer/maintenance/` |
| Journal migration | `/var/lib/wbalancer/operations/migration-journal.json` |
| Cooldown VM | `/var/lib/wbalancer/vm-cooldown.json` |
| Lock | `/var/lock/wbalancer-maint.lock`, `/var/lock/wbalancer-balancer.lock` |

Đừng xóa hay sửa tay checkpoint/journal khi còn lần chạy dang dở.

## 9. Exit code

| Lệnh | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `wbalancer` (balance) | Xong, kể cả không có gì để làm | Lỗi: config, dữ liệu, migration, hoặc lock đang bị giữ | — | — |
| `wbalancer --drain-host` | Host đã trống | Drain lỗi giữa chừng, hoặc bạn trả lời `N` | Không lập được kế hoạch, chưa migrate gì; **hoặc** đã migrate xong mọi VM live-migrate được nhưng host còn VM không live-migrate được (bị lock, SHUTOFF…) | — |
| `wbalancer --monitor-only` | OK | WARNING | CRITICAL | UNKNOWN |
| `rolling-maint` | Mọi host DONE hoặc SKIPPED | STOPPED (xem mục 7.3) | — | — |

Với nhiều aggregate, `--monitor-only` exit theo aggregate nghiêm trọng nhất, xếp CRITICAL > WARNING > UNKNOWN > OK: một aggregate CRITICAL và một aggregate UNKNOWN (lỗi dữ liệu) thì exit 2, không phải 3.

## 10. Lỗi thường gặp

| Thông báo | Nguyên nhân | Cách xử lý |
|---|---|---|
| `openstacksdk is not installed in this Python` | Chưa tự cài client, hoặc cài vào venv khác | Chạy lệnh cài ở mục 3, bước 2, đúng venv của tool |
| `the 'openstack' CLI … was not found` | Thiếu `python-openstackclient` | Như trên |
| `… older than the OpenStack 2026.1 baseline …` | Client cũ hơn chuẩn đã kiểm định | Nâng lên theo constraints 2026.1, hoặc tự kiểm lại trên lab |
| `Not running: state from an earlier version was found in /var/lib/wbalancer-v1 …` | Nâng cấp từ 1.0.0rc8 trở về trước; trạng thái còn nằm ở thư mục `-v1` cũ | Chuyển theo lệnh được in ra ([vận hành](operations.vi.md#2-file-trạng-thái-và-lock)) |
| `PROMETHEUS_QUERY_URL is required` | Thiếu URL trong config | Thêm URL đầy đủ, kết thúc bằng `/api/v1/query` |
| `PROMETHEUS_NODE_JOB is required` | Thiếu tên job | Thêm `PROMETHEUS_NODE_JOB` và `PROMETHEUS_LIBVIRT_JOB` |
| `Aggregate X skipped: …` / monitor trả 3 | Một host thiếu metric, thiếu label `alias`, hoặc thiếu Placement | Chạy query A, B ở mục 2. Các aggregate khác vẫn được xử lý |
| `Not running: …` | Đang có `rolling-maint` hoặc một lần balance khác chạy | Bình thường, chờ lượt sau |
| `Cloud <tên> was not found` (monitor exit 3) | `OS_CLOUD` / `--os-cloud` trỏ tới cloud không có trong `clouds.yaml` đang dùng | Sửa tên, hoặc trỏ `OS_CLIENT_CONFIG_FILE` tới đúng file |
| `rolling-maint`: `command failed rc=255: ssh …` kèm `Host key verification failed` | Host key của compute chưa có trong `known_hosts` của coordinator, hoặc đã đổi (host cài lại) | Xác minh key rồi thêm vào (`ssh-keyscan`), hoặc xóa key cũ trước (`ssh-keygen -R <host>`); hoặc đặt `StrictHostKeyChecking` trong `ssh_config` (mục 7.2) |
| `rolling-maint: error: unrecognized arguments: --ssh-strict-host-key` | Tùy chọn đã bị bỏ từ 1.0.0rc10 | Xóa nó khỏi lệnh; chính sách host key lấy từ `ssh_config` |
| `RAM_CRITICAL_THRESHOLD (…) must be above RAM_HOTSPOT_THRESHOLD`, `… must be > 0 and <= 100` | Ngưỡng trong config không hợp lệ | Sửa giá trị; tool không chạy gì cho tới khi hợp lệ |
| `ansible-inventory not found` | Dùng `-i` ngoài venv Kolla-Ansible | Thêm venv Kolla vào `PATH`, hoặc dùng `--host-address` |
| Drain exit 2 | Không đủ chỗ, hoặc vướng server group / AZ / VM unmovable | Đọc danh sách lý do; thêm capacity hoặc cân nhắc `--drain-allow-hotspot` |
| `migration_timeout` trong events | Migration không có tiến độ trong 600 s (`MIGRATION_TIMEOUT`), hoặc sau 30 phút vẫn chưa xong (`MIGRATION_MAX_WAIT`) | Kiểm migration trên Nova; lần chạy sau tự đối chiếu |

## 11. Giới hạn cần biết

- Chỉ chạy từ **một coordinator**; không có lock phân tán. Không chạy song song với bản legacy.
- Chỉ live migration với storage dùng chung; không hỗ trợ block migration.
- Không mô hình hóa VM có CPU pinning, PCI passthrough hay `resources:*`.
- Planner là heuristic: không tìm thấy kế hoạch không có nghĩa là không có cách xếp nào khác.
- Group soft-anti-affinity bị vỡ do bảo trì được tách lại (rc11) nhưng mỗi aggregate chỉ một VM mỗi lần chạy, chỉ khi không có host hotspot, và chỉ khi MAD CPU và MAD RAM đều không vượt `TARGET_AVG_MAD` (hoặc, nếu đã vượt từ trước, chỉ tăng tối đa `SOFT_AA_HEAL_MAD_TOLERANCE`); group gồm VM rất lớn mà chuyển đi sẽ làm aggregate mất cân bằng thì được báo "deferred: … mad_tolerance". Group `soft-affinity` không bao giờ được gom lại.
- Đã kiểm định trên lab Kolla-Ansible 2026.1, Ceph, 3 compute, 47 VM. Môi trường khác thì làm lại các bước ở mục 5.

Chi tiết thuật toán: [algorithm.vi.md](algorithm.vi.md). Vận hành và recovery: [operations.vi.md](operations.vi.md). Thay đổi và kết quả kiểm định: [CHANGELOG](../CHANGELOG.md).
