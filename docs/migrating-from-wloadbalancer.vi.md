# Chuyển từ OpenstackWBalancer (`wloadbalancer.py`) sang v1

[English](migrating-from-wloadbalancer.md) | Tiếng Việt

Dành cho người đang chạy bản script đơn [OpenstackWBalancer](https://github.com/nguyenhuukhoi/OpenstackWBalancer). v1 là bản nâng cấp của script đó: cùng thuật toán MAD/pressure/hotspot, cùng các tùy chọn CLI, thêm drain host và bảo trì cuốn chiếu.

## Khác biệt cần biết

| Mục | Bản cũ `wloadbalancer.py` | v1 |
|---|---|---|
| Cách chạy | `python wloadbalancer.py …` | Cài package, chạy lệnh `wbalancer …` (thêm lệnh `rolling-maint`) |
| Python | ≥ 3.10 | 3.12 |
| `openstacksdk` | Tự cài | Vẫn tự cài, chuẩn OpenStack 2026.1 (xem [hướng dẫn sử dụng mục 3](user-guide.vi.md#3-cài-đặt)) |
| Kết nối OpenStack | `openstack.connect(cloud='openstack')` | Cùng lời gọi, nên credential đang chạy được thì vẫn chạy được; tên cloud nay đổi được bằng `OS_CLOUD` / `--os-cloud` |
| File config mặc định | `/etc/loadleveller-secrets.conf` | `/etc/wbalancer/wbalancer.conf`, hoặc dùng `--config` trỏ về file cũ |
| Tên job Prometheus | Sửa thẳng trong code (`PROMETHEUS_NODE_JOB = '…'`) | **Bắt buộc** khai trong config: `PROMETHEUS_NODE_JOB`, `PROMETHEUS_LIBVIRT_JOB` |
| `PROMETHEUS_QUERY_URL` | Có giá trị mặc định | **Bắt buộc**, thiếu thì báo lỗi |
| Allocation ratio | Lấy từ openstack-exporter (`PROMETHEUS_CPU_RATIO`, `PROMETHEUS_MEM_RATIO`) | Lấy thẳng từ **Placement**; không cần openstack-exporter, các khóa cũ bị bỏ qua kèm cảnh báo |
| File cooldown | `/var/log/loadleveller_vm_cooldown.json` | `/var/lib/wbalancer/vm-cooldown.json`. **Cùng định dạng**, copy sang là giữ được cooldown |
| Log sự kiện | `/var/log/loadleveller_migration_events.jsonl` | `/var/log/wbalancer/migrations.jsonl` |
| Chống chạy chồng | Không có | Có lock; hai lượt cron không chạy chồng nhau |
| Mỗi lượt balance | Tối đa 1 migration mỗi aggregate | Tối đa 1 kế hoạch: 1 migration, hoặc 2 migration kiểu MAKE_ROOM (dời VM nhỏ lấy chỗ) |
| Tính năng mới | — | `--drain-host`, `rolling-maint` + profile, journal tự đối chiếu migration dang dở, timeout theo tiến độ, tôn trọng availability zone và VM bị lock |

Các tùy chọn CLI cũ đều còn, cùng tên: `--config`, `--verbose`, `--dry-run`, `--monitor-only`, `-a/--aggregate`, `-y/--yes`, `--send-all`, `--send-error`, `--no-server-groups`, `--use-vm-ram-rss`, `--cooldown-file`, `--events-file`. Exit code của `--monitor-only` vẫn theo chuẩn Icinga/Nagios (0–3).

## Ngưỡng đã đổi: kết quả có thể khác bản cũ

| Ngưỡng | Bản cũ | v1 |
|---|---|---|
| RAM hotspot | > 75% | > 80% |
| RAM critical | > 85% | > 95% |
| CPU hotspot | > 85% | > 70% |
| Cải thiện tối thiểu để move | 0,05 | 0,15 |
| Lệch (MAD) | > 3 điểm | > 3 điểm (giữ nguyên) |
| Pressure (trung bình) | > 80% | > 80% (giữ nguyên, nhưng không bao giờ tới lượt: host > 80% đã là hotspot) |

Từ rc7 các ngưỡng này là khóa config. Muốn giữ đúng hành vi cũ thì thêm vào `wbalancer.conf`:

```ini
RAM_HOTSPOT_THRESHOLD=75
RAM_CRITICAL_THRESHOLD=85
CPU_HOTSPOT_THRESHOLD=85
MIN_IMPROVEMENT_THRESHOLD=0.05
```

Với ngưỡng mặc định của v1, ví dụ host RAM 78%, CPU 75%:
- Bản cũ: RAM hotspot (vượt 75%), CPU bình thường (chưa tới 85%).
- v1: RAM bình thường (chưa tới 80%), CPU hotspot (vượt 70%).

Ngưỡng cải thiện cao hơn nên v1 migrate **ít hơn**: move chỉ cải thiện 0,1 thì bản cũ vẫn làm, v1 bỏ qua.

## Các bước chuyển

1. **Cài v1** vào venv riêng theo [hướng dẫn sử dụng mục 3](user-guide.vi.md#3-cài-đặt). Giữ nguyên bản cũ.
2. **Chuyển config:**

   ```bash
   mkdir -p /etc/wbalancer
   cp /etc/loadleveller-secrets.conf /etc/wbalancer/wbalancer.conf
   chmod 600 /etc/wbalancer/wbalancer.conf
   ```

   Thêm 2 dòng, lấy đúng giá trị đang sửa cứng trong `wloadbalancer.py`:

   ```ini
   PROMETHEUS_NODE_JOB=<giá trị PROMETHEUS_NODE_JOB trong script cũ>
   PROMETHEUS_LIBVIRT_JOB=<giá trị PROMETHEUS_LIBVIRT_JOB trong script cũ>
   ```

   Xóa `PROMETHEUS_CPU_RATIO`, `PROMETHEUS_MEM_RATIO`, `PROMETHEUS_OPENSTACK_EXPORTER_JOB` (để lại thì chỉ bị cảnh báo).
3. **So sánh song song, chỉ đọc:**

   ```bash
   python wloadbalancer.py --monitor-only -a AGG; echo "old=$?"
   /opt/wbalancer/venv/bin/wbalancer --monitor-only -a AGG; echo "v1=$?"
   /opt/wbalancer/venv/bin/wbalancer --dry-run -a AGG
   ```

   Kết quả khác nhau là bình thường nếu có host nằm giữa hai bộ ngưỡng ở bảng trên.
4. **Tắt cron của bản cũ**, chờ lượt đang chạy (nếu có) kết thúc. Hai bản không dùng chung lock, nên **không bao giờ để cả hai cùng migrate**.
5. **Giữ cooldown:**

   ```bash
   mkdir -p /var/lib/wbalancer
   cp /var/log/loadleveller_vm_cooldown.json /var/lib/wbalancer/vm-cooldown.json
   ```

   Như vậy VM vừa được bản cũ chuyển sẽ không bị v1 chuyển lại trong 4 giờ.
6. **Bật cron v1** theo [hướng dẫn sử dụng mục 6.1](user-guide.vi.md#61-balance). Cấu hình Icinga/Nagios chỉ cần đổi lệnh check sang `wbalancer --monitor-only`.

## Quay lại bản cũ

Tắt cron v1 rồi bật lại cron cũ. v1 không sửa file config, cooldown hay log của bản cũ.
