# OpenStack Workload Balancer

[English](README.md) | Tiếng Việt

Giữ cho các compute aggregate của OpenStack cân bằng, và drain hoặc bảo trì compute host mà không gây downtime, bằng live migration.

v1 thay cho script một file `wloadbalancer.py`. Script cũ vẫn còn trong repo nhưng đã deprecated; xem [hướng dẫn chuyển đổi](docs/migrating-from-wloadbalancer.vi.md).

## Tính năng

| Lệnh | Làm gì |
|---|---|
| `wbalancer` | **Cân bằng**: mỗi aggregate, mỗi lượt chạy làm tối đa 1 kế hoạch, gồm 1 live migration hoặc 2 bước kiểu "make room". Chỉ ra tay khi có host hotspot hoặc CPU/RAM lệch giữa các host (MAD), và chỉ khi move cải thiện đủ nhiều |
| `wbalancer --drain-host HOST` | **Drain**: chuyển hết VM khỏi host và rải đều để các host còn lại vẫn cân. Fail-closed: không lập được kế hoạch an toàn cho toàn bộ thì không chuyển VM nào |
| `rolling-maint` | **Bảo trì cuốn chiếu**: lần lượt từng host: disable, drain, chạy profile bảo trì, reboot, kiểm health, enable lại. Chạy lại đúng lệnh cũ là tự tiếp tục từ chỗ dừng |
| `wbalancer --monitor-only` | **Giám sát**: chỉ đọc, trả exit code chuẩn Nagios/Icinga (0 OK, 1 WARNING, 2 CRITICAL, 3 UNKNOWN) |

Mọi lệnh đều có `--dry-run`.

Mỗi move đều phải qua các quy tắc an toàn:

- **Capacity:** lấy từ Placement, chừa 5% headroom.
- **Chính sách:** server group theo đúng cách Nova xét, kể cả soft (anti-)affinity; availability zone; giới hạn số VM mỗi host.
- **Không tạo hotspot mới:** host đích không được thành hotspot sau move.
- **VM bị bỏ qua:** VM bị lock, đang có tác vụ, hoặc có flavor không mô hình hóa được (CPU pinning, PCI, `resources:*`) không bao giờ bị chuyển.
- **Khi chạy:** cooldown cho từng VM, migration journal bền vững qua các lượt chạy, timeout migration chờ tiếp chừng nào còn tiến độ.

Cách ra quyết định, kèm ví dụ số: [docs/algorithm.vi.md](docs/algorithm.vi.md).

## Yêu cầu

- **OpenStack:** Nova và Placement, quyền admin, live migration chạy được giữa các compute với **storage dùng chung** (ví dụ Ceph RBD), compute nằm trong host aggregate.
- **Prometheus:**
  - [node_exporter](https://github.com/prometheus/node_exporter) có label `alias` trùng tên host Nova;
  - libvirt exporter có metric `libvirt_domain_*` với label `host` và `instanceId` (ví dụ [prometheus-libvirt-exporter](https://github.com/zhangjianweibj/prometheus-libvirt-exporter)).
- **Coordinator:** máy Linux có Python 3.12. `rolling-maint` cần thêm SSH vào các compute.
- **Client OpenStack:** package này **không** cài `openstacksdk` và `python-openstackclient`; bạn tự cài cho hợp với cloud. Chuẩn đã kiểm định là OpenStack **2026.1** (openstacksdk 4.10.0, python-openstackclient 9.0.1).

Checklist đầy đủ, kèm query PromQL để tự kiểm metric, ở [hướng dẫn sử dụng](docs/user-guide.vi.md#2-hệ-thống-của-bạn-có-dùng-được-không).

## Bắt đầu nhanh

```bash
git clone https://github.com/nguyenhuukhoi/OpenstackWBalancer.git && cd OpenstackWBalancer
python3 -m venv /opt/wbalancer/venv            # python3 phải là 3.12 (requires-python >=3.12,<3.13)
VENV=/opt/wbalancer/venv

$VENV/bin/pip install --require-hashes -r requirements.lock.txt      # thư viện riêng của tool, đã khóa
$VENV/bin/pip install --no-deps .
$VENV/bin/pip install -c https://releases.openstack.org/constraints/upper/2026.1 openstacksdk python-openstackclient   # bạn tự chọn

mkdir -p /etc/wbalancer && install -m 600 wbalancer.conf.example /etc/wbalancer/wbalancer.conf
nano /etc/wbalancer/wbalancer.conf   # sửa PROMETHEUS_QUERY_URL và 2 tên job; các khóa khác là tùy chọn

$VENV/bin/wbalancer --monitor-only                 # chỉ đọc trạng thái, exit 0-3
$VENV/bin/wbalancer --dry-run -a my-aggregate      # in kế hoạch, không đổi gì
```

- **Credential:** cloud bất kỳ trong `clouds.yaml` (mặc định `openstack`; chọn cloud khác bằng `OS_CLOUD` hoặc `--os-cloud`), hoặc môi trường openrc với `--os-cloud envvars` (xem [hướng dẫn sử dụng](docs/user-guide.vi.md#4-cấu-hình)).
- **Ngưỡng:** các ngưỡng hotspot, pressure, MAD và cải thiện tối thiểu là khóa config (xem [hướng dẫn sử dụng](docs/user-guide.vi.md#42-file-cấu-hình-etcwbalancerwbalancerconf)).
- **Trước khi bật bất cứ thứ gì tự động,** đi theo [lộ trình chạy lần đầu](docs/user-guide.vi.md#5-lộ-trình-chạy-lần-đầu): chỉ đọc trước, rồi 1 move có người giám sát, 1 lần drain, 1 lần canary bảo trì.

## Tài liệu

| Tài liệu | Dùng để |
|---|---|
| [Hướng dẫn sử dụng](docs/user-guide.vi.md) | Yêu cầu, cài đặt, cấu hình, lộ trình chạy lần đầu, tra cứu lệnh, exit code, lỗi thường gặp |
| [Thuật toán](docs/algorithm.vi.md) | Các chế độ, cách chấm điểm, bộ lọc an toàn, make-room, kế hoạch drain, kèm ví dụ số |
| [Kiến trúc](docs/architecture.vi.md) | Phân tầng code, bất biến, file state, công thức mở rộng — cho người phát triển |
| [Vận hành](docs/operations.vi.md) | File state và lock, timeout, xử lý khi một lượt chạy dừng, recovery |
| [Profile bảo trì](docs/profiles.vi.md) | Tự viết profile và hook cho `rolling-maint` |
| [Chuyển từ `wloadbalancer.py`](docs/migrating-from-wloadbalancer.vi.md) | Khác biệt config và ngưỡng, các bước chuyển, quay lại bản cũ |
| [CHANGELOG](CHANGELOG.md) | Ghi chú phát hành và kết quả kiểm định |

## Đã kiểm định với

OpenStack 2026.1 triển khai bằng Kolla-Ansible, storage dùng chung Ceph RBD, OVN, trên lab 3 compute và 47 VM:

- **Canary drain + reboot:** mỗi lượt 19 VM, khoảng 70 giây/VM, không mất VM nào.
- **Thử lỗi:** migration lỗi và bị abort thật, migration bị treo, VM bị lock.
- **Bù tải khi drain:** sai số đo được trong khoảng ±0,03 điểm %.

Môi trường khác nên đi lại lộ trình chạy lần đầu trước khi dùng production.

## Phát triển

```bash
uv sync --extra test --locked            # nhóm dev pin client OpenStack 2026.1, chỉ dùng cho test
uv run --extra test python -m pytest -q
uv build --wheel
```

Bộ test không bao giờ nối vào cloud. Test lock POSIX chỉ chạy trên Linux.

## License

[MIT](LICENSE)
