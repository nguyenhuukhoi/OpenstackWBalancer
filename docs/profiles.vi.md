# Tạo profile bảo trì

[English](profiles.md) | Tiếng Việt

Profile [dummy](../profiles/dummy.yaml) có sẵn để thử tích hợp mà không thay đổi host: `pre_drain` local trả rc3 nên runner ghi `SKIPPED` trước disable/drain, không migration/reboot. Runner vẫn cần cloud/inventory hợp lệ cho preflight và vẫn ghi log/checkpoint local. Dummy không kiểm định toàn bộ workflow maintenance. Dùng `--profile profiles/dummy.yaml` cùng các tham số coordinator/aggregate/host thông thường; không đổi profile của một pending run sang dummy.

Mỗi tác vụ là một thư mục gồm `profile.yaml` và `scripts/`. Runner lo lock, checkpoint, disable, drain, reboot, health và restore; profile mô tả phần bảo trì riêng. Không cần sửa Python để thêm một tác vụ dùng các hook hiện có.

## Bắt đầu

Từ thư mục gốc của repo, copy toàn bộ template:

```powershell
Copy-Item -Recurse profile-workspace/template profile-workspace/my-maintenance
```

Linux:

```bash
cp -R profile-workspace/template profile-workspace/my-maintenance
```

Đổi `name`, `description`, chính sách reboot và nội dung bốn script. Template cố ý dừng với `PROFILE_NOT_IMPLEMENTED`, không phải một profile chạy được ngay. Đặt `name` dễ nhận diện, ví dụ `update-multipath-config`; dùng hook `id` ổn định và `name` mô tả ngắn để log dễ đọc.

Kiểm tra offline từ thư mục gốc của repo:

```bash
uv run python tools/profile_check.py profile-workspace/my-maintenance/profile.yaml
```

Tool dùng chính loader của runner: kiểm tra schema, duplicate key/ID, file script, path containment, types và timings; in hook plan và settings trước CLI override. Không chạy script, SSH, API hay disable host. Exit 0 = schema hợp lệ và không còn marker template; exit 2 = lỗi hoặc còn placeholder. Không chứng minh logic shell đúng hoặc tác vụ an toàn.

## Chọn hook

| Hook | Thời điểm | Nội dung phù hợp |
|---|---|---|
| `pre_drain` | Host còn enabled, trước disable/drain | Kiểm tra điều kiện, phiên bản, approved plan; xác định có việc cần làm |
| `maintenance` | Host đã disabled/drained và có baseline | Thay config, cập nhật package, thực hiện thay đổi đã duyệt |
| `post_boot` | Sau health gate, trước restore; vẫn chạy nếu bỏ qua reboot | Xác minh phiên bản/config/kết quả mong muốn |
| `post_restore` | Sau enable | Kiểm tra acceptance cần host phục vụ thật; lỗi sẽ quarantine |

Đặt kiểm tra trước enable ở `post_boot` nếu làm được. `post_restore` có khoảng thời gian host enabled, có thể nhận VM mới; recovery sẽ kiểm tra inventory và không tự drain/reboot workload mới.

## Contract cần giữ

| Thành phần | Quy ước |
|---|---|
| Hook thường | rc0 = thành công; khác 0 = lỗi |
| Riêng `pre_drain` | rc3 = không cần bảo trì, skip host; không dùng để che lỗi |
| `reboot_required_cmd` | rc0 = cần reboot; rc1 = không cần; mã khác = UNKNOWN |
| Recovery `check` | rc0 = đã hoàn tất và kiểm chứng; rc10 = chứng minh an toàn chạy lại; mã khác = UNKNOWN |
| `run: ssh` | Mặc định, chạy trên compute theo SSH identity của runner |
| `run: local` | Chạy trên coordinator, cần bash; không phải chạy trên compute |
| `idempotency: reconcile` | Cần `check`; phù hợp thao tác có side effect và có thể kiểm chứng kết quả |
| `idempotency: idempotent` | Chỉ dùng nếu chạy lại thực sự an toàn; không tự cho phép replay timeout UNKNOWN khi thiếu check |
| `idempotency: never` | Mặc định; không tự replay một kết quả trước đó chưa được xác minh |

Check reconciliation chỉ được gọi khi hook có journal từ lần chạy trước, không phải precheck mặc định lần đầu. Một hook đã ghi `done` sẽ được bỏ qua khi resume. Kiểm tra trực tiếp tài nguyên đích; file marker đơn thuần không chứng minh update/config đã thành công.

Script được nạp thành nội dung lệnh, gửi tới shell; không được copy như một thư mục lên host. Đừng dùng đường dẫn tương đối, `$0` hoặc `source ./helper.sh` với giả định đang đứng trong thư mục profile. Viết script tự chứa hoặc dùng dependency đã có tại đường dẫn tuyệt đối trên target. YAML `script` và `check` phải nằm trong thư mục chứa profile, kể cả sau khi resolve symlink.

Placeholder `{host}`, `{address}`, `{profile}` được runner shell-quote; dùng như một token không bọc thêm nháy, ví dụ `printf '%s\n' {host}`. Biến shell khác vẫn theo quy tắc Bash thông thường. `capture` hiện chỉ hỗ trợ `expected_kernel` và `package_plan`; stdout phải chỉ chứa giá trị cần lưu, diagnostic ghi stderr. Các hook sau đọc `WLB_EXPECTED_KERNEL` / `WLB_PACKAGE_PLAN`. Không giả định có cơ chế capture biến tùy ý.

Dùng `set -euo pipefail`, nhưng vẫn kiểm tra rõ exit codes của lệnh quan trọng. Không ghi secret vào YAML/output. Runner hiện gom output hook và chỉ log 20 dòng cuối sau khi command kết thúc, chưa stream tiến độ từng dòng. Thời gian chờ lâu cần timeout phù hợp và check recovery.

## Đưa vào coordinator

Copy cả thư mục đã review vào một đường dẫn riêng, ví dụ `/etc/wbalancer/profiles/my-maintenance/`; giữ file thuộc owner quản trị. Cài đặt và config theo [user-guide.vi.md](user-guide.vi.md).

```bash
rolling-maint --profile /etc/wbalancer/profiles/my-maintenance/profile.yaml \
  -a YOUR_AGGREGATE --host-address COMPUTE=IP --ssh-key /path/to/key \
  --prom-url http://PROMETHEUS:9090 --notify off --dry-run COMPUTE
```

`--dry-run` của runner có thể chạy hook `pre_drain`: chỉ dùng hook read-only ở đó. Muốn chỉ kiểm tra file thì dùng `profile_check.py`. Kiểm định canary một host được phép trước rollout; sau review bỏ `--dry-run` và dùng `-y` cho thực thi đã duyệt.

Không sửa YAML hoặc script giữa một pending run: identity bao gồm nội dung đã nạp. Giữ nguyên profile/artifact/settings để resume. Muốn thay profile, hoàn tất hoặc reconcile lượt cũ trước. Không xóa checkpoint để ép chạy lại.

Mẫu thực đã có: [drain-reboot](../profiles/drain-reboot.yaml) không maintenance hook; [kernel-update](../profiles/kernel-update.yaml) có approved package plan, reconcile và kiểm tra kernel. Profile package upgrade chưa được kiểm định bằng một lần upgrade thật; chạy thử trên lab trước.
