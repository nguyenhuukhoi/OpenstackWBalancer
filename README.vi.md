# OpenStack Workload Balancer

[English](README.md) | Tiếng Việt

Một script Python hướng tới operator, dùng để đánh giá các compute aggregate trong OpenStack và thực hiện một lần live migration khi migration đó được kỳ vọng sẽ cải thiện trạng thái cân bằng của cluster một cách an toàn.

Script này được thiết kế cho môi trường production, nơi tính dễ dự đoán quan trọng hơn việc rebalance quá mạnh tay. Nó dùng metric từ Prometheus, inventory từ OpenStack, kiểm tra server group, kiểm tra readiness của host đích, và cooldown window để giảm migration churn.

## Script này làm gì

Balancer hoạt động ở cấp độ aggregate.

Với mỗi aggregate được chọn, nó sẽ:

1. Thu thập mức sử dụng CPU và RAM của host từ Prometheus.
2. Thu thập capacity của host và allocation ratio từ OpenStack và Prometheus.
3. Xây dựng inventory các VM đang chạy trong aggregate.
4. Chọn mode vận hành:
   - RAM critical hotspot
   - RAM hotspot
   - CPU hotspot
   - Pressure
   - Cân bằng dựa trên MAD thông thường
5. Tìm một VM move tốt nhất có thể cải thiện trạng thái cân bằng mà vẫn tôn trọng các luật an toàn.
6. Sau đó:
   - trả kết quả ở chế độ monitoring,
   - hiển thị migration được đề xuất ở chế độ dry-run,
   - hoặc thực hiện migration và theo dõi kết quả.

Script chỉ thực hiện tối đa một migration cho mỗi aggregate trong mỗi lần chạy.

## Bắt đầu nhanh

Hãy dùng các lệnh này trước nếu bạn muốn hiểu nhanh hoặc vận hành script ngay.

Chỉ kiểm tra health, không thay đổi gì:

```bash
python wloadbalancer.py --monitor-only
```

Xem migration được đề xuất mà không thực thi:

```bash
python wloadbalancer.py --dry-run --aggregate my-compute-aggregate
```

Chạy một migration ở chế độ interactive:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate
```

## Cách đọc tài liệu này

README này cố ý viết khá chi tiết. Cách đọc nhanh nhất sẽ phụ thuộc vào nhu cầu của bạn.

Nếu bạn là operator:

- bắt đầu với `Script này làm gì`
- sau đó đọc `Bắt đầu nhanh`
- rồi đọc `Cách đọc một quyết định dưới góc nhìn operator`
- rồi đọc `Chế độ monitoring và execution`

Nếu bạn đang review hoặc maintain thuật toán:

- đọc `Logic ra quyết định cốt lõi`
- sau đó `Thuật toán “nghĩ” như thế nào`
- rồi `Tham chiếu toán học và cách chấm điểm`
- rồi `Mô phỏng move trong ba trường hợp chính`

Nếu bạn đang deploy script:

- đọc `Yêu cầu`
- sau đó `Yêu cầu OpenStack`
- rồi `Yêu cầu Prometheus`
- rồi `Cấu hình`
- rồi `Cách dùng command line`

## Logic ra quyết định cốt lõi

Script sử dụng cả tín hiệu về độ phân tán và tín hiệu về áp lực tài nguyên.

### 1. Phát hiện hotspot luôn được ưu tiên trước

Nếu bất kỳ host nào vượt ngưỡng hotspot, script sẽ chuyển từ balancing thông thường sang xử lý hotspot.

Ngưỡng hotspot mặc định:

- RAM hotspot: `75%`
- RAM critical hotspot: `85%`
- CPU hotspot: `85%`

Khi có hotspot, balancer sẽ ưu tiên tìm move từ host nóng nhất trước, sau đó mới fallback sang tìm kiếm trên toàn aggregate.

### 2. Tiếp theo là pressure mode

Nếu không có hotspot nhưng trung bình CPU hoặc RAM của cluster vượt ngưỡng pressure, script sẽ vào `pressure` mode.

Ngưỡng pressure mặc định:

- CPU hoặc RAM trung bình của cluster lớn hơn `80%`

### 3. Nếu không, dùng cân bằng dựa trên MAD

Nếu không có hotspot và không có pressure, script dùng mô hình Mean Absolute Deviation (MAD) có trọng số để đo mức mất cân bằng giữa các host.

Ngưỡng lệch chấp nhận được mặc định:

- Target MAD: `3`

### 4. Một candidate move phải vừa an toàn vừa hữu ích

Một VM move chỉ được xem xét nếu tất cả các điều sau đều đúng:

- VM không đang trong cooldown sau lần migration thành công gần đây.
- VM không đang trong cooldown sau lần move thất bại gần đây.
- Host đích có dịch vụ `nova-compute` sẵn sàng.
- Host đích có đủ CPU và RAM allocatable cho flavor của VM.
- Move đó không vi phạm affinity hoặc anti-affinity của server group.
- Trong hotspot mode, hotspot thực sự phải được cải thiện và host đích không được trở thành hotspot.

### 5. Một candidate move phải vượt ngưỡng improvement

Script không migrate chỉ vì move đó “khả thi về mặt kỹ thuật”.

Hiện tại script dùng một improvement threshold cố định cho mọi mode:

- `0.05`

## Thuật toán “nghĩ” như thế nào

Phần này giải thích luồng ra quyết định bằng ngôn ngữ đời thường.

### Bước 1. Đọc trạng thái hiện tại của cluster

Script trước tiên đọc trạng thái hiện tại của aggregate:

- phần trăm CPU của host
- phần trăm RAM của host
- phần trăm CPU của từng VM
- phần trăm RAM của từng VM
- capacity CPU và RAM allocatable của host

Tại thời điểm này, script có hai góc nhìn về cluster:

- góc nhìn ở mức host: pressure và spread
- góc nhìn ở mức VM: workload nào có thể được move

### Bước 2. Chọn operating mode

Script không phải lúc nào cũng tối ưu cluster theo cùng một cách.

Nó chọn một mode dựa trên mức độ khẩn cấp:

1. Hotspot mode
   - dùng khi một host đã quá nóng
   - ưu tiên là làm host đó dịu đi
2. Pressure mode
   - dùng khi trung bình toàn aggregate đang chạy nóng
   - ưu tiên là giảm áp lực tổng thể
3. MAD mode
   - dùng khi không có hotspot và không có pressure
   - ưu tiên là giảm độ lệch giữa các host

Tóm lại:

- hotspot mode hỏi: “Tôi có thể làm host nóng nhất dịu đi một cách an toàn không?”
- pressure mode hỏi: “Tôi có thể giảm áp lực tài nguyên toàn cụm không?”
- MAD mode hỏi: “Tôi có thể làm các host đều nhau hơn không?”

### Bước 3. Chuyển mức dùng của VM thành tác động lên host

Với mỗi VM, script ước lượng mức load của host sẽ được mang đi cùng với VM đó.

Nó dùng:

- số vCPU trong flavor của VM
- RAM trong flavor của VM
- phần trăm CPU sử dụng thực tế của VM
- phần trăm RAM sử dụng thực tế của VM

Về mặt khái niệm:

- real CPU moved = `vcpus * vm_cpu_pct`
- real RAM moved = `vm_ram_gb * vm_ram_pct`

Sau đó các giá trị này được chuyển thành phần trăm tác động ở cấp độ host cho:

- host nguồn
- host đích

Hai giá trị phần trăm này có thể khác nhau. Cùng một lượng tải của VM sẽ được chia theo capacity của host nguồn khi rời đi và theo capacity của host đích khi đi tới, nên nếu các host không đồng nhất thì source impact và destination impact khác nhau là bình thường.

Điều này quan trọng vì script không chấm điểm migration chỉ dựa trên kích thước flavor. Nó chấm điểm bằng tải thực tế mà VM đang tiêu thụ.

### Optional RAM impact source: guest RAM usage vs host-side RSS

Mặc định, script ước lượng RAM impact của VM từ guest-visible RAM usage:

- `real RAM moved = vm_ram_gb * vm_ram_pct`

Đây là default hợp lý khi guest RAM usage phản ánh khá sát mức RAM pressure mà VM tạo ra trên source host.

Tuy nhiên, một số workload giữ cache footprint lớn. Khi đó:

- guest RAM usage có thể nhìn vẫn vừa phải
- nhưng process `qemu` ở host side lại giữ resident memory lớn hơn nhiều

Điều đó có thể làm mô hình RAM impact mặc định bị thấp hơn thực tế cho mục đích balancing.

Để xử lý case này, script có thể optional dùng:

- `libvirt_domain_stat_memory_rss_bytes`

Khi được bật, script dùng host-side RSS làm giá trị `real_vm_ram` cho phần RAM impact scoring. Điều này chỉ thay đổi cách ước lượng RAM impact. Nó không thay đổi:

- destination feasibility checks
- improvement threshold logic
- hotspot / pressure / MAD mode selection
- candidate ranking flow

Nói ngắn gọn:

- flavor RAM vẫn được dùng để quyết định destination có thể host VM hay không
- RSS chỉ được dùng để ước lượng có bao nhiêu RAM pressure di chuyển theo VM

Đây thường là lựa chọn hợp lý hơn khi:

- VM có workload cache-heavy
- host memory pressure quan trọng hơn guest-reported RAM usage
- bạn có metric `libvirt_domain_stat_memory_rss_bytes` ổn định với labels `instanceId` và `host`

Default model vẫn hợp lý khi:

- guest RAM usage đã phản ánh đủ tốt actual host pressure
- libvirt exporter của bạn chưa expose RSS metric ổn định
- bạn muốn giữ behavior gốc, bảo thủ hơn

Quick decision guide:

- Giữ default model nếu bạn chủ yếu tin guest-visible RAM usage và muốn giữ behavior gốc.
- Dùng `--use-vm-ram-rss` nếu bạn quan tâm tới host-side memory pressure và có VM giữ cache footprint lớn.
- Chỉ nên ưu tiên RSS sau khi xác nhận libvirt exporter của bạn report ổn định labels `instanceId` và `host` cho `libvirt_domain_stat_memory_rss_bytes`.

### Bước 4. Mô phỏng từng destination có thể

Với mỗi VM trên các source host được phép, script mô phỏng việc move VM đó sang mọi host khác trong aggregate.

Với mỗi move mô phỏng, nó kiểm tra:

- cooldown rules
- destination host readiness
- allocatable capacity của destination
- việc tuân thủ server-group policy
- hotspot guardrail nếu aggregate đang ở hotspot mode

Nếu bất kỳ kiểm tra an toàn nào thất bại, candidate đó bị loại ngay.

### Bước 5. Tính lại cluster sau move mô phỏng

Nếu một candidate vượt qua các kiểm tra an toàn, script sẽ tính lại cluster như thể move đó đã xảy ra.

Sau đó nó tính lại:

- CPU MAD sau move
- RAM MAD sau move
- weighted MAD sau move

Improvement được định nghĩa là:

```text
improvement = current_weighted_mad - new_weighted_mad
```

Vì vậy:

- improvement dương nghĩa là move đó có ích
- improvement bằng 0 hoặc âm nghĩa là move đó không giúp ích

### Bước 6. Chọn best valid move

Script chỉ giữ move có improvement cao nhất.

Sau đó nó áp improvement threshold:

- nếu best move thấp hơn threshold, sẽ không migrate
- nếu best move đạt hoặc vượt threshold, move đó trở thành migration được đề xuất

Đây là cách script giữ tính bảo thủ:

- chỉ khả thi về kỹ thuật là chưa đủ
- move đó còn phải đủ đáng giá để bù cho chi phí migration

## Tham chiếu toán học và cách chấm điểm

Phần này giải thích các công thức phía sau mô hình chấm điểm và đi qua chúng bằng ví dụ tính tay.

### Ví dụ A. Mean và MAD

Hãy tưởng tượng ba host có mức CPU như sau:

- Host A: `70%`
- Host B: `40%`
- Host C: `35%`

Đầu tiên, tính mean CPU:

```text
mean_cpu = (70 + 40 + 35) / 3 = 48.33
```

Sau đó tính khoảng cách tuyệt đối so với mean:

```text
|70 - 48.33| = 21.67
|40 - 48.33| = 8.33
|35 - 48.33| = 13.33
```

Bây giờ lấy trung bình các khoảng cách đó:

```text
cpu_mad = (21.67 + 8.33 + 13.33) / 3 = 14.44
```

Làm tương tự với RAM. Nếu RAM usage là:

- Host A: `68%`
- Host B: `42%`
- Host C: `38%`

Thì:

```text
mean_ram = (68 + 42 + 38) / 3 = 49.33

|68 - 49.33| = 18.67
|42 - 49.33| = 7.33
|38 - 49.33| = 11.33

ram_mad = (18.67 + 7.33 + 11.33) / 3 = 12.44
```

Diễn giải:

- MAD càng cao thì các host càng lệch
- MAD càng thấp thì cluster càng cân bằng

### Ví dụ B. Adaptive weight trong MAD mode

Trong MAD mode bình thường, script dùng CPU MAD và RAM MAD như hai tín hiệu.

Dùng các giá trị ở trên:

```text
cpu_signal = 14.44
ram_signal = 12.44
total = 26.88
```

Tỷ lệ thô:

```text
cpu_share = 14.44 / 26.88 = 0.537
ram_share = 12.44 / 26.88 = 0.463
```

### Signal -> Share -> Weight

Ba khái niệm này liên quan chặt chẽ với nhau, nhưng không phải là một.

`Signal`

- Signal là điểm ưu tiên thô ban đầu trước khi normalize.
- Trong normal MAD mode:

```text
cpu_signal = cpu_mad
ram_signal = ram_mad
```

Nên trong ví dụ này:

```text
cpu_signal = 14.44
ram_signal = 12.44
```

Ở bước này, script mới chỉ đang nói rằng:

- CPU imbalance quan trọng ở mức `14.44` đơn vị
- RAM imbalance quan trọng ở mức `12.44` đơn vị

`Share`

- Share là phần của tổng signal thuộc về CPU hoặc RAM.
- Script chuyển raw signal thành tỷ lệ tương đối:

```text
cpu_share = cpu_signal / (cpu_signal + ram_signal)
ram_share = ram_signal / (cpu_signal + ram_signal)
```

Nên ở đây:

```text
cpu_share = 14.44 / (14.44 + 12.44) = 0.537
ram_share = 12.44 / (14.44 + 12.44) = 0.463
```

Điều đó có nghĩa là:

- CPU chiếm `53.7%` tổng mức ưu tiên
- RAM chiếm `46.3%` tổng mức ưu tiên

`Weight`

- Weight là giá trị cuối cùng thực sự được dùng trong công thức chấm điểm.
- Weight bắt đầu từ share, nhưng có thể bị điều chỉnh bởi policy minimum-weight:

```text
cpu_weight = max(cpu_share, MIN_CPU_WEIGHT)
ram_weight = max(ram_share, MIN_RAM_WEIGHT)
```

Sau đó chúng được normalize lại để tổng bằng đúng `1.0`.

Trong ví dụ này, cả hai share đều đã lớn hơn minimum `0.2`, nên:

```text
cpu_weight = 0.537
ram_weight = 0.463
```

Nói ngắn gọn:

- `signal` trả lời: “CPU hoặc RAM có bao nhiêu mức ưu tiên thô?”
- `share` trả lời: “CPU hoặc RAM chiếm bao nhiêu phần trong tổng mức ưu tiên đó?”
- `weight` trả lời: “Công thức chấm điểm cuối cùng sẽ thực sự dùng tỷ lệ nào?”

Script cũng áp minimum weight:

- minimum CPU weight: `0.2`
- minimum RAM weight: `0.2`

Trong trường hợp này, cả hai tỷ lệ thô đều đã lớn hơn mức tối thiểu, nên final weight gần như giữ nguyên:

```text
cpu_weight = 0.537
ram_weight = 0.463
```

Weighted MAD trở thành:

```text
weighted_mad = (cpu_mad * cpu_weight) + (ram_mad * ram_weight)
weighted_mad = (14.44 * 0.537) + (12.44 * 0.463)
weighted_mad = 13.52
```

Đây là score mà script cố gắng làm giảm.

### Vì sao công thức này tồn tại

Script cần một score cuối cùng để xếp hạng các migration candidate một cách công bằng.

Điều này là cần thiết vì một move có thể cải thiện CPU nhưng làm RAM xấu đi, hoặc cải thiện RAM nhưng làm CPU xấu đi.
Nếu script chấm điểm chỉ bằng CPU, nó có thể vô tình chọn một move làm RAM tệ hơn nhiều.
Nếu script chấm điểm chỉ bằng RAM, nó có thể vô tình chọn một move làm CPU tệ hơn nhiều.

`weighted_mad` giải quyết vấn đề đó bằng cách gộp cả hai chiều vào một score duy nhất có thể so sánh.

Ý nghĩa thực tế:

- `cpu_mad` cho biết CPU đang lệch thế nào giữa các host
- `ram_mad` cho biết RAM đang lệch thế nào giữa các host
- `weighted_mad` gộp cả hai thành một score để script có thể so candidate bằng một con số duy nhất

Bạn có thể hình dung nó như:

```text
overall imbalance score = CPU imbalance contribution + RAM imbalance contribution
```

Trong ví dụ này:

```text
CPU contribution = 14.44 * 0.537 = 7.75
RAM contribution = 12.44 * 0.463 = 5.76
Total weighted_mad = 7.75 + 5.76 = 13.52
```

Tức là script đang nói rằng:

- CPU quan trọng hơn RAM một chút ở thời điểm này
- cả CPU và RAM imbalance đều vẫn quan trọng
- overall cluster imbalance score hiện tại là `13.52`

Điều này quan trọng vì script không chọn move chỉ vì nó cải thiện CPU hoặc chỉ vì nó cải thiện RAM.
Nó chọn move làm giảm score tổng hợp nhiều nhất.

Ví dụ:

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

Dù Candidate 2 cải thiện CPU mạnh hơn, nó làm RAM xấu đi đủ nhiều để kết quả tổng thể tệ hơn.
Vì vậy Candidate 1 là move tốt hơn.

Đây là lý do chính khiến `weighted_mad` tồn tại:

- nếu không có nó, script sẽ cần hai logic riêng cho “best CPU move” và “best RAM move”
- có nó, script có thể xếp hạng mọi candidate nhất quán bằng một score cuối cùng
- `weighted_mad` càng thấp thì càng tốt

### Ví dụ C. Adaptive weight trong pressure mode

Trong pressure mode, script tính weight dựa trên cả:

- spread (`MAD`)
- average utilization (`mean`)

Công thức là:

```text
cpu_signal = (cpu_mad * 0.7) + (cpu_mean * 0.3)
ram_signal = (ram_mad * 0.7) + (ram_mean * 0.3)
```

Giả sử:

- `cpu_mean = 85`
- `ram_mean = 60`
- `cpu_mad = 7`
- `ram_mad = 5`

Khi đó:

```text
cpu_signal = (7 * 0.7) + (85 * 0.3) = 4.9 + 25.5 = 30.4
ram_signal = (5 * 0.7) + (60 * 0.3) = 3.5 + 18.0 = 21.5
```

Bây giờ đổi signal thành weight:

```text
total = 30.4 + 21.5 = 51.9

cpu_weight = 30.4 / 51.9 = 0.586
ram_weight = 21.5 / 51.9 = 0.414
```

### Pressure mode: Signal -> Share -> Weight

Cùng một ý tưởng ba bước đó cũng được dùng trong pressure mode, nhưng đầu vào khác đi.

`Signal`

- Trong pressure mode, script không dùng riêng MAD.
- Thay vào đó nó kết hợp cả spread và overall resource pressure:

```text
cpu_signal = (cpu_mad * 0.7) + (cpu_mean * 0.3)
ram_signal = (ram_mad * 0.7) + (ram_mean * 0.3)
```

Nên trong ví dụ này:

```text
cpu_signal = 30.4
ram_signal = 21.5
```

Ở bước này, script đang nói rằng:

- CPU đáng được chú ý nhiều hơn vì nó nóng hơn trên toàn cục và vẫn đủ lệch để tạo khác biệt
- RAM vẫn quan trọng, nhưng thấp hơn CPU trong trạng thái cụ thể này

`Share`

- Khi đã có signal, script chuyển chúng thành tỷ lệ tương đối:

```text
cpu_share = cpu_signal / (cpu_signal + ram_signal)
ram_share = ram_signal / (cpu_signal + ram_signal)
```

Nên ở đây:

```text
cpu_share = 30.4 / (30.4 + 21.5) = 0.586
ram_share = 21.5 / (30.4 + 21.5) = 0.414
```

Điều đó có nghĩa là:

- CPU chiếm `58.6%` tổng mức ưu tiên trong pressure mode
- RAM chiếm `41.4%` tổng mức ưu tiên trong pressure mode

`Weight`

- Trong ví dụ này, cả hai share đều lớn hơn mức tối thiểu.
- Vì vậy final weight giữ nguyên bằng share:

```text
cpu_weight = 0.586
ram_weight = 0.414
```

Nói ngắn gọn:

- `signal` trả lời: “Sau khi gộp spread và pressure, CPU hoặc RAM có bao nhiêu mức ưu tiên thô?”
- `share` trả lời: “CPU hoặc RAM chiếm bao nhiêu phần trong tổng mức ưu tiên đó?”
- `weight` trả lời: “Công thức weighted MAD cuối cùng sẽ thực sự dùng tỷ lệ nào?”

Diễn giải:

- CPU nhận weight lớn hơn vì aggregate đang bị áp lực CPU nhiều hơn RAM
- điều này khiến script ưu tiên các move giúp giảm CPU pressure mạnh hơn

### Vì sao pressure mode dùng `cpu_signal` và `ram_signal`

Pressure mode tồn tại cho một tình huống rất cụ thể:
aggregate không chỉ đang lệch, mà còn đang chạy nóng trên trung bình.

Điều đó có nghĩa là script không thể chỉ nhìn `MAD`.
Nếu nó chỉ dùng `MAD`, nó có thể bỏ qua việc một resource đã quá bận trên toàn aggregate.

Nhưng script cũng không thể chỉ nhìn `mean`.
Nếu nó chỉ dùng average utilization, nó có thể bỏ qua việc một resource đang phân bố lệch hơn nhiều và vì vậy dễ gây local overload hơn trên một số host.

Vì vậy pressure mode kết hợp cả hai:

- `mean` trả lời: “Resource này nóng đến mức nào trên toàn aggregate?”
- `MAD` trả lời: “Resource này phân bố lệch đến mức nào giữa các host?”
- `signal` trả lời: “Lúc này resource nào nên được ưu tiên hơn khi chấm điểm migration?”

Đó là lý do script dùng:

```text
cpu_signal = (cpu_mad * 0.7) + (cpu_mean * 0.3)
ram_signal = (ram_mad * 0.7) + (ram_mean * 0.3)
```

Đây là một heuristic, không phải là một định luật toán học bắt buộc.
Lựa chọn thiết kế ở đây là:

- vẫn giữ balancing là mục tiêu chính, nên `MAD` có phần lớn hơn (`0.7`)
- nhưng đưa cả overall pressure vào quyết định, nên `mean` vẫn đóng góp (`0.3`)

Nói cách khác, pressure mode có nghĩa là:

```text
vẫn rebalance cluster,
nhưng cho resource đang chịu áp lực thực tế cao hơn mức ưu tiên lớn hơn.
```

#### Ví dụ trade-off cho `0.7 / 0.3`

Hai số `0.7` và `0.3` là một lựa chọn policy.
Chúng quyết định pressure mode sẽ cư xử giống một rebalancer nhiều hơn, hay giống một cơ chế ưu tiên theo pressure nhiều hơn.

Bạn có thể hình dung các lựa chọn khác như sau:

- `0.9 / 0.1`
  - rất nghiêng về MAD
  - pressure mode sẽ gần giống normal balancing hơn nhiều
  - mức nóng chung của cluster chỉ ảnh hưởng nhẹ tới ưu tiên CPU hay RAM
- `0.8 / 0.2`
  - vẫn rất balance-first
  - pressure có tác động, nhưng chỉ như một hiệu chỉnh nhỏ
- `0.7 / 0.3`
  - hành vi hiện tại
  - balancing vẫn là mục tiêu chính, nhưng overall pressure ảnh hưởng rõ tới mức ưu tiên
- `0.6 / 0.4`
  - nhạy hơn với pressure
  - ưu tiên CPU hay RAM có thể đổi nhanh hơn khi average của aggregate tăng
- `0.5 / 0.5`
  - nhấn mạnh đều giữa spread và average pressure
  - pressure mode bắt đầu bớt giống balancing và giống general resource prioritization hơn

Dưới đây là một ví dụ cụ thể:

```text
cpu_mean = 90
ram_mean = 60
cpu_mad = 4
ram_mad = 8
```

Với `0.9 / 0.1`:

```text
cpu_signal = (4 * 0.9) + (90 * 0.1) = 12.6
ram_signal = (8 * 0.9) + (60 * 0.1) = 13.2
```

Kết quả:

- RAM vẫn thắng
- CPU average dù rất cao vẫn chưa đủ để lật ưu tiên từ RAM MAD lớn hơn

Với `0.7 / 0.3`:

```text
cpu_signal = (4 * 0.7) + (90 * 0.3) = 29.8
ram_signal = (8 * 0.7) + (60 * 0.3) = 23.6
```

Kết quả:

- CPU bây giờ thắng
- overall CPU pressure đã đủ mạnh để tạo khác biệt

Với `0.5 / 0.5`:

```text
cpu_signal = (4 * 0.5) + (90 * 0.5) = 47.0
ram_signal = (8 * 0.5) + (60 * 0.5) = 34.0
```

Kết quả:

- CPU thắng với khoảng cách còn lớn hơn
- pressure mode trở nên bị chi phối nhiều hơn bởi mức nóng chung của cluster

Vì vậy trade-off thực tế là:

- hệ số MAD càng lớn -> hành vi càng bảo thủ, càng balance-first
- hệ số mean càng lớn -> hành vi càng nhạy với pressure, càng heat-aware
- `0.7 / 0.3` là điểm trung dung: vẫn còn mang bản chất balancing, nhưng không bỏ qua pressure thực tế của aggregate

Vì sao không `0.5 / 0.5`?

- Vì như vậy pressure mode sẽ dễ trôi sang kiểu hành vi "ưu tiên average pressure là chính".
- Trong khi script này về bản chất vẫn là một balancer, không phải pure capacity-pressure solver.

Vì sao không `0.9 / 0.1`?

- Vì như vậy pressure mode sẽ quá giống normal MAD mode.
- Nó sẽ gần như không phản ánh đủ việc aggregate đang nóng trên toàn cục.

Vậy vì sao là `0.7 / 0.3`?

- Nó vẫn thiên về balancing.
- Nhưng nó có đủ độ nhạy với pressure thật để đổi ưu tiên CPU hay RAM khi aggregate đang nóng.

Nói thẳng hơn:

- đây là một policy choice, không phải scientific constant
- đây là một giá trị mặc định hợp lý nếu bạn muốn hành vi bảo thủ
- và hoàn toàn có thể tune lại theo thực tế cluster của bạn

#### Vì sao không dùng chỉ MAD?

Giả sử:

- `cpu_mean = 90`
- `ram_mean = 60`
- `cpu_mad = 4`
- `ram_mad = 8`

Nếu chỉ dùng MAD:

```text
CPU priority = 4
RAM priority = 8
```

Điều đó sẽ khiến script tập trung vào RAM.
Nhưng về mặt vận hành, CPU mới là vấn đề khẩn cấp hơn vì aggregate đang rất nóng về CPU trên toàn cục.

Với pressure formula:

```text
cpu_signal = (4 * 0.7) + (90 * 0.3) = 2.8 + 27.0 = 29.8
ram_signal = (8 * 0.7) + (60 * 0.3) = 5.6 + 18.0 = 23.6
```

Bây giờ CPU có mức ưu tiên cao hơn, phản ánh rủi ro thực tế tốt hơn.

#### Vì sao không dùng chỉ mean?

Giả sử:

- `cpu_mean = 82`
- `ram_mean = 78`
- `cpu_mad = 3`
- `ram_mad = 14`

Nếu chỉ dùng mean:

```text
CPU priority = 82
RAM priority = 78
```

Điều đó sẽ khiến script tập trung vào CPU.
Nhưng RAM đang phân bố lệch hơn rất nhiều, nên dễ tạo hotspot cục bộ trên một hoặc hai host hơn.

Với pressure formula:

```text
cpu_signal = (3 * 0.7) + (82 * 0.3) = 2.1 + 24.6 = 26.7
ram_signal = (14 * 0.7) + (78 * 0.3) = 9.8 + 23.4 = 33.2
```

Bây giờ RAM có mức ưu tiên cao hơn, phản ánh đúng hơn vấn đề phân bố.

#### Phiên bản ngắn gọn

Trong pressure mode:

- `mean` cho script biết resource nào đang nóng hơn trên toàn cục
- `MAD` cho script biết resource nào đang lệch hơn
- `signal` gộp cả hai để script chọn weight tốt hơn

Sau đó các signal này được chuyển thành:

```text
cpu_weight = cpu_signal / (cpu_signal + ram_signal)
ram_weight = ram_signal / (cpu_signal + ram_signal)
```

Các weight đó sau cùng sẽ được dùng trong `weighted_mad` để xếp hạng migration candidate.

### Cách tính weight trong mọi mode

Script không tính weight giống nhau ở mọi mode.
Bảng dưới đây cho thấy chính xác hành vi đó.

| Mode | Cách chọn weight | Kết quả |
| --- | --- | --- |
| Normal MAD mode | Dùng `cpu_mad` và `ram_mad` làm signal, rồi normalize | Adaptive |
| Pressure mode | Dùng `(mad * 0.7) + (mean * 0.3)` làm signal, rồi normalize | Adaptive |
| RAM hotspot | Giá trị cố định | `cpu_weight = 0.2`, `ram_weight = 0.8` |
| RAM critical hotspot | Giá trị cố định | `cpu_weight = 0.1`, `ram_weight = 0.9` |
| CPU hotspot | Giá trị cố định | `cpu_weight = 0.8`, `ram_weight = 0.2` |
| Zero-signal fallback | Nếu cả hai signal đều bằng `0` | `cpu_weight = 0.5`, `ram_weight = 0.5` |

### Ví dụ D. Normal mode với minimum-weight protection

Trong normal mode, script bắt đầu từ:

```text
cpu_signal = cpu_mad
ram_signal = ram_mad
```

Giả sử:

- `cpu_mad = 1`
- `ram_mad = 9`

Initial shares:

```text
cpu_share = 1 / (1 + 9) = 0.10
ram_share = 9 / (1 + 9) = 0.90
```

Nhưng script áp minimum weight:

- `MIN_CPU_WEIGHT = 0.2`
- `MIN_RAM_WEIGHT = 0.2`

Nên trước khi normalize:

```text
cpu_weight = max(0.10, 0.2) = 0.2
ram_weight = max(0.90, 0.2) = 0.9
```

Bây giờ normalize để tổng bằng `1`:

```text
weight_sum = 0.2 + 0.9 = 1.1

cpu_weight = 0.2 / 1.1 = 0.182
ram_weight = 0.9 / 1.1 = 0.818
```

Diễn giải:

- RAM vẫn chiếm ưu thế vì RAM imbalance lớn hơn nhiều
- CPU không bị phép rơi về mức gần như không còn ý nghĩa
- điều này giữ cho score không trở thành hoàn toàn một chiều

### Ví dụ E. RAM hotspot mode

Giả sử một host vượt ngưỡng RAM hotspot.
Trong mode này, script không suy ra weight từ MAD hiện tại nữa.
Nó dùng weight cố định:

```text
cpu_weight = 0.2
ram_weight = 0.8
```

Nếu một move mô phỏng cho ra:

```text
new_cpu_mad = 8
new_ram_mad = 6
```

Thì:

```text
new_weighted_mad = (8 * 0.2) + (6 * 0.8)
                 = 1.6 + 4.8
                 = 6.4
```

Diễn giải:

- RAM là vấn đề khẩn cấp chính
- CPU vẫn quan trọng, nhưng ít hơn nhiều
- move nào làm RAM imbalance giảm sẽ được ưu tiên, miễn là không tạo side effect CPU xấu

### Ví dụ F. RAM critical hotspot mode

Nếu một host vượt ngưỡng critical RAM, RAM còn được ưu tiên mạnh hơn nữa:

```text
cpu_weight = 0.1
ram_weight = 0.9
```

Ví dụ:

```text
new_cpu_mad = 10
new_ram_mad = 5

new_weighted_mad = (10 * 0.1) + (5 * 0.9)
                 = 1.0 + 4.5
                 = 5.5
```

Diễn giải:

- trong critical RAM hotspot, thuật toán cố ý theo hướng RAM-first
- CPU imbalance vẫn được tính, nhưng chỉ nhẹ

### Ví dụ G. CPU hotspot mode

Nếu một host vượt ngưỡng CPU hotspot, fixed weight sẽ đảo lại:

```text
cpu_weight = 0.8
ram_weight = 0.2
```

Ví dụ:

```text
new_cpu_mad = 4
new_ram_mad = 9

new_weighted_mad = (4 * 0.8) + (9 * 0.2)
                 = 3.2 + 1.8
                 = 5.0
```

Diễn giải:

- CPU là ưu tiên chính
- RAM vẫn đóng góp vào score
- script vẫn tránh các move chỉ sửa CPU bằng cách tạo ra tradeoff RAM quá xấu

### Ví dụ H. Zero-signal fallback

Nếu cả hai signal đều bằng `0`, script fallback về:

```text
cpu_weight = 0.5
ram_weight = 0.5
```

Đây chủ yếu là edge case để phòng thủ.
Nó có nghĩa là CPU và RAM được đối xử ngang nhau nếu không có signal nào cho script biết nên ưu tiên bên nào.

### Ví dụ I. Vì sao một move có thể làm một host tốt hơn nhưng vẫn thua tổng thể

Giả sử weighted MAD hiện tại là:

```text
current_weighted_mad = 13.52
```

Sau khi mô phỏng một VM move, phần trăm host mới tạo ra:

```text
new_cpu_mad = 10.20
new_ram_mad = 11.40
cpu_weight = 0.537
ram_weight = 0.463
```

Khi đó:

```text
new_weighted_mad = (10.20 * 0.537) + (11.40 * 0.463)
new_weighted_mad = 10.76
```

Improvement:

```text
improvement = 13.52 - 10.76 = 2.76
```

Đây là một candidate mạnh.

Nhưng nếu một move khác cho ra:

```text
new_weighted_mad = 13.49
improvement = 0.03
```

Thì script vẫn có thể từ chối nếu threshold là `0.05`.

Vì vậy script không hỏi:

- “Một host có tốt hơn không?”

Nó hỏi:

- “Toàn aggregate có tốt hơn đủ nhiều để justify migration không?”

## Mô phỏng move trong ba trường hợp chính

Các ví dụ dưới đây đi qua một simulated move trong từng nhóm quyết định chính:

- normal MAD balancing
- pressure mode
- hotspot mode

Các ví dụ này được đơn giản hóa có chủ đích, nhưng vẫn bám đủ sát logic của script để giải thích cách một candidate được đánh giá.

### Simulation 1. Normal MAD mode

Giả sử aggregate có ba host:

- Host A: CPU `70%`, RAM `68%`
- Host B: CPU `40%`, RAM `42%`
- Host C: CPU `35%`, RAM `38%`

Không có hotspot và không có pressure, nên script giữ nguyên ở normal MAD mode.

Từ phần tính toán phía trên:

- `cpu_mad = 14.44`
- `ram_mad = 12.44`
- `cpu_weight = 0.537`
- `ram_weight = 0.463`
- `baseline_weighted_mad = 13.52`

Công thức baseline:

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

Bây giờ mô phỏng việc move một VM từ Host A sang Host B.
Giả sử VM đó gây ra:

- CPU impact: `12%`
- RAM impact: `12%`

Sau move mô phỏng:

```text
CPU: [70, 40, 35] -> [58, 52, 35]
RAM: [68, 42, 38] -> [56, 54, 38]
```

Tính lại MAD:

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

Tính lại weighted MAD:

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

Diễn giải:

- các host trở nên đều hơn nhiều
- cả CPU và RAM đều được cải thiện
- đây chính là kiểu move mà normal MAD mode muốn tìm

### Simulation 2. Pressure mode

Giả sử aggregate có ba host:

- Host A: CPU `84%`, RAM `74%`
- Host B: CPU `82%`, RAM `70%`
- Host C: CPU `80%`, RAM `68%`

Không có hotspot:

- max CPU là `84%`, vẫn thấp hơn ngưỡng CPU hotspot
- max RAM là `74%`, vẫn thấp hơn ngưỡng RAM hotspot

Nhưng CPU trung bình lại cao:

```text
cpu_mean = (84 + 82 + 80) / 3 = 82.00
```

Vì vậy script chuyển sang `pressure` mode.

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

Pressure signal:

```text
cpu_mad = 1.33
ram_mad = 2.22
cpu_signal = (1.33 * 0.7) + (82.00 * 0.3) = 25.53
ram_signal = (2.22 * 0.7) + (70.67 * 0.3) = 22.76
```

Weight:

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

Bây giờ mô phỏng move một VM từ Host A sang Host C.
Giả sử move đó làm phần trăm host đổi như sau:

```text
CPU: [84, 82, 80] -> [82, 82, 82]
RAM: [74, 70, 68] -> [71, 70, 71]
```

Tính lại MAD:

```text
new_cpu_mean = (82 + 82 + 82) / 3 = 82.00
new_ram_mean = (71 + 70 + 71) / 3 = 70.67

new_cpu_mad = (|82 - 82.00| + |82 - 82.00| + |82 - 82.00|) / 3
            = 0.00

new_ram_mad = (|71 - 70.67| + |70 - 70.67| + |71 - 70.67|) / 3
            = (0.33 + 0.67 + 0.33) / 3
            = 0.44
```

Weighted MAD mới:

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

Diễn giải:

- cluster đang nóng trên trung bình, nên pressure mode là đúng
- move vẫn được xếp hạng bằng improvement trong weighted MAD
- CPU được ưu tiên hơn RAM một chút vì CPU đang là resource nóng hơn trên toàn cục

### Simulation 3. Hotspot mode

Giả sử aggregate có ba host:

- Host A: CPU `55%`, RAM `88%`
- Host B: CPU `50%`, RAM `52%`
- Host C: CPU `48%`, RAM `49%`

Điều này lập tức kích hoạt `ram_critical_hotspot`.

Trong mode này, weight là cố định:

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

Bây giờ mô phỏng việc move một VM từ Host A sang Host B.
Giả sử VM đó làm phần trăm host đổi như sau:

```text
cpu_mad = 2.67
ram_mad = 16.67
CPU: [55, 50, 48] -> [51, 54, 48]
RAM: [88, 52, 49] -> [70, 70, 49]
```

Đầu tiên kiểm tra hotspot guardrail:

- source hotspot được cải thiện: `88% -> 70%`
- destination không trở thành RAM hotspot: `70% < 75%`

Vì vậy move này được phép đi tiếp.

Tính lại MAD:

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

Weighted MAD mới:

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

Diễn giải:

- host nóng nhất được hạ nhiệt mạnh
- RAM chiếm ưu thế trong score vì đây là một tình huống RAM emergency
- move là hợp lệ vì nó giảm hotspot mà không tạo hotspot mới ở nơi khác

## Kịch bản vận hành

Các kịch bản dưới đây vẫn được đơn giản hóa, nhưng được viết theo góc nhìn của operator hơn là chỉ đi theo công thức toán.

### Ví dụ 1. Cân bằng MAD đơn giản

Giả sử một aggregate có ba host:

- Host A: CPU `70%`, RAM `68%`
- Host B: CPU `40%`, RAM `42%`
- Host C: CPU `35%`, RAM `38%`

Không có hotspot và không có pressure, nên script dùng MAD mode.

Bây giờ giả sử một VM trên Host A đang tiêu thụ xấp xỉ:

- `2.0` real CPU cores
- `6.0` real GB RAM

Script mô phỏng việc move VM đó từ Host A sang Host B.

Sau move mô phỏng:

- Host A bớt tải hơn
- Host B tải tăng lên
- phần trăm của các host trở nên đều hơn về tổng thể

Nếu weighted MAD mới thấp hơn trước đủ nhiều, move đó trở thành một candidate mạnh.

Nếu cùng VM đó được move từ Host A sang Host C và cho ra weighted MAD còn thấp hơn nữa, thì Host C sẽ trở thành destination tốt hơn.

### Ví dụ 2. Xử lý hotspot

Giả sử aggregate này:

- Host A RAM: `88%`
- Host B RAM: `52%`
- Host C RAM: `49%`

Điều này lập tức kích hoạt `ram_critical_hotspot`.

Script sẽ tập trung vào Host A như source host trước tiên.

Bây giờ hãy xét hai candidate VM trên Host A:

- VM 1 làm Host A giảm từ `88%` xuống `82%`, nhưng đẩy Host B lên `77%`
- VM 2 làm Host A giảm từ `88%` xuống `84%`, và giữ Host B ở `70%`

Dù VM 1 nhìn có vẻ “lớn” hơn, nó vẫn bị loại vì destination sẽ trở thành RAM hotspot.

VM 2 an toàn hơn vì:

- host hotspot được cải thiện
- destination không vượt ngưỡng hotspot

Đó chính là điều mà hotspot guardrail được thiết kế để ép buộc.

### Ví dụ 3. Khả thi về kỹ thuật nhưng vẫn bị từ chối

Giả sử một move được phép bởi:

- capacity
- server-group policy
- destination readiness

Nhưng sau khi mô phỏng, improvement rất nhỏ:

- current weighted MAD: `6.20`
- post-move weighted MAD: `6.17`
- improvement: `0.03`

Nếu threshold yêu cầu là `0.05`, script sẽ từ chối move đó.

Vì vậy kết quả là:

- có candidate tồn tại
- nhưng không có valid move nào vượt threshold

Đó là lý do output monitor có thể nói cluster đang unbalanced nhưng vẫn từ chối migrate.

### Ví dụ 4. Vì sao một VM có thể bị skip dù nhìn có vẻ hữu ích

Một VM có thể trông như một balancing candidate hoàn hảo nhưng vẫn bị skip vì:

- nó vừa được migrate gần đây và vẫn đang trong cooldown
- lần migration trước của nó bị fail và failed-move cooldown vẫn còn
- host đích không phải `nova-compute` khỏe mạnh
- destination không có đủ allocatable flavor capacity
- affinity hoặc anti-affinity rules sẽ bị vi phạm

Đây là chủ ý thiết kế. Script ưu tiên hành vi an toàn và dễ dự đoán hơn là rebalance quá mạnh tay.

## Cách đọc một quyết định dưới góc nhìn operator

Khi nhìn vào output, hãy đọc theo thứ tự này:

1. Mode
   - cho biết script đang phản ứng với hotspot, pressure, hay spread
2. Status / severity
   - cho biết đây là trạng thái khỏe, warning-level, hay critical
3. Candidate line
   - cho biết có move nào tồn tại không và nó có vượt threshold không
4. Reason
   - giải thích vì sao không move nào được chấp nhận nếu cluster vẫn trông không khỏe
5. Search scope
   - cho biết script có bám hotspot host trước hay dùng full aggregate search

Như vậy sẽ dễ phân biệt giữa:

- “cluster đang ổn”
- “cluster chưa lý tưởng nhưng không có safe move”
- “cluster không khỏe và có safe corrective move”

## Tính năng an toàn

Script này được thiết kế một cách bảo thủ có chủ đích.

Nó bao gồm:

- cooldown cho VM sau migration thành công: `1800` giây
- cooldown cho VM sau failed migration: `3600` giây
- kiểm tra readiness của destination host bằng `nova-compute`
- kiểm tra server-group policy theo mặc định
- theo dõi kết quả migration với timeout detection
- tùy chọn reset từ `ERROR` về `ACTIVE` sau failed migration
- persistent cooldown state lưu trên đĩa
- migration event logging ra JSONL

## Chế độ monitoring và execution

### `--monitor-only`

Dùng mode này khi bạn muốn kết quả health tương thích Icinga/Nagios mà không thay đổi gì.

Nó in ra một bản tóm tắt dễ đọc và thoát với:

- `0` cho `OK`
- `1` cho `WARNING`
- `2` cho `CRITICAL`
- `3` cho `UNKNOWN`

### `--dry-run`

Dùng mode này để xem script sẽ migrate gì mà không thực hiện migration thật.

Nó vẫn đánh giá aggregate đầy đủ và ghi một event `dry_run` nếu tồn tại valid move.

### Default execution mode

Nếu không dùng `--monitor-only` hoặc `--dry-run`, script sẽ:

- đánh giá aggregate,
- hiển thị move được đề xuất,
- hỏi xác nhận,
- thực hiện một live migration,
- theo dõi kết quả,
- cập nhật cooldown state,
- và có thể gửi email alert.

Dùng `-y` hoặc `--yes` để bỏ qua prompt xác nhận.

## Yêu cầu

Bạn cần:

- Python 3
- quyền truy cập OpenStack qua `openstacksdk`
- một OpenStack cloud profile tên `openstack`
- các Prometheus endpoint và metric có thể truy cập được từ máy chạy script

Các package Python script dùng:

- `openstacksdk`
- `requests`
- `python-dotenv`
- `click`
- `urllib3`

Cài bằng package manager bạn muốn, ví dụ:

```bash
pip install openstacksdk requests python-dotenv click urllib3
```

## Yêu cầu OpenStack

Script kết nối với:

- cloud name: `openstack`
- compute microversion: `2.87`

Hãy chắc chắn môi trường của bạn có `clouds.yaml` hợp lệ hoặc cấu hình OpenStack tương đương, trong đó định nghĩa một cloud tên `openstack`.

## Yêu cầu Prometheus

Script cần dữ liệu Prometheus cho:

- host CPU usage
- host RAM usage
- OpenStack placement allocation ratio
- per-VM CPU usage
- per-VM RAM usage

Default query giả định có metric từ:

- một node exporter job
- một libvirt exporter job
- một OpenStack exporter job

Script cũng kỳ vọng một số label cụ thể tồn tại trong kết quả Prometheus:

- `alias` cho query usage ở mức host
- `hostname` cho query allocation ratio
- `instanceId` và `host` cho query per-VM

Nếu bạn bật host-side VM RSS cho RAM impact scoring, RSS metric cũng phải expose:

- `instanceId`
- `host`

Nếu metric name hoặc label của bạn khác, hãy override Prometheus query trong config file.

## Cấu hình

Script nạp cấu hình từ một file theo kiểu dotenv.

Đường dẫn mặc định của file config:

```bash
/etc/loadleveller-secrets.conf
```

Các biến môi trường được hỗ trợ:

- `PROMETHEUS_QUERY_URL`
- `PROMETHEUS_MEM_USED`
- `PROMETHEUS_CPU_USED`
- `PROMETHEUS_VM_RAM_RSS`
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

Ví dụ:

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

Optional RSS query override:

```dotenv
PROMETHEUS_VM_RAM_RSS=avg_over_time(libvirt_domain_stat_memory_rss_bytes{job="Prod-Openstack-LibVirt-Exporter-DC11", host=~"__HOST_RE__"}[5m])
```

Notes:

- Dùng `__HOST_RE__` làm placeholder cho aggregate host regex.
- Metric value được kỳ vọng là bytes.
- Nếu biến này để trống, script sẽ tiếp tục dùng guest-RAM-based RAM impact model gốc, trừ khi bạn bật `--use-vm-ram-rss` để dùng built-in default query.

## Các file script sẽ ghi ra

Theo mặc định, script ghi:

- cooldown state: `/var/log/loadleveller_vm_cooldown.json`
- migration events: `/var/log/loadleveller_migration_events.jsonl`

Bạn có thể override cả hai khi chạy:

- `--cooldown-file`
- `--events-file`

## Cách dùng command line

Hiển thị tất cả option:

```bash
python wloadbalancer.py --help
```

Monitor tất cả aggregate:

```bash
python wloadbalancer.py --monitor-only
```

Monitor một aggregate:

```bash
python wloadbalancer.py --monitor-only --aggregate my-compute-aggregate
```

Xem best move mà không thay đổi gì:

```bash
python wloadbalancer.py --dry-run --aggregate my-compute-aggregate
```

Chạy interactive và hỏi trước khi migrate:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate
```

Chạy non-interactive:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --yes
```

Gửi email alert chỉ cho migration fail:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --send-error
```

Gửi email alert cho cả migration thành công và thất bại:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --send-all
```

Dùng host-side VM RSS cho RAM impact scoring:

```bash
python wloadbalancer.py --dry-run --aggregate my-compute-aggregate --use-vm-ram-rss
```

Chỉ tắt server-group checks nếu bạn hiểu rõ rủi ro:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --no-server-groups
```

## Quy trình operator được khuyến nghị

Trong production, workflow an toàn nhất là:

1. Bắt đầu bằng `--monitor-only` để hiểu trạng thái cluster hiện tại.
2. Chạy `--dry-run` để kiểm tra candidate được đề xuất và improvement kỳ vọng.
3. Chạy interactive không có `-y` ở lần đầu tiên.
4. Bật `--send-error` hoặc `--send-all` nếu bạn muốn có email notification.
5. Dùng đường dẫn `--cooldown-file` và `--events-file` tùy chỉnh nếu môi trường của bạn cần lưu vào vị trí khác.

## Hiểu output như thế nào

Bản tóm tắt monitoring bao gồm:

- decision mode được chọn
- peak và average của cluster
- CPU và RAM MAD
- adaptive weight
- có valid move hay không
- best candidate move
- reason nếu không chọn move nào

Khi một migration được đề xuất, runtime output còn hiển thị:

- source và destination host
- CPU và RAM impact ước lượng
- phần trăm host sau migration
- weighted MAD sau migration
- improvement của migration round đó

## Định dạng event log

File JSONL events là append-only và ghi lại các kết quả vận hành như:

- `dry_run`
- `cancelled`
- `skipped_destination_not_ready`
- `submit_or_monitor_failed`
- `migration_success`
- `migration_timeout`
- `migration_failed`
- `stuck_active`

Mỗi event có thể chứa:

- aggregate name
- result
- detail status
- severity
- duration
- timestamp
- improvement value
- VM metadata
- source và destination host

## Ghi chú vận hành

- Script không rebalance liên tục cho tới khi cluster hoàn hảo.
- Nó chọn một best move cho mỗi aggregate trong mỗi lần chạy.
- Nó ưu tiên improvement có thể dự đoán hơn là churn mạnh tay.
- Nó được thiết kế để được chạy lặp lại bởi operator, scheduler, hoặc monitoring system.

## Xử lý sự cố

### Script báo không có aggregate nào khớp lựa chọn

Hãy kiểm tra aggregate name truyền qua `--aggregate` và xác nhận nó tồn tại trong OpenStack.

### Script bỏ qua một destination host

Các lý do phổ biến nhất là:

- `nova-compute` không được enable
- `nova-compute` không ở trạng thái up
- host đang bị forced down
- destination thiếu allocatable capacity

### Script không tìm thấy valid move

Các lý do phổ biến:

- mọi candidate đều thấp hơn improvement threshold
- server-group rules chặn move
- cooldown chặn VM
- destination host không sẵn sàng
- không có migration nào khả thi về mặt kỹ thuật

### Migration có vẻ bắt đầu nhưng bị timeout

Mặc định script theo dõi trạng thái live migration trong tối đa `600` giây. Timeout nghĩa là VM không đi tới trạng thái thành công hoặc thất bại cuối cùng trong khoảng thời gian đó.

## Tóm tắt

Hãy dùng script này khi bạn muốn một công cụ balancing thận trọng, dễ giải thích, và chỉ thực hiện từng bước một cho OpenStack aggregate.

Nó phù hợp nhất với các môi trường mà:

- cho phép live migration,
- metric Prometheus được tin cậy,
- server-group policy có ý nghĩa,
- và operator ưu tiên balancing có kiểm soát, quan sát được, hơn là automation quá nhanh.
