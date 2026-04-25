# Phím Tắt TUI `robot`

## Mở ứng dụng

```powershell
robot
```

Hoặc:

```powershell
robot-tui
```

## Giao diện Hacker Pro

TUI dùng layout 3 vùng để xem nhiều thông tin cùng lúc mà không phải mở từng file:

| Vùng | Nội dung | Cách dùng nhanh |
|---|---|---|
| Control deck bên trái | Dataset, workflow, variant, preset, budget, picker HTML/JSON/log | Chọn cấu hình, bấm `1` để chạy hoặc `F5` để làm mới artifact |
| Workspace ở giữa | Status bar, command palette, output stack của workflow/log/history | Đọc kết quả chi tiết, bật `5` nếu cần xem JSON thô |
| Intel rail bên phải | Session, best model, FAMD cluster, artifact inventory, workflow map, fast path | Xem tóm tắt nhanh mà không cần scroll qua toàn bộ output |

Hero phía trên đã được rút gọn để dành nhiều diện tích hơn cho dashboard. Các panel bên phải đọc trực tiếp các artifact hiện có trong `results/`, ví dụ:

- `results/best_model_selection.json`
- `results/model_comparison_report.json`
- `results/visualizations/famd_clustering_results.json`
- `results/visualizations/famd_summary.json`

## Hotkeys hiện tại

| Phím | Chức năng |
|---|---|
| `1` | Chạy workflow đang chọn |
| `2` | Mở HTML artifact mới nhất |
| `3` | Mở HTML artifact đang chọn trong `html picker` |
| `4` | Load JSON history |
| `5` | Bật hoặc tắt forensic JSON dump |
| `6` | Load console log đã lưu |
| `P` | Chạy dự đoán sàng lọc từ form prediction và artifact best model |
| `F5` | Refresh danh sách HTML / JSON / LOG artifact |
| `r` | Chạy lại workflow gần nhất |
| `:` | Mở command palette |
| `q` | Thoát TUI |

## Intel rail đọc gì?

| Panel | Ý nghĩa |
|---|---|
| `SESSION` | Trạng thái hiện tại, workflow, variant, preset, budget, kết quả đang load |
| `BEST MODEL` | Model được chọn, ROC-AUC, PR-AUC, F1, Brier score, gap với model đứng sau và lift so với Dummy |
| `FAMD CLUSTER` | Số mẫu, số component, K-Means tốt nhất, DBSCAN có tìm được cụm mật độ ổn định không |
| `ARTIFACTS` | Số HTML/JSON/log đang index và artifact mới nhất |
| `WORKFLOW MAP` | Workflow hiện tại thuộc modern hay legacy, có hỗ trợ variant/budget/export HTML không |
| `FAST PATH` | Nhắc nhanh các phím quan trọng nhất |

## Khi dùng phím `6`

Sau khi load console log, workspace ở giữa hiển thị theo thứ tự:

1. `CONSOLE TRACE`
2. `OVERALL ASSESSMENT`
3. `FLAG BENCHMARK`

`OVERALL ASSESSMENT` và `FLAG BENCHMARK` được dựng theo workflow đang xem, không chỉ riêng `eda`.

## Command palette

Nhấn `:` rồi gõ:

```text
:help
:run
:rerun
:html latest
:html open
:history latest
:history load
:log latest
:log load
:predict
:json toggle
:json on
:json off
:refresh artifacts
:set workflow profile
:set workflow analysis
:set variant A
:set variant B
:set preset quick
:set preset research
:set budget default
:set budget auto
:set dataset Student_Depression_Dataset.csv
```

## Luồng thao tác nhanh

Xem profile:

1. Chọn `workflow = profile`
2. Bấm `1`

Chạy pipeline:

1. Chọn `workflow = run`
2. Chọn `variant = A` hoặc `B`
3. Chọn `preset`
4. Bấm `1`

Dự đoán một hồ sơ:

1. Train artifact trước bằng `robot train-best --dataset Student_Depression_Dataset.csv --preset research --budget auto`
2. Mở TUI bằng `robot`
3. Điền các trường trong vùng `prediction`
4. Bấm `P` hoặc gõ `:predict`
5. Đọc `probability`, `threshold`, `model`, `profile` và kết luận sàng lọc trong workspace

Xem lại artifact cũ:

1. Chọn file trong `history json` rồi bấm `4`
2. Hoặc chọn file trong `console log` rồi bấm `6`
3. Nếu muốn mở dashboard tương ứng thì chọn ở `html picker` rồi bấm `3`

## Cuộn và điều hướng

- Control deck, workspace và intel rail đều có scrollbar riêng.
- Dùng chuột hoặc con lăn để cuộn từng vùng đang trỏ vào.
- Dùng `PgUp`, `PgDn`, `Home`, `End` để cuộn đồng bộ cả 3 vùng.

## Mẹo dùng nhanh

- Nếu `html picker` đang trống, bấm `F5` rồi bấm `3`.
- Nếu muốn mở HTML mới nhất ngay, bấm `2`.
- Nếu chỉ muốn kiểm tra lại kết quả cũ, ưu tiên `4` và `6` thay vì chạy lại workflow.
- Nếu đang cần xem model nào mạnh/yếu, nhìn `BEST MODEL` ở intel rail trước, rồi mở `results/final_report.html` hoặc `results/model_evidence_metrics.html` để xem bằng chứng đầy đủ.

## Tài liệu liên quan

- [CLI_APP.md](./CLI_APP.md)
- [README.md](./README.md)
