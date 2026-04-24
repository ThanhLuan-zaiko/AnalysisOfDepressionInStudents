# Hướng Dẫn CLI App `robot`

## Mục tiêu

`robot` là lệnh chung để:

- Mở TUI theo phong cách terminal monitor.
- Chạy workflow modern và legacy trong cùng một hub.
- Xem lại artifact cũ mà không cần chạy lại.
- Mở HTML report/dashboard trên trình duyệt mặc định.

Nếu máy đã có `textual`, `robot` sẽ mở TUI. Nếu chưa có, lệnh sẽ rơi về console fallback.

## Khởi động nhanh

Trong thư mục repo:

```powershell
.\setup_with_gpu.ps1
```

Sau đó mở PowerShell mới rồi chạy:

```powershell
robot
```

## TUI Hacker Pro

TUI mới dùng layout 3 vùng để giảm cảm giác bị ngợp khi kết quả dài:

| Vùng | Vai trò |
|---|---|
| Control deck | Chọn dataset, workflow, variant, preset, budget, HTML/JSON/log artifact |
| Workspace | Xem status, command palette, output stack, JSON dump, console trace |
| Intel rail | Xem nhanh session, best model, FAMD clustering, artifact inventory, workflow map |

Intel rail đọc trực tiếp các artifact trong `results/`:

- `best_model_selection.json`: model được chọn, profile, metric và lý do.
- `model_comparison_report.json`: ranking Dummy, Logistic Regression, GAM, CatBoost.
- `visualizations/famd_clustering_results.json`: K-Means/DBSCAN trên tọa độ FAMD.
- `visualizations/famd_summary.json`: variance/cumulative variance của các component FAMD.

Nhờ vậy bạn có thể giữ workspace cho log/report chi tiết, còn rail bên phải dùng để xem model nào mạnh/yếu và FAMD clustering đang nói gì.

## Quy ước cấu hình

- `A = safe = conservative`
- `B = full = default`
- `budget = default | auto`

`budget=auto` tự chỉnh ngân sách train cho các model mà workflow đang dùng.

## Các lệnh chính

Mở TUI:

```powershell
robot
robot-tui
```

Profile dataset:

```powershell
robot profile --dataset Student_Depression_Dataset.csv --console-only
```

Chạy pipeline modern:

```powershell
robot run --dataset Student_Depression_Dataset.csv --variant A --preset quick --budget auto
robot run --dataset Student_Depression_Dataset.csv --variant B --preset research
```

So sánh A/B:

```powershell
robot compare --dataset Student_Depression_Dataset.csv --preset quick --console-only
```

Chạy workflow bất kỳ trong hub:

```powershell
robot task eda --dataset Student_Depression_Dataset.csv
robot task famd --dataset Student_Depression_Dataset.csv
robot task models --dataset Student_Depression_Dataset.csv --variant A --budget auto
robot task report --dataset Student_Depression_Dataset.csv
robot task analysis --dataset Student_Depression_Dataset.csv --variant A --budget auto
```

Mở HTML:

```powershell
robot open-html --latest
robot open-html results\final_report.html
```

Xem lại JSON history:

```powershell
robot history --latest
robot history results\app\run_safe_quick.json
```

## Artifact review trong TUI

TUI có 3 lane review artifact:

- `html picker`: chọn file `.html` rồi mở bằng phím `3`.
- `history json`: nạp lại artifact `.json` bằng phím `4`.
- `console log`: nạp lại `.log` đã lưu bằng phím `6`.

Mỗi lần chạy workflow, app lưu console log vào:

```text
results/app/console_logs/
```

Khi nạp console log bằng phím `6`, workspace hiển thị:

- `CONSOLE TRACE`
- `OVERALL ASSESSMENT`
- `FLAG BENCHMARK`

Nếu workflow chưa có parser chuyên biệt, app vẫn tạo đánh giá tổng quát bằng tiếng Việt từ log và artifact.

## Hotkeys quan trọng trong TUI

| Phím | Chức năng |
|---|---|
| `1` | Chạy workflow đang chọn |
| `2` | Mở HTML mới nhất |
| `3` | Mở HTML đang chọn trong picker |
| `4` | Load JSON history |
| `5` | Bật/tắt forensic JSON dump |
| `6` | Load console log đã lưu |
| `F5` | Refresh danh sách HTML / JSON / LOG |
| `r` | Chạy lại workflow gần nhất |
| `:` | Mở command palette |
| `q` | Thoát |

Điều hướng:

- Dùng con lăn để cuộn vùng đang trỏ vào.
- Dùng `PgUp`, `PgDn`, `Home`, `End` để cuộn đồng bộ control deck, workspace và intel rail.

## Command palette

Nhấn `:` rồi gõ một trong các lệnh sau:

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

## Gợi ý thao tác

Luồng kiểm tra nhanh:

1. `robot`
2. Chọn `workflow=profile`
3. Bấm `1`
4. Chuyển sang `run`, `variant=A`, `preset=quick`
5. Bấm `1`
6. Nhìn `BEST MODEL` và `ARTIFACTS` ở intel rail
7. Bấm `2` hoặc `3` để mở HTML report liên quan

Luồng nghiên cứu FAMD và model evidence:

1. Chạy `famd` để tạo component HTML, K-Means và DBSCAN report.
2. Chạy `models` hoặc `report` để tạo model comparison và best model selection.
3. Dùng intel rail để kiểm tra nhanh `FAMD CLUSTER` và `BEST MODEL`.
4. Mở `results/final_report.html`, `results/model_evidence_metrics.html` hoặc `results/visualizations/famd_clustering_report.html` để đọc đầy đủ.

## Xử lý sự cố nhanh

`robot` không mở TUI:

```powershell
uv pip install "textual>=0.86.0"
```

Phím `3` không mở HTML:

- Kiểm tra `html picker` có file đang chọn hay chưa.
- Bấm `F5` để refresh danh sách.
- Bấm `3` lại, app sẽ fallback sang file HTML mới nhất nếu picker đang trống.

Muốn chỉ xem lại kết quả cũ:

- Dùng `4` cho JSON history.
- Dùng `6` cho console log.
- Dùng `3` hoặc `2` để mở HTML liên quan.

## Tài liệu liên quan

- [TUI_HOTKEYS.md](./TUI_HOTKEYS.md)
- [README.md](./README.md)
