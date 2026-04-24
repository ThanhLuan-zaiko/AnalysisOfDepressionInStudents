# Hướng Dẫn CLI App `robot`

## Mục tiêu

`robot` là lệnh chung để:

- mở TUI theo phong cách terminal monitor
- chạy workflow modern và legacy trong cùng một hub
- xem lại artifact cũ mà không cần chạy lại
- mở HTML report/dashboard trên trình duyệt mặc định

Nếu máy đã có `textual`, `robot` sẽ mở TUI. Nếu chưa có, lệnh sẽ rơi về console fallback.

## Quy ước cấu hình

- `A = safe = conservative`
- `B = full = default`
- `budget = default | auto`

`budget=auto` sẽ tự chỉnh ngân sách train cho các model đang được workflow đó dùng.

## Khởi động nhanh

Trong thư mục repo:

```powershell
.\setup_with_gpu.ps1
```

Sau đó mở PowerShell mới rồi chạy:

```powershell
robot
```

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
robot task fairness --dataset Student_Depression_Dataset.csv --variant A --budget auto
robot task robustness --dataset Student_Depression_Dataset.csv --variant B --budget auto
robot task analysis --dataset Student_Depression_Dataset.csv --variant A --budget auto
robot task report --dataset Student_Depression_Dataset.csv
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

TUI hiện có 3 lane review artifact:

- `html picker`: chọn file `.html` rồi mở bằng phím `3`
- `history json`: nạp lại artifact `.json`
- `console log`: nạp lại `.log` đã lưu

Mỗi lần chạy workflow, app sẽ lưu console log vào:

```text
results/app/console_logs/
```

Khi nạp console log bằng phím `6`, output stack sẽ hiển thị:

- `CONSOLE TRACE`
- `OVERALL ASSESSMENT`
- `FLAG BENCHMARK`

Hai panel đánh giá này không chỉ dành cho `--eda`. Chúng sẽ tự chọn cách diễn giải theo workflow, gồm:

- `profile`
- `run`
- `compare`
- `eda`
- `models`
- `full`
- `fairness`
- `subgroups`
- `robustness`
- `analysis`
- `review`
- `stats`
- `split`
- `famd`
- `standardize`
- `report`

Nếu workflow chưa có parser chuyên biệt, app vẫn tạo đánh giá tổng quát bằng tiếng Việt từ log và artifact.

## Hotkeys quan trọng trong TUI

- `1`: chạy workflow đang chọn
- `2`: mở HTML mới nhất
- `3`: mở HTML đang chọn trong picker
- `4`: load JSON history
- `5`: bật/tắt forensic JSON dump
- `6`: load console log đã lưu
- `F5`: refresh danh sách HTML / JSON / LOG
- `r`: chạy lại workflow gần nhất
- `:`: mở command palette
- `q`: thoát

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

Luồng an toàn khi kiểm tra nhanh:

1. `robot`
2. chọn `workflow=profile`
3. bấm `1`
4. chuyển sang `run`, `variant=A`, `preset=quick`
5. bấm `1`
6. bấm `6` nếu muốn xem lại log cũ có benchmark

Luồng nghiên cứu leakage:

1. chạy `profile`
2. chạy `compare`
3. mở `eda` hoặc `analysis`
4. dùng `3` để mở HTML đang chọn
5. dùng `4` hoặc `6` để xem lại artifact cũ mà không rerun

## Xử lý sự cố nhanh

`robot` không mở TUI:

```powershell
uv pip install "textual>=0.86.0"
```

Phím `3` không mở HTML:

- kiểm tra `html picker` có file đang chọn hay chưa
- bấm `F5` để refresh danh sách
- bấm `3` lại, app sẽ fallback sang file HTML mới nhất nếu picker đang trống

Muốn chỉ xem lại kết quả cũ:

- dùng `4` cho JSON history
- dùng `6` cho console log
- dùng `3` hoặc `2` để mở HTML liên quan

## Tài liệu liên quan

- [TUI_HOTKEYS.md](./TUI_HOTKEYS.md)
- [README.md](./README.md)
