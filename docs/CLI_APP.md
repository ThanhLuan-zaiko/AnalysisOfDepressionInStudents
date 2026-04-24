# Hướng Dẫn CLI App `robot`

## Mục tiêu

`robot` là lệnh ngắn để mở ứng dụng phân tích trong repo này. Nếu máy đã có `textual`, `robot` sẽ mở TUI. Nếu chưa có, `robot` sẽ tự chuyển sang menu console.

Người mới nên bắt đầu từ `robot` thay vì gọi trực tiếp `main.py`.

Nếu bạn chỉ cần nhớ thao tác bàn phím trong giao diện, xem thêm [TUI_HOTKEYS.md](./TUI_HOTKEYS.md).

## Workflow Hub mới

`robot` giờ dùng cùng một workflow hub cho cả TUI lẫn fallback console:

- hỗ trợ cả nhánh modern và legacy trong `README.md`
- hiển thị cấu hình `A/B`
- hỗ trợ `training budget = default|auto`
- có HTML picker để mở report/dashboard trên browser mặc định

Quy đổi nhanh:

- `A = safe = conservative`
- `B = full = default`

## Cài nhanh

Trong thư mục repo:

```powershell
.\setup_with_gpu.ps1
```

Script này hiện đã:

- tạo `.venv`
- cài dependency chính
- cài `rich` và `textual`
- tạo launcher `robot` và `robot-tui`
- thêm thư mục launcher vào `PATH` người dùng nếu cần

Sau khi setup xong, mở một cửa sổ PowerShell mới rồi chạy:

```powershell
robot
```

## Dataset mặc định

Nếu không truyền `--dataset`, app sẽ dùng:

```text
Student_Depression_Dataset.csv
```

Bạn vẫn có thể chỉ định file khác bằng `--dataset`.

## Chạy từng chức năng

### 1. Mở ứng dụng

```powershell
robot
```

Ứng dụng sẽ mở TUI nếu `textual` đã có. Trong TUI:

- `1`: chạy hồ sơ dữ liệu
- `2`: chạy pipeline theo profile và preset đang chọn
- `3`: so sánh `safe` và `full`
- `r`: chạy lại tác vụ hiện tại
- `:`: mở command palette
- `q`: thoát

### 2. Hồ sơ dữ liệu

```powershell
robot profile --dataset Student_Depression_Dataset.csv
```

Chức năng này dùng để:

- xem số dòng, số cột
- xem tỷ lệ nhãn `Depression`
- xem cột nào được chọn cho `safe` và `full`
- hiện cảnh báo leakage, missing value, rare category

Nếu chỉ muốn in ra terminal, không ghi file:

```powershell
robot profile --dataset Student_Depression_Dataset.csv --console-only
```

Nếu muốn xuất HTML EDA:

```powershell
robot profile --dataset Student_Depression_Dataset.csv --full-export --export-html
```

### 3. Chạy nhanh an toàn

```powershell
robot run --dataset Student_Depression_Dataset.csv --profile safe --preset quick --console-only
```

Đây là lệnh nên dùng đầu tiên khi muốn đánh giá mô hình mà hạn chế leakage.

`safe` bỏ biến:

```text
Have you ever had suicidal thoughts ?
```

### 4. Chạy nhanh đầy đủ

```powershell
robot run --dataset Student_Depression_Dataset.csv --profile full --preset quick --console-only
```

`full` giữ biến nhạy cảm để bạn nhìn thấy mức tăng accuracy và rủi ro leakage.

### 5. Chạy nghiên cứu

```powershell
robot run --dataset Student_Depression_Dataset.csv --profile safe --preset research --console-only
```

Preset `research` sẽ sâu hơn `quick`:

- thêm GAM
- hiện metadata nghiên cứu nhiều hơn
- ưu tiên Rust engine nếu điều kiện phù hợp
- fallback về `pyGAM` nếu Rust không sẵn sàng

### 6. So sánh `safe` và `full`

```powershell
robot compare --dataset Student_Depression_Dataset.csv --preset quick --console-only
```

Chức năng này chạy cùng một split cho hai profile để bạn xem:

- `roc_auc`
- `f1`
- `recall`
- độ lệch giữa `full` và `safe`

### 7. Mở thẳng TUI

```powershell
robot-tui
```

Hoặc:

```powershell
robot
```

Nếu `textual` đã được cài.

## Command Palette Trong TUI

Nhấn `:` rồi nhập một trong các lệnh sau:

```text
:help
:run
:rerun
:html latest
:html open
:refresh html
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

## Ý nghĩa profile

### `safe`

- phù hợp đánh giá thực dụng
- giảm nguy cơ leakage
- nên dùng khi báo cáo hoặc demo chính

### `full`

- dùng để so sánh nghiên cứu
- giữ biến nhạy cảm
- dễ có accuracy cao hơn nhưng rủi ro leakage lớn hơn

## Ý nghĩa preset

### `quick`

- chạy nhanh hơn
- phù hợp smoke test, demo, kiểm tra pipeline
- thường gồm `logistic` và `catboost`

### `research`

- chạy sâu hơn
- thêm `gam`
- có thêm metadata phục vụ phân tích

## Khi nào dùng `main.py`

`main.py` vẫn còn để tương thích luồng cũ, nhưng với người mới thì nên ưu tiên:

```powershell
robot
```

Nếu cần kiểm tra đường legacy:

```powershell
.\.venv\Scripts\python.exe main.py --quick --profile safe --console-only
```

## Lệnh nên nhớ

```powershell
robot
robot profile --dataset Student_Depression_Dataset.csv --console-only
robot run --dataset Student_Depression_Dataset.csv --variant A --preset quick --budget auto
robot run --dataset Student_Depression_Dataset.csv --variant B --preset quick
robot compare --dataset Student_Depression_Dataset.csv --preset quick --console-only
robot task analysis --dataset Student_Depression_Dataset.csv --variant A --budget auto
robot open-html --latest
robot-tui
```

## Xử lý sự cố nhanh

### `robot` không mở TUI

Chạy lại setup:

```powershell
.\setup_with_gpu.ps1
```

Hoặc cài nhanh riêng `textual`:

```powershell
uv pip install "textual>=0.86.0"
```

### Không ghi được file trong `results/` hoặc `logs/`

Dùng chế độ chỉ in terminal:

```powershell
robot run --profile safe --preset quick --console-only
```

### Muốn biết GAM có dùng Rust hay không

Chạy:

```powershell
robot run --dataset Student_Depression_Dataset.csv --profile safe --preset research --console-only
```

Sau đó xem phần metadata của model `gam`.

## Gợi ý cho người mới

Thứ tự đọc hợp lý:

1. `docs/CLI_APP.md`
2. `README.md`
3. `src/cli/entrypoint.py`
4. `src/cli/textual_app.py`
5. `src/app/services.py`
6. `src/entrypoints/main_dispatch.py`
