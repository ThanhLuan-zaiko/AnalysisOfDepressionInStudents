# Hướng Dẫn CLI App `robot`

## Mục tiêu

CLI App của repo này là lớp chạy gọn cho hệ thống phân tích trầm cảm dành cho dữ liệu sinh viên Việt Nam. Thay vì gọi trực tiếp `main.py` với nhiều cờ, người dùng mới có thể bắt đầu từ lệnh ngắn `robot`.

## Yêu cầu tối thiểu

- Python theo phiên bản repo đang khai báo trong `.python-version`
- `uv`
- Môi trường ảo `.venv` đã được đồng bộ dependency

## Cài nhanh

Trong thư mục repo:

```powershell
uv sync
```

Nếu muốn giao diện TUI:

```powershell
uv sync --extra tui
```

## Chạy nhanh nhất

```powershell
robot
```

Hành vi hiện tại:

- Nếu đã cài `textual`, `robot` sẽ mở giao diện TUI.
- Nếu chưa cài `textual`, `robot` sẽ tự rơi về console CLI và chạy preset mặc định.

## Các chế độ chạy chính

### 1. Xem hồ sơ dữ liệu

```powershell
robot profile --dataset Student_Depression_Dataset.csv
```

Lệnh này dùng để:

- xem số dòng, số cột
- xem tỷ lệ dương tính của nhãn `Depression`
- xem các cột được chọn cho profile `safe` và `full`
- phát hiện cảnh báo leakage hoặc dữ liệu hiếm

### 2. Chạy pipeline nhanh

```powershell
robot run --profile safe --preset quick --console-only
```

Ý nghĩa:

- `--profile safe`: bỏ biến nhạy cảm có nguy cơ leakage cao
- `--preset quick`: chạy nhanh với đường đánh giá holdout-first
- `--console-only`: chỉ in kết quả ra terminal, không cố ghi artifact JSON/HTML

### 3. Chạy pipeline nghiên cứu

```powershell
robot run --profile safe --preset research --console-only
```

Khác biệt của `research`:

- có thêm GAM
- ưu tiên dùng `rust_engine` khi dữ liệu đủ lớn
- nếu Rust không sẵn sàng hoặc gặp lỗi, app tự fallback về `pyGAM`

### 4. So sánh `safe` và `full`

```powershell
robot compare --preset quick --console-only
```

Lệnh này giúp thấy rõ:

- AUC/F1 của profile an toàn
- AUC/F1 của profile đầy đủ
- phần chênh lệch có thể đến từ leakage

## Khi nào nên dùng `main.py`

`main.py` vẫn được giữ để tương thích với luồng cũ, nhưng người mới nên ưu tiên `robot`.

Ví dụ tương đương:

```powershell
.\.venv\Scripts\python.exe main.py --quick --profile safe --console-only
```

## Ý nghĩa các profile

### `safe`

- dùng cho đánh giá thực dụng
- tránh dựa trực tiếp vào biến nhạy cảm `Have you ever had suicidal thoughts ?`

### `full`

- dùng để so sánh nghiên cứu
- giữ biến nhạy cảm để đo mức chênh accuracy và rủi ro leakage

## Ý nghĩa các preset

### `quick`

- nhanh hơn
- phù hợp để smoke test hoặc trình diễn
- thường chạy `logistic` và `catboost`

### `research`

- sâu hơn
- có thêm `gam`
- có research summary và metadata về Rust engine

## Một số lệnh nên nhớ

```powershell
robot
robot profile --dataset Student_Depression_Dataset.csv
robot run --dataset Student_Depression_Dataset.csv --profile safe --preset quick --console-only
robot run --dataset Student_Depression_Dataset.csv --profile safe --preset research --console-only
robot compare --dataset Student_Depression_Dataset.csv --preset quick --console-only
robot-tui
```

## Xử lý sự cố nhanh

### `robot` báo chưa có `textual`

Chạy:

```powershell
uv sync --extra tui
```

Hoặc cứ dùng CLI console vì app đã có fallback.

### Không ghi được file vào `results/` hoặc `logs/`

Dùng:

```powershell
robot run --profile safe --preset quick --console-only
```

Tùy chọn này rất hữu ích trong môi trường sandbox hoặc máy có quyền ghi hạn chế.

### Muốn biết GAM có dùng Rust hay không

Chạy preset `research`. Report sẽ in metadata của model và trạng thái `rust_engine`.

## Gợi ý cho người mới trong repo

Thứ tự đọc hợp lý:

1. `docs/CLI_APP.md`
2. `README.md`
3. `src/cli/entrypoint.py`
4. `src/app/services.py`
5. `src/entrypoints/main_dispatch.py`
