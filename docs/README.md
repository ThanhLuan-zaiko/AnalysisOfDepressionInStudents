# Tài Liệu Ứng Dụng CLI

Thư mục này dành cho người mới clone repo và cần chạy ứng dụng nhanh mà không phải đọc toàn bộ mã nguồn ngay từ đầu.

## Bắt đầu từ đâu

- Đọc [CLI_APP.md](./CLI_APP.md) để biết cách chạy ứng dụng bằng lệnh ngắn `robot`.
- Đọc [GITHUB_DOCS_QUYEN.md](./GITHUB_DOCS_QUYEN.md) để hiểu cách repo đang khóa review riêng cho thư mục `docs/`.

## Lối chạy ngắn nhất

```powershell
robot
```

Nếu máy chưa cài giao diện TUI `Textual`, lệnh trên sẽ tự rơi về CLI console.

## Lệnh quan trọng

```powershell
robot profile --dataset Student_Depression_Dataset.csv
robot run --profile safe --preset quick --console-only
robot compare --preset quick --console-only
```

## Ghi chú cho người mới

- `safe`: bỏ cột `Have you ever had suicidal thoughts ?` để giảm nguy cơ leakage.
- `full`: giữ cột đó để phục vụ so sánh nghiên cứu.
- `quick`: chạy nhanh, phù hợp để xem pipeline hoạt động ra sao.
- `research`: chạy sâu hơn, có thêm GAM và kiểm tra nghiên cứu.
