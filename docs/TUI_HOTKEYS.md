# Phím Tắt TUI `robot`

## Mục tiêu

Trang này chỉ tập trung vào thao tác nhanh trong giao diện TUI của `robot`.

Mở TUI bằng:

```powershell
robot
```

Hoặc:

```powershell
robot-tui
```

## Phím tắt chính

| Phím | Chức năng |
|---|---|
| `1` | Chạy workflow đang chọn |
| `2` | Mở HTML mới nhất |
| `3` | Refresh danh sách HTML |
| `r` | Chạy lại tác vụ gần nhất |
| `:` | Mở command palette |
| `q` | Thoát ứng dụng |

## Luồng thao tác nhanh nhất

### Xem hồ sơ dữ liệu

1. Gõ `robot`
2. Kiểm tra `dataset`
3. Bấm `1`

### Chạy pipeline an toàn

1. Gõ `robot`
2. Để `profile = safe`
3. Để `preset = quick`
4. Bấm `2`

### So sánh `safe` và `full`

1. Gõ `robot`
2. Chọn `preset`
3. Bấm `3`

## Command Palette

Nhấn `:` để mở ô lệnh ở vùng trên cùng của khu vực report, sau đó nhập một trong các lệnh sau:

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

## Command palette dùng khi nào

- Dùng `:profile` khi muốn chạy nhanh mà không cần rê focus vào nút.
- Dùng `:run` khi đã chỉnh `profile` và `preset`.
- Dùng `:compare` khi muốn sang màn hình so sánh ngay.
- Dùng `:rerun` khi muốn lặp lại tác vụ gần nhất.
- Dùng `:set profile ...` và `:set preset ...` khi muốn đổi cấu hình thật nhanh từ bàn phím.
- Dùng `:set dataset ...` khi cần đổi file CSV mà không phải click vào ô nhập liệu.
- Dùng `:help` khi quên cú pháp.

## Điều hướng và cuộn

- Cột điều khiển bên trái đã có thể cuộn lên xuống khi màn hình thấp.
- Khu vực kết quả bên phải cũng có cuộn riêng.
- Nếu dùng chuột, ưu tiên con lăn hoặc trackpad để cuộn.
- Nếu dùng bàn phím, có thể kết hợp `Tab` để chuyển focus giữa các ô nhập, select, nút và command palette.

## Gợi ý thao tác

### Khi mới mở app

- Bấm `1` để xem dữ liệu trước.
- Sau đó bấm `2` để chạy `safe + quick`.
- Nếu muốn xem mức chênh do leakage, bấm `3`.

### Khi đang demo

- Giữ `profile = safe`
- Giữ `preset = quick`
- Dùng `2` để chạy nhanh
- Dùng `r` để lặp lại ngay sau khi đổi dataset

### Khi đang nghiên cứu

- Đổi `preset = research`
- Dùng `2` để chạy pipeline sâu hơn
- Dùng `3` để nhìn sự khác biệt giữa `safe` và `full`

## Lỗi thường gặp

### Nhấn `:` nhưng không thấy gì

- Hãy nhìn vùng ngay dưới status bar.
- Gõ lại `:`.
- Nếu vẫn không thấy, thử bấm `q` để thoát rồi mở lại `robot`.

### Report dài quá

- Cuộn trong panel bên phải.
- Nếu sidebar bị dài, cuộn riêng bên trái.

### Quên phím tắt

- Nhìn footer cuối màn hình.
- Hoặc nhấn `:` rồi gõ `help`.

## Tài liệu liên quan

- [CLI_APP.md](./CLI_APP.md)
- [README.md](./README.md)
