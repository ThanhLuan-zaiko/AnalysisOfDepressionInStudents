# Phím Tắt TUI `robot`

## Mở ứng dụng

```powershell
robot
```

Hoặc:

```powershell
robot-tui
```

## Hotkeys hiện tại

| Phím | Chức năng |
|---|---|
| `1` | Chạy workflow đang chọn |
| `2` | Mở HTML artifact mới nhất |
| `3` | Mở HTML artifact đang chọn trong `html picker` |
| `4` | Load JSON history |
| `5` | Bật hoặc tắt forensic JSON dump |
| `6` | Load console log đã lưu |
| `F5` | Refresh danh sách HTML / JSON / LOG artifact |
| `r` | Chạy lại workflow gần nhất |
| `:` | Mở command palette |
| `q` | Thoát TUI |

## Khi dùng phím `6`

Sau khi load console log, vùng output sẽ hiển thị theo thứ tự:

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

1. chọn `workflow = profile`
2. bấm `1`

Chạy pipeline:

1. chọn `workflow = run`
2. chọn `variant = A` hoặc `B`
3. chọn `preset`
4. bấm `1`

Xem lại artifact cũ:

1. chọn file trong `history json` rồi bấm `4`
2. hoặc chọn file trong `console log` rồi bấm `6`
3. nếu muốn mở dashboard tương ứng thì chọn ở `html picker` rồi bấm `3`

## Cuộn và điều hướng

- Sidebar bên trái có thể cuộn riêng.
- Output stack bên phải có thể cuộn riêng.
- Có thể dùng chuột, con lăn, `PgUp`, `PgDn`, `Home`, `End`.

## Mẹo dùng nhanh

- Nếu `html picker` đang trống, bấm `F5` rồi bấm `3`.
- Nếu muốn mở HTML mới nhất ngay, bấm `2`.
- Nếu chỉ muốn kiểm tra lại kết quả cũ, ưu tiên `4` và `6` thay vì chạy lại workflow.

## Tài liệu liên quan

- [CLI_APP.md](./CLI_APP.md)
- [README.md](./README.md)
