# Quyền Và Review Cho Thư Mục `docs/`

## Điều đã được chuẩn bị trong repo

Repo đã được thêm file:

```text
.github/CODEOWNERS
```

Trong đó path `/docs/` được gán cho chủ repo để GitHub có thể yêu cầu review riêng cho tài liệu.

## Điều cần hiểu đúng

GitHub repository thông thường không hỗ trợ cấp quyền đọc/ghi riêng theo từng thư mục trong cùng một repo.

Điều khả thi nhất ở mức thư mục là:

- dùng `CODEOWNERS` để chỉ định người review cho `docs/`
- bật branch protection và yêu cầu `Require review from Code Owners`

## Cấu hình nên bật trên GitHub

Trong `Settings -> Branches -> Branch protection rules`:

1. Tạo hoặc sửa rule cho nhánh chính.
2. Bật `Require a pull request before merging`.
3. Bật `Require review from Code Owners`.

Sau đó mọi thay đổi trong `docs/` sẽ đi qua luồng review riêng theo `CODEOWNERS`.

## Nếu cần quyền tách biệt thật sự

Nếu bạn cần phân quyền ghi khác nhau giữa code và tài liệu, hướng đúng hơn là:

- tách docs sang repo riêng
- hoặc dùng GitHub Team kết hợp quy trình PR bắt buộc

Trong phạm vi một repo chuẩn, `CODEOWNERS + branch protection` là cách thực tế nhất.
