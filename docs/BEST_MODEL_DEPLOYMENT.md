# Best Model Deployment

Tài liệu này mô tả đầu ra triển khai của repo: một artifact model đã fit để dự đoán nguy cơ trầm cảm từ hồ sơ học sinh/sinh viên.

## Cảnh báo sử dụng

Artifact này chỉ là công cụ hỗ trợ sàng lọc/nghiên cứu. Kết quả `prediction=1` nghĩa là mẫu bị gắn cờ nguy cơ theo mô hình, không phải chẩn đoán lâm sàng và không thay thế đánh giá của chuyên gia.

## Train Best Model

Chạy so sánh Safe A và Full B, chọn model tốt nhất theo holdout ROC-AUC, rồi fit lại model cuối trên toàn bộ dữ liệu của profile được chọn:

```powershell
robot train-best --dataset Student_Depression_Dataset.csv --preset research --budget auto
```

Đầu ra chính:

```text
models/best_depression_model.joblib
models/best_depression_model.json
results/best_model_selection.json
```

Luật chọn mặc định:

- Chỉ chọn trong `logistic`, `gam`, `catboost`; không deploy `dummy`.
- Xếp hạng theo holdout ROC-AUC trên cả Safe A và Full B.
- Tie-breaker: PR-AUC, F1, ưu tiên CatBoost rồi GAM rồi Logistic.
- Ngưỡng mặc định là `screening`, ưu tiên recall để giảm bỏ sót ca nguy cơ.

## Predict JSON

`sample.json` có thể là một record:

```json
{
  "Gender": "Female",
  "Age": 22,
  "City": "Hanoi",
  "Academic Pressure": 4,
  "CGPA": 7.8,
  "Study Satisfaction": 2,
  "Sleep Duration": "5-6 hours",
  "Dietary Habits": "Moderate",
  "Degree": "Bachelor",
  "Work/Study Hours": 8,
  "Financial Stress": 4,
  "Family History of Mental Illness": "No",
  "Have you ever had suicidal thoughts ?": "No"
}
```

Chạy:

```powershell
robot predict --model models/best_depression_model.joblib --input sample.json
```

Nếu artifact chọn Safe A, cột suicidal-thoughts sẽ bị bỏ qua. Nếu artifact chọn Full B, cột này là bắt buộc.

## Predict CSV

```powershell
robot predict --input students.csv --output predictions.csv
```

Output gồm:

- `probability`: xác suất model dự đoán cho class nguy cơ.
- `prediction`: `1` nếu vượt ngưỡng sàng lọc, `0` nếu không.
- `risk_label`: `screening_flag` hoặc `not_flagged`.
- `threshold`, `threshold_policy`, `model`, `profile`.
- `disclaimer`: nhắc lại giới hạn không dùng để chẩn đoán lâm sàng.

## Predict Trong TUI

Sau khi đã có `models/best_depression_model.joblib`:

```powershell
robot
```

Nếu chưa có artifact, bấm `T` trong TUI để train/export best model từ dataset/preset/budget đang chọn.

Trong control deck bên trái, điền vùng `prediction`, rồi bấm `P` hoặc gõ `:predict`. Workspace sẽ hiển thị xác suất, ngưỡng đang dùng, model/profile được chọn và kết luận sàng lọc.
