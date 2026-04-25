# Machine Learning Và Deep Learning Trong Repo

Tài liệu này mô tả cách repo áp dụng machine learning và deep learning cho bài toán phân tích nguy cơ trầm cảm ở sinh viên. Trọng tâm hiện tại của pipeline chính là bài toán phân loại nhị phân với target `Depression`, còn deep learning được chuẩn bị như một module mở rộng bằng PyTorch.

## 1. Bài Toán Và Luồng Dữ Liệu

Repo làm việc với dataset `Student_Depression_Dataset.csv`. Biến mục tiêu chính là:

```text
Depression
```

Đây là bài toán binary classification:

- `0`: không thuộc nhóm trầm cảm theo nhãn dữ liệu.
- `1`: thuộc nhóm trầm cảm theo nhãn dữ liệu.

Pipeline có hai cách nhìn dữ liệu:

| Profile | Ý nghĩa | Cách chọn biến |
|---|---|---|
| `safe` / `A` / `conservative` | Phương án an toàn hơn | Loại biến `Have you ever had suicidal thoughts ?` để giảm nguy cơ leakage |
| `full` / `B` / `default` | Phương án đầy đủ để nghiên cứu so sánh | Giữ thêm biến nhạy cảm `Have you ever had suicidal thoughts ?` |

Các cột bị loại khỏi feature set trong pipeline risk-model:

```text
id
Profession
Work Pressure
Job Satisfaction
```

Các nhóm feature chính:

| Nhóm | Cột |
|---|---|
| Numeric | `Age`, `CGPA`, `Work/Study Hours` |
| Ordinal | `Academic Pressure`, `Study Satisfaction`, `Financial Stress` |
| Nominal | `Gender`, `City`, `Degree`, `Sleep Duration`, `Dietary Habits`, `Family History of Mental Illness` |
| Sensitive tùy chọn | `Have you ever had suicidal thoughts ?` |

## 2. Hai Pipeline Huấn Luyện

Repo có hai luồng huấn luyện cùng tồn tại.

### Modern pipeline

Mã chính nằm ở:

```text
src/app/services.py
```

Modern pipeline dùng split holdout trước:

1. Chia train/test bằng `train_test_split`, có `stratify=y`.
2. Huấn luyện model trên train.
3. Đánh giá bằng out-of-fold score trên train và holdout score trên test.
4. Xuất kết quả theo `RunReport` vào `results/app/`.

Preset mặc định:

| Preset | Model mặc định |
|---|---|
| `quick` | `dummy`, `logistic`, `catboost` |
| `research` | `dummy`, `logistic`, `gam`, `catboost` |

Lệnh chạy modern pipeline có đủ 4 model:

```powershell
robot run --dataset Student_Depression_Dataset.csv --variant A --preset research --budget auto
```

### Legacy risk-model pipeline

Mã chính nằm ở:

```text
src/ml_models/risk_model.py
main.py
```

Legacy pipeline chạy theo thứ tự:

```text
Dummy -> Logistic Regression -> GAM -> CatBoost
```

Lệnh chạy legacy workflow:

```powershell
robot task models --dataset Student_Depression_Dataset.csv --variant A --budget auto
```

Hoặc gọi trực tiếp:

```powershell
uv run python main.py --models --conservative
```

## 3. Machine Learning Được Áp Dụng

Pipeline chính dùng 4 model:

| Model | Vai trò | Lý do dùng |
|---|---|---|
| Dummy | Baseline | Kiểm tra model thật có vượt qua dự đoán theo prior/phân phối lớp hay không |
| Logistic Regression | Model trung tâm | Dễ giải thích bằng hệ số và odds ratio |
| GAM | Model giải thích phi tuyến | Cho phép quan hệ phi tuyến nhưng vẫn có partial dependence/feature effect |
| CatBoost | Model dự báo mạnh | Phù hợp dữ liệu bảng hỗn hợp numeric/categorical, có hỗ trợ GPU |

Ngoài 4 model trên, repo còn có toolkit phụ trong `src/ml_models/predictor.py` cho XGBoost và LightGBM. Hai model này không phải luồng chính của pipeline 4 model, nhưng có sẵn nếu cần thử nghiệm thêm.

## 4. Tiền Xử Lý Feature

### Logistic Regression

Modern pipeline dùng `Pipeline` và `ColumnTransformer`:

- Numeric: `SimpleImputer(strategy="median")` rồi `StandardScaler`.
- Ordinal: `SimpleImputer(strategy="most_frequent")` rồi `StandardScaler`.
- Nominal: `SimpleImputer(strategy="most_frequent")` rồi `OneHotEncoder(handle_unknown="ignore")`.
- Model: `LogisticRegression(class_weight="balanced", solver="lbfgs")`.

Legacy pipeline tự tạo feature matrix:

- Numeric được chuẩn hóa bằng `StandardScaler`.
- Ordinal được encode theo thứ tự giá trị.
- Nominal được one-hot bằng `pandas.get_dummies`.
- Biến nhị phân được map về `0/1`.

### GAM

Modern pipeline dùng `_build_gam_design()`:

- Numeric: fill median, chuẩn hóa bằng mean/std của train.
- Ordinal: fill mode, giữ dạng số.
- Nominal: fill mode, map category thành index số.
- Tạo `feature_types` gồm `numeric`, `ordinal`, `nominal`.

Legacy pipeline truyền toàn bộ feature matrix đã chuẩn bị vào `GAMClassifier`, đồng thời tự suy luận type cho từng feature.

### CatBoost

CatBoost giữ được lợi thế với dữ liệu bảng hỗn hợp:

- Numeric được ép kiểu số và fill median.
- Categorical được ép về string và fill mode.
- CatBoost nhận danh sách `cat_features` trong modern pipeline.
- Legacy pipeline đang đưa vào matrix đã encode, nhưng vẫn dùng CatBoost để tận dụng gradient boosting mạnh cho dữ liệu bảng.

## 5. Đánh Giá Model

Các metric chính được dùng xuyên suốt repo:

| Metric | Ý nghĩa |
|---|---|
| ROC-AUC | Khả năng xếp hạng positive cao hơn negative |
| PR-AUC | Hữu ích khi lớp positive cần chú ý hơn accuracy |
| F1 | Cân bằng precision và recall |
| Recall | Tỷ lệ phát hiện đúng positive |
| Precision | Tỷ lệ dự đoán positive là đúng |
| Brier score | Đánh giá chất lượng xác suất dự báo |
| Calibration curve | Kiểm tra xác suất dự báo có đáng tin hay không |
| Threshold report | So sánh các ngưỡng quyết định khác nhau |
| Fairness/subgroup metrics | So sánh hiệu năng theo nhóm giới tính, tuổi, tiền sử gia đình |

Modern pipeline lưu metric trong `ModelResult`:

```text
oof
holdout
thresholds
fairness
feature_importance
metadata
```

Legacy pipeline lưu kết quả tổng hợp tại:

```text
results/model_results_conservative.json
results/model_results_full.json
```

## 6. GAM Và Rust Engine

GAM được bọc bởi:

```text
src/ml_models/gam_model.py
```

`GAMClassifier.train()` sẽ thử dùng Rust engine trước nếu được bật:

```text
rust_engine -> pyGAM fallback
```

Modern pipeline chỉ dùng Rust GAM khi:

- `rust_engine` import được.
- Số dòng train đạt ngưỡng `MIN_RUST_GAM_ROWS = 200`.

Nếu Rust engine lỗi hoặc không đủ điều kiện, pipeline fallback về `pyGAM`. Metadata của GAM ghi lại engine đã dùng:

```text
engine
rust_available
rust_used
rust_error
n_splines
optimize_splines
```

## 7. GPU Được Dùng Ở Đâu

GPU không được dùng cho mọi model.

| Thành phần | GPU | Ghi chú |
|---|---|---|
| Dummy | Không | Model baseline rất nhẹ |
| Logistic Regression | Không | scikit-learn Logistic Regression chạy CPU |
| GAM | Không trực tiếp | Rust engine tăng tốc CPU; pyGAM cũng chạy CPU |
| CatBoost | Có nếu CUDA sẵn sàng | Tự set `task_type="GPU"` khi `torch.cuda.is_available()` |
| PyTorch deep learning | Có | Tự chọn CUDA, MPS hoặc CPU |

Repo có các file hỗ trợ kiểm tra GPU:

```text
config.py
verify_gpu.py
verify_gpu_usage.py
setup_with_gpu.ps1
```

## 8. Deep Learning Trong Repo

Deep learning nằm ở:

```text
src/deep_learning/trainer.py
```

Module này chưa phải luồng mặc định của pipeline 4 model, nhưng đã có đủ thành phần để huấn luyện neural network bằng PyTorch.

Các class chính:

| Class | Vai trò |
|---|---|
| `DepressionNet` | Mạng feed-forward cơ bản |
| `DepressionNetDeep` | Mạng sâu hơn với nhiều hidden layer |
| `DepressionNN` | Toolkit chuẩn bị DataLoader, tạo model, train, evaluate, predict, save/load |

Kiến trúc `DepressionNet`:

```text
Linear -> BatchNorm1d -> ReLU -> Dropout
Linear -> BatchNorm1d -> ReLU -> Dropout
Linear -> BatchNorm1d -> ReLU -> Dropout
Linear -> Sigmoid
```

Quy trình train trong `DepressionNN.train()`:

1. Dùng `BCELoss` cho binary classification.
2. Dùng `Adam` optimizer.
3. Có `weight_decay` để regularize.
4. Có `ReduceLROnPlateau` scheduler nếu bật.
5. Có early stopping theo validation loss.
6. Lưu history gồm train/validation loss và accuracy.

Thiết bị được auto-detect:

```text
CUDA -> MPS -> CPU
```

Điểm cần lưu ý: deep learning hiện là module mở rộng. Nếu muốn so sánh neural network với 4 model chính, cần viết thêm wrapper để đưa kết quả neural network vào cùng format `ModelResult` hoặc `ModelComparator`.

## 9. Artifact Quan Trọng

Modern pipeline:

```text
results/app/run_safe_quick.json
results/app/run_safe_research.json
results/app/run_full_quick.json
results/app/run_full_research.json
results/app/compare_profiles_quick.json
results/app/compare_profiles_research.json
```

Legacy pipeline:

```text
results/model_results_conservative.json
results/model_results_full.json
results/model_comparison.html
results/calibration_curves.html
results/decision_curves.html
results/gam_feature_effects.html
results/gam_interpretation.json
results/gam_plots/
```

## 10. Cảnh Báo Diễn Giải

Các model trong repo dùng cho phân tích và hỗ trợ sàng lọc, không thay thế đánh giá lâm sàng. Kết quả là quan hệ dự báo/liên quan trong dữ liệu, không khẳng định nguyên nhân. Khi dùng profile `full`, cần đặc biệt chú ý biến `Have you ever had suicidal thoughts ?` vì biến này có thể làm tăng metric nhưng cũng có nguy cơ leakage hoặc tạo kết luận thiếu an toàn nếu diễn giải sai.
