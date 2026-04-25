# Cách Huấn Luyện 4 Model: Dummy, Logistic, GAM, CatBoost

Tài liệu này mô tả cách repo huấn luyện 4 model chính cho bài toán dự báo `Depression`: `dummy`, `logistic`, `gam`, `catboost`.

Tên người dùng thường gõ nhầm `logictist` hoặc `catboot`; trong code repo, tên đúng là:

```text
logistic
catboost
```

## 1. Chạy Nhanh 4 Model

Từ thư mục gốc repo:

```powershell
robot run --dataset Student_Depression_Dataset.csv --variant A --preset research --budget auto
```

Lệnh trên chạy modern pipeline với:

- `variant A`: profile `safe`, loại biến `Have you ever had suicidal thoughts ?`.
- `preset research`: bao gồm đủ 4 model `dummy`, `logistic`, `gam`, `catboost`.
- `budget auto`: tự chỉnh số vòng lặp/spline/early stopping theo kích thước train.

Nếu muốn chạy profile đầy đủ:

```powershell
robot run --dataset Student_Depression_Dataset.csv --variant B --preset research --budget auto
```

Nếu muốn chạy legacy pipeline 4 model:

```powershell
robot task models --dataset Student_Depression_Dataset.csv --variant A --budget auto
```

Hoặc gọi trực tiếp `main.py`:

```powershell
uv run python main.py --models --conservative
```

## 2. File Code Liên Quan

| Thành phần | File |
|---|---|
| Modern service pipeline | `src/app/services.py` |
| Contract kết quả | `src/app/contracts.py` |
| Legacy risk-model pipeline | `src/ml_models/risk_model.py` |
| GAM wrapper | `src/ml_models/gam_model.py` |
| Rust GAM engine | `rust_engine/` |
| Training budget | `src/training_budget.py` |
| CLI entry point | `src/cli/entrypoint.py` |
| Workflow hub | `src/cli/workflows.py` |

## 3. Tổng Quan Quy Trình Huấn Luyện

Modern pipeline trong `src/app/services.py` chạy theo thứ tự:

1. Load dataset.
2. Chia train/test bằng stratified split.
3. Chọn feature theo profile `safe` hoặc `full`.
4. Resolve training budget.
5. Huấn luyện từng model.
6. Tính metric out-of-fold hoặc CV summary trên train.
7. Fit model cuối cùng và đánh giá trên holdout test.
8. Tính threshold report, fairness/subgroup metrics và feature importance.
9. Lưu JSON artifact vào `results/app/`.

Legacy pipeline trong `src/ml_models/risk_model.py` chạy theo thứ tự:

```text
prepare_features()
train_dummy()
train_logistic()
train_gam()
train_catboost()
calibration_analysis()
print_report()
save results/model_results_*.json
```

## 4. Training Budget

Training budget được resolve tại:

```text
src/training_budget.py
```

Modern defaults:

| Preset | Logistic | CatBoost | GAM |
|---|---:|---:|---:|
| `quick` | `max_iter=1500` | `iterations=300`, `depth=6`, `lr=0.05` | `n_splines=10`, không optimize spline mặc định |
| `research` | `max_iter=1500` | `iterations=300`, `depth=6`, `lr=0.05` | `n_splines=12`, optimize spline |

Legacy defaults:

| Model | Budget |
|---|---|
| Logistic | `max_iter=1000` |
| CatBoost | `iterations=500`, `early_stopping_rounds=30`, `learning_rate=0.05`, `depth=6` |
| GAM | `n_splines=15`, `optimize_splines=True` |

Khi dùng `--budget auto`, repo tự tăng/giảm budget theo số dòng train.

## 5. Model 0: Dummy Baseline

### Mục đích

Dummy là baseline để trả lời câu hỏi: model thật có học được tín hiệu nào hay chỉ đang dự đoán theo phân phối lớp?

### Modern pipeline

Hàm:

```text
DepressionAnalysisService._run_dummy()
```

Cách train:

```text
DummyClassifier(strategy="prior")
```

Quy trình:

1. Fit trên `y_train` với feature giả toàn số 0.
2. Lấy prior của class positive từ train.
3. Gán cùng một xác suất cho mọi mẫu train/test.
4. Dự đoán class bằng ngưỡng `0.5`.
5. Tính metric trên train và holdout.

Metadata chính:

```text
strategy = "prior"
class_prior_positive
evaluation = "constant_prior_baseline"
```

### Legacy pipeline

Hàm:

```text
DepressionRiskModeler.train_dummy()
```

Cách train:

```text
DummyClassifier(strategy="stratified")
```

Quy trình:

1. Chạy `StratifiedKFold(n_splits=5)`.
2. Dùng `cross_validate()` với các metric ROC-AUC, average precision, F1, recall, precision.
3. Fit lại trên toàn bộ dữ liệu.
4. Xác suất dummy được đặt bằng tỷ lệ positive `y.mean()`.
5. Lưu metric vào `self.results["dummy"]`.

## 6. Model 1: Logistic Regression

### Mục đích

Logistic Regression là model trung tâm vì dễ giải thích. Hệ số có thể chuyển thành odds ratio để xem feature nào làm tăng/giảm xác suất dự báo class positive.

### Modern pipeline

Hàm:

```text
DepressionAnalysisService._run_logistic()
```

Preprocessing:

| Nhóm feature | Xử lý |
|---|---|
| Numeric | median imputation + `StandardScaler` |
| Ordinal | most frequent imputation + `StandardScaler` |
| Nominal | most frequent imputation + `OneHotEncoder(handle_unknown="ignore")` |

Model:

```text
LogisticRegression(
    class_weight="balanced",
    solver="lbfgs",
    max_iter=<training_budget>,
    random_state=42,
)
```

Quy trình:

1. Tạo sklearn `Pipeline(preprocessor, model)`.
2. Chạy `StratifiedKFold`.
3. Lấy out-of-fold probability bằng `cross_val_predict(..., method="predict_proba")`.
4. Fit pipeline cuối cùng trên toàn bộ train.
5. Predict probability trên holdout test.
6. Trích hệ số từ `pipeline.named_steps["model"].coef_`.
7. Tính `coefficient`, `abs_coefficient`, `odds_ratio`.
8. Tính threshold report và subgroup/fairness metrics.

Kết quả feature importance là top hệ số lớn nhất theo `abs_coefficient`.

### Legacy pipeline

Hàm:

```text
DepressionRiskModeler.train_logistic()
```

Model:

```text
LogisticRegression(
    C=1.0,
    max_iter=<training_budget>,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
)
```

Quy trình:

1. Nhận `X`, `y`, `feature_names` từ `prepare_features()`.
2. Chạy 5-fold stratified CV bằng `cross_validate()`.
3. Fit lại trên toàn bộ data.
4. Tính probability, prediction và Brier score.
5. Trích hệ số bằng `_extract_coefficients()`.
6. Lưu kết quả vào `self.models["logistic"]` và `self.results["logistic"]`.

## 7. Model 2: GAM

### Mục đích

GAM nằm giữa Logistic Regression và CatBoost:

- Linh hoạt hơn Logistic Regression vì mô hình hóa được quan hệ phi tuyến.
- Dễ giải thích hơn tree ensemble vì mỗi feature có effect riêng.
- Hỗ trợ partial dependence/feature effect khi dùng pyGAM.

### Modern pipeline

Hàm:

```text
DepressionAnalysisService._run_gam()
```

Thiết kế dữ liệu:

```text
_build_gam_design(X_train, X_test)
```

Quy trình tiền xử lý:

1. Numeric: fill median, chuẩn hóa bằng mean/std của train.
2. Ordinal: fill mode, giữ dạng số.
3. Nominal: fill mode, map category thành index.
4. Tạo `feature_names`.
5. Tạo `feature_types` với các giá trị `numeric`, `ordinal`, `nominal`.

Model wrapper:

```text
GAMClassifier(random_state=42)
```

Huấn luyện:

```text
gam.train(
    X_train_matrix,
    y_train,
    feature_types=feature_types,
    feature_names=feature_names,
    n_splines=<training_budget>,
    optimize_splines=<training_budget>,
    use_rust=<true_or_false>,
)
```

Luồng engine:

1. Kiểm tra `rust_engine`.
2. Nếu Rust sẵn sàng và train đủ ít nhất `MIN_RUST_GAM_ROWS = 200`, thử Rust GAM.
3. Nếu Rust lỗi, retry bằng pyGAM.
4. Predict probability trên holdout.
5. Tính threshold, fairness và feature importance.

Metadata quan trọng:

```text
engine
rust_available
rust_used
rust_version
rust_error
n_splines
optimize_splines
evaluation = "cv_summary_on_train_plus_holdout"
```

### Legacy pipeline

Hàm:

```text
DepressionRiskModeler.train_gam()
```

Quy trình:

1. Suy luận `feature_types` nếu chưa truyền vào.
2. Khởi tạo `GAMClassifier`.
3. Lấy budget `n_splines` và `optimize_splines`.
4. Gọi `gam.train()`.
5. Lưu model vào `self.models["gam"]`.
6. Lưu metrics vào `self.results["gam"]`.
7. Lưu `gam_feature_types` trong `self.preprocessors`.

Trong `main.py`, legacy pipeline còn tạo thêm artifact giải thích GAM:

```text
results/gam_plots/
results/gam_feature_effects.html
results/gam_interpretation.json
```

Lưu ý: nếu GAM chạy bằng Rust engine, một số API partial dependence kiểu pyGAM chưa có, nên một số biểu đồ có thể bị bỏ qua.

## 8. Model 3: CatBoost

### Mục đích

CatBoost là model dự báo mạnh cho dữ liệu bảng hỗn hợp. Nó thường dùng để kiểm tra mức hiệu năng cao hơn so với model tuyến tính, đồng thời vẫn cung cấp feature importance.

### Modern pipeline

Hàm:

```text
DepressionAnalysisService._run_catboost()
```

Preprocessing:

```text
_prepare_catboost_frame()
```

Quy trình tiền xử lý:

1. Tách categorical columns và numeric columns.
2. Numeric: ép kiểu số, fill median.
3. Categorical: ép string, fill mode.
4. Tạo `Pool(..., cat_features=categorical_cols)`.

Model params:

```text
CatBoostClassifier(
    iterations=<training_budget>,
    depth=<training_budget>,
    learning_rate=<training_budget>,
    loss_function="Logloss",
    eval_metric="AUC",
    allow_writing_files=False,
    verbose=False,
    random_seed=42,
    early_stopping_rounds=<training_budget>,
)
```

Nếu CUDA sẵn sàng:

```text
task_type = "GPU"
devices = "0"
```

Quy trình:

1. Chạy `StratifiedKFold`.
2. Với từng fold, fit CatBoost trên fold train và eval trên fold validation.
3. Lưu out-of-fold probability.
4. Fit final model trên toàn bộ train.
5. Eval trên holdout test.
6. Lấy feature importance bằng `get_feature_importance()`.
7. Tính threshold report và subgroup/fairness metrics.

Metadata chính:

```text
cat_features
used_gpu
evaluation = "oof_on_train_plus_holdout"
```

### Legacy pipeline

Hàm:

```text
DepressionRiskModeler.train_catboost()
```

Quy trình:

1. Import `CatBoostClassifier`.
2. Dùng `torch.cuda.is_available()` để quyết định GPU.
3. Tính `class_weights` theo phân phối lớp.
4. Chạy 5-fold stratified CV thủ công.
5. Fit model cuối trên toàn bộ data.
6. Tính probability, prediction, Brier score.
7. Lấy feature importance.
8. Lưu vào `self.models["catboost"]` và `self.results["catboost"]`.

Nếu chưa cài CatBoost, legacy pipeline trả về:

```text
{"error": "CatBoost not installed"}
```

## 9. Metric Và Threshold Sau Khi Train

Sau khi model tạo probability score, repo tính các nhóm kết quả sau:

| Nhóm | Nội dung |
|---|---|
| Binary metrics | accuracy, F1, recall, precision, specificity, FPR, FNR |
| Ranking metrics | ROC-AUC, PR-AUC |
| Calibration | Brier score, calibration curve |
| Threshold report | best F1, best Youden J, screening threshold |
| Fairness/subgroup | metric theo Gender, nhóm Age, Family History |
| Feature importance | odds ratio, variance importance hoặc CatBoost importance |

Modern pipeline lưu threshold trong:

```text
ModelResult.thresholds
```

Legacy pipeline in báo cáo threshold cho:

```text
logistic
gam
catboost
```

## 10. Artifact Đầu Ra

Modern pipeline:

```text
results/app/run_safe_research.json
results/app/run_full_research.json
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

## 11. Khi Nào Dùng Lệnh Nào

| Mục tiêu | Lệnh |
|---|---|
| Chạy nhanh nhưng không có GAM | `robot run --variant A --preset quick` |
| Chạy đủ 4 model bằng modern pipeline | `robot run --variant A --preset research --budget auto` |
| So sánh safe và full cùng split | `robot compare --preset research --budget auto` |
| Chạy legacy models + fairness + threshold + GAM plots | `robot task models --variant A --budget auto` |
| Chạy trực tiếp legacy bằng Python | `uv run python main.py --models --conservative` |

## 12. Lưu Ý Thực Hành

- Dùng `variant A` khi muốn giảm nguy cơ leakage từ biến suicidal thoughts.
- Dùng `variant B` khi mục tiêu là nghiên cứu so sánh full feature set, không nên diễn giải như cấu hình triển khai an toàn.
- `quick` không huấn luyện GAM trong modern pipeline; dùng `research` nếu cần đủ 4 model.
- CatBoost chỉ dùng GPU nếu PyTorch thấy CUDA qua `torch.cuda.is_available()`.
- GAM có thể dùng Rust engine hoặc pyGAM fallback; xem metadata `engine` để biết thực tế đã chạy bằng gì.
- Với bài toán sàng lọc, không chỉ xem accuracy. Cần ưu tiên ROC-AUC, PR-AUC, recall, FNR, Brier score và calibration.
