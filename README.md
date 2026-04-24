# 🧠 Phân Tích Trầm Cảm Ở Học Sinh Sinh Viên

> Dự án phân tích dữ liệu và ứng dụng machine learning để phát hiện, phân tích các mẫu hình trầm cảm trong học sinh sinh viên.

## Tài Liệu Nhanh

- CLI App và TUI Hacker Pro: [docs/CLI_APP.md](docs/CLI_APP.md)
- Phím tắt TUI: [docs/TUI_HOTKEYS.md](docs/TUI_HOTKEYS.md)
- Tổng quan tài liệu: [docs/README.md](docs/README.md)
- Review và quyền cho `docs/`: [docs/GITHUB_DOCS_QUYEN.md](docs/GITHUB_DOCS_QUYEN.md)

## 📋 Tổng Quan

Dự án này cung cấp một bộ công cụ hoàn chỉnh để phân tích trầm cảm trong cộng đồng học sinh sinh viên, bao gồm:

- **🔧 Xử lý dữ liệu**: Thao tác và phân tích dữ liệu nhanh chóng với Polars (Rust-based)
- **📊 Phân tích thống kê**: Kiểm định thống kê toàn diện với Pingouin, Statsmodels
- **🤖 Machine Learning**: XGBoost, LightGBM cho dự đoán trầm cảm
- **🔥 Deep Learning**: PyTorch với **GPU acceleration tự động** (CUDA 12.6)
- **⚡ Rust Engine**: GAM (Generalized Additive Model) viết bằng Rust — **nhanh hơn 15-50x** so với pyGAM
- **⏱️ Phân tích chuỗi thời gian**: Darts, NeuralProphet
- **📈 Trực quan hóa**: Plotly, Datashader, HoloViews

---

## 🏗️ Cấu Trúc Module

```
AnalysisOfDepressionInStudents/
├── rust_engine/                # ⚡ Rust GAM engine (15-50x nhanh hơn pyGAM)
│   ├── Cargo.toml              #   PyO3 + nalgebra + rayon
│   └── src/
│       ├── bspline.rs          #   B-spline basis functions
│       ├── pirls.rs            #   P-IRLS solver (logistic GAM)
│       ├── gam.rs              #   GAMClassifier API
│       ├── cross_val.rs        #   Parallel cross-validation (Rayon)
│       └── python_bindings.rs  #   PyO3 exports → Python
│
src/
├── data_processing/          # Polars-based data processing
├── visualization/            # Plotly interactive charts + EDA
│   └── plots.py              #   + ExploratoryAnalyzer
├── statistical_analysis/     # Pingouin, Statsmodels
├── ml_models/                # Risk modeling
│   ├── risk_model.py         #   ★ Pipeline: Baseline → Logistic → GAM → CatBoost
│   ├── gam_model.py          #   ★ GAM — tự động dùng Rust engine (fallback pyGAM)
│   ├── model_comparator.py   #   ★ So sánh mô hình toàn diện (mới)
│   ├── predictor.py          #   XGBoost/LightGBM (legacy)
│   ├── optimizer.py          #   Optuna hyperparameter tuning
│   ├── explainer.py          #   SHAP explainability
│   └── imbalanced.py         #   SMOTE handling
├── evaluation/               # Model evaluation metrics
└── utils/                    # Logging, timing, helpers
```

**Pipeline hiện tại (4 mô hình tuần tự):**

| # | Mô hình | Vai trò | Ưu điểm |
|---|---------|---------|---------|
| 0 | **Dummy Baseline** | Reference point | Luôn dự đoán lớp đa số — tránh ảo tưởng |
| 1 | **Logistic Regression** | Trung tâm, giải thích được | Odds ratio, dễ diễn giải, chuẩn thống kê |
| 2 | **GAM** (Generalized Additive Model) | Linh hoạt + interpretability | Capture nonlinear, partial dependence plots |
| 3 | **CatBoost** | Dự báo bổ sung, mạnh nhất | Xử lý tốt dữ liệu bảng hỗn hợp |

**Model Comparison (so sánh toàn diện):**
- **McNemar's test** — so sánh disagreement giữa 2 models
- **DeLong's test** — so sánh ROC-AUC curves
- **Calibration curves** — xác suất dự báo có đáng tin?
- **Decision curve analysis** — clinical utility
- **Feature importance ranking** — nhất quán giữa các models?

**Fairness analysis** theo Gender, Age group, Family History
**Threshold optimization** — chọn ngưỡng phù hợp bài toán sàng lọc
**GAM interpretability** — partial dependence plots, feature effects

**📖 Chi tiết:** Đọc [USAGE.md](USAGE.md) và [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 🚀 Quick Start

### Chạy Nhanh Nhất

```bash
# Pipeline hoàn chỉnh với sample data
uv run python main.py --sample

# Hoặc phân tích tệp dữ liệu tùy chỉnh
uv run python main.py --eda --dataset "du_lieu_cua_ban.csv"
```

### Yêu Cầu

| Thành phần | Version | Ghi chú |
|------------|---------|---------|
| **Python** | 3.14+ | Khai báo trong `.python-version` |
| **uv** | Mới nhất | `winget install astral-sh.uv` |
| **Rust** | 1.70+ | `rustup install stable` — cần thiết cho rust_engine (GAM nhanh hơn 15-50x) |
| **GPU** | NVIDIA (khuyến nghị) | GTX 1060 trở lên |
| **NVIDIA Driver** | 560.94+ | Hỗ trợ CUDA 12.6 |
| **CUDA Toolkit** | 13.x | Script tự phát hiện và chọn wheel phù hợp |

### Cài Đặt Nhanh

```powershell
# 1. Clone và vào thư mục
cd AnalysisOfDepressionInStudents

# 2. Chạy script tự động (cần quyền Administrator)
# Script sẽ tự động xin UAC nếu chưa có quyền admin
.\setup_with_gpu.ps1
```

**Script sẽ tự động làm:**
1. ✅ Phát hiện GPU + driver version qua `nvidia-smi`
2. ✅ Chọn PyTorch wheel phù hợp (cu126/cu124/cu121)
3. ✅ Cài 16 packages cần thiết
4. ✅ Cài PyTorch GPU
5. ✅ **Build rust_engine** (GAM engine bằng Rust — nhanh hơn 15-50x)
6. ✅ Verify GPU + Rust engine hoạt động
7. ✅ Unblock Python DLLs (Smart App Control)

### Kiểm Tra Cài Đặt

```bash
# Kiểm tra GPU
uv run python verify_gpu.py

# Kiểm tra Rust engine (GAM)
uv run python -c "from rust_engine import PyGAMClassifier; print('✅ rust_engine OK')"

# Kiểm tra GPU usage khi training
uv run python verify_gpu_usage.py

# Chạy config để xem device info
uv run python config.py
```

---

## 📦 Packages Đã Cài

### 🔧 Xử Lý Dữ Liệu

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[Polars](https://pola.rs/)** | DataFrame siêu nhanh (Rust) | Xử lý dữ liệu lớn nhanh gấp 10x pandas |
| **[PyArrow](https://arrow.apache.org/)** | Định dạng dữ liệu cột | Lưu trữ hiệu quả, trao đổi dữ liệu |
| **[DuckDB](https://duckdb.org/)** | SQL OLAP engine | Chạy SQL trực tiếp trên DataFrame |

### 📈 Trực Quan Hóa

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[Plotly](https://plotly.com/python/)** | Biểu đồ tương tác | Zoom, hover, click - perfect cho dashboard |
| **[Datashader](https://datashader.org/)** | Render dataset khổng lồ | Vẽ hàng triệu điểm không chồng chéo |
| **[HoloViews](https://holoviews.org/)** | Trực quan hóa khai báo | Tạo biểu đồ phức tạp với ít code |

### 🤖 Machine Learning & Thống Kê

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[CatBoost](https://catboost.ai/)** | Gradient Boosting | **Mô hình dự báo chính** — xử lý tốt dữ liệu bảng hỗn hợp, tự động xử lý biến phân loại |
| **[pyGAM](https://pygam.readthedocs.io/)** | Generalized Additive Models | **Mô hình mới** — spline-based, capture nonlinear relationships, vẫn interpretability được |
| **[scikit-learn](https://scikit-learn.org/)** | ML toolkit | Dummy baseline, Logistic Regression, cross-validation, metrics |
| **[SHAP](https://shap.readthedocs.io/)** | Explainable AI | Giải thích prediction — feature importance, force plots |
| **[XGBoost](https://xgboost.readthedocs.io/)** | Gradient Boosting | Mô hình dự đoán trầm cảm (legacy) |
| **[LightGBM](https://lightgbm.readthedocs.io/)** | Gradient Boosting nhanh | Training nhanh hơn XGBoost (legacy) |
| **[Statsmodels](https://www.statsmodels.org/)** | Mô hình thống kê | Hồi quy, kiểm định, phân tích phương sai |
| **[Factor Analyzer](https://github.com/EducationalTestingService/factor_analyzer)** | Phân tích nhân tố | Tìm yếu tố tiềm ẩn (stress, lo âu, cô đơn) |
| **[Pingouin](https://pingouin-stats.org/)** | Kiểm định thống kê | ANOVA, T-test, tương quan - dễ dùng hơn scipy |

**Core ML dependencies (transitive):** `numpy`, `pandas`, `joblib`, `scipy` — được cài tự động bởi các package trên.

### 🎯 Hyperparameter Optimization

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[Optuna](https://optuna.org/)** | Auto-tune hyperparameters | Tìm bộ tham số tốt nhất tự động |

### ⏱️ Time Series

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[Darts](https://unit8co.github.io/darts/)** | Forecasting framework | Dự đoán xu hướng trầm cảm theo thời gian |
| **[NeuralProphet](https://neuralprophet.com/)** | Neural forecasting | Deep learning cho time series |

### 🔄 Online Learning

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[VowpalWabbit](https://vowpalwabbit.org/)** | Online learning nhanh | Cập nhật mô hình real-time |

### 🏥 Survival Analysis

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[Lifelines](https://lifelines.readthedocs.io/)** | Survival analysis | Phân tích thời gian phục hồi từ trầm cảm |

### 🔥 Deep Learning

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[PyTorch](https://pytorch.org/)** | Deep learning framework | Neural networks với **GPU acceleration** |
| **torchvision** | Computer vision | Xử lý ảnh phân tích cảm xúc |
| **torchaudio** | Audio processing | Phân tích giọng nói |

---

## 🖥️ Cấu Trúc Dự Án

```
AnalysisOfDepressionInStudents/
│
├── main.py                      # Entry point - chạy phân tích
├── config.py                    # ⭐ Auto-detect CPU/GPU + Config trung tâm
├── pyproject.toml               # Dependencies khai báo
├── uv.lock                      # Lock file - đảm bảo reproducibility
├── .python-version              # Python 3.14
├── .gitignore
│
├── rust_engine/                 # ⚡ Rust GAM engine (mới)
│   ├── Cargo.toml               #   Rust dependencies
│   └── src/
│       ├── lib.rs               #   Module entry
│       ├── bspline.rs           #   B-spline basis functions
│       ├── pirls.rs             #   P-IRLS solver
│       ├── gam.rs               #   GAMClassifier
│       ├── cross_val.rs         #   Parallel CV (Rayon)
│       └── python_bindings.rs   #   PyO3 Python bindings
│
├── setup_with_gpu.ps1           # ⭐ Script tự động setup GPU + Rust
├── verify_gpu.py                # Script kiểm tra GPU
│
├── src/                         # Python source code
│   ├── data_processing/
│   ├── visualization/
│   ├── statistical_analysis/
│   ├── ml_models/
│   │   ├── risk_model.py        #   Main pipeline
│   │   └── gam_model.py         #   GAM — auto-detect Rust vs pyGAM
│   └── ...
│
└── README.md                    # File này
```

---

## 🎯 GPU Setup - Chi Tiết

### Script Tự Động (`setup_with_gpu.ps1`)

Script này là **trái tim** của việc setup GPU. Nó tự động:

```
Bước 1: Phát hiện GPU
  ├─ Đọc nvidia-smi → lấy driver max CUDA version
  ├─ Kiểm tra CUDA Toolkit đã cài
  └─ Chọn wheel phù hợp (cu126/cu124/cu121)

Bước 2: Xin quyền Administrator
  ├─ Tự động prompt UAC nếu chưa có quyền admin
  └─ Cần admin để unblock Python DLLs

Bước 3: Xử lý Smart App Control (Windows)
  ├─ Phát hiện SAC đang active hay không
  ├─ Unblock tất cả .pyd files trong Python path
  └─ Cảnh báo nếu SAC chặn Python 3.14

Bước 4: Cài 16 packages
  ├─ Thử bulk install (nhanh)
  └─ Fallback: install từng package nếu lỗi

Bước 5: Cài PyTorch GPU
  ├─ Thử uv add --index pytorch=<url>
  ├─ Fallback: uv pip install --index-url --force-reinstall
  └─ Verify: import torch → torch.cuda.is_available()

Bước 6: Auto-fix nếu driver không khớp
  ├─ Nếu cu130 không hoạt động → thử cu126/cu124/cu121
  └─ Chọn wheel khớp với driver capability

Bước 7: Build rust_engine (⭐ MỚI)
  ├─ Kiểm tra Rust/Cargo đã cài chưa
  ├─ Cài maturin (build tool)
  ├─ Build rust_engine --release
  └─ Verify: import rust_engine → PyGAMClassifier
```

### Bảng CUDA Wheels Theo Driver

| NVIDIA Driver | Max CUDA | PyTorch Wheel | Ghi chú |
|---------------|----------|---------------|---------|
| 560.94+ | 12.6 | `cu126` | ✅ GTX 1060 hỗ trợ |
| 550.x | 12.4 | `cu124` | |
| 535.x | 12.2 | `cu121` | |
| 525.x | 12.0 | `cu121` | |
| 470.x | 11.4 | `cu118` | GPU cũ |

### Smart App Control (SAC) - Vấn Đề Python 3.14

**Vấn đề:** Windows Smart App Control chặn `_overlapped.pyd` của Python 3.14, khiến PyTorch không import được.

**Giải pháp đã implement trong script:**
1. ✅ Script tự động xin quyền Administrator
2. ✅ Tự động unblock tất cả `.pyd` files
3. ✅ Phát hiện SAC và cảnh báo rõ ràng

**Nếu vẫn bị chặn sau khi chạy script:**
```
1. Windows Security → App & browser control
2. Smart App Control settings → Turn off
3. Restart computer
4. Chạy lại: .\setup_with_gpu.ps1
```

---

## ⚡ Rust Engine — GAM Nhanh Hơn 15-50x

### Tại Sao Cần Rust Engine?

Mô hình **GAM (Generalized Additive Model)** là phần chậm nhất trong pipeline:
- **pyGAM** (Python): 20-40 giây cho 1 model với cross-validation
- **rust_engine** (Rust): **0.5-2 giây** cho cùng kích thước dữ liệu

Lý do: pyGAM dùng grid search tuần tự, trong khi rust_engine:
- **P-IRLS solver** viết bằng Rust — tối ưu matrix operations với `nalgebra`
- **Cross-validation song song** với `rayon` (tự động dùng tất cả CPU cores)
- **B-spline basis** tính toán trực tiếp trong Rust — không qua Python overhead

### Cấu Trúc Rust Engine

```
rust_engine/
├── Cargo.toml              # PyO3 + nalgebra + rayon
└── src/
    ├── lib.rs              # Module entry point
    ├── bspline.rs          # B-spline basis + penalty matrix
    ├── pirls.rs            # P-IRLS solver (logistic GAM)
    ├── gam.rs              # GAMClassifier (high-level API)
    ├── cross_val.rs        # Stratified K-Fold CV (parallel)
    └── python_bindings.rs  # PyO3 exports → Python
```

### Cài Đặt Rust

```bash
# 1. Cài Rust toolchain
rustup install stable

# 2. Build rust_engine (tự động build + cài vào Python venv)
cd rust_engine
uv run maturin develop --release

# 3. Kiểm tra
uv run python -c "from rust_engine import PyGAMClassifier; print('✅ OK')"
```

**Hoặc đơn giản hơn:** chạy `.\setup_with_gpu.ps1` — script sẽ tự động build rust_engine.

### Lệnh Phát Triển (cho developer)

```bash
cd rust_engine

# Build Rust library (kiểm tra lỗi, không cài vào Python)
cargo build --release

# Build + cài vào Python venv (dùng khi sửa code Rust)
uv run maturin develop --release

# Chạy Rust tests
cargo test

# Build tối ưu (release mode)
cargo build --release
```

> **Mẹo:** Mỗi khi sửa code Rust (`.rs` files), chạy lại `uv run maturin develop --release` để rebuild và cài lại.

### Fallback Tự Động

Nếu Rust engine không build được, code sẽ **tự động fallback về pyGAM**:

```python
from src.ml_models.gam_model import GAMClassifier

gam = GAMClassifier()
# Tự động: thử rust_engine trước → nếu lỗi → dùng pyGAM
result = gam.train(X, y, feature_types, feature_names)
# result["_engine"] = "rust" hoặc "pygam"
```

### Benchmark Thực Tế

| Dataset | Rust Engine | pyGAM | Speedup |
|---------|-------------|-------|---------|
| 500 samples × 5 features | **0.07s** | ~5s | ~70x |
| 1000 samples × 6 features | **0.27s** | ~15s | ~55x |
| 2000 samples × 8 features | **0.68s** | ~30s | ~44x |
| 5000 samples × 6 features | **1.26s** | ~60s | ~48x |

> Benchmark trên Windows 11, CPU Intel i7, không có GPU.

---

## 🏃 Cách Chạy

### Chi tiết 11 chế độ chạy của `main.py`

#### 1. Không flag (mặc định) = `--eda`

```bash
uv run python main.py
```

**Chạy:** Giai đoạn 0 (cảnh báo đạo đức) + Giai đoạn 1 (EDA) + Giai đoạn 2-3 (rà soát dữ liệu)

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức & giới hạn sử dụng | In console |
| **1** | EDA — 6 biểu đồ HTML | `results/visualizations/eda_*.html` + `eda_data_profile.json` |
| **2-3** | Rà soát dữ liệu — loại cột hằng, missing, rare categories | In console |

**Không chạy:** Thống kê, mô hình ML, fairness, threshold

**Dùng khi:** Mới mở dataset lần đầu, muốn hiểu dữ liệu trước khi làm gì tiếp. ~2-3 giây.

---

#### 2. `--eda` (tường minh)

```bash
uv run python main.py --eda
```

**Giống hệt không flag.** Chỉ khác là viết rõ ràng hơn trong script/automation.

---

#### 3. `--stats`

```bash
uv run python main.py --stats
```

**Chạy:** Giai đoạn 0 + Giai đoạn 2-3 + Giai đoạn 4 (kiểm định thống kê + cỡ ảnh hưởng)

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức | In console |
| **2-3** | Rà soát dữ liệu | In console |
| **4** | **Thống kê mô tả** theo nhóm (mean, SD, median, min, max) | Bảng |
| | **Mann-Whitney U** — so sánh biến số giữa 2 nhóm | U, p, Cohen d, RBC |
| | **Chi-square** — liên hệ biến phân loại với trầm cảm | χ², df, p, Cramer's V, OR |
| | **Tương quan Spearman** — mức độ liên quan của từng biến | rho, p, 95% CI |

**Không chạy:** EDA (không sinh biểu đồ), mô hình ML

**Dùng khi:** Cần phân tích thống kê đầy đủ để viết báo cáo — có cả **kiểm định ý nghĩa** (p-value) và **cỡ ảnh hưởng** (effect size: Cohen d, Cramer's V, Odds Ratio). ~5-10 giây.

**Ngưỡng đánh giá effect size:**
- **Cohen d**: <0.2 rất nhỏ, <0.5 nhỏ, <0.8 trung bình, ≥0.8 LỚN
- **Cramer's V**: <0.1 rất nhỏ, <0.3 nhỏ, <0.5 trung bình, ≥0.5 LỚN
- **Odds Ratio**: ~1 = không liên quan, >2 = liên quan đáng kể, >5 = rất mạnh

---

#### 4. `--models`

```bash
uv run python main.py --models
```

**Chạy:** Giai đoạn 0 + Giai đoạn 2-3 + Giai đoạn 7-8-9-10-11 (4 mô hình + fairness + threshold + GAM viz + model comparison)

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức | In console |
| **2-3** | Rà soát dữ liệu | In console |
| **7-8** | Huấn luyện **4 mô hình** (Dummy → Logistic → **GAM** → CatBoost) | `results/model_results_full.json` |
| **9** | Fairness analysis — ROC-AUC, FPR, FNR theo Gender, Age group, Family History | In console — cảnh báo bias |
| **8b** | Threshold analysis — thử 30 ngưỡng (0.20→0.80) cho **từng mô hình** | In console — khuyến nghị |
| **10** | **GAM Visualizations** — partial dependence plots, feature effects summary | `results/gam_plots/*.html`, `results/gam_feature_effects.html`, `results/gam_interpretation.json` |
| **11** | **Model Comparison** — McNemar's test, DeLong's test, calibration curves, decision curves | `results/model_comparison.html`, `results/calibration_curves.html`, `results/decision_curves.html`, `results/model_comparison_report.json` |

**Không chạy:** EDA (không sinh biểu đồ), thống kê mô tả (Giai đoạn 4)

**Dùng khi:** Đã có EDA rồi, giờ muốn xây dựng mô hình và đánh giá toàn diện.

| Rust engine | Thời gian |
|-------------|-----------|
| ✅ **Có Rust** | **~10-20 giây** (GAM: ~2-3s) |
| ❌ Không (pyGAM fallback) | ~30-60 giây (GAM: ~20-40s) |

---

#### 5. `--full`

```bash
uv run python main.py --full
```

**Chạy: TẤT CẢ** — Giai đoạn 0 → 1 → 2-3 → 4 → 7-8-9-10-11

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức | In console |
| **1** | EDA — 6 biểu đồ | 6 file HTML + 1 JSON |
| **2-3** | Rà soát dữ liệu | In console |
| **4** | Thống kê mô tả theo nhóm | In console |
| **7-8** | **4 mô hình** (Dummy → Logistic → GAM → CatBoost) | `results/model_results_full.json` |
| **9** | Fairness analysis | In console |
| **8b** | Threshold analysis | In console |
| **10** | GAM Visualizations | `results/gam_plots/*.html`, `results/gam_feature_effects.html`, `results/gam_interpretation.json` |
| **11** | Model Comparison | `results/model_comparison.html`, `results/calibration_curves.html`, `results/decision_curves.html`, `results/model_comparison_report.json` |

**Dùng khi:** Muốn chạy trọn vẹn từ đầu đến cuối — sinh báo cáo hoàn chỉnh.

| Rust engine | Thời gian |
|-------------|-----------|
| ✅ **Có Rust** | **~20-30 giây** |
| ❌ Không (pyGAM fallback) | ~40-90 giây |

---

#### 6. `--review`

```bash
uv run python main.py --review
```

**Chạy:** Giai đoạn 0 + Giai đoạn 2-3 (rà soát dữ liệu chi tiết)

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức | In console |
| **2-3** | Rà soát dữ liệu — biến hằng số, biến gần-hằng-số, missing, rare categories | In console — báo cáo chi tiết |

**Không chạy:** EDA (không sinh biểu đồ), thống kê mô tả, mô hình ML

**Dùng khi:** Mới mở dataset lần đầu, muốn kiểm tra chất lượng dữ liệu trước khi quyết định phân tích gì tiếp. ~2-3 giây.

**Báo cáo bao gồm:**
- **Biến đã loại** — cột phương sai quá thấp / định danh (Profession, Work Pressure, ...)
- **Biến gần-hằng-số** — gợi ý xem xét loại thêm (>95% cùng 1 giá trị)
- **Missing values** — cột nào thiếu, bao nhiêu dòng, tỷ lệ %
- **Rare categories** — category xuất hiện <1% trong dữ liệu
- **Cảnh báo** — class imbalance, rò rỉ nhãn, ...

---

#### 7. `--leakage`

```bash
uv run python main.py --leakage
```

**Chạy:** Giai đoạn 0 + Điều tra rò rỉ nhãn (label leakage) từ biến `Suicidal thoughts`

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức | In console |
| **Leakage** | 4 phân tích: Odds Ratio, Stress Test (rò rỉ từng hàng), Cross-Tab, Synthetic Check | `results/leakage_investigation.json` + console |

**Không chạy:** EDA, thống kê mô tả, mô hình ML

**Dùng khi:** Muốn đánh giá rủi ro rò rỉ nhãn trước khi quyết định đưa biến `Suicidal thoughts` vào mô hình. ~5-10 giây.

**Kết quả bao gồm:**
- **Odds Ratio** — mức độ liên quan giữa Suicidal thoughts và Depression
- **Stress Test** — phát hiện hàng cụ thể bị rò rỉ nhãn
- **Cross-Tabulation** — bảng chéo Suicidal × Depression
- **Synthetic Check** — kiểm tra xem biến có phải là "proxy" cho nhãn không

⚠️ Nếu Odds Ratio > 10 → **nguy cơ cao** rò rỉ nhãn, cần cân nhắc Phiên bản A (không có biến này).

---

#### 8. `--standardize`

```bash
uv run python main.py --standardize
```

**Chạy:** Giai đoạn 0 + Chuẩn hóa biểu diễn dữ liệu

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức | In console |
| **Chuẩn hóa** | 3 bước: rename cột → chuẩn hóa giá trị → phân loại biến | In console — báo cáo chi tiết |

**Không chạy:** EDA, review, stats, models, leakage

**Dùng khi:** Muốn xem dataset sẽ trông như thế nào sau khi chuẩn hóa, và biết trước feature matrix sẽ có bao nhiêu cột. ~2-3 giây.

**Báo cáo bao gồm:**
- **Rename cột** — tên cũ → tên mới (snake_case tiếng Việt không dấu)
- **Chuẩn hóa giá trị** — Male→Nam, Yes→Co, Less than 5 hours→Duoi_5h, ...
- **Phân loại biến** — ID, Target, Numeric, Ordinal, Nominal
- **Ước lượng feature matrix** — ~104 features sau encoding (3 numeric + 7 ordinal + 94 one-hot)

**💡 Mẹo:** `--review` tự động chạy kèm standardize, không cần gõ cả 2 flag.

---

#### 9. `--famd`

```bash
uv run python main.py --famd
```

**Chạy:** Giai đoạn 0 + FAMD (Factor Analysis of Mixed Data)

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức | In console |
| **FAMD** | Giảm chiều dữ liệu hỗn hợp (numeric + categorical) + clustering | HTML cho từng cặp component liền kề + K-Means/DBSCAN report |

**Không chạy:** EDA, stats, models, leakage

**Dùng khi:** Muốn hiểu cấu trúc tiềm ẩn của dữ liệu — biến nào nhóm lại với nhau, sample nào cụm lại, cần bao nhiêu chiều để giữ 80% phương sai. ~10 giây.

**Kết quả bao gồm:**
- **Scree plot** — phương sai giải thích bởi mỗi thành phần
- **Contribution plots** — top biến đóng góp cho từng component F1..F10
- **Correlation circle / sample projection** — các cặp F1-F2, F2-F3, ..., F9-F10
- **Clustering** — K-Means + DBSCAN trên tọa độ FAMD, giải thích cụm bằng tiếng Việt
- **Sample projection** — phân bố mẫu trên mặt phẳng PC, tô màu trầm cảm
- **Top contributing variables** — biến nào đóng góp nhiều nhất vào mỗi PC
- **Bảng eigenvalues** — eigenvalue, variance %, cumulative %

---

#### 10. `--split`

```bash
uv run python main.py --split
```

**Chạy:** Giai đoạn 0 + Chia tập train/test stratified

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức | In console |
| **Split** | Chia 80/20, stratify theo Depression + Gender + Family History | Console + `results/split_report.json` |

**Không chạy:** EDA, stats, models, leakage, famd

**Dùng khi:** Kiểm tra xem việc chia tập có cân bằng phân phối không trước khi huấn luyện mô hình. ~3 giây.

**Báo cáo bao gồm:**
- **Kích thước** — tổng, train, test và tỷ lệ %
- **Phân phối target** — so sánh tỷ lệ lớp giữa total/train/test
- **Cân bằng biến nhân khẩu học** — Gender, Family History (max diff ≤ 1% là OK)
- **Kiểm định KS** — so sánh phân phối từng biến số giữa train/test (p > 0.05 → giống nhau)

---

#### 11. `--no-ethical`

```bash
uv run python main.py --models --no-ethical
```

**Tác dụng duy nhất:** Bỏ qua phần in cảnh báo đạo đức ở Giai đoạn 0.

**Không thay đổi:** Mọi thứ khác (EDA, stats, models, fairness, threshold) chạy y hệt.

**Dùng khi:** Chạy script tự động, CI/CD, hoặc đã quá quen thuộc và không muốn đọc cảnh báo mỗi lần.

**⚠️ Không khuyến nghị** vì cảnh báo đạo đức là phần cốt lõi định hướng triết lý của dự án.

---

#### 12. `--fairness`

```bash
uv run python main.py --fairness --conservative
```

**Chạy:** ⚖️ Fairness Analysis — đánh giá tính công bằng của mô hình theo các nhóm nhân khẩu học.

| Phân tích | Nội dung | Output |
|-----------|----------|--------|
| **Demographic Parity** | Tỷ lệ dự đoán dương tính có giống nhau giữa các nhóm? | JSON + HTML dashboard |
| **Equalized Odds** | TPR (Recall) và FPR có tương đương giữa các nhóm? | JSON + HTML dashboard |
| **Predictive Parity** | Precision có tương đương giữa các nhóm? | JSON + HTML dashboard |
| **Disparate Impact** | Tỷ lệ chấp nhận (4/5 rule) — nhóm thiểu số có bị thiệt thòi? | JSON + HTML dashboard |
| **Theil Index** | Độ bất bình đẳng trong confidence của dự đoán | JSON + HTML dashboard |

**Nhóm được phân tích:** Gender, Age, Family History of Mental Illness

**Dùng khi:** Muốn đảm bảo mô hình không phân biệt đối xử — đặc biệt quan trọng cho ứng dụng y tế/học đường.

**File sinh ra:**
- `results/fairness_logistic.json`, `results/fairness_catboost.json` — metrics chi tiết
- `results/fairness_dashboard_logistic.html`, `results/fairness_dashboard_catboost.html` — interactive dashboard

**️ Thời gian:** ~1-2 giây (sau khi models đã train).

---

#### 13. `--subgroups`

```bash
uv run python main.py --subgroups --conservative
```

**Chạy:** 🔍 Subgroup Analysis — phân tích hiệu suất mô hình chi tiết theo từng nhóm con.

| Phân tích | Nội dung | Output |
|-----------|----------|--------|
| **Performance breakdown** | AUC, F1, Precision, Recall, FPR, FNR theo từng subgroup | JSON + HTML dashboard |
| **Error analysis** | False negatives/positives — nhóm nào bị bỏ sót nhiều nhất? Confidence của lỗi? | JSON + HTML dashboard |
| **Calibration by subgroup** | ECE (Expected Calibration Error) — mô hình có calibrated tốt cho mọi nhóm? | JSON + HTML dashboard |
| **Threshold recommendations** | Ngưỡng tối ưu (F1-optimal, Cost-optimal) cho từng nhóm | JSON + HTML dashboard |

**Nhóm được phân tích:** Gender, Age groups, Family History, Top 10 Cities, Academic Pressure levels

**Dùng khi:** Muốn hiểu mô hình hoạt động thế nào với từng nhóm cụ thể — tìm nhóm yếu để cải thiện.

**File sinh ra:**
- `results/subgroup_logistic.json`, `results/subgroup_catboost.json` — metrics chi tiết
- `results/subgroup_dashboard_logistic.html`, `results/subgroup_dashboard_catboost.html` — interactive dashboard

**⏱️ Thời gian:** ~3-5 giây (sau khi models đã train).

---

#### 14. `--robustness`

```bash
uv run python main.py --robustness --conservative
```

**Chạy:** 🛡️ Robustness Analysis — kiểm tra độ bền của mô hình trước nhiễu và biến đổi dữ liệu.

| Phân tích | Nội dung | Output |
|-----------|----------|--------|
| **Bootstrap CI** | Confidence intervals qua 500 bootstrap samples — độ ổn định của metrics? | JSON + HTML dashboard |
| **CV Stability** | Performance qua 5-fold cross-validation — mô hình có ổn định? | JSON + HTML dashboard |
| **Noise Injection** | Thêm nhiễu Gaussian vào features — AUC giảm bao nhiêu? | JSON + HTML dashboard |
| **Feature Ablation** | Bỏ từng nhóm features (City, Academic, Sleep...) — impact lên AUC? | JSON + HTML dashboard |
| **Adversarial Label Flip** | Đảo ngẫu nhiên labels — mô hình sụp ở mức nhiễu nào? | JSON + HTML dashboard |

**Dùng khi:** Muốn đảm bảo mô hình đáng tin cậy — không quá nhạy cảm với nhiễu hoặc thay đổi nhỏ.

**File sinh ra:**
- `results/robustness_logistic.json`, `results/robustness_catboost.json` — metrics chi tiết
- `results/robustness_dashboard_logistic.html`, `results/robustness_dashboard_catboost.html` — interactive dashboard

**⏱️ Thời gian:** ~2-5 phút (Logistic) / ~5-10 phút (CatBoost) — do cần retrain nhiều lần.

---

#### 15. `--analysis`

```bash
uv run python main.py --analysis --conservative
```

**Chạy:** 📊 **Tất cả 3 phân tích nâng cao** — Fairness + Subgroup + Robustness.

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **Models** | Huấn luyện 4 mô hình (Dummy → Logistic → GAM → CatBoost) | `results/model_results_*.json` |
| **Fairness** | Demographic Parity, Equalized Odds, Disparate Impact, Theil Index | 2 JSON + 2 HTML |
| **Subgroup** | Performance breakdown, Error analysis, Calibration, Threshold recs | 2 JSON + 2 HTML |
| **Robustness** | Bootstrap CI, CV Stability, Noise Injection, Feature Ablation, Label Flip | 2 JSON + 2 HTML |

**Không chạy:** EDA, stats, leakage, review, standardize, famd, split

**Dùng khi:** Muốn đánh giá toàn diện mô hình — từ fairness đến robustness — trước khi deploy hoặc báo cáo.

**File sinh ra:** **12 file** (6 JSON + 6 HTML dashboards) + output console chi tiết.

**⏱️ Thời gian:** ~8-15 phút (tùy model — CatBoost robustness lâu nhất).

---

#### 16. `--report`

```bash
uv run python main.py --report
```

**Chạy:** 📝 Auto-generate comprehensive report — tổng hợp tất cả kết quả thành 1 file Markdown + HTML.

| Nội dung | Mô tả |
|----------|-------|
| **Executive Summary** | Key findings ở 1 trang — mô hình tốt nhất, fairness issues, robustness grade |
| **Model Performance** | Bảng so sánh Dummy, Logistic Regression, GAM, CatBoost trên Safe A / Full B |
| **Model Evidence** | Biểu đồ metric, feature importance, calibration/Brier, confusion matrix, lý do model yếu/mạnh |
| **Fairness Analysis** | Disparate Impact, Equalized Odds warnings cho từng model |
| **Subgroup Analysis** | Performance breakdown + Error analysis + Threshold recommendations |
| **Robustness Analysis** | Bootstrap CI, CV Stability, Noise robustness, Feature ablation |
| **Recommendations** | Khuyến nghị actionable — model selection, fairness, data quality |
| **Appendix** | Links đến tất cả file JSON + HTML dashboards |

**Dùng khi:** Cần báo cáo tổng hợp để trình bày, chia sẻ, hoặc lưu trữ — không cần mở từng file riêng lẻ.

**File sinh ra:**
- `results/final_report.md` — Markdown report (dễ đọc, dễ chia sẻ)
- `results/final_report.html` — HTML report (có styling, dễ xem trên browser)
- `results/best_model_selection.json` — model được chọn, profile, metric và lý do
- `results/model_evidence_metrics.html`, `results/model_feature_importance_safe.html` — biểu đồ bằng chứng

**⏱️ Thời gian:** ~2 giây (sau khi đã có kết quả analysis).

---

### Bảng tổng hợp

| Flag | EDA | Stats | Models (4) | GAM Viz | Model Comp | Fairness | Threshold | Leakage | Review | Standardize | FAMD | Split | Fair | Subgroup | Robust | Thời gian ~ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *(không flag)* | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 2-3s |
| `--eda` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 2-3s |
| `--review` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 2-3s |
| `--standardize` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 2-3s |
| `--stats` | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 5-10s |
| `--famd` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | 10s |
| `--split` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 3s |
| `--models` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 10-20s* |
| `--leakage` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 5-10s |
| `--full` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 20-30s* |
| `--fairness` | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | 1-2s** |
| `--subgroups` | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 3-5s** |
| `--robustness` | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 5-15m** |
| `--analysis` | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | 8-15m** |
| `--report` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 2s*** |
| `--no-ethical` | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | Bỏ cảnh báo |

> **\*** Thời gian với Rust engine. Không có Rust (pyGAM fallback): gấp 2-4x chậm hơn.
> **\*\*** Thời gian sau khi models đã train (train models ~2-5 phút).
> **\*\*\*** Thời gian sau khi đã có kết quả analysis (load JSON + generate report).

### Flag bổ trợ

| Flag | Tác dụng | Dùng với |
|------|----------|----------|
| `--conservative` | Dùng **Phiên bản A** — không có biến `Suicidal thoughts` (an toàn, không rủi ro rò rỉ nhãn) | `--models`, `--full`, `--fairness`, `--subgroups`, `--robustness`, `--analysis` |
| `--no-ethical` | Bỏ qua cảnh báo đạo đức ở Giai đoạn 0 | Mọi flag |
| `--standardize` | Chuẩn hóa tên cột + giá trị + phân loại biến | Đứng riêng hoặc kèm `--review`, `--full` |

**Ví dụ:**
```bash
uv run python main.py --models --conservative        # Huấn luyện Phiên bản A
uv run python main.py --full --conservative          # Toàn bộ pipeline, Phiên bản A
uv run python main.py --analysis --conservative      # Fairness + Subgroup + Robustness, Phiên bản A
uv run python main.py --fairness --conservative      # Chỉ Fairness, Phiên bản A
uv run python main.py --robustness --no-ethical      # Robustness, bỏ cảnh báo
uv run python main.py --report                       # Tạo báo cáo tổng hợp từ tất cả kết quả
uv run python main.py --analysis --report            # Chạy analysis + tạo report luôn
```

### 📁 Tổng hợp Output theo Flag

| Flag | File sinh ra | Output console |
|------|-------------|----------------|
| *(không flag)* / `--eda` | 6 HTML + 1 JSON | Cảnh báo đạo đức, EDA summary, data review |
| `--review` | — | Biến hằng số, missing values, rare categories, cảnh báo |
| `--standardize` | — | Rename cột, chuẩn hóa giá trị, phân loại biến, feature estimate |
| `--stats` | — | Thống kê mô tả, Mann-Whitney U (Cohen d), Chi-square (Cramer V, OR), Spearman |
| `--famd` | 20+ HTML + JSON/CSV | Eigenvalues, top biến đóng góp từng component, FAMD adjacent pairs, K-Means/DBSCAN clustering |
| `--split` | 1 JSON (`split_report.json`) | Kích thước train/test, phân phối target, KS test |
| `--leakage` | 1 JSON (`leakage_investigation.json`) | Odds Ratio, Stress Test, Cross-Tab, Synthetic Check |
| `--models` | 1 JSON + **10+ HTML** + 2 JSON | 4 mô hình, Fairness, Threshold, **GAM plots**, **Model comparison charts** |
| `--full` | **16+ HTML** + 5-6 JSON | **Tất cả** các output trên |
| `--fairness` | 2 JSON + 2 HTML | Fairness metrics (Demographic Parity, Equalized Odds, Disparate Impact, Theil Index) |
| `--subgroups` | 2 JSON + 2 HTML | Subgroup performance, error analysis, calibration, threshold recommendations |
| `--robustness` | 2 JSON + 2 HTML | Bootstrap CI, CV stability, noise injection, feature ablation, adversarial label flip |
| `--analysis` | **6 JSON + 6 HTML** | **Tất cả 3 phân tích nâng cao** — Fairness + Subgroup + Robustness |

**Chi tiết file từ `--models` và `--full`:**
- `results/model_results_*.json` — metrics của 4 mô hình
- `results/gam_plots/gam_partial_dependence_*.html` — partial dependence plots (top 8 features)
- `results/gam_plots/gam_partial_dependence_combined.html` — combined GAM plots
- `results/gam_feature_effects.html` — feature importance ranking
- `results/gam_interpretation.json` — GAM interpretation report
- `results/model_comparison.html` — bar chart so sánh models
- `results/calibration_curves.html` — calibration curves
- `results/decision_curves.html` — decision curve analysis
- `results/model_comparison_report.json` — McNemar's test, DeLong's test, rankings

**Chi tiết file từ `--analysis`, `--fairness`, `--subgroups`, `--robustness`:**
- `results/fairness_logistic.json`, `results/fairness_catboost.json` — fairness metrics chi tiết
- `results/fairness_dashboard_*.html` — interactive fairness dashboard (6 biểu đồ + table)
- `results/subgroup_logistic.json`, `results/subgroup_catboost.json` — subgroup performance chi tiết
- `results/subgroup_dashboard_*.html` — interactive subgroup dashboard (6 biểu đồ + table)
- `results/robustness_logistic.json`, `results/robustness_catboost.json` — robustness metrics chi tiết
- `results/robustness_dashboard_*.html` — interactive robustness dashboard (6 biểu đồ + summary table)

**Chi tiết file từ `--report`:**
- `results/final_report.md` — Markdown report (dễ đọc, copy, share)
- `results/final_report.html` — HTML report (có styling, mở trên browser)

---

## 📖 Hướng Dẫn Đọc & Diễn Giải Kết Quả

### 1. Dashboard HTML — Cách đọc

Mỗi dashboard HTML có 6 biểu đồ + 1 bảng. Cách đọc nhanh:

| Biểu đồ | Ý nghĩa | Cách đọc |
|---------|---------|----------|
| **Demographic Parity** | Tỷ lệ dự đoán dương tính giữa các nhóm | Cột cao hơn = nhóm bị dự đoán dương tính nhiều hơn. Lý tưởng: các cột bằng nhau |
| **Equalized Odds** | TPR vs FPR scatter plot | Điểm gần nhau = công bằng. Điểm xa = bias |
| **Disparate Impact** | Tỷ lệ chấp nhận theo nhóm | Đường đỏ = ngưỡng 0.80. Cột dưới đường = vi phạm 4/5 rule |
| **Predictive Parity** | Precision theo nhóm | Cột chênh lệch lớn = precision không đồng đều |
| **Theil Index** | Bất bình đẳng confidence | Cột cao = nhóm có confidence phân bố không đều |
| **Performance Table** | AUC, F1, FPR, FNR theo subgroup | Dòng có FNR cao = nhóm bị bỏ sót nhiều |

### 2. Ngưỡng Cảnh Báo

| Metric | Tốt | Cảnh báo | Nguy hiểm |
|--------|-----|----------|-----------|
| **ROC-AUC** | > 0.90 | 0.80-0.90 | < 0.80 |
| **Brier Score** | < 0.15 | 0.15-0.25 | > 0.25 |
| **Disparate Impact** | > 0.80 | 0.70-0.80 | < 0.70 |
| **FPR difference** | < 0.05 | 0.05-0.10 | > 0.10 |
| **FNR difference** | < 0.05 | 0.05-0.10 | > 0.10 |
| **ECE (calibration)** | < 0.05 | 0.05-0.10 | > 0.10 |
| **Bootstrap CI width** | < 0.02 | 0.02-0.05 | > 0.05 |
| **CV AUC range** | < 0.05 | 0.05-0.10 | > 0.10 |
| **Noise robustness** | Grade A-B | Grade C | Grade D-F |

### 3. Diễn Giải Fairness

**Disparate Impact (DI) < 0.80:**
- Nghĩa là: Nhóm thiểu số được dự đoán dương tính ít hơn 80% so với nhóm đa số
- Nguyên nhân thường: Prevalence khác nhau giữa các nhóm, hoặc model học bias từ data
- Hành động: Fine-tune threshold per subgroup, hoặc dùng reweighting

**Equalized Odds violation (FPR/FNR khác biệt > 0.10):**
- FPR cao hơn ở nhóm A = Nhóm A bị báo động giả nhiều hơn → tốn chi phí follow-up
- FNR cao hơn ở nhóm A = Nhóm A bị bỏ sót nhiều hơn → rủi ro an toàn
- Hành động: Thu thập thêm data cho nhóm yếu, hoặc dùng separate models

### 4. Diễn Giải Subgroup

**FNR cao (> 0.30) ở nhóm nào:**
- Nhóm đó đang bị **bỏ sót** — nhiều ca trầm cảm không được phát hiện
- Trong screening, FNR cao = rủi ro cao → cần ưu tiên fix

**FPR cao (> 0.30) ở nhóm nào:**
- Nhóm đó đang bị **báo động giả** nhiều — tốn chi phí đánh giá lại
- Acceptable trong screening (thà báo nhầm còn hơn bỏ sót), nhưng cần monitoring

**Threshold khác nhau giữa nhóm:**
- Nếu F1-optimal threshold khác nhau > 0.10 → nên dùng **separate thresholds**
- Ví dụ: Age 18-22 threshold=0.44, Age 31+ threshold=0.30

### 5. Diễn Giải Robustness

**Bootstrap CI width > 0.05:**
- Model không ổn định — kết quả có thể khác nhiều với data mới
- Nguyên nhân: Dataset nhỏ, hoặc model overfit
- Hành động: Thu thập thêm data, hoặc đơn giản hóa model

**Noise robustness Grade D-F:**
- Model dễ bị ảnh hưởng bởi nhiễu — không đáng tin trong production
- Nguyên nhân: Model phức tạp quá, hoặc features có nhiễu cao
- Hành động: Feature selection, regularization, hoặc dùng model đơn giản hơn

**Feature ablation ΔAUC > 0.05:**
- Feature group đó rất quan trọng — không nên bỏ
- Nếu ΔAUC < 0.01 → feature group đó ít đóng góp, có thể bỏ để đơn giản hóa

### 6. Workflow Khuyến Nghị

```
Bước 1: Chạy analysis
  uv run python main.py --analysis --conservative --no-ethical

Bước 2: Xem report tổng hợp
  uv run python main.py --report

Bước 3: Mở HTML dashboards để zoom vào chi tiết
  • results/fairness_dashboard_catboost.html
  • results/subgroup_dashboard_catboost.html
  • results/robustness_dashboard_catboost.html

Bước 4: Đọc final_report.html để có overview + recommendations

Bước 5: Nếu có vấn đề → fine-tune threshold hoặc thu thập thêm data
```

### File khác

```bash
uv run python config.py       # Xem device info (CPU/GPU)
uv run python verify_gpu.py   # Kiểm tra GPU hoạt động
```

### Mô hình mặc định: Phiên bản B (Đầy đủ)

Pipeline mặc định dùng **Phiên bản B** — có biến `Have you ever had suicidal thoughts ?`.
Đây là phiên bản có hiệu năng cao nhất nhưng cần lưu ý nguy cơ rò rỉ nhãn
(OR = 12.388 cho biến này).

**Ngưỡng khuyến nghị:**
- **Logistic Regression**: threshold = 0.36 (Recall = 0.91, FNR = 9.2%)
- **CatBoost**: threshold = 0.38 (Recall = 0.92, FNR = 8.0%)

---

## 💻 GPU Code - Không Cần Viết Lại!

### `config.py` - Auto-Detect Device

File này tự động detect GPU và cung cấp helper functions:

```python
from config import Config, to_device, get_device

# Xem device info
Config.print_config()
# Output:
# 🎯 DEVICE: cuda:0
#    Name: NVIDIA GeForce GTX 1060 3GB
#    CUDA Version: 12.6
#    ✅ GPU acceleration enabled!

# Sử dụng trong code
model = MyModel().to(Config.DEVICE)  # Tự động CPU hoặc GPU
X = to_device(X)                      # Shortcut
```

### XGBoost/LightGBM - Tự Động GPU

```python
import xgboost as xgb
from config import Config

use_gpu = Config.DEVICE.type == "cuda"

model = xgb.XGBClassifier(
    tree_method='gpu_hist' if use_gpu else 'hist',
    n_estimators=100
)
model.fit(X_train, y_train)  # Tự động dùng GPU!
```

### CatBoost - Tự Động GPU

```python
import torch
from src.ml_models import DepressionRiskModeler

# Pipeline tự động detect GPU cho CatBoost
modeler = DepressionRiskModeler()
results = modeler.run_full_pipeline(df)

# Console output:
# 💻 DEVICE INFO:
#      ✅ GPU: NVIDIA GeForce GTX 1060 3GB
#      ✅ CUDA: 12.6
#      ✅ CatBoost sẽ dùng GPU acceleration
```

CatBoost sẽ **tự động bật GPU** nếu `torch.cuda.is_available() == True`.
Không cần config thêm — pipeline tự detect và set `task_type='GPU'`.

### ⚡ Rust Engine - GAM Tự Động Dùng

```python
from src.ml_models.gam_model import GAMClassifier

gam = GAMClassifier()
result = gam.train(X, y, feature_types, feature_names)
# result["_engine"] = "rust" hoặc "pygam" (fallback tự động)
```

> ⚠️ **Lưu ý về hiệu năng GAM:**
> - GAM **không phải mô hình mạnh nhất** trong pipeline này
> - Trên dữ liệu thực tế (100 features, 27K samples): GAM ~0.80 AUC, Logistic ~0.92, CatBoost ~0.94
> - Rust engine (block-diagonal P-IRLS) **nhanh hơn pyGAM 15-50x** nhưng accuracy thấp hơn do bỏ qua cross-feature correlations
> - **Nên dùng GAM cho**: interpretability, partial dependence plots, visualizations
> - **Nên dùng CatBoost/Logistic cho**: accuracy cao nhất, production deployment
>
> GAM vẫn hoạt động và có giá trị riêng — chỉ đừng kỳ vọng nó đánh bại tree-based models.

### PyTorch - Code Giống Hệt CPU/GPU

```python
import torch
from config import Config

# Model
model = torch.nn.Linear(10, 5).to(Config.DEVICE)

# Data
X = torch.randn(32, 10).to(Config.DEVICE)

# Training loop - GIỐNG HỆT!
output = model(X)
loss = criterion(output, y)
loss.backward()
optimizer.step()
```

---

## 🔍 Ví Dụ Thực Tế

### Phân Tích Dữ Liệu

```python
import polars as pl

df = pl.read_csv("data/depression_scores.csv")
print(df.describe())

# Lọc sinh viên có nguy cơ
at_risk = df.filter(pl.col("depression_score") > 16)
print(f"Số sinh viên có nguy cơ: {len(at_risk)}")

# Nhóm theo giới tính
by_gender = df.group_by("gender").agg(
    pl.col("depression_score").mean().alias("avg_score")
)
```

### Thống Kê

```python
import pingouin as pg

# T-test: So sánh nam/nữ
t_test = pg.ttest(
    df.filter(pl.col("gender") == "Nam")["depression_score"],
    df.filter(pl.col("gender") == "Nữ")["depression_score"]
)

# Tương quan
corr = df.select(["sleep_hours", "depression_score"]).corr()
```

### Huấn Luyện Mô Hình

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Trực Quan Hóa

```python
import plotly.express as px

fig = px.scatter(
    df.to_pandas(),
    x="gpa", y="depression_score", color="gender",
    title="Mối Quan Hệ GPA và Điểm Trầm Cảm"
)
fig.show()
```

---

## 🎯 GPU Usage Verification

### Pipeline dùng GPU như thế nào?

| Mô hình | Device | Lý do |
|---------|--------|-------|
| Dummy Baseline | CPU | Trivial model, không cần GPU |
| Logistic Regression (sklearn) | CPU | scikit-learn không hỗ trợ GPU cho LR |
| GAM (pygam) | CPU | pygam không hỗ trợ GPU |
| **CatBoost** | **✅ GPU** | Hỗ trợ CUDA acceleration |

### Cách kiểm tra GPU đang được dùng

**Cách 1: Chạy script verify**
```bash
uv run python verify_gpu_usage.py
```

**Cách 2: Xem console output khi chạy `--models`**
```
💻 DEVICE INFO:
     ✅ GPU: NVIDIA GeForce GTX 1060 3GB
     ✅ CUDA: 12.6
     ✅ CatBoost sẽ dùng GPU acceleration
```

**Cách 3: Monitor Task Manager**
1. Chạy: `uv run python main.py --models`
2. Mở Task Manager → Performance → GPU 0
3. Xem GPU utilization tăng khi CatBoost training (30-60s)

**Cách 4: Check log file**
```powershell
# Sau khi chạy --models, xem logs/analysis.log
findstr /c:"CatBoost using device" /c:"GPU enabled" logs/analysis.log
```

Expected output:
```
CatBoost using device: GPU
  ✅ CatBoost GPU enabled: NVIDIA GeForce GTX 1060 3GB
```

### Nếu GPU không được dùng

**Kiểm tra:**
```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

- Nếu `False` → Chạy lại `.\setup_with_gpu.ps1` với quyền Administrator
- Nếu `True` nhưng CatBoost vẫn CPU → Kiểm tra log để debug

---

## 🔧 Troubleshooting

### Lỗi: Module Not Found

```bash
uv sync              # Cài lại packages
uv run python main.py
```

### Lỗi: PyTorch Không Nhận GPU

```bash
# Kiểm tra
uv run python verify_gpu.py

# Chạy lại setup với admin
# Right-click PowerShell → Run as Administrator
.\setup_with_gpu.ps1
```

### Lỗi: Smart App Control Chặn Python

```
1. Windows Security → App & browser control
2. Smart App Control → Turn off
3. Restart
4. Chạy lại: .\setup_with_gpu.ps1 (với quyền admin)
```

### Lỗi: Hết RAM

```python
# Dùng Polars lazy evaluation
df = pl.scan_csv("large_file.csv")
result = df.filter(pl.col("score") > 10).collect()
```

---

## 📝 Lưu Ý Quan Trọng

1. **Tiếp cận đạo đức**: Dự án này **KHÔNG** nhằm chẩn đoán trầm cảm.
   Mô hình là công cụ **HỖ TRỢ sàng lọc**, không thay thế đánh giá lâm sàng.
   Đọc `SCIENTIFIC_ANALYSIS_PLAN.md` để hiểu triết lý và giới hạn sử dụng.

2. **Phiên bản mô hình mặc định**: Phiên bản B (có `Suicidal thoughts`).
   Biến này có **OR = 12.388** — nguy cơ rò rỉ nhãn.
   Nếu muốn dùng phiên bản an toàn hơn, sửa `main.py` → `include_suicidal=False`.

3. **Python 3.14**: Dự án dùng Python 3.14. Một số packages cũ có thể cần build từ source.

4. **GPU Support**:
   - **NVIDIA**: Hoạt động tốt nhất (CUDA)
   - **AMD**: Cần WSL2 trên Windows
   - **Intel**: Hỗ trợ hạn chế

5. **Windows + SAC**: Smart App Control có thể chặn Python 3.14.
   Script tự động unblock nhưng nếu vẫn lỗi, cần tắt SAC thủ công.

6. **uv.lock**: Luôn commit file này để đảm bảo reproducibility.

7. **VowpalWabbit**: Thay thế cho River (không tương thích Python 3.14).

8. **Dữ liệu không commit**: File `*.csv`, `results/`, `logs/`, `models/`
   đều nằm trong `.gitignore` — tự sinh lại được.

---

## 🤝 Đóng Góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Tạo Pull Request

---

## 📄 License

Dự án phục vụ mục đích học thuật và nghiên cứu.

---

**Chúc Phân Tích Vui Vẻ! 🎓✨**
