# 🧠 Phân Tích Trầm Cảm Ở Học Sinh Sinh Viên

> Dự án phân tích dữ liệu và ứng dụng machine learning để phát hiện, phân tích các mẫu hình trầm cảm trong học sinh sinh viên.

## 📋 Tổng Quan

Dự án này cung cấp một bộ công cụ hoàn chỉnh để phân tích trầm cảm trong cộng đồng học sinh sinh viên, bao gồm:

- **🔧 Xử lý dữ liệu**: Thao tác và phân tích dữ liệu nhanh chóng với Polars (Rust-based)
- **📊 Phân tích thống kê**: Kiểm định thống kê toàn diện với Pingouin, Statsmodels
- **🤖 Machine Learning**: XGBoost, LightGBM cho dự đoán trầm cảm
- **🔥 Deep Learning**: PyTorch với **GPU acceleration tự động** (CUDA 12.6)
- **⏱️ Phân tích chuỗi thời gian**: Darts, NeuralProphet
- **📈 Trực quan hóa**: Plotly, Datashader, HoloViews

---

## 🏗️ Cấu Trúc Module

```
src/
├── data_processing/          # Polars-based data processing
├── visualization/            # Plotly interactive charts + EDA
│   └── plots.py              #   + ExploratoryAnalyzer
├── statistical_analysis/     # Pingouin, Statsmodels
├── ml_models/                # Risk modeling
│   ├── risk_model.py         #   ★ Logistic Regression + CatBoost + Fairness
│   ├── predictor.py          #   XGBoost/LightGBM (legacy)
│   ├── optimizer.py          #   Optuna hyperparameter tuning
│   ├── explainer.py          #   SHAP explainability
│   └── imbalanced.py         #   SMOTE handling
├── evaluation/               # Model evaluation metrics
└── utils/                    # Logging, timing, helpers
```

**Phương pháp hiện tại:**
- **Logistic Regression** (trung tâm) — dễ giải thích, báo cáo odds ratio
- **CatBoost** (dự báo bổ sung) — mạnh nhất trên dữ liệu bảng hỗn hợp
- **Fairness analysis** theo Gender, Age group, Family History
- **Threshold optimization** — chọn ngưỡng phù hợp bài toán sàng lọc

**📖 Chi tiết:** Đọc [USAGE.md](USAGE.md) và [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 🚀 Quick Start

### Chạy Nhanh Nhất

```bash
# Pipeline hoàn chỉnh với sample data
uv run python main.py --sample
```

### Yêu Cầu

| Thành phần | Version | Ghi chú |
|------------|---------|---------|
| **Python** | 3.14+ | Khai báo trong `.python-version` |
| **uv** | Mới nhất | `winget install astral-sh.uv` |
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
5. ✅ Verify GPU hoạt động
6. ✅ Unblock Python DLLs (Smart App Control)

### Kiểm Tra Cài Đặt

```bash
# Kiểm tra GPU
uv run python verify_gpu.py

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
| **[XGBoost](https://xgboost.readthedocs.io/)** | Gradient Boosting | Mô hình dự đoán trầm cảm |
| **[LightGBM](https://lightgbm.readthedocs.io/)** | Gradient Boosting nhanh | Training nhanh hơn XGBoost |
| **[Statsmodels](https://www.statsmodels.org/)** | Mô hình thống kê | Hồi quy, kiểm định, phân tích phương sai |
| **[Factor Analyzer](https://github.com/EducationalTestingService/factor_analyzer)** | Phân tích nhân tố | Tìm yếu tố tiềm ẩn (stress, lo âu, cô đơn) |
| **[Pingouin](https://pingouin-stats.org/)** | Kiểm định thống kê | ANOVA, T-test, tương quan - dễ dùng hơn scipy |

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
├── setup_with_gpu.ps1           # ⭐ Script tự động setup GPU
├── verify_gpu.py                # Script kiểm tra GPU
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

## 🏃 Cách Chạy

### Chi tiết 8 chế độ chạy của `main.py`

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

**Chạy:** Giai đoạn 0 + Giai đoạn 2-3 + Giai đoạn 4 (thống kê mô tả theo nhóm trầm cảm)

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức | In console |
| **2-3** | Rà soát dữ liệu | In console |
| **4** | Thống kê mô tả theo nhóm (mean, std, min, max cho từng biến số, tách Depression=0 vs 1) | In console — bảng polars |

**Không chạy:** EDA (không sinh biểu đồ), mô hình ML

**Dùng khi:** Cần số liệu thống kê nhanh để viết báo cáo, không cần biểu đồ hay mô hình. ~5-10 giây.

---

#### 4. `--models`

```bash
uv run python main.py --models
```

**Chạy:** Giai đoạn 0 + Giai đoạn 2-3 + Giai đoạn 7-8-9 (mô hình + fairness + threshold)

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức | In console |
| **2-3** | Rà soát dữ liệu | In console |
| **7-8** | Huấn luyện 3 mô hình (Dummy, Logistic Regression, CatBoost) | `results/model_results_full.json` |
| **9** | Fairness analysis — ROC-AUC, FPR, FNR theo Gender, Age group, Family History | In console — cảnh báo bias |
| **8b** | Threshold analysis — thử 30 ngưỡng (0.20→0.80), khuyến nghị ngưỡng tối ưu | In console — khuyến nghị |

**Không chạy:** EDA (không sinh biểu đồ), thống kê mô tả (Giai đoạn 4)

**Dùng khi:** Đã có EDA rồi, giờ muốn xây dựng mô hình và đánh giá. ~15-25 giây.

---

#### 5. `--full`

```bash
uv run python main.py --full
```

**Chạy: TẤT CẢ** — Giai đoạn 0 → 1 → 2-3 → 4 → 7-8-9

| Giai đoạn | Nội dung | Output |
|-----------|----------|--------|
| **0** | Cảnh báo đạo đức | In console |
| **1** | EDA — 6 biểu đồ | 6 file HTML + 1 JSON |
| **2-3** | Rà soát dữ liệu | In console |
| **4** | Thống kê mô tả theo nhóm | In console |
| **7-8** | 3 mô hình | `results/model_results_full.json` |
| **9** | Fairness analysis | In console |
| **8b** | Threshold analysis | In console |

**Dùng khi:** Muốn chạy trọn vẹn từ đầu đến cuối — sinh báo cáo hoàn chỉnh. ~20-40 giây.

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

#### 7. `--no-ethical`

```bash
uv run python main.py --models --no-ethical
```

**Tác dụng duy nhất:** Bỏ qua phần in cảnh báo đạo đức ở Giai đoạn 0.

**Không thay đổi:** Mọi thứ khác (EDA, stats, models, fairness, threshold) chạy y hệt.

**Dùng khi:** Chạy script tự động, CI/CD, hoặc đã quá quen thuộc và không muốn đọc cảnh báo mỗi lần.

**⚠️ Không khuyến nghị** vì cảnh báo đạo đức là phần cốt lõi định hướng triết lý của dự án.

---

### Bảng tổng hợp

| Flag | EDA (biểu đồ) | Stats | Models | Fairness | Threshold | Leakage | Review | Thời gian ~ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *(không flag)* | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 2-3s |
| `--eda` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 2-3s |
| `--review` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 2-3s |
| `--stats` | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 5-10s |
| `--models` | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | 15-25s |
| `--leakage` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 5-10s |
| `--full` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 20-40s |
| `--no-ethical` | — | — | — | — | — | — | — | Bỏ cảnh báo |

### Flag bổ trợ

| Flag | Tác dụng | Dùng với |
|------|----------|----------|
| `--conservative` | Dùng **Phiên bản A** — không có biến `Suicidal thoughts` (an toàn, không rủi ro rò rỉ nhãn) | `--models`, `--full` |
| `--no-ethical` | Bỏ qua cảnh báo đạo đức ở Giai đoạn 0 | Mọi flag |

**Ví dụ:**
```bash
uv run python main.py --models --conservative    # Huấn luyện Phiên bản A
uv run python main.py --full --conservative      # Toàn bộ pipeline, Phiên bản A
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
