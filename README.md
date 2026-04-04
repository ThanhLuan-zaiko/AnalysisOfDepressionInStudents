# 🧠 Phân Tích Trầm Cảm Ở Học Sinh Sinh Viên

> Dự án phân tích dữ liệu và ứng dụng machine learning để phát hiện, phân tích các mẫu hình trầm cảm trong học sinh sinh viên.

## 📋 Tổng Quan

Dự án này cung cấp một bộ công cụ hoàn chỉnh để phân tích trầm cảm trong cộng đồng học sinh sinh viên, bao gồm:

- **🔧 Xử lý dữ liệu**: Thao tác và phân tích dữ liệu nhanh chóng, hiệu quả
- **📊 Phân tích thống kê**: Kiểm định thống kê toàn diện và xác nhận giả thuyết
- **🤖 Machine Learning**: Các mô hình ML tiên tiến để dự đoán trầm cảm
- **⏱️ Phân tích chuỗi thời gian**: Theo dõi xu hướng trầm cảm theo thời gian
- **📈 Trực quan hóa**: Biểu đồ tương tác, chất lượng cao để báo cáo

---

## 📦 Các Packages Và Mục Đích Sử Dụng

### 🔧 Xử Lý Dữ Liệu

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[Polars](https://pola.rs/)** | Thư viện DataFrame siêu nhanh (viết bằng Rust) | Thay thế pandas, xử lý dữ liệu lớn nhanh gấp 10x |
| **[PyArrow](https://arrow.apache.org/docs/python/)** | Định dạng dữ liệu cột trong bộ nhớ | Lưu trữ hiệu quả, trao đổi dữ liệu giữa các công cụ |
| **[DuckDB](https://duckdb.org/)** | Cơ sở dữ liệu SQL OLAP | Chạy SQL trực tiếp trên DataFrame, truy vấn phân tích nhanh |

**Ví dụ thực tế:** Khi bạn có file CSV với 1 triệu dòng dữ liệu khảo sát trầm cảm, Polars đọc và xử lý nhanh hơn pandas rất nhiều, DuckDB cho phép bạn viết SQL để lọc, nhóm dữ liệu mà không cần chuyển đổi.

### 📈 Trực Quan Hóa Dữ Liệu

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[Plotly](https://plotly.com/python/)** | Thư viện biểu đồ tương tác | Tạo biểu đồ có thể zoom, hover, click - perfect cho dashboard |
| **[Datashader](https://datashader.org/)** | Render tập dữ liệu khổng lồ | Vẽ hàng triệu điểm dữ liệu mà không bị chồng chéo, mờ |
| **[HoloViews](https://holoviews.org/)** | Trực quan hóa khai báo | Tạo biểu đồ phức tạp với ít code, kết hợp với Bokeh/Plotly |

**Ví dụ thực tế:** Plotly giúp tạo scatter plot để xem mối quan hệ giữa điểm trầm cảm và GPA, người xem có thể hover để xem chi tiết từng sinh viên. Datashader giúp vẽ heatmap cho hàng nghìn điểm khảo sát.

### 🤖 Machine Learning & Thống Kê

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[XGBoost](https://xgboost.readthedocs.io/)** | Gradient Boosting mạnh nhất | Mô hình chính để dự đoán trầm cảm, đạt accuracy cao |
| **[LightGBM](https://lightgbm.readthedocs.io/)** | Gradient Boosting nhanh | Training nhanh hơn XGBoost với dataset lớn |
| **[Statsmodels](https://www.statsmodels.org/)** | Mô hình thống kê | Hồi quy, kiểm định thống kê, phân tích phương quy |
| **[Factor Analyzer](https://github.com/EducationalTestingService/factor_analyzer)** | Phân tích nhân tố | Tìm các yếu tố tiềm ẩn (stress, lo âu, cô đơn) trong bảng câu hỏi |
| **[Pingouin](https://pingouin-stats.org/)** | Kiểm định thống kê | ANOVA, T-test, tương quan, effect sizes - dễ dùng hơn scipy |

**Ví dụ thực tế:** XGBoost huấn luyện mô hình dự đoán sinh viên có nguy cơ trầm cảm dựa trên các features: giờ ngủ, điểm số, mối quan hệ, v.v. Factor Analyzer giúp tìm ra các yếu tố cốt lõi gây trầm cảm.

### 🎯 Tinh Chỉnh Siêu Tham Số

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[Optuna](https://optuna.org/)** | Tự động tối ưu hyperparameter | Tìm bộ tham số tốt nhất cho XGBoost/LightGBm tự động |

**Ví dụ thực tế:** Thay vì thử thủ công `learning_rate=0.01, 0.05, 0.1`, Optuna tự động thử hàng trăm combinations và tìm ra bộ tham số tối ưu nhất.

### ⏱️ Phân Tích Chuỗi Thời Gian

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[Darts](https://unit8co.github.io/darts/)** | Forecasting chuỗi thời gian | Dự đoán xu hướng trầm cảm theo thời gian, theo học kỳ |
| **[NeuralProphet](https://neuralprophet.com/)** | Forecasting dùng neural network | Dự báo với deep learning, tự động bắt pattern phức tạp |

**Ví dụ thực tế:** Darts giúp dự đoán số sinh viên có trầm cảm sẽ tăng/giảm trong học kỳ tới dựa trên dữ liệu lịch sử. NeuralProphet tự động phát hiện seasonal patterns (trầm cảm tăng mùa thi).

### 🔄 Online/Streaming Machine Learning

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[VowpalWabbit](https://vowpalwabbit.org/)** | Hệ thống online learning cực nhanh | Huấn luyện real-time khi có dữ liệu mới, không cần train lại từ đầu |

**Ví dụ thực tế:** Khi có sinh viên mới làm khảo sát, VowpalWabbit cập nhật mô hình ngay lập tức mà không cần train lại toàn bộ dữ liệu. Nhanh hơn river và production-ready.

### 🏥 Phân Tích Survival

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[Lifelines](https://lifelines.readthedocs.io/)** | Survival analysis | Phân tích thời gian cho đến khi có biến cố (khỏi trầm cảm, tái phát) |

**Ví dụ thực tế:** Ước lượng thời gian trung bình sinh viên cần để phục hồi từ trầm cảm, các factors ảnh hưởng đến tốc độ phục hồi.

### 🔥 Deep Learning

| Package | Là Gì? | Dùng Để Làm Gì? |
|---------|---------|-----------------|
| **[PyTorch](https://pytorch.org/)** | Framework deep learning | Xây dựng neural networks phức tạp, custom architectures |
| **torchvision** | Computer vision | Xử lý ảnh (nếu có dữ liệu ảnh/video phân tích cảm xúc) |
| **torchaudio** | Xử lý audio | Phân tích giọng nói để phát hiện dấu hiệu trầm cảm |

**Ví dụ thực tế:** PyTorch cho phép xây dựng neural network để phát hiện pattern phức tạp trong dữ liệu mà traditional ML bỏ qua.

---

## 🚀 Hướng Dẫn Cài Đặt (Cho Người Mới)

### ⚠️ Yêu Cầu Trước Khi Bắt Đầu

1. **Python 3.14 trở lên** (đã khai báo trong file `.python-version`)
2. **uv** - Trình quản lý gói Python siêu nhanh
   
   Cài uv trên Windows:
   ```powershell
   # Dùng winget
   winget install --id=astral-sh.uv
   
   # HOẶC dùng pip
   pip install uv
   
   # HOẶC tải từ trang chủ
   # https://docs.astral.sh/uv/getting-started/installation/
   ```

---

### 🎯 2 Cách Cài Đặt (Chọn 1)

> **❓ `uv sync` và `.\setup_with_gpu.ps1` khác nhau chỗ nào?**
> 
> Đây là câu hỏi phổ biến! Đây là sự khác biệt:
>
> | | `uv sync` | `.\setup_with_gpu.ps1` |
> |---|-----------|------------------------|
> | **Là gì?** | Lệnh uv cơ bản | Script thông minh tự viết |
> | **Cách hoạt động** | Đọc file `pyproject.toml` và `uv.lock`, cài đúng phiên bản đã lock | Tự động phát hiện GPU → cài packages → cấu hình PyTorch cho GPU phù hợp |
> | **GPU** | ❌ Chỉ cài PyTorch CPU | ✅ Tự động chọn CUDA/ROCm/XPU/CPU |
> | **Tốc độ** | ⚡ Nhanh nhất (vì đã có lock file) | 🐢 Chậm hơn (vì phải detect + resolve) |
> | **Dành cho ai?** | Developer đã quen uv, CI/CD | Người mới, muốn setup tự động |
> | **Khi nào dùng?** | Sau khi git clone, cần nhanh | Khi mới bắt đầu dự án, hoặc chuyển GPU |
>
> **💡 Tóm lại:**
> - Dùng **`.\setup_with_gpu.ps1`** nếu: Bạn mới clone repo, muốn mọi thứ tự động, có GPU
> - Dùng **`uv sync`** nếu: Bạn cần nhanh, đã có lock file, không cần GPU setup

---

#### ✅ Cách 1: Tự Động (Khuyến Nghị Cho Người Mới)

Chạy script tự động setup mọi thứ:

**Trên Windows:**
```powershell
# Mở PowerShell trong thư mục dự án
.\setup_with_gpu.ps1
```

**Script sẽ tự động làm gì?**

```
Bước 1: Phát hiện GPU của bạn
  ├─ Kiểm tra: Có NVIDIA GPU không?
  ├─ Kiểm tra: Có AMD GPU không?
  ├─ Kiểm tra: Có Intel GPU không?
  └─ Không có → Dùng CPU

Bước 2: Cài đặt 16 packages cần thiết
  ├─ Thử cài tất cả cùng lúc (nhanh)
  └─ Nếu lỗi → Cài từng cái để xác định package lỗi

Bước 3: Cài PyTorch với GPU phù hợp
  ├─ NVIDIA → Cài CUDA 12.1 version
  ├─ AMD → Cài ROCm version
  ├─ Intel → Cài XPU version
  └─ Không GPU → Cài CPU version

Bước 4: Kiểm tra lại
  └─ Chạy verify_gpu.py để đảm bảo mọi thứ hoạt động
```

**Ưu điểm:**
- ✅ Không cần biết gì về packages
- ✅ Tự động phát hiện và cấu hình GPU
- ✅ Có fallback nếu cài lỗi
- ✅ Báo cáo chi tiết những gì đã cài

**Nhược điểm:**
- ⏱️ Chậm hơn `uv sync` (vì phải detect GPU + resolve dependencies)

---

#### ⚡ Cách 2: Thủ Công (Nhanh Hơn)

Nếu bạn muốn kiểm soát hoàn toàn hoặc cần setup nhanh:

**Bước 1: Sync toàn bộ dependencies từ lock file**

```bash
# Lệnh này đọc file uv.lock và cài đúng phiên bản đã được lock
uv sync
```

**`uv sync` làm gì?**
- Đọc file `pyproject.toml` để biết cần packages gì
- Đọc file `uv.lock` để biết chính xác version nào
- Tạo virtual environment (nếu chưa có)
- Cài đặt tất cả packages với versions đã lock
- ⚡ **Rất nhanh** vì không phải resolve dependencies

**Khi nào `uv sync` không đủ?**
- ❌ Bạn muốn dùng GPU (NVIDIA/AMD/Intel) → PyTorch trong lock file là CPU version
- ❌ Lock file chưa có hoặc outdated
- ❌ Bạn muốn thêm packages mới

**Bước 2: (Tùy chọn) Cài PyTorch với GPU support**

Mặc định PyTorch trong lock file là **CPU version**. Nếu bạn có GPU và muốn tăng tốc:

**Cho NVIDIA GPU (phổ biến nhất):**
```bash
# Cài PyTorch với CUDA 12.1 - sẽ chạy nhanh hơn nhiều trên NVIDIA GPU
uv add --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

**Cho AMD GPU:**
```bash
# Cài PyTorch với ROCm 6.0 - chỉ hoạt động tốt trên Linux/WSL2
uv add --index-url https://download.pytorch.org/whl/rocm6.0 torch torchvision torchaudio
```

**Cho Intel GPU (Arc, Iris):**
```bash
# Cài PyTorch với Intel XPU support
uv add --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ torch torchvision torchaudio
```

**Nếu không có GPU hoặc không quan tâm:**
```bash
# Giữ nguyên CPU version - vẫn chạy tốt, chỉ chậm hơn
# Không cần làm gì thêm!
```

**Bước 3: Kiểm tra cài đặt**

```bash
# Chạy script kiểm tra GPU
uv run python verify_gpu.py
```

**Kết quả mong đợi:**

✅ **Nếu có NVIDIA GPU + CUDA:**
```
✅ CUDA is available!
   CUDA version: 12.1
   GPU count: 1
   GPU 0: NVIDIA GeForce RTX 3060
✅ GPU computation test: PASSED
✅ GPU acceleration is READY!
```

⚠️ **Nếu dùng CPU:**
```
ℹ️  CUDA is NOT available
   CPU tensor test: PASSED
ℹ️  Running on CPU mode. Performance may be slower for large datasets.
```

---

## 📁 Cấu Trúc Dự Án

```
AnalysisOfDepressionInStudents/
│
├── main.py                      # File chính - chạy phân tích tại đây
├── config.py                    # ⭐ Auto-detect CPU/GPU + Config
├── pyproject.toml               # Khai báo dependencies và config dự án
├── uv.lock                      # ⚠️ QUAN TRỌNG: Lock file - commit lên Git!
├── .python-version              # Yêu cầu Python 3.14
├── .gitignore                   # Files không cần commit
│
├── setup_with_gpu.ps1           # ⭐ Script tự động setup với GPU detection
├── verify_gpu.py                # Script kiểm tra PyTorch có nhận GPU không
│
└── README.md                    # File hướng dẫn này
```

### ⚠️ Về `uv.lock` File

**CÓ commit lên GitHub!** File này đảm bảo:
- ✅ Mọi người cài **đúng cùng versions**
- ✅ Không bị breaking changes khi packages update
- ✅ Reproducible builds - "máy tôi chạy được" = "máy bạn cũng chạy được"
- ✅ CI/CD ổn định, dễ debug

**Đừng thêm `uv.lock` vào `.gitignore`!**

---

## 🏃 Cách Chạy Chương Trình

### `uv run` Là Gì?

`uv run` là lệnh đặc biệt của uv giúp bạn:
- ✅ **Tự động kích hoạt virtual environment** (không cần `source .venv/bin/activate`)
- ✅ **Dùng đúng Python version** từ `.python-version`
- ✅ **Dùng đúng packages** đã cài trong project
- ✅ **Không ảnh hưởng hệ thống** - chạy trong môi trường isolated

> **💡 So sánh:**
> ```bash
> # Cách thường (phải activate venv trước)
> .venv\Scripts\activate
> python main.py
> 
> # Cách uv (nhanh hơn, không cần activate)
> uv run python main.py
> ```

---

### 3 Cách Chạy Chương Trình

#### Cách 1: Chạy với `uv run` (Khuyến Nghị ⭐)

```bash
# Chạy file chính
uv run python main.py

# Chạy file khác
uv run python analyze_data.py
uv run python train_model.py
```

**Khi nào dùng?**
- ✅ Mới setup xong, muốn chạy thử
- ✅ Chạy scripts một lần
- ✅ Không muốn activate venv

---

#### Cách 2: Activate Virtual Environment Trước

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Sau đó chạy bình thường
python main.py
python analyze_data.py
```

**Khi nào dùng?**
- ✅ Development liên tục (không phải gõ `uv run` mỗi lần)
- ✅ Dùng IDE (VS Code, PyCharm) - chọn interpreter từ `.venv`
- ✅ Cần interactive Python (REPL)

---

#### Cách 3: Chạy Trực Tiếp Script Setup

```powershell
# Script này vừa cài packages vừa chạy verify
.\setup_with_gpu.ps1
```

---

### 📝 Ví Dụ Thực Tế

**Ví dụ 1: Chạy phân tích nhanh**
```bash
# Sau khi git clone và uv sync
uv run python main.py
```

**Ví dụ 2: Chạy với GPU verification**
```bash
# Kiểm tra PyTorch có nhận GPU không
uv run python verify_gpu.py
```

**Ví dụ 3: Interactive Python (REPL)**
```bash
# Activate venv
.venv\Scripts\Activate.ps1

# Vào Python interactive mode
python

# Giờ import và test được
>>> import torch
>>> print(torch.__version__)
>>> import polars as pl
>>> df = pl.DataFrame({"a": [1, 2, 3]})
```

---

### ⚠️ Lỗi Thường Gặp Khi Chạy

**Lỗi 1: `ModuleNotFoundError: No module named 'polars'`**
```bash
# Nguyên nhân: Chưa cài packages hoặc dùng system Python thay vì venv
# Giải pháp:
uv sync              # Cài packages
uv run python main.py  # Chạy với uv run
```

**Lỗi 2: `python: command not found`**
```bash
# Nguyên nhân: Python không trong PATH
# Giải pháp:
uv run python main.py  # uv tự tìm Python đúng version
```

**Lỗi 3: IDE không nhận packages**
```bash
# Trong VS Code:
# 1. Ctrl+Shift+P → "Python: Select Interpreter"
# 2. Chọn interpreter từ .venv
# 3. Restart terminal

# Trong PyCharm:
# 1. Settings → Project → Python Interpreter
# 2. Add interpreter → Existing environment
# 3. Chọn .venv/Scripts/python.exe
```

---

## 💻 Code Cho CPU & GPU (Không Cần Viết Lại!)

### ❓ Câu Hỏi: Code Có Cần Thay Đổi Khi Chạy Trên GPU Không?

**TRẢ LỜI NGẮN: ❌ KHÔNG!** Chỉ cần thêm **1-2 dòng code** để tự động switch giữa CPU/GPU.

---

### 🎯 Auto-Detect Device Template

Dự án đã có sẵn file `config.py` tự động detect và cấu hình device:

```python
# Cách dùng đơn giản nhất
from config import Config, to_device

Config.print_config()  # In thông tin device

# Tạo model
model = MyModel()
model = model.to(Config.DEVICE)  # ← Tự động CPU hoặc GPU

# Đưa data lên device
X = to_device(X)  # ← Shortcut cho X.to(Config.DEVICE)
y = to_device(y)

# Training loop - GIỐNG HỆT cho CPU và GPU!
predictions = model(X)
loss = criterion(predictions, y)
```

---

### 📝 Cách Hoạt Động

**Step 1: Import config**
```python
from config import Config, get_device, to_device
```

**Step 2: Config tự động detect**
```python
# Config.py tự làm:
if torch.cuda.is_available():
    DEVICE = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    DEVICE = "mps"   # Apple Silicon
else:
    DEVICE = "cpu"   # Fallback
```

**Step 3: Sử dụng trong code**
```python
import torch
from config import Config

# Model
model = torch.nn.Linear(10, 5)
model = model.to(Config.DEVICE)  # ← 1 dòng duy nhất!

# Data
X = torch.randn(32, 10)
X = X.to(Config.DEVICE)  # ← 1 dòng duy nhất!

# Training loop - KHÔNG CẦN THAY ĐỔI GÌ!
output = model(X)  # ← Giống hệt CPU và GPU
loss = loss_fn(output, y)
loss.backward()
optimizer.step()
```

---

### ⚡ So Sánh Code CPU vs GPU

| Phần | CPU | GPU | Khác Biệt? |
|------|-----|-----|-----------|
| **Import** | `from config import Config` | Giống hệT | ❌ KHÔNG |
| **Device** | `Config.DEVICE` → `"cpu"` | `Config.DEVICE` → `"cuda"` | ✅ Tự động |
| **Model** | `model.to(Config.DEVICE)` | `model.to(Config.DEVICE)` | ❌ GIỐNG HỆT |
| **Data** | `X.to(Config.DEVICE)` | `X.to(Config.DEVICE)` | ❌ GIỐNG HỆT |
| **Forward** | `output = model(X)` | `output = model(X)` | ❌ GIỐNG HỆT |
| **Loss** | `loss = criterion(out, y)` | `loss = criterion(out, y)` | ❌ GIỐNG HỆT |
| **Backward** | `loss.backward()` | `loss.backward()` | ❌ GIỐNG HỆT |

---

### 📊 Cho XGBoost & LightGBM

Các library này **tự động detect GPU**, chỉ cần setting:

```python
import xgboost as xgb
from config import Config

# XGBoost - tự động dùng GPU nếu có
use_gpu = Config.DEVICE.type == "cuda"

model = xgb.XGBClassifier(
    n_estimators=100,
    tree_method='gpu_hist' if use_gpu else 'hist',  # ← 1 dòng
    eval_metric='logloss'
)

# Training - GIỐNG HỆT!
model.fit(X_train, y_train)
```

```python
import lightgbm as lgb
from config import Config

# LightGBM - tự động dùng GPU
model = lgb.LGBMClassifier(
    device='gpu' if Config.DEVICE.type == "cuda" else 'cpu',  # ← 1 dòng
    n_estimators=100
)

# Training - GIỐNG HỆT!
model.fit(X_train, y_train)
```

---

### 🚀 Best Practices

✅ **NÊN LÀM:**
```python
# 1. Dùng Config.DEVICE mọi lúc
model = model.to(Config.DEVICE)
data = data.to(Config.DEVICE)

# 2. Dùng helper functions từ config.py
from config import to_device, create_dataloader

X = to_device(X)
loader = create_dataloader(dataset)

# 3. Print config khi start
Config.print_config()
```

❌ **KHÔNG NÊN:**
```python
# 1. Hard-code device
model.to("cuda")  # ❌ Sẽ crash nếu không có GPU

# 2. Check GPU nhiều lần
if torch.cuda.is_available():  # ❌ Lặp lại không cần thiết
    model.to("cuda")

# 3. Quên .to(device)
model = MyModel()
X = X.to(device)
output = model(X)  # ❌ Error: model on CPU, data on GPU!
```

---

### 📋 Template Code Cho Dự Án

```python
# main.py - Template hoàn chỉnh
import torch
import polars as pl
from config import Config, to_device

# Print cấu hình
Config.print_config()

# Load data
df = pl.read_csv("data/depression_scores.csv")
X = torch.tensor(df.drop("label").to_numpy(), dtype=torch.float32)
y = torch.tensor(df["label"].to_numpy(), dtype=torch.int64)

# Move to device
X = to_device(X)
y = to_device(y)

# Create model
model = torch.nn.Sequential(
    torch.nn.Linear(X.shape[1], Config.MODEL_PARAMS["hidden_dim"]),
    torch.nn.ReLU(),
    torch.nn.Dropout(Config.MODEL_PARAMS["dropout"]),
    torch.nn.Linear(Config.MODEL_PARAMS["hidden_dim"], 2)
)
model = model.to(Config.DEVICE)  # ← Quan trọng!

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=Config.MODEL_PARAMS["learning_rate"])
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(Config.MODEL_PARAMS["epochs"]):
    # Forward
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X)
    accuracy = (preds.argmax(dim=1) == y).float().mean()
    print(f"\n✅ Accuracy: {accuracy:.4f}")
```

---

### 💡 Key Takeaways

| Điểm | Giải Thích |
|------|-----------|
| **Chỉ 1-2 dòng thay đổi** | `.to(Config.DEVICE)` cho model và data |
| **Training loop giống hệt** | Forward/backward/optimizer không đổi |
| **Auto-detect** | Config.py tự động nhận GPU/CPU |
| **No code duplication** | Không cần viết 2 versions CPU/GPU |
| **Easy to maintain** | Change 1 variable → toàn bộ code adapt |

---

## 🔍 Ví Dụ Sử Dụng

### 1. Phân Tích Dữ Liệu Cơ Bản

```python
import polars as pl

# Đọc dữ liệu khảo sát trầm cảm
df = pl.read_csv("data/depression_scores.csv")

# Thống kê nhanh
print(df.describe())

# Lọc sinh viên có nguy cơ trầm cảm (điểm > 16)
at_risk = df.filter(pl.col("depression_score") > 16)
print(f"Số sinh viên có nguy cơ: {len(at_risk)}")

# Nhóm theo giới tính và tính điểm trung bình
by_gender = df.group_by("gender").agg(
    pl.col("depression_score").mean().alias("avg_score"),
    pl.col("age").mean().alias("avg_age")
)
print(by_gender)
```

### 2. Phân Tích Thống Kê

```python
import pingouin as pg

# So sánh điểm trầm cảm giữa nam và nữ (T-test)
t_test = pg.ttest(
    df.filter(pl.col("gender") == "Nam")["depression_score"],
    df.filter(pl.col("gender") == "Nữ")["depression_score"]
)
print(t_test)

# Tương quan giữa giờ ngủ và điểm trầm cảm
corr = df.select(["sleep_hours", "depression_score"]).corr()
print(corr)

# ANOVA: So sánh nhiều nhóm (năm 1, 2, 3, 4)
anova = pg.anova(data=df.to_pandas(), dv='depression_score', between='year')
print(anova)
```

### 3. Huấn Luyện Mô hình Machine Learning

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Chuẩn bị dữ liệu
X = df.drop("depression_label", axis=1)  # Features: tuổi, giờ ngủ, GPA, v.v.
y = df["depression_label"]               # Label: Có trầm cảm (1) / Không (0)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Đánh giá
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Xem feature importance (yếu tố nào quan trọng nhất)
xgb.plot_importance(model)
```

### 4. Dự Báo Chuỗi Thời Gian

```python
from darts import TimeSeries
from darts.models import ExponentialSmoothing

# Tạo time series từ dữ liệu theo tháng
monthly_data = df.group_by("month").agg(
    pl.col("depression_score").mean().alias("avg_score")
)

ts = TimeSeries.from_dataframe(
    monthly_data.to_pandas(),
    time_col="month",
    value_cols="avg_score"
)

# Huấn luyện mô hình dự báo
model = ExponentialSmoothing()
model.fit(ts)

# Dự báo 6 tháng tiếp theo
forecast = model.predict(6)
forecast.plot(label="Dự báo")
```

### 5. Trực Quan Hóa Tương Tác

```python
import plotly.express as px

# Scatter plot: Điểm trầm cảm vs GPA, tô màu theo giới tính
fig = px.scatter(
    df.to_pandas(),
    x="gpa",
    y="depression_score",
    color="gender",
    size="sleep_hours",
    hover_data=["age", "year"],
    title="Mối Quan Hệ Giữa GPA và Điểm Trầm Cảm",
    labels={"gpa": "Điểm Trung Bình", "depression_score": "Điểm Trầm Cảm"}
)
fig.show()

# Histogram: Phân bố điểm trầm cảm
fig_hist = px.histogram(
    df.to_pandas(),
    x="depression_score",
    color="gender",
    nbins=30,
    title="Phân Bố Điểm Trầm Cảm"
)
fig_hist.show()
```

---

## 🎯 Các Luồng Làm Việc Thường Gặp

### Luồng 1: Chạy Phân Tích Nhanh

```bash
# Chạy script chính
uv run python main.py
```

### Luồng 2: Tinh Chỉnh Mô Hình

```bash
# Chạy script tối ưu hyperparameters với Optuna
uv run python optimize_model.py
```

### Luồng 3: Tạo Báo Cáo

```bash
# Tạo báo cáo phân tích
uv run python generate_report.py
```

---

## 🔧 Xử Lý Sự Cố

### ❌ Lỗi: Cài đặt package thất bại

**Nguyên nhân:** 
- Network không ổn định
- Package không tương thích với Python 3.14
- Thiếu Visual Studio Build Tools (cho packages có C/C++ extensions)

**Cách sửa:**

```bash
# Cài từng package một để xác định package lỗi
uv add polars
uv add pyarrow
uv add duckdb
# ... tiếp tục
```

### ❌ Lỗi: PyTorch không nhận GPU

**Kiểm tra:**

```bash
# Chạy script kiểm tra
uv run python verify_gpu.py
```

**Cách sửa theo GPU:**

**NVIDIA:**
1. Cài [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)
2. Cài [cuDNN](https://developer.nvidia.com/cudnn)
3. Chạy lại: `.\setup_with_gpu.ps1`

**AMD:**
- Windows cần dùng WSL2 để có ROCm support đầy đủ
- Hướng dẫn: https://rocm.docs.amd.com/projects/install-on-windows/en/latest/

**Intel:**
- Đảm bảo driver Intel Arc mới nhất
- Một số Intel UHD/Iris không được support đầy đủ

### ❌ Lỗi: Hết bộ nhớ (Out of Memory)

**Giải pháp:** Dùng Polars thay vì pandas (tiết kiệm RAM hơn)

```python
import polars as pl

# Thay vì đọc toàn bộ vào RAM
# df = pd.read_csv("large_file.csv")  # ❌ Tốn RAM

# Dùng lazy evaluation - chỉ load khi cần
df = pl.scan_csv("large_file.csv")    # ✅ Tiết kiệm RAM
df_filtered = df.filter(pl.col("score") > 10).collect()
```

### ❌ Lỗi: Training quá chậm

**Tăng tốc với GPU:**

```python
# XGBoost với GPU
import xgboost as xgb
model = xgb.XGBClassifier(
    tree_method='gpu_hist',  # Dùng GPU thay vì CPU
    gpu_id=0
)

# PyTorch với GPU
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

---

## 📝 Lưu Ý Quan Trọng

1. **Python 3.14**: Dự án dùng Python 3.14 (mới nhất). Một số packages cũ có thể không tương thích.

2. **GPU Support**: 
   - **NVIDIA** hoạt động tốt nhất (CUDA support đầy đủ)
   - **AMD** cần WSL2 trên Windows
   - **Intel** còn hạn chế support

3. **Windows Only**: Script `setup_with_gpu.ps1` tối ưu cho Windows. Linux/macOS cần điều chỉnh.

4. **Package River**: KHÔNG dùng được với Python 3.14. Đã thay thế bằng **VowpalWabbit** (nhanh hơn, production-ready).

5. **uv.lock File**: Luôn commit file này để đảm bảo mọi người cài cùng versions.

---

## 🤝 Đóng Góp Dự Án

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/TenTinhNang`)
3. Commit thay đổi (`git commit -m 'Thêm tính năng ABC'`)
4. Push lên branch (`git push origin feature/TenTinhNang`)
5. Tạo Pull Request

---

## 📄 License

Dự án phục vụ mục đích học thuật và nghiên cứu.

---

## 📧 Liên Hệ

Có câu hỏi hoặc cần hỗ trợ? Vui lòng tạo issue trong repository.

---

**Chúc Phân Tích Vui Vẻ! 🎓🧠✨**
