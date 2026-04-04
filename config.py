"""
Cấu hình chung cho dự án Phân Tích Trầm Cảm
Auto-detect device (CPU/GPU) và các parameters mặc định
"""

import torch


class Config:
    """
    Cấu hình trung tâm cho toàn bộ dự án.
    Tự động detect GPU và điều chỉnh code phù hợp.
    """
    
    # ==========================================
    # 🎯 AUTO-DETECT DEVICE (CPU/GPU)
    # ==========================================
    
    # Tự động chọn: CUDA (NVIDIA) → MPS (Apple) → CPU
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        DEVICE_NAME = torch.cuda.get_device_name(0)
        DEVICE_COUNT = torch.cuda.device_count()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        DEVICE_NAME = "Apple M-series (MPS)"
        DEVICE_COUNT = 1
    else:
        DEVICE = torch.device("cpu")
        DEVICE_NAME = "CPU"
        DEVICE_COUNT = 0
    
    # ==========================================
    # 📊 Model Hyperparameters
    # ==========================================
    
    MODEL_PARAMS = {
        # Neural Network
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "hidden_dim": 128,
        "dropout": 0.3,
        
        # XGBoost
        "xgb_n_estimators": 100,
        "xgb_max_depth": 6,
        "xgb_learning_rate": 0.1,
        
        # LightGBM
        "lgb_n_estimators": 100,
        "lgb_learning_rate": 0.1,
        "lgb_num_leaves": 31,
        
        # Data
        "test_size": 0.2,
        "random_state": 42,
    }
    
    # ==========================================
    # 📁 Đường Dẫn Dữ Liệu
    # ==========================================
    
    DATA_PATHS = {
        "raw": "data/raw/",
        "processed": "data/processed/",
        "models": "models/",
        "results": "results/",
    }
    
    # ==========================================
    # 🖨️ Print Configuration
    # ==========================================
    
    @classmethod
    def print_config(cls):
        """In thông tin cấu hình hiện tại"""
        print("=" * 60)
        print(" ⚙️  DỰ ÁN PHÂN TÍCH TRẦM CẢM - CẤU HÌNH")
        print("=" * 60)
        print()
        
        # Device info
        print("🎯 DEVICE:")
        print(f"   Using: {cls.DEVICE}")
        print(f"   Name: {cls.DEVICE_NAME}")
        if cls.DEVICE.type == "cuda":
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {cls.DEVICE_COUNT}")
            print(f"   ✅ GPU acceleration enabled!")
        else:
            print(f"   ⚠️  Running on CPU (slower but stable)")
        print()
        
        # Model params
        print("📊 MODEL PARAMETERS:")
        print(f"   Learning Rate: {cls.MODEL_PARAMS['learning_rate']}")
        print(f"   Batch Size: {cls.MODEL_PARAMS['batch_size']}")
        print(f"   Epochs: {cls.MODEL_PARAMS['epochs']}")
        print(f"   Hidden Dim: {cls.MODEL_PARAMS['hidden_dim']}")
        print(f"   Dropout: {cls.MODEL_PARAMS['dropout']}")
        print()
        
        # Data paths
        print("📁 DATA PATHS:")
        for name, path in cls.DATA_PATHS.items():
            print(f"   {name.capitalize()}: {path}")
        print()
        
        print("=" * 60)
    
    @classmethod
    def get_device_info(cls):
        """Trả về dict thông tin device (dùng cho logging/reporting)"""
        return {
            "device": str(cls.DEVICE),
            "device_name": cls.DEVICE_NAME,
            "device_count": cls.DEVICE_COUNT,
            "cuda_available": torch.cuda.is_available(),
            "pytorch_version": torch.__version__,
        }


# ==========================================
# 🛠️ Helper Functions
# ==========================================

def get_device():
    """
    Trả về device hiện tại (CPU/GPU/MPS)
    Shortcut cho Config.DEVICE
    """
    return Config.DEVICE


def to_device(tensor):
    """
    Đưa tensor lên device hiện tại (CPU/GPU)
    
    Usage:
        x = torch.randn(32, 10)
        x = to_device(x)  # Tự động lên GPU nếu có
    """
    return tensor.to(Config.DEVICE)


def create_dataloader(dataset, batch_size=None, shuffle=True):
    """
    Tạo DataLoader với batch_size phù hợp device
    
    Usage:
        loader = create_dataloader(my_dataset)
    """
    from torch.utils.data import DataLoader
    
    if batch_size is None:
        # GPU có thể xử lý batch lớn hơn
        batch_size = Config.MODEL_PARAMS["batch_size"]
        if Config.DEVICE.type == "cuda":
            batch_size *= 2  # GPU xử lý được batch lớn hơn
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ==========================================
# 🚀 Auto-run khi import
# ==========================================

if __name__ == "__main__":
    # Khi chạy trực tiếp: python config.py
    Config.print_config()
    
    # Test device
    print("\n🧪 Testing device transfer:")
    x = torch.randn(100, 100)
    x = to_device(x)
    print(f"   Tensor device: {x.device}")
    print(f"   Expected: {Config.DEVICE}")
    print(f"   Match: {x.device.type == Config.DEVICE.type}")
