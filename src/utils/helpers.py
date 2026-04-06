"""
Utils Module - Helper Functions
Logging, device management, data paths and other utilities

Usage:
    from src.utils import setup_logging, Timer

    logger = setup_logging()
    with Timer():
        # code here
        pass
"""

import logging
import time
from pathlib import Path
from typing import Optional
import json


# ==========================================
# 📝 LOGGING
# ==========================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging for the entire project

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: File to save logs (optional)
        format: Custom log format
    
    Usage:
        logger = setup_logging(level="INFO", log_file="logs/analysis.log")
        logger.info("Starting analysis...")
    """
    if format is None:
        format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    
    # Setup handlers
    handlers = [logging.StreamHandler()]  # Console
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format,
        handlers=handlers,
        force=True  # Override existing config
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup: level={level}, file={log_file}")
    
    return logger


# ==========================================
# ⏱️ TIMER
# ==========================================

class Timer:
    """
    Context manager to measure execution time

    Usage:
        with Timer("Data loading"):
            df = load_data()
        
        # Output: ⏱️ Data loading completed in 2.34s
    """
    
    def __init__(self, task_name: str = "Task"):
        self.task_name = task_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"⏱️  {self.task_name} started...")
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"⏱️  {self.task_name} completed in {elapsed:.2f}s")
    
    @property
    def elapsed(self) -> float:
        """Return elapsed time"""
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


# ==========================================
# 📁 PATH MANAGEMENT
# ==========================================

class PathManager:
    """
    Path management for the project
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        
        # Create directory structure
        self.paths = {
            'data': self.base_dir / 'data',
            'data_raw': self.base_dir / 'data' / 'raw',
            'data_processed': self.base_dir / 'data' / 'processed',
            'models': self.base_dir / 'models',
            'results': self.base_dir / 'results',
            'visualizations': self.base_dir / 'results' / 'visualizations',
            'logs': self.base_dir / 'logs',
            'configs': self.base_dir / 'configs'
        }
        
        # Create all directories
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Path:
        """Get path"""
        if key not in self.paths:
            raise KeyError(f"Path key '{key}' not found. Available: {list(self.paths.keys())}")
        return self.paths[key]
    
    def list_files(self, directory: str, extension: Optional[str] = None) -> list:
        """
        List files in directory

        Args:
            directory: Key in paths dict
            extension: Filter by extension (e.g., ".csv")
        """
        dir_path = self.get(directory)
        
        if not dir_path.exists():
            return []
        
        files = list(dir_path.iterdir())
        
        if extension:
            files = [f for f in files if f.suffix == extension]
        
        return sorted(files)


# ==========================================
# 💾 DATA SERIALIZATION
# ==========================================

def save_json(data: dict, filepath: str, indent: int = 2):
    """
    Save dict to JSON file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    print(f"💾 Saved JSON: {filepath}")


def load_json(filepath: str) -> dict:
    """
    Load JSON file to dict
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"💾 Loaded JSON: {filepath}")
    
    return data


# ==========================================
# 🎯 DEVICE INFO
# ==========================================

def get_device_info() -> dict:
    """
    Get device information (CPU/GPU)
    """
    import torch
    
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'device_name': 'CPU'
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
    
    return info


def print_device_info():
    """
    Print device information
    """
    info = get_device_info()
    
    print("=" * 60)
    print(" 🖥️  DEVICE INFORMATION")
    print("=" * 60)
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"GPU Count: {info['device_count']}")
        print(f"GPU Name: {info['device_name']}")
        print(f"CUDA Version: {info['cuda_version']}")
        print("✅ GPU acceleration enabled!")
    else:
        print("⚠️  Running on CPU")
    
    print("=" * 60)


# ==========================================
# 📊 DATA VALIDATION
# ==========================================

def validate_dataframe(df, required_columns: list) -> bool:
    """
    Check if DataFrame has required columns
    """
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return True


def check_data_quality(df) -> dict:
    """
    Check data quality
    
    Returns:
        Dict with data quality metrics
    """
    quality = {
        'total_rows': len(df),
        'total_columns': df.width,
        'missing_values': {},
        'duplicate_rows': 0,
        'completeness': 0.0
    }
    
    # Missing values
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            quality['missing_values'][col] = null_count
    
    # Duplicate rows
    quality['duplicate_rows'] = df.n_unique()
    
    # Completeness
    total_cells = df.height * df.width
    non_null_cells = sum(df[col].drop_nulls().len() for col in df.columns)
    quality['completeness'] = non_null_cells / total_cells if total_cells > 0 else 0
    
    return quality


# ==========================================
# 🎨 FORMATTING
# ==========================================

def format_number(num: float, decimals: int = 2) -> str:
    """
    Format numbers for readability
    """
    if num >= 1_000_000:
        return f"{num / 1_000_000:.{decimals}f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def format_time(seconds: float) -> str:
    """
    Format time for readability
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
