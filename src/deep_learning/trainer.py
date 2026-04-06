"""
Deep Learning Module - PyTorch-based
Neural networks for depression prediction with GPU acceleration

Usage:
    from src.deep_learning import DepressionNN

    nn_model = DepressionNN()
    model = nn_model.create_model(input_dim=20)
    history = nn_model.train(model, train_loader, val_loader)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import polars as pl
import numpy as np
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ==========================================
# 🏗️ NEURAL NETWORK ARCHITECTURES
# ==========================================

class DepressionNet(nn.Module):
    """
    Basic neural network for depression prediction
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super(DepressionNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()


class DepressionNetDeep(nn.Module):
    """
    Deep neural network (more layers, more complex)
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], dropout: float = 0.4):
        super(DepressionNetDeep, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


# ==========================================
# 🎯 TRAINER CLASS
# ==========================================

class DepressionNN:
    """
    Neural network training toolkit for depression prediction.
    Auto-detect GPU from config.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() 
                                     else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                                     else 'cpu')
        else:
            self.device = device
        
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DepressionNN initialized on {self.device}")
    
    # ==========================================
    # 📊 DATA PREPARATION
    # ==========================================
    
    def prepare_dataloaders(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str = "depression_score",
        threshold: int = 16,
        batch_size: int = 32,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare DataLoaders for training

        Returns:
            train_loader, test_loader
        """
        # Set seed for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Extract features and target
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        
        # Binary classification
        if df[target_col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            y = (df[target_col].to_numpy() >= threshold).astype(np.float32)
        else:
            y = df[target_col].to_numpy().astype(np.float32)
        
        # Train/test split
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * (1 - test_size))
        
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Create tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"DataLoaders prepared: {len(train_dataset)} train, {len(test_dataset)} test")
        
        return train_loader, test_loader
    
    # ==========================================
    # 🏗️ MODEL CREATION
    # ==========================================
    
    def create_model(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        model_type: str = "basic"
    ) -> nn.Module:
        """
        Create model architecture

        Args:
            model_type: "basic" or "deep"
        """
        if model_type == "basic":
            model = DepressionNet(input_dim, hidden_dim, dropout)
        elif model_type == "deep":
            model = DepressionNetDeep(input_dim, [hidden_dim * 2, hidden_dim, hidden_dim // 2], dropout)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        model = model.to(self.device)
        
        logger.info(f"Model created: {model_type}, input_dim={input_dim}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    # ==========================================
    # 🏋️ TRAINING
    # ==========================================
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.001,
        epochs: int = 100,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 10,
        use_scheduler: bool = True
    ) -> Dict:
        """
        Train neural network

        Args:
            model: Neural network model
            train_loader: Training data
            val_loader: Validation data (optional)
            learning_rate: Learning rate
            epochs: Number of epochs
            weight_decay: L2 regularization
            early_stopping_patience: Stop early if val_loss doesn't improve
            use_scheduler: Use learning rate scheduler

        Returns:
            History dict with loss, accuracy across epochs
        """
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = None
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        logger.info(f"Training started: {epochs} epochs, lr={learning_rate}")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(epochs):
            # ===== TRAINING PHASE =====
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Calculate train metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            
            # ===== VALIDATION PHASE =====
            if val_loader is not None:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        predicted = (outputs > 0.5).float()
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_accuracy)
                
                # Learning rate scheduling
                if scheduler is not None:
                    scheduler.step(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Log progress every 10 epochs
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{epochs}] | "
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                        f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                    )
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
            else:
                # No validation, just log
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{epochs}] | "
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
                    )
        
        logger.info(f"Training completed. Best Val Loss: {best_val_loss:.4f}")
        
        return history
    
    # ==========================================
    # 📈 EVALUATION
    # ==========================================
    
    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict:
        """
        Evaluate model on test set

        Returns:
            Dict with metrics
        """
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                probs = outputs
                preds = (outputs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
        }
        
        logger.info(f"Test Evaluation:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    # ==========================================
    # 🔮 PREDICTION
    # ==========================================
    
    def predict(
        self,
        model: nn.Module,
        X: np.ndarray,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Predict with model

        Args:
            X: Input features (numpy array)
            return_proba: Return probabilities

        Returns:
            Predictions (and probabilities if return_proba=True)
        """
        model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
        
        if return_proba:
            return preds, probs
        
        return preds
    
    # ==========================================
    # 💾 SAVE/LOAD MODELS
    # ==========================================
    
    def save_model(self, model: nn.Module, filename: str) -> Path:
        """
        Save model checkpoint
        """
        filepath = self.models_dir / filename
        torch.save({
            'model_state_dict': model.state_dict(),
            'device': str(self.device)
        }, filepath)
        
        logger.info(f"Model saved: {filepath}")
        return filepath
    
    def load_model(self, model: nn.Module, filename: str) -> nn.Module:
        """
        Load model checkpoint
        """
        filepath = self.models_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded: {filepath}")
        
        return model
    
    # ==========================================
    # 📊 VISUALIZE TRAINING
    # ==========================================
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None):
        """
        Plot training history (loss and accuracy)
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        if history['val_loss']:
            ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_title('Training & Validation Loss', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        if history['val_acc']:
            ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        ax2.set_title('Training & Validation Accuracy', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved: {save_path}")
        
        plt.show()
