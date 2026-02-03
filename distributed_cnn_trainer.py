import numpy as np
import redis
import pickle
import json
import pandas as pd
from typing import Dict, Tuple, Optional, List
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import gc      # For garbage collection
from sklearn.metrics import f1_score, precision_score, recall_score

class DistributedCNN1DTrainer:
    """Distributed 1D CNN trainer using KeyDB for shared weight storage across servers - EXACTLY like MLP trainer"""
    
    def __init__(self, keydb_host='localhost', keydb_port=6379, keydb_db=0, server_id='unknown'):
        """Initialize trainer with KeyDB connection and server identification"""
        self.redis_client = redis.Redis(
            host=keydb_host, 
            port=keydb_port, 
            db=keydb_db,
            decode_responses=False
        )
        self.weights = {}
        self.device = 'cpu'  # Use CPU for training to avoid CUDA memory issues
        self.model: Optional[nn.Module] = None
        self.experiment_name = "amber_large_cnn"  # Large CNN experiment name (MLP comparable)
        self.server_id = server_id
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        self.log_file = f'logs/training_log_{server_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize CSV log file with headers"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'server_id', 'iteration', 'epoch', 'batch_loss', 'batch_acc',
                'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'val_precision', 'val_recall',
                'learning_rate', 'time_elapsed'
            ])
    
    def log_training_step(self, iteration: int, epoch: int, batch_loss: float, batch_acc: float,
                         train_loss: float, train_acc: float, val_loss: float, val_acc: float,
                         val_f1: float, val_precision: float, val_recall: float,
                         learning_rate: float, time_elapsed: float):
        """Log training step to CSV file"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), self.server_id, iteration, epoch,
                batch_loss, batch_acc, train_loss, train_acc, val_loss, val_acc,
                val_f1, val_precision, val_recall, learning_rate, time_elapsed
            ])
    
    def update_training_status(self, status: str, iteration: int = 0, loss: float = 0.0, accuracy: float = 0.0):
        """Update training status in KeyDB for monitoring"""
        try:
            status_data = {
                'server_id': self.server_id,
                'status': status,  # 'running', 'completed', 'failed', 'stopped'
                'iteration': iteration,
                'current_loss': loss,
                'current_accuracy': accuracy,
                'last_update': datetime.now().isoformat(),
                'start_time': getattr(self, 'start_time', datetime.now().isoformat())
            }
            
            status_key = f"{self.experiment_name}:training_status:{self.server_id}"
            self.redis_client.set(status_key, json.dumps(status_data))
            return True
        except Exception as e:
            print(f"[{self.server_id}] ❌ Error updating status: {e}")
            return False
    
    def get_all_training_status(self) -> dict:
        """Get training status of all servers"""
        try:
            all_status = {}
            for key in self.redis_client.keys(f"{self.experiment_name}:training_status:*"):
                server_id = key.decode('utf-8').split(':')[-1]
                status_data = self.redis_client.get(key)
                if status_data:
                    all_status[server_id] = json.loads(status_data.decode('utf-8'))
            return all_status
        except Exception as e:
            print(f"[{self.server_id}] ❌ Error getting status: {e}")
            return {}
    
    def check_training_completion(self) -> bool:
        """Check if all servers have completed training"""
        try:
            all_status = self.get_all_training_status()
            if not all_status:
                return False
            
            # Check if all servers are completed
            all_completed = all(
                status['status'] in ['completed', 'failed', 'stopped'] 
                for status in all_status.values()
            )
            
            return all_completed
        except Exception as e:
            print(f"[{self.server_id}] ❌ Error checking completion: {e}")
            return False
    
    def load_weights(self) -> bool:
        """Load current weights from KeyDB"""
        try:
            # Load metadata
            metadata_key = f"{self.experiment_name}:metadata"
            metadata_bytes = self.redis_client.get(metadata_key)
            if metadata_bytes is None:
                print(f"[{self.server_id}] ❌ No weights found in KeyDB. Run initializer first.")
                return False
            
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            layer_keys = metadata['layer_keys']
            
            # Load weights
            self.weights = {}
            for key in layer_keys:
                redis_key = f"{self.experiment_name}:{key}"
                weight_bytes = self.redis_client.get(redis_key)
                if weight_bytes is None:
                    print(f"[{self.server_id}] ❌ Missing weight: {key}")
                    return False
                self.weights[key] = pickle.loads(weight_bytes)
            
            return True
            
        except Exception as e:
            print(f"[{self.server_id}] ❌ Error loading weights: {e}")
            return False
    
    def save_weights(self) -> bool:
        """Save updated weights back to KeyDB"""
        try:
            for key, weight_matrix in self.weights.items():
                redis_key = f"{self.experiment_name}:{key}"
                serialized_weights = pickle.dumps(weight_matrix)
                self.redis_client.set(redis_key, serialized_weights)
            return True
        except Exception as e:
            print(f"[{self.server_id}] ❌ Error saving weights: {e}")
            return False
    
    class _LargeCNN1D(nn.Module):
        def __init__(self, input_features: int = 2381, num_classes: int = 2, dropout_p: float = 0.1):
            super().__init__()
            
            # LARGE CNN architecture - ~3M parameters
            # Target: ~3M parameters for high capacity training
            
            # First convolutional layer
            self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)      # 1 → 64 channels
            self.bn1 = nn.BatchNorm1d(64)
            
            # Second convolutional layer
            self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)     # 64 → 128 channels
            self.bn2 = nn.BatchNorm1d(128)
            
            # Third convolutional layer
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)    # 128 → 256 channels
            self.bn3 = nn.BatchNorm1d(256)
            
            # Fourth convolutional layer
            self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)   # 256 → 512 channels
            self.bn4 = nn.BatchNorm1d(512)
            
            # Pooling layers
            self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)            # After conv1
            self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)            # After conv2
            self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)            # After conv3
            self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4)            # After conv4
            
            # Calculate feature size after convolutions and pooling
            # Starting: input_features
            # After conv1 (kernel=7, stride=1, padding=3): input_features (no change)
            # After maxpool1 (kernel=4, stride=4): input_features/4
            # After conv2 (kernel=5, stride=1, padding=2): input_features/4 (no change)
            # After maxpool2 (kernel=4, stride=4): input_features/16
            # After conv3 (kernel=3, stride=1, padding=1): input_features/16 (no change)
            # After maxpool3 (kernel=4, stride=4): input_features/64
            # After conv4 (kernel=3, stride=1, padding=1): input_features/64 (no change)
            # After maxpool4 (kernel=4, stride=4): input_features/256
            fc_input_size = 512 * (input_features // 256)                 # 512 channels × (2381/256) = 512 × 9 = 4,608
            
            # Very large fully connected layers for ~3M parameters
            self.fc1 = nn.Linear(fc_input_size, 1024)                     # Hidden layer 1: 4,608 → 1024
            self.fc2 = nn.Linear(1024, 512)                               # Hidden layer 2: 1024 → 512
            self.fc3 = nn.Linear(512, num_classes)                       # Output layer: 512 → 2
            
            # Dropout
            self.dropout = nn.Dropout(p=dropout_p)
            
            # Activation
            self.relu = nn.ReLU()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Input shape: (batch_size, 1, input_features)
            # Add channel dimension if needed
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (batch_size, 1, input_features)
            
            # First convolutional layer
            x = self.pool1(self.relu(self.bn1(self.conv1(x))))
            
            # Second convolutional layer
            x = self.pool2(self.relu(self.bn2(self.conv2(x))))
            
            # Third convolutional layer
            x = self.pool3(self.relu(self.bn3(self.conv3(x))))
            
            # Fourth convolutional layer
            x = self.pool4(self.relu(self.bn4(self.conv4(x))))
            
            # Flatten for fully connected layers
            x = x.view(x.size(0), -1)
            
            # Fully connected layers
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.fc3(x)
            
            return x

    def _ensure_model(self, input_features: int = 2381, num_classes: int = 2, dropout_p: float = 0.1):
        """Ensure model exists, create EVERY time (like MLP trainer)"""
        # Always create new model (like MLP trainer)
        # print(f"[{self.server_id}] 🔨 Creating new LARGE 4-layer CNN model...")
        self.model = self._LargeCNN1D(input_features, num_classes, dropout_p=dropout_p).to(self.device)
        # print(f"[{self.server_id}] ✅ LARGE 4-layer CNN model created successfully")

    def _load_weights_into_model(self):
        assert self.model is not None, "Model must be created before loading weights"
        
        # Check if existing weights match our LARGE architecture
        if 'conv1_weights' in self.weights:
            existing_conv1_shape = self.weights['conv1_weights'].shape
            if existing_conv1_shape[0] != 64:  # Different number of channels
                print(f"[{self.server_id}] ⚠️  Existing weights have {existing_conv1_shape[0]} channels, but model expects 64")
                print(f"[{self.server_id}] 🔄 Creating new weights for LARGE architecture...")
                self._create_new_weights()
                return
        
        # Load conv1 weights (if they exist and match)
        if 'conv1_weights' in self.weights:
            self.model.conv1.weight.data = torch.from_numpy(self.weights['conv1_weights']).to(self.device)
            self.model.conv1.bias.data = torch.from_numpy(self.weights['conv1_bias']).to(self.device)
        
        # Load conv2 weights (if they exist and match)
        if 'conv2_weights' in self.weights:
            self.model.conv2.weight.data = torch.from_numpy(self.weights['conv2_weights']).to(self.device)
            self.model.conv2.bias.data = torch.from_numpy(self.weights['conv2_bias']).to(self.device)
        
        # Load conv3 weights (if they exist and match)
        if 'conv3_weights' in self.weights:
            self.model.conv3.weight.data = torch.from_numpy(self.weights['conv3_weights']).to(self.device)
            self.model.conv3.bias.data = torch.from_numpy(self.weights['conv3_bias']).to(self.device)
        
        # Load conv4 weights (if they exist and match)
        if 'conv4_weights' in self.weights:
            self.model.conv4.weight.data = torch.from_numpy(self.weights['conv4_weights']).to(self.device)
            self.model.conv4.bias.data = torch.from_numpy(self.weights['conv4_bias']).to(self.device)
        
        # Load FC weights (if they exist and match)
        if 'fc1_weights' in self.weights:
            self.model.fc1.weight.data = torch.from_numpy(self.weights['fc1_weights'].T).to(self.device)
            self.model.fc1.bias.data = torch.from_numpy(self.weights['fc1_bias']).to(self.device)
        
        if 'fc2_weights' in self.weights:
            self.model.fc2.weight.data = torch.from_numpy(self.weights['fc2_weights'].T).to(self.device)
            self.model.fc2.bias.data = torch.from_numpy(self.weights['fc2_bias']).to(self.device)
        
        if 'fc3_weights' in self.weights:
            self.model.fc3.weight.data = torch.from_numpy(self.weights['fc3_weights'].T).to(self.device)
            self.model.fc3.bias.data = torch.from_numpy(self.weights['fc3_bias']).to(self.device)

    def _create_new_weights(self):
        """Create new weights for the large architecture"""
        print(f"[{self.server_id}] 🔨 Initializing new weights for large 4-layer CNN...")
        
        # Initialize conv1 weights
        torch.nn.init.xavier_uniform_(self.model.conv1.weight)
        torch.nn.init.zeros_(self.model.conv1.bias)
        
        # Initialize conv2 weights
        torch.nn.init.xavier_uniform_(self.model.conv2.weight)
        torch.nn.init.zeros_(self.model.conv2.bias)
        
        # Initialize conv3 weights
        torch.nn.init.xavier_uniform_(self.model.conv3.weight)
        torch.nn.init.zeros_(self.model.conv3.bias)
        
        # Initialize conv4 weights
        torch.nn.init.xavier_uniform_(self.model.conv4.weight)
        torch.nn.init.zeros_(self.model.conv4.bias)
        
        # Initialize FC weights
        torch.nn.init.xavier_uniform_(self.model.fc1.weight)
        torch.nn.init.zeros_(self.model.fc1.bias)
        torch.nn.init.xavier_uniform_(self.model.fc2.weight)
        torch.nn.init.zeros_(self.model.fc2.bias)
        torch.nn.init.xavier_uniform_(self.model.fc3.weight)
        torch.nn.init.zeros_(self.model.fc3.bias)
        
        print(f"[{self.server_id}] ✅ New weights initialized successfully")

    def _extract_model_weights(self):
        assert self.model is not None, "Model must be created before extracting weights"
        extracted: Dict[str, np.ndarray] = {}
        
        # Extract conv1 weights
        extracted['conv1_weights'] = self.model.conv1.weight.detach().to('cpu').numpy()
        extracted['conv1_bias'] = self.model.conv1.bias.detach().to('cpu').numpy()
        
        # Extract conv2 weights
        extracted['conv2_weights'] = self.model.conv2.weight.detach().to('cpu').numpy()
        extracted['conv2_bias'] = self.model.conv2.bias.detach().to('cpu').numpy()
        
        # Extract conv3 weights
        extracted['conv3_weights'] = self.model.conv3.weight.detach().to('cpu').numpy()
        extracted['conv3_bias'] = self.model.conv3.bias.detach().to('cpu').numpy()
        
        # Extract conv4 weights
        extracted['conv4_weights'] = self.model.conv4.weight.detach().to('cpu').numpy()
        extracted['conv4_bias'] = self.model.conv4.bias.detach().to('cpu').numpy()
        
        # Extract FC weights
        extracted['fc1_weights'] = self.model.fc1.weight.detach().to('cpu').numpy().T
        extracted['fc1_bias'] = self.model.fc1.bias.detach().to('cpu').numpy()
        extracted['fc2_weights'] = self.model.fc2.weight.detach().to('cpu').numpy().T
        extracted['fc2_bias'] = self.model.fc2.bias.detach().to('cpu').numpy()
        extracted['fc3_weights'] = self.model.fc3.weight.detach().to('cpu').numpy().T
        extracted['fc3_bias'] = self.model.fc3.bias.detach().to('cpu').numpy()
        
        self.weights = extracted

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Evaluate current KeyDB weights on validation dataset and compute metrics"""
        if not self.load_weights():
            return float('inf'), 0.0, 0.0, 0.0, 0.0
        
        self._ensure_model(input_features=X.shape[1], num_classes=2, dropout_p=0.0)
        self._load_weights_into_model()
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        # Preprocess
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - mean) / std
        X = np.clip(X, -10.0, 10.0)

        if y.ndim > 1:
            y_idx = np.argmax(y, axis=1)
        else:
            y_idx = y
        y_idx = y_idx.astype(np.int64)
        valid = (y_idx >= 0) & (y_idx < 2)
        X = X[valid]
        y_idx = y_idx[valid]

        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y_idx).to(self.device)
        
        logits = self.model(X_t)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = torch.clamp(logits, -30.0, 30.0)
        loss_t = criterion(logits, y_t)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_t).float().mean().item()
        
        # Additional metrics
        y_true = y_t.cpu().numpy()
        y_pred = preds.cpu().numpy()
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        # Store values before cleanup
        loss_value = float(loss_t.item())
        acc_value = float(acc)
        f1_value = float(f1)
        precision_value = float(precision)
        recall_value = float(recall)
        
        # Clean up tensors to free memory
        del X_t, y_t, logits, loss_t, preds
        # CPU training - no CUDA cache to clear
        gc.collect()
        
        return loss_value, acc_value, f1_value, precision_value, recall_value
    
    @torch.no_grad()
    def calculate_detailed_metrics(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Calculate detailed metrics including F1-score, precision, and recall"""
        if not self.load_weights():
            return float('inf'), 0.0, 0.0, 0.0, 0.0
        
        self._ensure_model(input_features=X.shape[1], num_classes=2, dropout_p=0.0)
        self._load_weights_into_model()
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        # Preprocess
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - mean) / std
        X = np.clip(X, -10.0, 10.0)

        if y.ndim > 1:
            y_idx = np.argmax(y, axis=1)
        else:
            y_idx = y
        y_idx = y_idx.astype(np.int64)
        valid = (y_idx >= 0) & (y_idx < 2)
        X = X[valid]
        y_idx = y_idx[valid]

        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y_idx).to(self.device)
        
        logits = self.model(X_t)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = torch.clamp(logits, -30.0, 30.0)
        loss_t = criterion(logits, y_t)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_t).float().mean().item()
        
        # Convert to numpy for sklearn metrics
        y_true = y_t.cpu().numpy()
        y_pred = preds.cpu().numpy()
        
        # Calculate additional metrics
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Store values before cleanup
        loss_value = float(loss_t.item())
        acc_value = float(acc)
        
        # Clean up tensors to free memory
        del X_t, y_t, logits, loss_t, preds
        gc.collect()
        
        return loss_value, acc_value, f1, precision, recall
    
    def train_single_step(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                         learning_rate: float = 0.001,
                         dropout_p: float = 0.1) -> Tuple[float, float]:
        """Perform single training step with shared memory pattern - EXACTLY like MLP trainer"""
        # Load weights from KeyDB EVERY step (like MLP trainer)
        if not self.load_weights():
            return float('inf'), 0.0
        
        # Build/Load model EVERY step (like MLP trainer)
        input_features = X_batch.shape[1]
        num_classes = 2
        self._ensure_model(input_features, num_classes, dropout_p=dropout_p)
        self._load_weights_into_model()

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=min(learning_rate, 1e-3), weight_decay=1e-4)
        max_grad_norm = 5.0
        criterion = nn.CrossEntropyLoss()

        # Preprocess batch
        X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)
        mean = X_batch.mean(axis=0, keepdims=True)
        std = X_batch.std(axis=0, keepdims=True) + 1e-6
        X_norm = (X_batch - mean) / std
        X_norm = np.clip(X_norm, -10.0, 10.0)
        X_t = torch.from_numpy(X_norm).to(self.device)
        
        # Sanitize labels
        if y_batch.ndim > 1:
            y_indices = np.argmax(y_batch, axis=1)
        else:
            y_indices = y_batch
        y_indices = y_indices.astype(np.int64)
        valid_idx = (y_indices >= 0) & (y_indices < num_classes)
        if not np.all(valid_idx):
            X_t = X_t[torch.from_numpy(valid_idx).to(self.device)]
            y_indices = y_indices[valid_idx]
        y_t = torch.from_numpy(y_indices).to(self.device)

        optimizer.zero_grad()
        logits = self.model(X_t)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = torch.clamp(logits, -30.0, 30.0)
        loss_t = criterion(logits, y_t)
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_t).float().mean().item()
            loss = float(loss_t.item()) if torch.isfinite(loss_t) else float('inf')

        # Extract and save weights EVERY step (like MLP trainer)
        self._extract_model_weights()
        if not self.save_weights():
            print(f"[{self.server_id}] ❌ Failed to save weights")
            return loss, accuracy
        
        # Reload weights EVERY step (like MLP trainer)
        if not self.load_weights():
            print(f"[{self.server_id}] ❌ Failed to reload weights after update")
            return loss, accuracy
        
        # Clear all tensors (like MLP trainer)
        del X_t, y_t, logits, loss_t, preds, optimizer
        del X_batch, y_batch, X_norm, mean, std, y_indices, valid_idx
        
        # CPU training - no CUDA cache to clear
        
        # Force garbage collection
        gc.collect()
        
        # Clear model gradients
        if self.model is not None:
            self.model.zero_grad()
        
        return loss, accuracy
    
    def create_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        """Create mini-batches for training (like MLP trainer)"""
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]
    
    def load_amber_parquet_clean(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load cleaned Parquet shard data (like MLP trainer)"""
        print(f"[{self.server_id}] Loading shard: {file_path}")
        df = pd.read_parquet(file_path, engine="pyarrow")
        
        if 'Label' not in df.columns:
            raise ValueError("Parquet file must contain a 'Label' column")
        
        y = df['Label'].astype('int64').to_numpy()
        X = df.drop(columns=['Label']).astype('float32').to_numpy()
        
        print(f"[{self.server_id}] Shard loaded: {X.shape[0]} samples, {X.shape[1]} features | labels: {np.unique(y)}")
        return X, y

    def train_distributed(self, 
                         target_loss: float = 0.1,
                         max_iterations: int = 10000,
                         batch_size: int = 32,
                         learning_rate: float = 0.01,
                         print_every: int = 50,
                         dropout_p: float = 0.1,
                         eval_every: int = 50):
        """Train on local shard data until target loss is reached (like MLP trainer)"""
        print(f"[{self.server_id}] 🚀 Starting distributed LARGE 4-layer CNN training on local shard...")
        print(f"[{self.server_id}] Target loss: {target_loss}")
        print(f"[{self.server_id}] Max iterations: {max_iterations}")
        print(f"[{self.server_id}] Batch size: {batch_size}")
        print(f"[{self.server_id}] Learning rate: {learning_rate}")
        print(f"[{self.server_id}] Log file: {self.log_file}")
        print("-" * 50)
        
        # Set start time for status tracking
        self.start_time = datetime.now().isoformat()
        
        # Report training started
        self.update_training_status('running', 0, 0.0, 0.0)
        
        # Load local shard data (like MLP trainer)
        train_file = "..//data/train_shard.parquet"
        test_file = "..//data/test_shard.parquet"
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"[{self.server_id}] ❌ Shard data not found! Expected: {train_file}, {test_file}")
            self.update_training_status('failed', 0, float('inf'), 0.0)
            return False
        
        print(f"[{self.server_id}] 📖 Loading training data...")
        X_train, y_train = self.load_amber_parquet_clean(train_file)
        print(f"[{self.server_id}] 📖 Loading test data...")
        X_test, y_test = self.load_amber_parquet_clean(test_file)
        
        print(f"[{self.server_id}] ✅ Local shard loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        # Check memory after loading data
        try:
            import psutil
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            print(f"[{self.server_id}] 📊 Memory after loading data:")
            print(f"[{self.server_id}] System RAM: {memory.used/1024/1024:.0f}MB / {memory.total/1024/1024:.0f}MB ({memory.percent:.1f}%)")
            print(f"[{self.server_id}] Process Memory: {process.memory_info().rss/1024/1024:.0f}MB")
        except:
            print(f"[{self.server_id}] ⚠️  Could not check memory")
        
        iteration = 0
        epoch = 0
        start_time = time.time()
        
        while iteration < max_iterations:
            epoch += 1
            epoch_losses = []
            epoch_accuracies = []
            
            for X_batch, y_batch in self.create_batches(X_train, y_train, batch_size):
                # Check memory before first training step
                if iteration == 0:
                    try:
                        import psutil
                        memory = psutil.virtual_memory()
                        process = psutil.Process(os.getpid())
                        print(f"[{self.server_id}] 📊 Memory before first training step:")
                        print(f"[{self.server_id}] System RAM: {memory.used/1024/1024:.0f}MB / {memory.total/1024/1024:.0f}MB ({memory.percent:.1f}%)")
                        print(f"[{self.server_id}] Process Memory: {process.memory_info().rss/1024/1024:.0f}MB")
                    except:
                        print(f"[{self.server_id}] ⚠️  Could not check memory")
                
                if iteration >= max_iterations:
                    break
                
                # Training step
                batch_loss, batch_acc = self.train_single_step(X_batch, y_batch, learning_rate, dropout_p)
                epoch_losses.append(batch_loss)
                epoch_accuracies.append(batch_acc)
                iteration += 1
                
                # Print progress
                if iteration == 1 or (iteration % print_every == 0):
                    avg_loss = np.mean(epoch_losses[-print_every:]) if len(epoch_losses) >= print_every else np.mean(epoch_losses)
                    avg_acc = np.mean(epoch_accuracies[-print_every:]) if len(epoch_accuracies) >= print_every else np.mean(epoch_accuracies)
                    elapsed_time = time.time() - start_time
                    
                    # Update training status in KeyDB
                    self.update_training_status('running', iteration, avg_loss, avg_acc)
                    
                    # Run validation
                    if iteration % eval_every == 0 or iteration == 1:
                        val_loss, val_acc, val_f1, val_precision, val_recall = self.evaluate(X_test, y_test)
                        
                        # Update status with validation results
                        self.update_training_status('running', iteration, val_loss, val_acc)
                        
                        print(f"[{self.server_id}] Iter {iteration:6d} | TrainLoss: {avg_loss:.6f} | TrainAcc: {avg_acc:.4f} | ValLoss: {val_loss:.6f} | ValAcc: {val_acc:.4f} | ValF1: {val_f1:.4f} | ValPrec: {val_precision:.4f} | ValRec: {val_recall:.4f} | Time: {elapsed_time:.1f}s")
                        
                        # Log to CSV
                        self.log_training_step(iteration, epoch, batch_loss, batch_acc, 
                                             avg_loss, avg_acc, val_loss, val_acc,
                                             val_f1, val_precision, val_recall,
                                             learning_rate, elapsed_time)
                        
                        # Check if target loss reached
                        if val_loss <= target_loss:
                            # Report completion
                            self.update_training_status('completed', iteration, val_loss, val_acc)
                            
                            # Calculate detailed metrics on test set
                            print(f"[{self.server_id}] 📊 Calculating detailed metrics on test set...")
                            test_loss, test_acc, test_f1, test_precision, test_recall = self.calculate_detailed_metrics(X_test, y_test)
                            
                            elapsed_time = time.time() - start_time
                            print(f"[{self.server_id}] 🎉 Target loss reached!")
                            print(f"[{self.server_id}] Final iteration: {iteration}")
                            print(f"[{self.server_id}] Final val loss: {val_loss:.6f}")
                            print(f"[{self.server_id}] Total training time: {elapsed_time:.2f} seconds")
                            print(f"[{self.server_id}] " + "="*60)
                            print(f"[{self.server_id}] 📊 FINAL TEST SET METRICS:")
                            print(f"[{self.server_id}] Test Loss: {test_loss:.6f}")
                            print(f"[{self.server_id}] Test Accuracy: {test_acc:.4f}")
                            print(f"[{self.server_id}] Test F1-Score: {test_f1:.4f}")
                            print(f"[{self.server_id}] Test Precision: {test_precision:.4f}")
                            print(f"[{self.server_id}] Test Recall: {test_recall:.4f}")
                            print(f"[{self.server_id}] " + "="*60)
                            # Log final test metrics to CSV (same file)
                            self.log_training_step(
                                iteration, epoch, batch_loss, batch_acc,
                                avg_loss, avg_acc, test_loss, test_acc,
                                test_f1, test_precision, test_recall,
                                learning_rate, elapsed_time
                            )
                            return True
                    else:
                        print(f"[{self.server_id}] Iter {iteration:6d} | Loss: {avg_loss:.6f} | Accuracy: {avg_acc:.4f} | Time: {elapsed_time:.1f}s")
                
                if iteration >= max_iterations:
                    break
            
            if iteration >= max_iterations:
                break
        
        # Report failure if max iterations reached
        final_loss = np.mean(epoch_losses[-10:]) if epoch_losses else float('inf')
        final_acc = np.mean(epoch_accuracies[-10:]) if epoch_accuracies else 0.0
        self.update_training_status('completed', iteration, final_loss, final_acc)
        
        # Calculate detailed metrics on test set
        print(f"[{self.server_id}] 📊 Calculating detailed metrics on test set...")
        test_loss, test_acc, test_f1, test_precision, test_recall = self.calculate_detailed_metrics(X_test, y_test)
        
        # Maximum iterations reached
        elapsed_time = time.time() - start_time
        
        print(f"[{self.server_id}] 🎉 Training completed after {max_iterations} iterations!")
        print(f"[{self.server_id}] Final training loss: {final_loss:.6f}")
        print(f"[{self.server_id}] Final training accuracy: {final_acc:.4f}")
        print(f"[{self.server_id}] Total training time: {elapsed_time:.2f} seconds")
        print(f"[{self.server_id}] " + "="*60)
        print(f"[{self.server_id}] 📊 FINAL TEST SET METRICS:")
        print(f"[{self.server_id}] Test Loss: {test_loss:.6f}")
        print(f"[{self.server_id}] Test Accuracy: {test_acc:.4f}")
        print(f"[{self.server_id}] Test F1-Score: {test_f1:.4f}")
        print(f"[{self.server_id}] Test Precision: {test_precision:.4f}")
        print(f"[{self.server_id}] Test Recall: {test_recall:.4f}")
        print(f"[{self.server_id}] " + "="*60)
        # Also append final test metrics to CSV
        self.log_training_step(
            iteration, epoch, '', '',
            final_loss, final_acc, test_loss, test_acc,
            test_f1, test_precision, test_recall,
            learning_rate, elapsed_time
        )
        return True

def main():
    """Main function for distributed training - run on each server"""
    
    # Get server ID from environment or use hostname
    import socket
    server_id = os.environ.get('SERVER_ID', socket.gethostname())
    
    print(f"🚀 Starting distributed large 1D CNN trainer on server: {server_id}")
    
    # Initialize trainer
    trainer = DistributedCNN1DTrainer(server_id=server_id)
    
    # Check if weights exist in KeyDB
    if not trainer.load_weights():
        print(f"[{server_id}] ❌ No weights found. Please run the large CNN initializer first!")
        return
    
    print(f"[{server_id}] ✅ Large CNN weights loaded successfully from KeyDB")
    
    # Start distributed training
    try:
        success = trainer.train_distributed(
            target_loss=0.2,
            max_iterations=1000,  # You can change this value
            batch_size=256,         # Smaller batch size for CPU training
            learning_rate=0.0001,  # Learning rate for 4-layer CNN
            print_every=50,       # Print every 50 iterations
            dropout_p=0.1,
            eval_every=50
        )
        
        if success:
            print(f"[{server_id}] ✅ Distributed large CNN training completed successfully!")
        else:
            print(f"[{server_id}] ⚠️  Distributed large CNN training stopped without reaching target loss")
            
    except Exception as e:
        print(f"[{server_id}] ❌ Fatal error during training: {e}")
        trainer.update_training_status('failed', 0, float('inf'), 0.0)

if __name__ == "__main__":
    main()