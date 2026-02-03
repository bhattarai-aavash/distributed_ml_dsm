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

class DistributedLSTMTrainer:
    """Distributed LSTM trainer using KeyDB for shared weight storage across servers - EXACTLY like MLP trainer"""
    
    def __init__(self, keydb_host='localhost', keydb_port=6379, keydb_db=0, server_id='unknown'):
        """Initialize trainer with KeyDB connection and server identification"""
        self.redis_client = redis.Redis(
            host=keydb_host, 
            port=keydb_port, 
            db=keydb_db,
            decode_responses=False
        )
        self.weights = {}
        self.device = 'cpu'  # Force CPU for consistency across servers
        self.model: Optional[nn.Module] = None
        self.experiment_name = "amber_simple_lstm"  # LSTM experiment name
        self.server_id = server_id
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        self.log_file = f'logs/lstm_training_log_{server_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize CSV log file with headers"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'server_id', 'iteration', 'epoch', 'batch_loss', 'batch_acc',
                'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate', 'time_elapsed'
            ])
    
    def log_training_step(self, iteration: int, epoch: int, batch_loss: float, batch_acc: float,
                         train_loss: float, train_acc: float, val_loss: float, val_acc: float,
                         learning_rate: float, time_elapsed: float):
        """Log training step to CSV file"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), self.server_id, iteration, epoch,
                batch_loss, batch_acc, train_loss, train_acc, val_loss, val_acc,
                learning_rate, time_elapsed
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
                print(f"[{self.server_id}] ❌ No weights found in KeyDB. Run LSTM initializer first.")
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
    
    class _SimpleLSTM(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 8, num_layers: int = 3, num_classes: int = 2, dropout_p: float = 0.1):
            super().__init__()
            
            # 3-LAYER ULTRA-COMPACT LSTM architecture - minimal memory usage
            # Memory usage: ~3K parameters (still extremely small)
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # LSTM layer - 3 layers for enhanced learning
            self.lstm = nn.LSTM(
                input_size=input_size,           # 1 feature per timestep
                hidden_size=hidden_size,         # 8 hidden units (was 32)
                num_layers=num_layers,           # 3 layers (was 2)
                dropout=dropout_p if num_layers > 1 else 0,
                batch_first=True
            )
            
            # Tiny fully connected layers
            self.fc1 = nn.Linear(hidden_size, 8)                       # Hidden layer: 8 → 8 (was 32 → 16)
            self.fc2 = nn.Linear(8, num_classes)                       # Output layer: 8 → 2 (was 16 → 2)
            
            # Dropout
            self.dropout = nn.Dropout(p=dropout_p)
            
            # Activation
            self.relu = nn.ReLU()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Input shape: (batch_size, 2381)
            # Process in smaller chunks to save memory
            batch_size = x.size(0)
            
            # Reshape: (batch_size, 2381, 1) - each feature becomes a timestep
            if x.dim() == 2:
                x = x.unsqueeze(-1)  # (batch_size, 2381, 1)
            
            # Process in chunks to save memory
            chunk_size = 100  # Process 100 timesteps at a time
            num_chunks = (x.size(1) + chunk_size - 1) // chunk_size
            
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            
            # Process chunks sequentially
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, x.size(1))
                chunk = x[:, start_idx:end_idx, :]
                
                if i == 0:
                    # First chunk
                    lstm_out, (hn, cn) = self.lstm(chunk, (h0, c0))
                else:
                    # Subsequent chunks - continue from previous state
                    lstm_out, (hn, cn) = self.lstm(chunk, (hn, cn))
            
            # Use last hidden state for classification
            last_output = hn[-1]  # (batch_size, hidden_size)
            
            # Fully connected layers
            x = self.dropout(self.relu(self.fc1(last_output)))
            x = self.fc2(x)
            
            return x

    def _ensure_model(self, input_size: int = 1, hidden_size: int = 8, num_layers: int = 3, num_classes: int = 2, dropout_p: float = 0.1):
        """Ensure model exists, create EVERY time (like MLP trainer)"""
        # Always create new model (like MLP trainer)
        # print(f"[{self.server_id}] 🔨 Creating new 3-LAYER ULTRA-COMPACT LSTM model...")
        self.model = self._SimpleLSTM(input_size, hidden_size, num_layers, num_classes, dropout_p=dropout_p).to(self.device)
        # print(f"[{self.server_id}] ✅ 3-LAYER ULTRA-COMPACT LSTM model created successfully")

    def _load_weights_into_model(self):
        assert self.model is not None, "Model must be created before loading weights"
        
        # Load LSTM weights for all three layers
        if 'lstm_weight_ih_l0' in self.weights:
            # Layer 0 weights
            self.model.lstm.weight_ih_l0.data = torch.from_numpy(self.weights['lstm_weight_ih_l0']).to(self.device)
            self.model.lstm.weight_hh_l0.data = torch.from_numpy(self.weights['lstm_weight_hh_l0']).to(self.device)
            self.model.lstm.bias_ih_l0.data = torch.from_numpy(self.weights['lstm_bias_ih_l0']).to(self.device)
            self.model.lstm.bias_hh_l0.data = torch.from_numpy(self.weights['lstm_bias_hh_l0']).to(self.device)
            
            # Layer 1 weights
            self.model.lstm.weight_ih_l1.data = torch.from_numpy(self.weights['lstm_weight_ih_l1']).to(self.device)
            self.model.lstm.weight_hh_l1.data = torch.from_numpy(self.weights['lstm_weight_hh_l1']).to(self.device)
            self.model.lstm.bias_ih_l1.data = torch.from_numpy(self.weights['lstm_bias_ih_l1']).to(self.device)
            self.model.lstm.bias_hh_l1.data = torch.from_numpy(self.weights['lstm_bias_hh_l1']).to(self.device)
            
            # Layer 2 weights
            self.model.lstm.weight_ih_l2.data = torch.from_numpy(self.weights['lstm_weight_ih_l2']).to(self.device)
            self.model.lstm.weight_hh_l2.data = torch.from_numpy(self.weights['lstm_weight_hh_l2']).to(self.device)
            self.model.lstm.bias_ih_l2.data = torch.from_numpy(self.weights['lstm_bias_ih_l2']).to(self.device)
            self.model.lstm.bias_hh_l2.data = torch.from_numpy(self.weights['lstm_bias_hh_l2']).to(self.device)
        
        # Load FC weights
        if 'fc1_weights' in self.weights:
            self.model.fc1.weight.data = torch.from_numpy(self.weights['fc1_weights'].T).to(self.device)
            self.model.fc1.bias.data = torch.from_numpy(self.weights['fc1_bias']).to(self.device)
        
        if 'fc2_weights' in self.weights:
            self.model.fc2.weight.data = torch.from_numpy(self.weights['fc2_weights'].T).to(self.device)
            self.model.fc2.bias.data = torch.from_numpy(self.weights['fc2_bias']).to(self.device)

    def _extract_model_weights(self):
        assert self.model is not None, "Model must be created before extracting weights"
        extracted: Dict[str, np.ndarray] = {}
        
        # Extract LSTM weights for all three layers
        extracted['lstm_weight_ih_l0'] = self.model.lstm.weight_ih_l0.detach().to('cpu').numpy()
        extracted['lstm_weight_hh_l0'] = self.model.lstm.weight_hh_l0.detach().to('cpu').numpy()
        extracted['lstm_bias_ih_l0'] = self.model.lstm.bias_ih_l0.detach().to('cpu').numpy()
        extracted['lstm_bias_hh_l0'] = self.model.lstm.bias_hh_l0.detach().to('cpu').numpy()
        
        extracted['lstm_weight_ih_l1'] = self.model.lstm.weight_ih_l1.detach().to('cpu').numpy()
        extracted['lstm_weight_hh_l1'] = self.model.lstm.weight_hh_l1.detach().to('cpu').numpy()
        extracted['lstm_bias_ih_l1'] = self.model.lstm.bias_ih_l1.detach().to('cpu').numpy()
        extracted['lstm_bias_hh_l1'] = self.model.lstm.bias_hh_l1.detach().to('cpu').numpy()
        
        extracted['lstm_weight_ih_l2'] = self.model.lstm.weight_ih_l2.detach().to('cpu').numpy()
        extracted['lstm_weight_hh_l2'] = self.model.lstm.weight_hh_l2.detach().to('cpu').numpy()
        extracted['lstm_bias_ih_l2'] = self.model.lstm.bias_ih_l2.detach().to('cpu').numpy()
        extracted['lstm_bias_hh_l2'] = self.model.lstm.bias_hh_l2.detach().to('cpu').numpy()
        
        # Extract FC weights
        extracted['fc1_weights'] = self.model.fc1.weight.detach().to('cpu').numpy().T
        extracted['fc1_bias'] = self.model.fc1.bias.detach().to('cpu').numpy()
        extracted['fc2_weights'] = self.model.fc2.weight.detach().to('cpu').numpy().T
        extracted['fc2_bias'] = self.model.fc2.bias.detach().to('cpu').numpy()
        
        self.weights = extracted

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate current KeyDB weights on validation dataset"""
        if not self.load_weights():
            return float('inf'), 0.0
        
        self._ensure_model(input_size=1, hidden_size=8, num_layers=3, num_classes=2, dropout_p=0.0)
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
        
        # Store values before cleanup
        loss_value = float(loss_t.item())
        acc_value = float(acc)
        
        # Clean up tensors to free memory
        del X_t, y_t, logits, loss_t, preds
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return loss_value, acc_value
    
    def train_single_step(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                         learning_rate: float = 0.001,
                         dropout_p: float = 0.1) -> Tuple[float, float]:
        """Perform single training step with shared memory pattern - EXACTLY like MLP trainer"""
        # Load weights from KeyDB EVERY step (like MLP trainer)
        if not self.load_weights():
            return float('inf'), 0.0
        
        # Build/Load model EVERY step (like MLP trainer)
        self._ensure_model(input_size=1, hidden_size=8, num_layers=3, num_classes=2, dropout_p=dropout_p)
        self._load_weights_into_model()

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=min(learning_rate, 1e-3), weight_decay=1e-4)
        max_grad_norm = 0.1  # Much tighter gradient clipping to prevent NaN values
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
        valid_idx = (y_indices >= 0) & (y_indices < 2)
        if not np.all(valid_idx):
            X_t = X_t[torch.from_numpy(valid_idx).to(self.device)]
            y_indices = y_indices[valid_idx]
        y_t = torch.from_numpy(y_indices).to(self.device)

        optimizer.zero_grad()
        logits = self.model(X_t)
        
        # Check for NaN values and prevent them
        if torch.isnan(logits).any():
            print(f"[{self.server_id}] 🚨 CRITICAL: NaN detected in model output! Stopping training.")
            return float('inf'), 0.0
        
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = torch.clamp(logits, -30.0, 30.0)
        loss_t = criterion(logits, y_t)
        
        # Check for NaN loss
        if torch.isnan(loss_t):
            print(f"[{self.server_id}] 🚨 CRITICAL: NaN detected in loss! Stopping training.")
            return float('inf'), 0.0
        
        loss_t.backward()
        
        # Check for NaN gradients
        has_nan_grad = False
        for p in self.model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"[{self.server_id}] 🚨 CRITICAL: NaN detected in gradients! Stopping training.")
            return float('inf'), 0.0
        
        # Debug: Check gradient norms before clipping
        with torch.no_grad():
            grad_norms = [p.grad.norm().item() if p.grad is not None else 0 for p in self.model.parameters()]
            # Print gradient norms for debugging (removed iteration reference)
            print(f"[{self.server_id}] Gradient norms: {grad_norms}")
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_t).float().mean().item()
            loss = float(loss_t.item()) if torch.isfinite(loss_t) else float('inf')
            
            # Debug: Check sample outputs and probabilities
            # Print sample outputs for debugging (removed iteration reference)
            sample_output = self.model(X_t[:1])  # Test first sample
            sample_probs = torch.softmax(sample_output, dim=1)
            print(f"[{self.server_id}] Sample output: {sample_output}")
            print(f"[{self.server_id}] Sample probabilities: {sample_probs}")
            print(f"[{self.server_id}] Expected vs Predicted: {y_t[:1]} vs {preds[:1]}")

        # Extract and save weights EVERY step (like MLP trainer)
        self._extract_model_weights()
        
        # Check for NaN weights and prevent corruption
        has_nan_weights = False
        for key, weight_matrix in self.weights.items():
            if np.isnan(weight_matrix).any():
                has_nan_weights = True
                print(f"[{self.server_id}] 🚨 CRITICAL: NaN detected in weights {key}!")
                break
        
        if has_nan_weights:
            print(f"[{self.server_id}] 🚨 CRITICAL: NaN weights detected! Stopping training.")
            return float('inf'), 0.0
        
        # Debug: Check if weights are actually changing
        # Print weight changes for debugging (removed iteration reference)
        weight_changes = []
        for key, weight_matrix in self.weights.items():
            if hasattr(self, 'prev_weights') and key in self.prev_weights:
                change = np.mean(np.abs(weight_matrix - self.prev_weights[key]))
                weight_changes.append(f"{key}: {change:.6f}")
            else:
                weight_changes.append(f"{key}: NEW")
        print(f"[{self.server_id}] Weight changes: {weight_changes}")
        
        # Store current weights for next comparison
        self.prev_weights = {k: v.copy() for k, v in self.weights.items()}
        
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
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
                         learning_rate: float = 0.001,
                         print_every: int = 50,
                         dropout_p: float = 0.1,
                         eval_every: int = 50):
        """Train on local shard data until target loss is reached (like MLP trainer)"""
        print(f"[{self.server_id}] 🚀 Starting distributed simple LSTM training on local shard...")
        print(f"[{self.server_id}] Target loss: {target_loss}")
        print(f"[{self.server_id}] Max iterations: {max_iterations}")
        print(f"[{self.server_id}] Batch size: {batch_size}")
        print(f"[{self.server_id}] Learning rate: {learning_rate}")
        print(f"[{self.server_id}] Gradient clipping: 0.1 (ultra-tight to prevent NaN values)")
        print(f"[{self.server_id}] Log file: {self.log_file}")
        print("-" * 50)
        
        # Set start time for status tracking
        self.start_time = datetime.now().isoformat()
        
        # Report training started
        self.update_training_status('running', 0, 0.0, 0.0)
        
        # Load local shard data (like MLP trainer)
        train_file = "../data/train_shard.parquet"
        test_file = "../data/test_shard.parquet"
        
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
                        val_loss, val_acc = self.evaluate(X_test, y_test)
                        
                        # Update status with validation results
                        self.update_training_status('running', iteration, val_loss, val_acc)
                        
                        print(f"[{self.server_id}] Iter {iteration:6d} | TrainLoss: {avg_loss:.6f} | TrainAcc: {avg_acc:.4f} | ValLoss: {val_loss:.6f} | ValAcc: {val_acc:.4f} | LR: {learning_rate:.6f} | Time: {elapsed_time:.1f}s")
                        
                        # Log to CSV
                        self.log_training_step(iteration, epoch, batch_loss, batch_acc, 
                                             avg_loss, avg_acc, val_loss, val_acc, 
                                             learning_rate, elapsed_time)
                        
                        # Check if target loss reached
                        if val_loss <= target_loss:
                            # Report completion
                            self.update_training_status('completed', iteration, val_loss, val_acc)
                            elapsed_time = time.time() - start_time
                            print(f"[{self.server_id}] 🎉 Target loss reached!")
                            print(f"[{self.server_id}] Final iteration: {iteration}")
                            print(f"[{self.server_id}] Final val loss: {val_loss:.6f}")
                            print(f"[{self.server_id}] Total training time: {elapsed_time:.2f} seconds")
                            return True
                    else:
                        print(f"[{self.server_id}] Iter {iteration:6d} | Loss: {avg_loss:.6f} | Accuracy: {avg_acc:.4f} | LR: {learning_rate:.6f} | Time: {elapsed_time:.1f}s")
                
                if iteration >= max_iterations:
                    break
            
            if iteration >= max_iterations:
                break
        
        # Report failure if max iterations reached
        final_loss = np.mean(epoch_losses[-10:]) if epoch_losses else float('inf')
        final_acc = np.mean(epoch_accuracies[-10:]) if epoch_accuracies else 0.0
        self.update_training_status('failed', iteration, final_loss, final_acc)
        
        # Maximum iterations reached
        elapsed_time = time.time() - start_time
        
        print(f"[{self.server_id}] ⚠️  Maximum iterations reached without reaching target loss")
        print(f"[{self.server_id}] Final loss: {final_loss:.6f}")
        print(f"[{self.server_id}] Final accuracy: {final_acc:.4f}")
        print(f"[{self.server_id}] Total training time: {elapsed_time:.2f} seconds")
        return False

def main():
    """Main function for distributed training - run on each server"""
    
    # Get server ID from environment or use hostname
    import socket
    server_id = os.environ.get('SERVER_ID', socket.gethostname())
    
    print(f"🚀 Starting distributed simple LSTM trainer on server: {server_id}")
    
    # Initialize trainer
    trainer = DistributedLSTMTrainer(server_id=server_id)
    
    # Check if weights exist in KeyDB
    if not trainer.load_weights():
        print(f"[{server_id}] ❌ No weights found. Please run the LSTM initializer first!")
        return
    
    print(f"[{server_id}] ✅ Simple LSTM weights loaded successfully from KeyDB")
    
    # Start distributed training
    try:
        success = trainer.train_distributed(
            target_loss=0.2,
            max_iterations=5000,  # You can change this value
            batch_size=32,        # Safe batch size for LSTM
            learning_rate=0.0001,  # Much lower learning rate for 3-layer LSTM
            print_every=50,
            dropout_p=0.1,
            eval_every=50
        )
        
        if success:
            print(f"[{server_id}] ✅ Distributed simple LSTM training completed successfully!")
        else:
            print(f"[{self.server_id}] ⚠️  Distributed simple LSTM training stopped without reaching target loss")
            
    except Exception as e:
        print(f"[{server_id}] ❌ Fatal error during training: {e}")
        trainer.update_training_status('failed', 0, float('inf'), 0.0)

if __name__ == "__main__":
    main()
