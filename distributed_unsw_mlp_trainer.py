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

class DistributedUNSWMLPTrainer:
    """Distributed MLP trainer for UNSW-NB15 dataset using KeyDB for shared weight storage across servers"""
    
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
        self.experiment_name = "unsw_mlp"
        self.server_id = server_id
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        self.log_file = f'logs/unsw_training_log_{server_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
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
    
    class _UNSWMLP(nn.Module):
        def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_p: float = 0.1):
            super().__init__()
            layers: List[nn.Module] = []
            layer_sizes = [input_size] + hidden_sizes
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                layers.append(nn.ReLU())
                if dropout_p > 0:
                    layers.append(nn.Dropout(p=dropout_p))
            layers.append(nn.Linear(layer_sizes[-1], output_size))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    def _ensure_model(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_p: float = 0.1):
        if self.model is None:
            self.model = self._UNSWMLP(input_size, hidden_sizes, output_size, dropout_p=dropout_p).to(self.device)

    def _load_weights_into_model(self):
        assert self.model is not None, "Model must be created before loading weights"
        # Determine layer sizes from stored weights
        num_layers = len([k for k in self.weights.keys() if 'weights' in k])
        linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        if len(linear_layers) != num_layers:
            # Rebuild model to match weights in KeyDB
            inferred_input, inferred_hidden, inferred_output = self._infer_sizes_from_weights()
            self.model = self._UNSWMLP(inferred_input, inferred_hidden, inferred_output, dropout_p=0.1).to(self.device)
            linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        
        with torch.no_grad():
            for i, lin in enumerate(linear_layers):
                w_np = self.weights[f"layer_{i}_weights"]
                b_np = self.weights[f"layer_{i}_bias"]
                # Convert to torch and transpose
                w_t = torch.from_numpy(w_np.T.copy()).to(self.device)
                b_t = torch.from_numpy(b_np.copy()).to(self.device)
                lin.weight.copy_(w_t)
                lin.bias.copy_(b_t)

    def _extract_model_weights(self):
        assert self.model is not None, "Model must be created before extracting weights"
        extracted: Dict[str, np.ndarray] = {}
        linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        for i, lin in enumerate(linear_layers):
            w = lin.weight.detach().to('cpu').numpy().T
            b = lin.bias.detach().to('cpu').numpy()
            extracted[f"layer_{i}_weights"] = w.astype(np.float32)
            extracted[f"layer_{i}_bias"] = b.astype(np.float32)
        self.weights = extracted

    def _infer_sizes_from_weights(self) -> Tuple[int, List[int], int]:
        """Infer (input_size, hidden_sizes, output_size) from stored weights"""
        layer_indices = sorted({int(k.split('_')[1]) for k in self.weights.keys() if k.startswith('layer_') and k.endswith('_weights')})
        hidden_sizes: List[int] = []
        input_size = None
        output_size = None
        
        for i in layer_indices:
            w = self.weights[f"layer_{i}_weights"]
            fan_in, fan_out = int(w.shape[0]), int(w.shape[1])
            if i == 0:
                input_size = fan_in
            hidden_sizes.append(fan_out)
        
        output_size = hidden_sizes[-1]
        hidden_sizes = hidden_sizes[:-1]
        
        assert input_size is not None and output_size is not None and len(hidden_sizes) >= 0
        return input_size, hidden_sizes, output_size

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray, hidden_sizes: List[int]) -> Tuple[float, float]:
        """Evaluate current KeyDB weights on validation dataset"""
        if not self.load_weights():
            return float('inf'), 0.0
        
        inferred_input, inferred_hidden, inferred_output = self._infer_sizes_from_weights()
        self._ensure_model(inferred_input, inferred_hidden or hidden_sizes, inferred_output, dropout_p=0.0)
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
        valid = (y_idx >= 0) & (y_idx < inferred_output)
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
        return float(loss_t.item()), float(acc)
    
    def train_single_step(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                         learning_rate: float = 0.001,
                         hidden_sizes: List[int] = [1024, 512, 256, 128],
                         dropout_p: float = 0.1) -> Tuple[float, float]:
        """Perform single training step with shared memory pattern"""
        # Load weights from KeyDB
        if not self.load_weights():
            return float('inf'), 0.0
        
        # Build/Load model
        input_size = X_batch.shape[1]
        output_size = 2  # UNSW-NB15 binary classification
        self._ensure_model(input_size, hidden_sizes, output_size, dropout_p=dropout_p)
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
        valid_idx = (y_indices >= 0) & (y_indices < output_size)
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

        # Extract and save weights
        self._extract_model_weights()
        if not self.save_weights():
            print(f"[{self.server_id}] ❌ Failed to save weights")
            return loss, accuracy
        
        # Reload weights (shared memory experiment)
        if not self.load_weights():
            print(f"[{self.server_id}] ❌ Failed to reload weights after update")
            return loss, accuracy
        
        return loss, accuracy
    
    def create_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        """Create mini-batches for training"""
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]
    
    def train_distributed(self, 
                         target_loss: float = 0.1,
                         max_iterations: int = 10000,
                         batch_size: int = 32,
                         learning_rate: float = 0.001,
                         print_every: int = 50,
                         hidden_sizes: List[int] = [1024, 512, 256, 128],
                         dropout_p: float = 0.1,
                         eval_every: int = 50):
        """Train on local UNSW shard data until target loss is reached"""
        print(f"[{self.server_id}] 🚀 Starting UNSW-NB15 distributed training on local shard...")
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
        
        # Load local UNSW shard data
        train_file = "../unsw_data/train_shard.parquet"
        test_file = "../unsw_data/test_shard.parquet"
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"[{self.server_id}] ❌ UNSW shard data not found! Expected: {train_file}, {test_file}")
            self.update_training_status('failed', 0, float('inf'), 0.0)
            return False
        
        X_train, y_train = self.load_unsw_parquet_clean(train_file)
        X_test, y_test = self.load_unsw_parquet_clean(test_file)
        
        print(f"[{self.server_id}] ✅ UNSW local shard loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        iteration = 0
        epoch = 0
        start_time = time.time()
        
        while iteration < max_iterations:
            epoch += 1
            epoch_losses = []
            epoch_accuracies = []
            
            for X_batch, y_batch in self.create_batches(X_train, y_train, batch_size):
                batch_loss, batch_acc = self.train_single_step(X_batch, y_batch, learning_rate, hidden_sizes, dropout_p)
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
                        val_loss, val_acc = self.evaluate(X_test, y_test, hidden_sizes)
                        
                        # Update status with validation results
                        self.update_training_status('running', iteration, val_loss, val_acc)
                        
                        print(f"[{self.server_id}] Iter {iteration:6d} | TrainLoss: {avg_loss:.6f} | TrainAcc: {avg_acc:.4f} | ValLoss: {val_loss:.6f} | ValAcc: {val_acc:.4f} | Time: {elapsed_time:.1f}s")
                        
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
                        print(f"[{self.server_id}] Iter {iteration:6d} | Loss: {avg_loss:.6f} | Accuracy: {avg_acc:.4f} | Time: {elapsed_time:.1f}s")
                
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
    
    def load_unsw_parquet_clean(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load cleaned UNSW Parquet shard data"""
        print(f"[{self.server_id}] Loading UNSW shard: {file_path}")
        df = pd.read_parquet(file_path, engine="pyarrow")
        
        print(f"[{self.server_id}] Original data shape: {df.shape}")
        print(f"[{self.server_id}] Original columns: {list(df.columns)}")
        
        # Use 'label' column for binary classification (0=normal, 1=attack)
        label_col = 'label'
        if label_col not in df.columns:
            print(f"[{self.server_id}] Available columns: {list(df.columns)}")
            raise ValueError(f"Could not find '{label_col}' column in UNSW data")
        
        print(f"[{self.server_id}] Using label column: '{label_col}' for binary classification")
        
        # Extract labels (0=normal, 1=attack)
        y = df[label_col].astype('int64').to_numpy()
        
        # Remove label column and 'id' column (not useful for training)
        feature_df = df.drop(columns=[label_col, 'id'])
        
        print(f"[{self.server_id}] Features after removing label and id: {feature_df.shape[1]} columns")
        
        # Convert categorical columns to numeric using label encoding
        categorical_columns = ['proto', 'service', 'state', 'attack_cat']
        for col in categorical_columns:
            if col in feature_df.columns:
                print(f"[{self.server_id}] Converting categorical column '{col}' to numeric")
                # Use pandas factorize to convert strings to numeric codes
                feature_df[col] = pd.factorize(feature_df[col])[0]
            else:
                print(f"[{self.server_id}] Warning: Expected categorical column '{col}' not found")
        
        # Convert all remaining columns to float32
        try:
            X = feature_df.astype('float32').to_numpy()
        except Exception as e:
            print(f"[{self.server_id}] Error converting to float32: {e}")
            print(f"[{self.server_id}] Data types: {feature_df.dtypes}")
            # Handle any remaining issues
            for col in feature_df.columns:
                try:
                    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                except:
                    print(f"[{self.server_id}] Could not convert column '{col}', using factorize")
                    feature_df[col] = pd.factorize(feature_df[col])[0]
            
            # Fill any NaN values with 0
            feature_df = feature_df.fillna(0)
            X = feature_df.astype('float32').to_numpy()
        
        print(f"[{self.server_id}] UNSW shard loaded: {X.shape[0]} samples, {X.shape[1]} features | labels: {np.unique(y)}")
        print(f"[{self.server_id}] Label distribution: {np.bincount(y)}")
        return X, y

def main():
    """Main function for UNSW-NB15 distributed training - run on each server"""
    
    # Get server ID from environment or use hostname
    import socket
    server_id = os.environ.get('SERVER_ID', socket.gethostname())
    
    print(f"🚀 Starting UNSW-NB15 distributed MLP trainer on server: {server_id}")
    
    # Initialize trainer
    trainer = DistributedUNSWMLPTrainer(server_id=server_id)
    
    # Check if weights exist in KeyDB
    if not trainer.load_weights():
        print(f"[{server_id}] ❌ No UNSW weights found. Please run the UNSW initializer first!")
        return
    
    print(f"[{server_id}] ✅ UNSW weights loaded successfully from KeyDB")
    
    # Start distributed training
    success = trainer.train_distributed(
        target_loss=0.1,
        max_iterations=5000,
        batch_size=256,
        learning_rate=0.0001,
        print_every=50,
        hidden_sizes=[1024, 512, 256, 128],
        eval_every=50
    )
    
    if success:
        print(f"[{server_id}] ✅ UNSW-NB15 distributed training completed successfully!")
    else:
        print(f"[{server_id}] ⚠️  UNSW-NB15 distributed training stopped without reaching target loss")

if __name__ == "__main__":
    main()
