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
from sklearn.metrics import f1_score, precision_score, recall_score


class DistributedCICMLPTrainer:
    """Distributed MLP trainer for CIC-IDS2017 using KeyDB for shared weight storage across servers"""

    def __init__(self, keydb_host='localhost', keydb_port=6379, keydb_db=0, server_id='unknown'):
        self.redis_client = redis.Redis(host=keydb_host, port=keydb_port, db=keydb_db, decode_responses=False)
        self.weights: Dict[str, np.ndarray] = {}
        self.device = 'cpu'
        self.model: Optional[nn.Module] = None
        self.experiment_name = "cic_mlp"
        self.server_id = server_id

        os.makedirs('logs', exist_ok=True)
        self.log_file = f'logs/cic_training_log_{server_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self._init_log_file()

    def _init_log_file(self):
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
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), self.server_id, iteration, epoch,
                batch_loss, batch_acc, train_loss, train_acc, val_loss, val_acc,
                val_f1, val_precision, val_recall, learning_rate, time_elapsed
            ])

    def update_training_status(self, status: str, iteration: int = 0, loss: float = 0.0, accuracy: float = 0.0) -> bool:
        try:
            status_data = {
                'server_id': self.server_id,
                'status': status,
                'iteration': iteration,
                'current_loss': loss,
                'current_accuracy': accuracy,
                'last_update': datetime.now().isoformat(),
                'start_time': getattr(self, 'start_time', datetime.now().isoformat())
            }
            self.redis_client.set(f"{self.experiment_name}:training_status:{self.server_id}", json.dumps(status_data))
            return True
        except Exception as e:
            print(f"[{self.server_id}] ❌ Error updating status: {e}")
            return False

    def load_weights(self) -> bool:
        try:
            metadata_bytes = self.redis_client.get(f"{self.experiment_name}:metadata")
            if metadata_bytes is None:
                print(f"[{self.server_id}] ❌ No weights found in KeyDB. Run initializer first.")
                return False
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            layer_keys = metadata['layer_keys']
            self.weights = {}
            for key in layer_keys:
                wb = self.redis_client.get(f"{self.experiment_name}:{key}")
                if wb is None:
                    print(f"[{self.server_id}] ❌ Missing weight: {key}")
                    return False
                self.weights[key] = pickle.loads(wb)
            return True
        except Exception as e:
            print(f"[{self.server_id}] ❌ Error loading weights: {e}")
            return False

    def save_weights(self) -> bool:
        try:
            for key, w in self.weights.items():
                self.redis_client.set(f"{self.experiment_name}:{key}", pickle.dumps(w))
            return True
        except Exception as e:
            print(f"[{self.server_id}] ❌ Error saving weights: {e}")
            return False

    class _MLP(nn.Module):
        def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_p: float = 0.1):
            super().__init__()
            layers: List[nn.Module] = []
            sizes = [input_size] + hidden_sizes
            for i in range(len(sizes) - 1):
                layers.append(nn.Linear(sizes[i], sizes[i+1]))
                layers.append(nn.ReLU())
                if dropout_p > 0:
                    layers.append(nn.Dropout(p=dropout_p))
            layers.append(nn.Linear(sizes[-1], output_size))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    def _ensure_model(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_p: float = 0.1):
        if self.model is None:
            self.model = self._MLP(input_size, hidden_sizes, output_size, dropout_p=dropout_p).to(self.device)

    def _infer_sizes_from_weights(self) -> Tuple[int, List[int], int]:
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
        assert input_size is not None and output_size is not None
        return input_size, hidden_sizes, output_size

    def _load_weights_into_model(self):
        assert self.model is not None
        num_layers = len([k for k in self.weights.keys() if 'weights' in k])
        linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        if len(linear_layers) != num_layers:
            inferred_input, inferred_hidden, inferred_output = self._infer_sizes_from_weights()
            self.model = self._MLP(inferred_input, inferred_hidden, inferred_output, dropout_p=0.1).to(self.device)
            linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        with torch.no_grad():
            for i, lin in enumerate(linear_layers):
                w_np = self.weights[f"layer_{i}_weights"]
                b_np = self.weights[f"layer_{i}_bias"]
                lin.weight.copy_(torch.from_numpy(w_np.T.copy()).to(self.device))
                lin.bias.copy_(torch.from_numpy(b_np.copy()).to(self.device))

    def _extract_model_weights(self):
        assert self.model is not None
        extracted: Dict[str, np.ndarray] = {}
        linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        for i, lin in enumerate(linear_layers):
            w = lin.weight.detach().to('cpu').numpy().T
            b = lin.bias.detach().to('cpu').numpy()
            extracted[f"layer_{i}_weights"] = w.astype(np.float32)
            extracted[f"layer_{i}_bias"] = b.astype(np.float32)
        self.weights = extracted

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray, hidden_sizes: List[int]) -> Tuple[float, float, float, float, float]:
        if not self.load_weights():
            return float('inf'), 0.0, 0.0, 0.0, 0.0
        inferred_input, inferred_hidden, inferred_output = self._infer_sizes_from_weights()
        self._ensure_model(inferred_input, inferred_hidden or hidden_sizes, inferred_output, dropout_p=0.0)
        self._load_weights_into_model()
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

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
        
        # Calculate additional metrics
        y_true = y_t.cpu().numpy()
        y_pred = preds.cpu().numpy()
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return float(loss_t.item()), float(acc), float(f1), float(precision), float(recall)
    
    @torch.no_grad()
    def calculate_detailed_metrics(self, X: np.ndarray, y: np.ndarray, hidden_sizes: List[int]) -> Tuple[float, float, float, float, float]:
        """Calculate detailed metrics on test set"""
        if not self.load_weights():
            return float('inf'), 0.0, 0.0, 0.0, 0.0
        inferred_input, inferred_hidden, inferred_output = self._infer_sizes_from_weights()
        self._ensure_model(inferred_input, inferred_hidden or hidden_sizes, inferred_output, dropout_p=0.0)
        self._load_weights_into_model()
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

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
        
        # Calculate additional metrics
        y_true = y_t.cpu().numpy()
        y_pred = preds.cpu().numpy()
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return float(loss_t.item()), float(acc), float(f1), float(precision), float(recall)

    def train_single_step(self, X_batch: np.ndarray, y_batch: np.ndarray,
                          learning_rate: float = 0.001,
                          hidden_sizes: List[int] = [1024, 512, 256, 128],
                          dropout_p: float = 0.1) -> Tuple[float, float]:
        if not self.load_weights():
            return float('inf'), 0.0

        input_size = X_batch.shape[1]
        output_size = 2
        self._ensure_model(input_size, hidden_sizes, output_size, dropout_p=dropout_p)
        self._load_weights_into_model()

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=min(learning_rate, 1e-3), weight_decay=1e-4)
        max_grad_norm = 5.0
        criterion = nn.CrossEntropyLoss()

        X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)
        mean = X_batch.mean(axis=0, keepdims=True)
        std = X_batch.std(axis=0, keepdims=True) + 1e-6
        X_norm = (X_batch - mean) / std
        X_norm = np.clip(X_norm, -10.0, 10.0)
        X_t = torch.from_numpy(X_norm).to(self.device)

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

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_t).float().mean().item()
            loss = float(loss_t.item()) if torch.isfinite(loss_t) else float('inf')

        self._extract_model_weights()
        if not self.save_weights():
            print(f"[{self.server_id}] ❌ Failed to save weights")
            return loss, accuracy
        if not self.load_weights():
            print(f"[{self.server_id}] ❌ Failed to reload weights after update")
            return loss, accuracy

        return loss, accuracy

    def create_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]

    def train_distributed(self,
                          target_loss: float = 0.1,
                          max_iterations: int = 1000,
                          batch_size: int = 256,
                          learning_rate: float = 0.0001,
                          print_every: int = 50,
                          hidden_sizes: List[int] = [1024, 512, 256, 128],
                          dropout_p: float = 0.1,
                          eval_every: int = 50):
        print(f"[{self.server_id}] 🚀 Starting CIC-IDS2017 distributed training on local shard...")
        self.start_time = datetime.now().isoformat()
        self.update_training_status('running', 0, 0.0, 0.0)

        # Allow overriding shard paths via environment; default to filenames copied by script.sh
        train_file = os.environ.get("CIC_TRAIN_FILE", "../cic_shard/cic_train.parquet")
        test_file = os.environ.get("CIC_TEST_FILE", "../cic_shard/cic_test.parquet")
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"[{self.server_id}] ❌ CIC shard data not found! Expected: {train_file}, {test_file}")
            self.update_training_status('failed', 0, float('inf'), 0.0)
            return False

        X_train, y_train = self.load_cic_parquet_clean(train_file)
        X_test, y_test = self.load_cic_parquet_clean(test_file)

        iteration = 0
        epoch = 0
        start_time = time.time()

        while iteration < max_iterations:
            epoch += 1
            epoch_losses: List[float] = []
            epoch_accuracies: List[float] = []

            for X_batch, y_batch in self.create_batches(X_train, y_train, batch_size):
                batch_loss, batch_acc = self.train_single_step(X_batch, y_batch, learning_rate, hidden_sizes, dropout_p)
                epoch_losses.append(batch_loss)
                epoch_accuracies.append(batch_acc)
                iteration += 1

                if iteration == 1 or (iteration % print_every == 0):
                    avg_loss = np.mean(epoch_losses[-print_every:]) if len(epoch_losses) >= print_every else np.mean(epoch_losses)
                    avg_acc = np.mean(epoch_accuracies[-print_every:]) if len(epoch_accuracies) >= print_every else np.mean(epoch_accuracies)
                    elapsed_time = time.time() - start_time
                    self.update_training_status('running', iteration, avg_loss, avg_acc)

                    if iteration % eval_every == 0 or iteration == 1:
                        val_loss, val_acc, val_f1, val_precision, val_recall = self.evaluate(X_test, y_test, hidden_sizes)
                        self.update_training_status('running', iteration, val_loss, val_acc)
                        print(f"[{self.server_id}] Iter {iteration:6d} | TrainLoss: {avg_loss:.6f} | TrainAcc: {avg_acc:.4f} | ValLoss: {val_loss:.6f} | ValAcc: {val_acc:.4f} | ValF1: {val_f1:.4f} | ValPrec: {val_precision:.4f} | ValRec: {val_recall:.4f} | Time: {elapsed_time:.1f}s")
                        self.log_training_step(iteration, epoch, batch_loss, batch_acc, avg_loss, avg_acc, val_loss, val_acc, val_f1, val_precision, val_recall, learning_rate, elapsed_time)
                        if val_loss <= target_loss:
                            self.update_training_status('completed', iteration, val_loss, val_acc)
                            print(f"[{self.server_id}] 🎉 Target loss reached!")
                            print(f"[{self.server_id}] Final iteration: {iteration}")
                            print(f"[{self.server_id}] Final val loss: {val_loss:.6f}")
                            print(f"[{self.server_id}] Final val F1: {val_f1:.4f}")
                            print(f"[{self.server_id}] Final val precision: {val_precision:.4f}")
                            print(f"[{self.server_id}] Final val recall: {val_recall:.4f}")
                            print(f"[{self.server_id}] Total training time: {elapsed_time:.2f} seconds")
                            return True
                    else:
                        print(f"[{self.server_id}] Iter {iteration:6d} | Loss: {avg_loss:.6f} | Accuracy: {avg_acc:.4f} | Time: {elapsed_time:.1f}s")

                if iteration >= max_iterations:
                    break
            if iteration >= max_iterations:
                break

        final_loss = np.mean(epoch_losses[-10:]) if epoch_losses else float('inf')
        final_acc = np.mean(epoch_accuracies[-10:]) if epoch_accuracies else 0.0
        self.update_training_status('completed', iteration, final_loss, final_acc)
        
        # Calculate detailed metrics on test set
        print(f"[{self.server_id}] 📊 Calculating detailed metrics on test set...")
        test_loss, test_acc, test_f1, test_precision, test_recall = self.calculate_detailed_metrics(X_test, y_test, hidden_sizes)
        
        # Maximum iterations reached
        elapsed_time = time.time() - start_time
        
        print(f"[{self.server_id}] ⚠️  Maximum iterations reached without reaching target loss")
        print(f"[{self.server_id}] Final training loss: {final_loss:.6f}")
        print(f"[{self.server_id}] Final training accuracy: {final_acc:.4f}")
        print(f"[{self.server_id}] 📊 Final Test Set Metrics:")
        print(f"[{self.server_id}]   Test Loss: {test_loss:.6f}")
        print(f"[{self.server_id}]   Test Accuracy: {test_acc:.4f}")
        print(f"[{self.server_id}]   Test F1-Score: {test_f1:.4f}")
        print(f"[{self.server_id}]   Test Precision: {test_precision:.4f}")
        print(f"[{self.server_id}]   Test Recall: {test_recall:.4f}")
        print(f"[{self.server_id}] Total training time: {elapsed_time:.2f} seconds")
        return False

    def load_cic_parquet_clean(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        print(f"[{self.server_id}] Loading CIC shard: {file_path}")
        df = pd.read_parquet(file_path, engine="pyarrow")

        if 'label' not in df.columns:
            raise ValueError("Parquet shard must contain a 'label' column (0/1)")

        y = df['label'].astype('int64').to_numpy()
        feature_df = df.drop(columns=['label'])

        # Convert non-numeric columns
        for col in feature_df.columns:
            if not np.issubdtype(feature_df[col].dtype, np.number):
                feature_df[col] = pd.factorize(feature_df[col])[0]

        X = feature_df.astype('float32').to_numpy()
        print(f"[{self.server_id}] CIC shard loaded: {X.shape[0]} samples, {X.shape[1]} features | labels: {np.unique(y)}")
        return X, y


def main():
    import socket
    server_id = os.environ.get('SERVER_ID', socket.gethostname())
    print(f"🚀 Starting CIC-IDS2017 distributed MLP trainer on server: {server_id}")
    trainer = DistributedCICMLPTrainer(server_id=server_id)
    if not trainer.load_weights():
        print(f"[{server_id}] ❌ No CIC weights found. Please run the CIC initializer first!")
        return
    print(f"[{server_id}] ✅ CIC weights loaded successfully from KeyDB")
    success = trainer.train_distributed()
    if success:
        print(f"[{server_id}] ✅ CIC-IDS2017 distributed training completed successfully!")
    else:
        print(f"[{server_id}] ⚠️  CIC-IDS2017 distributed training stopped without reaching target loss")


if __name__ == "__main__":
    main()


