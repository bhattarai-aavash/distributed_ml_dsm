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


class DistributedCICCNNTrainer:
    def __init__(self, keydb_host='localhost', keydb_port=6379, keydb_db=0, server_id='unknown'):
        self.redis_client = redis.Redis(host=keydb_host, port=keydb_port, db=keydb_db, decode_responses=False)
        self.weights = {}
        self.device = 'cpu'
        self.model: Optional[nn.Module] = None
        self.experiment_name = "cic_cnn"
        self.server_id = server_id

        os.makedirs('logs', exist_ok=True)
        self.log_file = f'logs/cic_cnn_training_log_{server_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
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

    class _CICCNN(nn.Module):
        def __init__(self, input_features: int = 78, num_classes: int = 2, dropout_p: float = 0.1):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # pool/2 each stage
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout_p)

            # After 4 pool/2: L_out = floor(input_features / 16) ~ 4
            fc_input = 128 * max(1, input_features // 16)
            self.fc1 = nn.Linear(fc_input, 896)
            self.fc2 = nn.Linear(896, 256)
            self.fc3 = nn.Linear(256, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = self.pool(self.relu(self.conv4(x)))
            x = x.view(x.size(0), -1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.fc3(x)
            return x

    def _ensure_model(self, input_features: int, num_classes: int = 2, dropout_p: float = 0.1):
        self.model = self._CICCNN(input_features, num_classes, dropout_p=dropout_p).to(self.device)

    def _load_weights_into_model(self):
        assert self.model is not None
        # convs
        if 'conv1_weights' in self.weights:
            self.model.conv1.weight.data = torch.from_numpy(self.weights['conv1_weights']).to(self.device)
            self.model.conv1.bias.data = torch.from_numpy(self.weights['conv1_bias']).to(self.device)
        if 'conv2_weights' in self.weights:
            self.model.conv2.weight.data = torch.from_numpy(self.weights['conv2_weights']).to(self.device)
            self.model.conv2.bias.data = torch.from_numpy(self.weights['conv2_bias']).to(self.device)
        if 'conv3_weights' in self.weights:
            self.model.conv3.weight.data = torch.from_numpy(self.weights['conv3_weights']).to(self.device)
            self.model.conv3.bias.data = torch.from_numpy(self.weights['conv3_bias']).to(self.device)
        if 'conv4_weights' in self.weights:
            self.model.conv4.weight.data = torch.from_numpy(self.weights['conv4_weights']).to(self.device)
            self.model.conv4.bias.data = torch.from_numpy(self.weights['conv4_bias']).to(self.device)
        # fcs
        if 'fc1_weights' in self.weights:
            self.model.fc1.weight.data = torch.from_numpy(self.weights['fc1_weights'].T).to(self.device)
            self.model.fc1.bias.data = torch.from_numpy(self.weights['fc1_bias']).to(self.device)
        if 'fc2_weights' in self.weights:
            self.model.fc2.weight.data = torch.from_numpy(self.weights['fc2_weights'].T).to(self.device)
            self.model.fc2.bias.data = torch.from_numpy(self.weights['fc2_bias']).to(self.device)
        if 'fc3_weights' in self.weights:
            self.model.fc3.weight.data = torch.from_numpy(self.weights['fc3_weights'].T).to(self.device)
            self.model.fc3.bias.data = torch.from_numpy(self.weights['fc3_bias']).to(self.device)

    def _extract_model_weights(self):
        assert self.model is not None
        extracted: Dict[str, np.ndarray] = {}
        extracted['conv1_weights'] = self.model.conv1.weight.detach().to('cpu').numpy()
        extracted['conv1_bias'] = self.model.conv1.bias.detach().to('cpu').numpy()
        extracted['conv2_weights'] = self.model.conv2.weight.detach().to('cpu').numpy()
        extracted['conv2_bias'] = self.model.conv2.bias.detach().to('cpu').numpy()
        extracted['conv3_weights'] = self.model.conv3.weight.detach().to('cpu').numpy()
        extracted['conv3_bias'] = self.model.conv3.bias.detach().to('cpu').numpy()
        extracted['conv4_weights'] = self.model.conv4.weight.detach().to('cpu').numpy()
        extracted['conv4_bias'] = self.model.conv4.bias.detach().to('cpu').numpy()
        extracted['fc1_weights'] = self.model.fc1.weight.detach().to('cpu').numpy().T
        extracted['fc1_bias'] = self.model.fc1.bias.detach().to('cpu').numpy()
        extracted['fc2_weights'] = self.model.fc2.weight.detach().to('cpu').numpy().T
        extracted['fc2_bias'] = self.model.fc2.bias.detach().to('cpu').numpy()
        extracted['fc3_weights'] = self.model.fc3.weight.detach().to('cpu').numpy().T
        extracted['fc3_bias'] = self.model.fc3.bias.detach().to('cpu').numpy()
        self.weights = extracted

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
        if not self.load_weights():
            return float('inf'), 0.0, 0.0, 0.0, 0.0
        self._ensure_model(input_features=X.shape[1], num_classes=2, dropout_p=0.0)
        self._load_weights_into_model()
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - mean) / std
        X = np.clip(X, -10.0, 10.0)

        y_idx = y.astype(np.int64)
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
        
        # Calculate additional metrics
        y_true = y_t.cpu().numpy()
        y_pred = preds.cpu().numpy()
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return float(loss_t.item()), float(acc), float(f1), float(precision), float(recall)
    
    @torch.no_grad()
    def calculate_detailed_metrics(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Calculate detailed metrics on test set"""
        if not self.load_weights():
            return float('inf'), 0.0, 0.0, 0.0, 0.0
        self._ensure_model(input_features=X.shape[1], num_classes=2, dropout_p=0.0)
        self._load_weights_into_model()
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - mean) / std
        X = np.clip(X, -10.0, 10.0)

        y_idx = y.astype(np.int64)
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
        
        # Calculate additional metrics
        y_true = y_t.cpu().numpy()
        y_pred = preds.cpu().numpy()
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return float(loss_t.item()), float(acc), float(f1), float(precision), float(recall)

    def train_single_step(self, X_batch: np.ndarray, y_batch: np.ndarray,
                          learning_rate: float = 0.001,
                          dropout_p: float = 0.1) -> Tuple[float, float]:
        if not self.load_weights():
            return float('inf'), 0.0
        input_features = X_batch.shape[1]
        self._ensure_model(input_features, num_classes=2, dropout_p=dropout_p)
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

        y_indices = y_batch.astype(np.int64)
        valid_idx = (y_indices >= 0) & (y_indices < 2)
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

    def load_cic_parquet_clean(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_parquet(file_path, engine="pyarrow")
        if 'label' not in df.columns:
            raise ValueError("Parquet shard must contain a 'label' column (0/1)")
        y = df['label'].astype('int64').to_numpy()
        feature_df = df.drop(columns=['label'])
        for col in feature_df.columns:
            if not np.issubdtype(feature_df[col].dtype, np.number):
                feature_df[col] = pd.factorize(feature_df[col])[0]
        X = feature_df.astype('float32').to_numpy()
        return X, y

    def train_distributed(self,
                          target_loss: float = 0.1,
                          max_iterations: int = 5000,
                          batch_size: int = 256,
                          learning_rate: float = 0.0001,
                          print_every: int = 50,
                          dropout_p: float = 0.1,
                          eval_every: int = 50):
        print(f"[{self.server_id}] 🚀 Starting CIC CNN distributed training on local shard...")
        self.start_time = datetime.now().isoformat()
        train_file = os.environ.get("CIC_TRAIN_FILE", "../cic_shard/cic_train.parquet")
        test_file = os.environ.get("CIC_TEST_FILE", "../cic_shard/cic_test.parquet")
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"[{self.server_id}] ❌ CIC shard data not found! Expected: {train_file}, {test_file}")
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
                batch_loss, batch_acc = self.train_single_step(X_batch, y_batch, learning_rate, dropout_p)
                epoch_losses.append(batch_loss)
                epoch_accuracies.append(batch_acc)
                iteration += 1
                if iteration == 1 or (iteration % print_every == 0):
                    avg_loss = np.mean(epoch_losses[-print_every:]) if len(epoch_losses) >= print_every else np.mean(epoch_losses)
                    avg_acc = np.mean(epoch_accuracies[-print_every:]) if len(epoch_accuracies) >= print_every else np.mean(epoch_accuracies)
                    elapsed_time = time.time() - start_time
                    if iteration % eval_every == 0 or iteration == 1:
                        val_loss, val_acc, val_f1, val_precision, val_recall = self.evaluate(X_test, y_test)
                        print(f"[{self.server_id}] Iter {iteration:6d} | TrainLoss: {avg_loss:.6f} | TrainAcc: {avg_acc:.4f} | ValLoss: {val_loss:.6f} | ValAcc: {val_acc:.4f} | ValF1: {val_f1:.4f} | ValPrec: {val_precision:.4f} | ValRec: {val_recall:.4f} | Time: {elapsed_time:.1f}s")
                        self.log_training_step(iteration, epoch, batch_loss, batch_acc, avg_loss, avg_acc, val_loss, val_acc, val_f1, val_precision, val_recall, learning_rate, elapsed_time)
                        if val_loss <= target_loss:
                            print(f"[{self.server_id}] 🎉 Target loss reached!")
                            print(f"[{self.server_id}] Final iteration: {iteration}")
                            print(f"[{self.server_id}] Final val loss: {val_loss:.6f}")
                            print(f"[{self.server_id}] Final val F1: {val_f1:.4f}")
                            print(f"[{self.server_id}] Final val precision: {val_precision:.4f}")
                            print(f"[{self.server_id}] Final val recall: {val_recall:.4f}")
                            print(f"[{self.server_id}] Total training time: {elapsed_time:.2f} seconds")
                            return True
                if iteration >= max_iterations:
                    break
            if iteration >= max_iterations:
                break
        
        # Calculate detailed metrics on test set
        print(f"[{self.server_id}] 📊 Calculating detailed metrics on test set...")
        test_loss, test_acc, test_f1, test_precision, test_recall = self.calculate_detailed_metrics(X_test, y_test)
        
        # Maximum iterations reached
        elapsed_time = time.time() - start_time
        
        print(f"[{self.server_id}] ⚠️  Maximum iterations reached without reaching target loss")
        print(f"[{self.server_id}] 📊 Final Test Set Metrics:")
        print(f"[{self.server_id}]   Test Loss: {test_loss:.6f}")
        print(f"[{self.server_id}]   Test Accuracy: {test_acc:.4f}")
        print(f"[{self.server_id}]   Test F1-Score: {test_f1:.4f}")
        print(f"[{self.server_id}]   Test Precision: {test_precision:.4f}")
        print(f"[{self.server_id}]   Test Recall: {test_recall:.4f}")
        print(f"[{self.server_id}] Total training time: {elapsed_time:.2f} seconds")
        return False


def main():
    import socket
    server_id = os.environ.get('SERVER_ID', socket.gethostname())
    print(f"🚀 Starting CIC CNN distributed trainer on server: {server_id}")
    trainer = DistributedCICCNNTrainer(server_id=server_id)
    if not trainer.load_weights():
        print(f"[{server_id}] ❌ No CIC CNN weights found. Please run the CNN initializer first!")
        return
    print(f"[{server_id}] ✅ CIC CNN weights loaded successfully from KeyDB")
    trainer.train_distributed()


if __name__ == "__main__":
    main()


