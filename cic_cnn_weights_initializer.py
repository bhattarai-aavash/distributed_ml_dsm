import numpy as np
import redis
import pickle
import json


class CICCNNWeightManager:
    """Manages CNN weights in KeyDB for CIC-IDS2017 distributed training."""

    def __init__(self, keydb_host: str = 'localhost', keydb_port: int = 6379, keydb_db: int = 0):
        self.redis_client = redis.Redis(host=keydb_host, port=keydb_port, db=keydb_db, decode_responses=False)

    def _xavier_init_conv1d(self, in_channels: int, out_channels: int, kernel_size: int) -> np.ndarray:
        fan_in = in_channels * kernel_size
        fan_out = out_channels * kernel_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(out_channels, in_channels, kernel_size)).astype(np.float32)

    def _xavier_init_linear(self, fan_in: int, fan_out: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

    def create_cic_cnn_weights(self, input_features: int = 78, num_classes: int = 2) -> dict:
        """
        Create a 1D CNN sized to be comparable (~0.75M params) to CIC MLP (~0.77M params).

        Architecture (1D):
          - Conv1: 1 -> 32, k=5, pool/2
          - Conv2: 32 -> 64, k=3, pool/2
          - Conv3: 64 -> 128, k=3, pool/2
          - Conv4: 128 -> 128, k=3, pool/2
          - Flatten (L_out ~= floor(78/2/2/2/2) = 4) => 128*4=512
          - FC: 512 -> 896 -> 256 -> 2
        """
        weights = {}

        # Convs
        weights["conv1_weights"] = self._xavier_init_conv1d(1, 32, 5)
        weights["conv1_bias"] = np.zeros(32, dtype=np.float32)

        weights["conv2_weights"] = self._xavier_init_conv1d(32, 64, 3)
        weights["conv2_bias"] = np.zeros(64, dtype=np.float32)

        weights["conv3_weights"] = self._xavier_init_conv1d(64, 128, 3)
        weights["conv3_bias"] = np.zeros(128, dtype=np.float32)

        weights["conv4_weights"] = self._xavier_init_conv1d(128, 128, 3)
        weights["conv4_bias"] = np.zeros(128, dtype=np.float32)

        # Fully connected
        fc_input = 512  # 128 channels * 4 length
        weights["fc1_weights"] = self._xavier_init_linear(fc_input, 896)
        weights["fc1_bias"] = np.zeros(896, dtype=np.float32)

        weights["fc2_weights"] = self._xavier_init_linear(896, 256)
        weights["fc2_bias"] = np.zeros(256, dtype=np.float32)

        weights["fc3_weights"] = self._xavier_init_linear(256, num_classes)
        weights["fc3_bias"] = np.zeros(num_classes, dtype=np.float32)

        return weights

    def store_weights_in_keydb(self, weights: dict, experiment_name: str = "cic_cnn") -> bool:
        try:
            for key, w in weights.items():
                self.redis_client.set(f"{experiment_name}:{key}", pickle.dumps(w))
            metadata = {
                'layer_keys': list(weights.keys()),
                'shapes': {k: v.shape for k, v in weights.items()},
                'total_parameters': int(sum(w.size for w in weights.values())),
                'architecture': {
                    'model_type': 'cic_cnn',
                    'input_features': 78,
                    'conv_channels': [32, 64, 128, 128],
                    'fc_sizes': [896, 256, 2]
                }
            }
            self.redis_client.set(f"{experiment_name}:metadata", json.dumps(metadata, default=str))
            return True
        except Exception as e:
            print(f"Error storing CIC CNN weights: {e}")
            return False


def main():
    mgr = CICCNNWeightManager()
    weights = mgr.create_cic_cnn_weights(input_features=78, num_classes=2)
    total_params = sum(w.size for w in weights.values())
    print(f"Total parameters (CIC CNN): {total_params:,}")
    ok = mgr.store_weights_in_keydb(weights, experiment_name="cic_cnn")
    if ok:
        print("✅ CIC CNN weights stored in KeyDB under experiment 'cic_cnn'")
    else:
        print("❌ Failed to store CIC CNN weights")


if __name__ == "__main__":
    main()


