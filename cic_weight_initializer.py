import numpy as np
import redis
import pickle
import json
from typing import List


class CICMLPWeightManager:
    """Manages MLP weights in KeyDB for CIC-IDS2017 distributed training"""

    def __init__(self, keydb_host: str = 'localhost', keydb_port: int = 6379, keydb_db: int = 0):
        self.redis_client = redis.Redis(
            host=keydb_host,
            port=keydb_port,
            db=keydb_db,
            decode_responses=False
        )

    def create_cic_mlp_weights(self, input_size: int, hidden_sizes: List[int], output_size: int) -> dict:
        """Create Xavier-initialized MLP weights for CIC-IDS2017"""
        weights = {}
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            layer_name = f"layer_{i}"
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights[f"{layer_name}_weights"] = np.random.uniform(
                -limit, limit, size=(fan_in, fan_out)
            ).astype(np.float32)
            weights[f"{layer_name}_bias"] = np.random.uniform(
                -0.1, 0.1, size=(fan_out,)
            ).astype(np.float32)

        return weights

    def store_weights_in_keydb(self, weights: dict, experiment_name: str = "cic_mlp") -> bool:
        try:
            for key, weight_matrix in weights.items():
                redis_key = f"{experiment_name}:{key}"
                self.redis_client.set(redis_key, pickle.dumps(weight_matrix))

            metadata = {
                'layer_keys': list(weights.keys()),
                'shapes': {k: v.shape for k, v in weights.items()},
                'total_parameters': int(sum(w.size for w in weights.values())),
                'architecture': {
                    'model_type': 'cic_mlp'
                }
            }
            self.redis_client.set(f"{experiment_name}:metadata", json.dumps(metadata, default=str))
            return True
        except Exception as e:
            print(f"Error storing CIC weights: {e}")
            return False


def main():
    # With current sharding, each shard has 79 columns total -> 78 features + 1 label
    INPUT_SIZE = 78
    HIDDEN_SIZES = [1024, 512, 256, 128]
    OUTPUT_SIZE = 2  # binary (0/1)

    print("Initializing CIC-IDS2017 MLP weights...")
    print(f"  Input size: {INPUT_SIZE}")
    print(f"  Hidden: {HIDDEN_SIZES}")
    print(f"  Output: {OUTPUT_SIZE}")

    mgr = CICMLPWeightManager()
    weights = mgr.create_cic_mlp_weights(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

    total_params = sum(w.size for w in weights.values())
    print(f"Total parameters: {total_params:,}")
    for k, v in weights.items():
        print(f"  {k}: {v.shape} -> {v.size:,}")

    ok = mgr.store_weights_in_keydb(weights, experiment_name="cic_mlp")
    if ok:
        print("✅ CIC-IDS2017 weights stored in KeyDB under experiment 'cic_mlp'")
    else:
        print("❌ Failed to store CIC-IDS2017 weights")


if __name__ == "__main__":
    main()


