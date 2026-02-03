import numpy as np
import redis
import pickle
import json
from typing import List, Tuple

class SimpleLSTMWeightManager:
    """Manages simple LSTM weights in KeyDB for distributed training"""
    
    def __init__(self, keydb_host='localhost', keydb_port=6379, keydb_db=0):
        """Initialize connection to KeyDB"""
        self.redis_client = redis.Redis(
            host=keydb_host, 
            port=keydb_port, 
            db=keydb_db,
            decode_responses=False  # Keep binary for pickle
        )
    
    def create_simple_lstm_weights(self, input_size: int = 1, hidden_size: int = 8, num_layers: int = 3, num_classes: int = 2) -> dict:
        """
        Create and initialize simple LSTM weights for Amber dataset (3-LAYER ULTRA-COMPACT MODEL)
        
        Args:
            input_size: Input dimension (1 for single feature per timestep)
            hidden_size: LSTM hidden state size (8 for ultra-compact model)
            num_layers: Number of LSTM layers (3 for enhanced learning)
            num_classes: Number of output classes (2 for binary classification)
            
        Returns:
            Dictionary containing weights and biases for each layer
        """
        weights = {}
        
        # LSTM weights (for 3 layers, 1 input, 8 hidden)
        # Each LSTM cell has 4 gates: input, forget, cell, output
        # Weights: (4 * hidden_size, input_size + hidden_size) for layer 0
        # Weights: (4 * hidden_size, hidden_size + hidden_size) for layers 1 & 2
        
        # Layer 0: Input to LSTM
        lstm_input_size_0 = input_size + hidden_size  # 1 + 8 = 9
        weights["lstm_weight_ih_l0"] = self._xavier_init_linear(4 * hidden_size, lstm_input_size_0)  # (32, 9)
        weights["lstm_weight_hh_l0"] = self._xavier_init_linear(4 * hidden_size, hidden_size)        # (32, 8)
        weights["lstm_bias_ih_l0"] = np.zeros(4 * hidden_size, dtype=np.float32)                     # (32,)
        weights["lstm_bias_hh_l0"] = np.zeros(4 * hidden_size, dtype=np.float32)                     # (32,)
        
        # Layer 1: LSTM to LSTM
        lstm_input_size_1 = hidden_size + hidden_size  # 8 + 8 = 16
        weights["lstm_weight_ih_l1"] = self._xavier_init_linear(4 * hidden_size, lstm_input_size_1)  # (32, 16)
        weights["lstm_weight_hh_l1"] = self._xavier_init_linear(4 * hidden_size, hidden_size)        # (32, 8)
        weights["lstm_bias_ih_l1"] = np.zeros(4 * hidden_size, dtype=np.float32)                     # (32,)
        weights["lstm_bias_hh_l1"] = np.zeros(4 * hidden_size, dtype=np.float32)                     # (32,)
        
        # Layer 2: LSTM to LSTM
        lstm_input_size_2 = hidden_size + hidden_size  # 8 + 8 = 16
        weights["lstm_weight_ih_l2"] = self._xavier_init_linear(4 * hidden_size, lstm_input_size_2)  # (32, 16)
        weights["lstm_weight_hh_l2"] = self._xavier_init_linear(4 * hidden_size, hidden_size)        # (32, 8)
        weights["lstm_bias_ih_l2"] = np.zeros(4 * hidden_size, dtype=np.float32)                     # (32,)
        weights["lstm_bias_hh_l2"] = np.zeros(4 * hidden_size, dtype=np.float32)                     # (32,)
        
        # Tiny fully connected layers
        weights["fc1_weights"] = self._xavier_init_linear(hidden_size, 8)                      # (8, 8)
        weights["fc1_bias"] = np.zeros(8, dtype=np.float32)                                   # (8,)
        
        weights["fc2_weights"] = self._xavier_init_linear(8, num_classes)                      # (8, 2)
        weights["fc2_bias"] = np.zeros(num_classes, dtype=np.float32)                          # (2,)
        
        return weights
    
    def get_experiment_info(self, experiment_name: str = "amber_simple_lstm") -> dict:
        """Get information about the simple LSTM experiment"""
        try:
            metadata_key = f"{experiment_name}:metadata"
            metadata_bytes = self.redis_client.get(metadata_key)
            if metadata_bytes is None:
                return {"status": "No experiment found"}
            
            return json.loads(metadata_bytes)
            
        except Exception as e:
            return {"error": str(e)}

    def _xavier_init_linear(self, fan_in: int, fan_out: int) -> np.ndarray:
        """Initialize linear layer weights using Xavier/Glorot initialization"""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        return np.random.uniform(
            -limit, limit, 
            size=(fan_in, fan_out)
        ).astype(np.float32)
    
    def store_weights_in_keydb(self, weights: dict, experiment_name: str = "amber_simple_lstm"):
        """Store simple LSTM weights in KeyDB"""
        try:
            # Store individual weight matrices
            for key, weight_matrix in weights.items():
                redis_key = f"{experiment_name}:{key}"
                serialized_weights = pickle.dumps(weight_matrix)
                self.redis_client.set(redis_key, serialized_weights)
                print(f"Stored {key} with shape {weight_matrix.shape}")
            
            # Store metadata
            metadata = {
                'layer_keys': list(weights.keys()),
                'shapes': {k: v.shape for k, v in weights.items()},
                'total_parameters': sum(w.size for w in weights.values()),
                'architecture': {
                    'input_size': 1,
                    'hidden_size': 8,
                    'num_layers': 3,
                    'num_classes': 2,
                    'fc_sizes': [8, 2],
                    'model_type': '3_layer_ultra_compact_lstm'
                }
            }
            
            metadata_key = f"{experiment_name}:metadata"
            self.redis_client.set(metadata_key, json.dumps(metadata, default=str))
            print(f"Stored metadata: {metadata}")
            
            return True
            
        except Exception as e:
            print(f"Error storing weights: {e}")
            return False

def main():
    """Initialize simple LSTM for Amber dataset and store in KeyDB (ULTRA-COMPACT MODEL)"""
    
    print("Initializing Simple LSTM Weight Manager (ULTRA-COMPACT MODEL)...")
    weight_manager = SimpleLSTMWeightManager()
    
    print(f"Creating simple LSTM weights (ULTRA-COMPACT VERSION):")
    print(f"  Input: 1 feature per timestep")
    print(f"  LSTM: 3 layers, 8 hidden units (ULTRA-COMPACT)")
    print(f"  FC layers: 8 → 8 → 2 (TINY)")
    print(f"  Output size: 2 classes (binary classification)")
    
    # Create simple LSTM weights
    weights = weight_manager.create_simple_lstm_weights(
        input_size=1,
        hidden_size=8,
        num_layers=3,
        num_classes=2
    )
    
    # Calculate total parameters
    total_params = sum(w.size for w in weights.values())
    print(f"Total parameters: {total_params:,} (ULTRA-COMPACT model, minimal memory!)")
    
    # Store in KeyDB
    print("\nStoring simple LSTM weights in KeyDB...")
    success = weight_manager.store_weights_in_keydb(weights, "amber_simple_lstm")
    
    if success:
        print("✅ Simple LSTM weights (ULTRA-COMPACT) successfully stored in KeyDB!")
        
        # Show experiment info
        info = weight_manager.get_experiment_info("amber_simple_lstm")
        print("\nSimple LSTM Experiment Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Failed to store simple LSTM weights")

if __name__ == "__main__":
    main()
