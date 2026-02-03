import numpy as np
import redis
import pickle
import json
from typing import List, Tuple

class UNSWMLPWeightManager:
    """Manages MLP weights in KeyDB for UNSW-NB15 distributed training"""
    
    def __init__(self, keydb_host='localhost', keydb_port=6379, keydb_db=0):
        """Initialize connection to KeyDB"""
        self.redis_client = redis.Redis(
            host=keydb_host, 
            port=keydb_port, 
            db=keydb_db,
            decode_responses=False  # Keep binary for pickle
        )
        
    def create_unsw_mlp_weights(self, input_size: int, hidden_sizes: List[int], output_size: int) -> dict:
        """
        Create and initialize MLP weights for UNSW-NB15 dataset
        
        Args:
            input_size: Number of input features (43 for UNSW-NB15 dataset: 45 columns - label - id)
            hidden_sizes: List of hidden layer sizes, e.g., [1024, 512, 256, 128]
            output_size: Number of output classes (2 for binary classification: 0=normal, 1=attack)
            
        Returns:
            Dictionary containing weights and biases for each layer
        """
        weights = {}
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights using Xavier/Glorot initialization
        for i in range(len(layer_sizes) - 1):
            layer_name = f"layer_{i}"
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            # Xavier initialization
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            
            weights[f"{layer_name}_weights"] = np.random.uniform(
                -limit, limit, size=(fan_in, fan_out)
            ).astype(np.float32)
            
            # Initialize biases to small positive values
            weights[f"{layer_name}_bias"] = np.random.uniform(
                -0.1, 0.1, size=(fan_out,)
            ).astype(np.float32)
        
        return weights
    
    def store_weights_in_keydb(self, weights: dict, experiment_name: str = "unsw_mlp"):
        """Store weights in KeyDB"""
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
                    'input_shape': (44,),  # UNSW-NB15 has 44 features
                    'num_classes': 2,      # Binary classification
                    'hidden_layers': len([k for k in weights.keys() if 'weights' in k]) - 1,
                    'model_type': 'unsw_mlp'
                }
            }
            
            metadata_key = f"{experiment_name}:metadata"
            self.redis_client.set(metadata_key, json.dumps(metadata, default=str))
            print(f"Stored metadata: {metadata}")
            
            return True
            
        except Exception as e:
            print(f"Error storing weights: {e}")
            return False
    
    def load_weights_from_keydb(self, experiment_name: str = "unsw_mlp") -> dict:
        """Load weights from KeyDB"""
        try:
            # Load metadata first
            metadata_key = f"{experiment_name}:metadata"
            metadata_bytes = self.redis_client.get(metadata_key)
            if metadata_bytes is None:
                raise ValueError(f"No metadata found for experiment {experiment_name}")
            
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            layer_keys = metadata['layer_keys']
            
            # Load individual weight matrices
            weights = {}
            for key in layer_keys:
                redis_key = f"{experiment_name}:{key}"
                weight_bytes = self.redis_client.get(redis_key)
                if weight_bytes is None:
                    raise ValueError(f"Weight matrix {key} not found")
                weights[key] = pickle.loads(weight_bytes)
                print(f"Loaded {key} with shape {weights[key].shape}")
            
            return weights
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            return {}
    
    def get_experiment_info(self, experiment_name: str = "unsw_mlp") -> dict:
        """Get information about the experiment"""
        try:
            metadata_key = f"{experiment_name}:metadata"
            metadata_bytes = self.redis_client.get(metadata_key)
            if metadata_bytes is None:
                return {"status": "No experiment found"}
            
            return json.loads(metadata_bytes)
            
        except Exception as e:
            return {"error": str(e)}

def main():
    """Initialize MLP for UNSW-NB15 dataset and store in KeyDB"""
    
    # UNSW-NB15 dataset parameters
    INPUT_SIZE = 43   # UNSW-NB15 has 43 features (45 columns - 1 label column - 1 id column)
    HIDDEN_SIZES = [1024, 512, 256, 128]  # Same hidden architecture as Amber dataset
    OUTPUT_SIZE = 2   # Binary classification (0=normal, 1=attack)
    
    print("Initializing UNSW-NB15 MLP Weight Manager...")
    weight_manager = UNSWMLPWeightManager()
    
    print(f"Creating UNSW-NB15 MLP weights:")
    print(f"  Input size: {INPUT_SIZE} (UNSW-NB15 features: 45 columns - label - id)")
    print(f"  Hidden layers: {HIDDEN_SIZES}")
    print(f"  Output size: {OUTPUT_SIZE} (binary classification: 0=normal, 1=attack)")
    print(f"  Categorical columns: proto, service, state, attack_cat (converted to numeric)")
    
    # Create weights
    weights = weight_manager.create_unsw_mlp_weights(
        input_size=INPUT_SIZE,
        hidden_sizes=HIDDEN_SIZES,
        output_size=OUTPUT_SIZE
    )
    
    # Calculate total parameters
    total_params = sum(w.size for w in weights.values())
    print(f"Total parameters: {total_params:,}")
    
    # Show parameter breakdown
    print("\nParameter breakdown:")
    for key, weight_matrix in weights.items():
        print(f"  {key}: {weight_matrix.shape} = {weight_matrix.size:,} parameters")
    
    # Store in KeyDB
    print("\nStoring weights in KeyDB...")
    success = weight_manager.store_weights_in_keydb(weights, "unsw_mlp")
    
    if success:
        print("✅ UNSW-NB15 weights successfully stored in KeyDB!")
        
        # Verify by loading back
        print("\nVerifying storage by loading weights back...")
        loaded_weights = weight_manager.load_weights_from_keydb("unsw_mlp")
        
        if loaded_weights:
            print("✅ Weights successfully verified!")
            
            # Show experiment info
            info = weight_manager.get_experiment_info("unsw_mlp")
            print("\nUNSW-NB15 Experiment Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("❌ Failed to verify weights")
    else:
        print("❌ Failed to store weights")

if __name__ == "__main__":
    main()
