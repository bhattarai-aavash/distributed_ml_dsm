import numpy as np
import redis
import pickle
import json
from typing import List, Tuple

class MLPWeightManager:
    """Manages MLP weights in KeyDB for distributed training"""
    
    def __init__(self, keydb_host='localhost', keydb_port=6379, keydb_db=0):
        """Initialize connection to KeyDB"""
        self.redis_client = redis.Redis(
            host=keydb_host, 
            port=keydb_port, 
            db=keydb_db,
            decode_responses=False  # Keep binary for pickle
        )
        
    def create_mlp_weights(self, input_size: int, hidden_sizes: List[int], output_size: int) -> dict:
        """
        Create and initialize MLP weights randomly
        
        Args:
            input_size: Number of input features (2381 for amber dataset)
            hidden_sizes: List of hidden layer sizes, e.g., [512, 256, 128]
            output_size: Number of output classes (2 for binary classification)
            
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
    
    def store_weights_in_keydb(self, weights: dict, experiment_name: str = "amber_mlp"):
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
                'total_parameters': sum(w.size for w in weights.values())
            }
            
            metadata_key = f"{experiment_name}:metadata"
            self.redis_client.set(metadata_key, json.dumps(metadata, default=str))
            print(f"Stored metadata: {metadata}")
            
            return True
            
        except Exception as e:
            print(f"Error storing weights: {e}")
            return False
    
    def load_weights_from_keydb(self, experiment_name: str = "amber_mlp") -> dict:
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
    
    def get_experiment_info(self, experiment_name: str = "amber_mlp") -> dict:
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
    """Initialize MLP for Amber dataset and store in KeyDB"""
    
    # Amber dataset parameters
    INPUT_SIZE = 2381  # F1 to F2381 (2382 columns - 1 label column)
    HIDDEN_SIZES = [1024, 512, 256, 128] # Larger network
    OUTPUT_SIZE = 2  # Binary classification (but using 2 neurons for softmax)
    
    print("Initializing MLP Weight Manager...")
    weight_manager = MLPWeightManager()
    
    print(f"Creating MLP weights:")
    print(f"  Input size: {INPUT_SIZE}")
    print(f"  Hidden layers: {HIDDEN_SIZES}")
    print(f"  Output size: {OUTPUT_SIZE}")
    
    
    # Create weights
    weights = weight_manager.create_mlp_weights(
        input_size=INPUT_SIZE,
        hidden_sizes=HIDDEN_SIZES,
        output_size=OUTPUT_SIZE
    )
    
    # Calculate total parameters
    total_params = sum(w.size for w in weights.values())
    print(f"Total parameters: {total_params:,}")
    
    # Store in KeyDB
    print("\nStoring weights in KeyDB...")
    success = weight_manager.store_weights_in_keydb(weights, "amber_mlp")
    
    if success:
        print("✅ Weights successfully stored in KeyDB!")
        
        # Verify by loading back
        print("\nVerifying storage by loading weights back...")
        loaded_weights = weight_manager.load_weights_from_keydb("amber_mlp")
        
        if loaded_weights:
            print("✅ Weights successfully verified!")
            
            # Show experiment info
            info = weight_manager.get_experiment_info("amber_mlp")
            print("\nExperiment Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("❌ Failed to verify weights")
    else:
        print("❌ Failed to store weights")

if __name__ == "__main__":
    main()