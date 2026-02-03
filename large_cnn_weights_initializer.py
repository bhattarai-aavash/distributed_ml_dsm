import numpy as np
import redis
import pickle
import json
from typing import List, Tuple

class LargeCNNWeightManager:
    """Manages large CNN weights in KeyDB for distributed training - comparable to MLP size"""
    
    def __init__(self, keydb_host='localhost', keydb_port=6379, keydb_db=0):
        """Initialize connection to KeyDB"""
        self.redis_client = redis.Redis(
            host=keydb_host, 
            port=keydb_port, 
            db=keydb_db,
            decode_responses=False  # Keep binary for pickle
        )
    
    def create_large_cnn_weights(self, input_features: int = 2381, num_classes: int = 2) -> dict:
        """
        Create and initialize LARGE CNN weights comparable to MLP (3.13M parameters)
        
        Args:
            input_features: Number of input features (2381 for amber dataset)
            num_classes: Number of output classes (2 for binary classification)
            
        Returns:
            Dictionary containing weights and biases for each layer
        """
        weights = {}
        
        # LARGE CNN architecture - comparable to MLP (3.13M parameters)
        # Target: ~3.1M parameters to match MLP
        
        # Layer 1: Conv1d (1 input channel -> 64 output channels, kernel_size=7)
        weights["conv1_weights"] = self._xavier_init_conv1d(1, 64, 7)      # 1 → 64 channels
        weights["conv1_bias"] = np.zeros(64, dtype=np.float32)              # 64 bias terms
        
        # Layer 2: Conv1d (64 input channels -> 128 output channels, kernel_size=5)
        weights["conv2_weights"] = self._xavier_init_conv1d(64, 128, 5)     # 64 → 128 channels
        weights["conv2_bias"] = np.zeros(128, dtype=np.float32)              # 128 bias terms
        
        # Layer 3: Conv1d (128 input channels -> 256 output channels, kernel_size=3)
        weights["conv3_weights"] = self._xavier_init_conv1d(128, 256, 3)    # 128 → 256 channels
        weights["conv3_bias"] = np.zeros(256, dtype=np.float32)              # 256 bias terms
        
        # Layer 4: Conv1d (256 input channels -> 512 output channels, kernel_size=3)
        weights["conv4_weights"] = self._xavier_init_conv1d(256, 512, 3)    # 256 → 512 channels
        weights["conv4_bias"] = np.zeros(512, dtype=np.float32)              # 512 bias terms
        
        # Calculate feature map size after convolutions and pooling
        # Starting: input_features (2381)
        # After conv1 (kernel=7, stride=1, padding=3): 2381 (no change)
        # After maxpool1 (kernel=4, stride=4): 2381/4 = 595
        # After conv2 (kernel=5, stride=1, padding=2): 595 (no change)
        # After maxpool2 (kernel=4, stride=4): 595/4 = 148
        # After conv3 (kernel=3, stride=1, padding=1): 148 (no change)
        # After maxpool3 (kernel=4, stride=4): 148/4 = 37
        # After conv4 (kernel=3, stride=1, padding=1): 37 (no change)
        # After maxpool4 (kernel=4, stride=4): 37/4 = 9
        
        # Moderately sized fully connected layers for 2-3M parameters
        fc_input_size = 512 * 9  # 512 channels × 9 spatial dimensions = 4,608
        
        # FC Layer 1: 4,608 -> 512 neurons (moderate hidden layer)
        weights["fc1_weights"] = self._xavier_init_linear(fc_input_size, 512)
        weights["fc1_bias"] = np.zeros(512, dtype=np.float32)
        
        # FC Layer 2: 512 -> 256 neurons (moderate hidden layer)
        weights["fc2_weights"] = self._xavier_init_linear(512, 256)
        weights["fc2_bias"] = np.zeros(256, dtype=np.float32)
        
        # Output Layer: 256 -> num_classes
        weights["fc3_weights"] = self._xavier_init_linear(256, num_classes)
        weights["fc3_bias"] = np.zeros(num_classes, dtype=np.float32)
        
        return weights
    
    def get_experiment_info(self, experiment_name: str = "amber_large_cnn") -> dict:
        """Get information about the large CNN experiment"""
        try:
            metadata_key = f"{experiment_name}:metadata"
            metadata_bytes = self.redis_client.get(metadata_key)
            if metadata_bytes is None:
                return {"status": "No experiment found"}
            
            return json.loads(metadata_bytes)
            
        except Exception as e:
            return {"error": str(e)}

    def _xavier_init_conv1d(self, in_channels: int, out_channels: int, kernel_size: int) -> np.ndarray:
        """Initialize 1D convolution weights using Xavier/Glorot initialization"""
        fan_in = in_channels * kernel_size
        fan_out = out_channels * kernel_size
        
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        return np.random.uniform(
            -limit, limit, 
            size=(out_channels, in_channels, kernel_size)
        ).astype(np.float32)
    
    def _xavier_init_linear(self, fan_in: int, fan_out: int) -> np.ndarray:
        """Initialize linear layer weights using Xavier/Glorot initialization"""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        return np.random.uniform(
            -limit, limit, 
            size=(fan_in, fan_out)
        ).astype(np.float32)
    
    def store_weights_in_keydb(self, weights: dict, experiment_name: str = "amber_large_cnn"):
        """Store large CNN weights in KeyDB"""
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
                    'input_shape': (1, 2381),
                    'conv_layers': 4,
                    'fc_layers': 3,    # 2 hidden + 1 output
                    'num_classes': 2,
                    'conv_kernel_sizes': [7, 5, 3, 3],
                    'conv_channels': [64, 128, 256, 512],  # Much larger channels
                    'fc_sizes': [512, 256, 2],            # Moderate FC layers
                    'model_type': 'large_cnn_mlp_comparable'
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
    """Initialize large CNN for Amber dataset and store in KeyDB (comparable to MLP)"""
    
    print("Initializing Large CNN Weight Manager (MLP-comparable size)...")
    weight_manager = LargeCNNWeightManager()
    
    print(f"Creating LARGE CNN weights (comparable to MLP):")
    print(f"  Input shape: (1, 2381) - 1 channel, 2381 features")
    print(f"  Architecture: 4 Conv1d layers + 3 FC layers")
    print(f"  Conv1: 1→64 channels (kernel=7) - Basic features")
    print(f"  Conv2: 64→128 channels (kernel=5) - Feature combinations")
    print(f"  Conv3: 128→256 channels (kernel=3) - Mid-level features")
    print(f"  Conv4: 256→512 channels (kernel=3) - High-level features")
    print(f"  FC layers: 4,608 → 512 → 256 → 2 - Moderate classification")
    print(f"  Output size: 2 classes (binary classification)")
    
    # Create large CNN weights
    weights = weight_manager.create_large_cnn_weights(
        input_features=2381,
        num_classes=2
    )
    
    # Calculate total parameters
    total_params = sum(w.size for w in weights.values())
    print(f"Total parameters: {total_params:,}")
    
    # Compare with MLP
    mlp_params = 3_128_450
    ratio = total_params / mlp_params
    print(f"MLP parameters: {mlp_params:,}")
    print(f"CNN/MLP ratio: {ratio:.2f}x")
    
    # Store in KeyDB
    print("\nStoring large CNN weights in KeyDB...")
    success = weight_manager.store_weights_in_keydb(weights, "amber_large_cnn")
    
    if success:
        print("✅ Large CNN weights successfully stored in KeyDB!")
        
        # Show experiment info
        info = weight_manager.get_experiment_info("amber_large_cnn")
        print("\nLarge CNN Experiment Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Failed to store large CNN weights")

if __name__ == "__main__":
    main()
