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
        Create and initialize LARGE CNN weights (~3M parameters)
        
        Args:
            input_features: Number of input features (2381 for amber dataset)
            num_classes: Number of output classes (2 for binary classification)
            
        Returns:
            Dictionary containing weights and biases for each layer
        """
        weights = {}
        
        # LARGE CNN architecture - ~3M parameters
        # Target: ~3M parameters for high capacity training
        
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
        
        # Very large fully connected layers for ~3M parameters
        fc_input_size = 512 * 9  # 512 channels × 9 spatial dimensions = 4,608
        
        # FC Layer 1: 4,608 -> 1024 neurons (very large hidden layer)
        weights["fc1_weights"] = self._xavier_init_linear(fc_input_size, 1024)
        weights["fc1_bias"] = np.zeros(1024, dtype=np.float32)
        
        # FC Layer 2: 1024 -> 512 neurons (large hidden layer)
        weights["fc2_weights"] = self._xavier_init_linear(1024, 512)
        weights["fc2_bias"] = np.zeros(512, dtype=np.float32)
        
        # Output Layer: 512 -> num_classes
        weights["fc3_weights"] = self._xavier_init_linear(512, num_classes)
        weights["fc3_bias"] = np.zeros(num_classes, dtype=np.float32)
        
        return weights
    
    def create_simple_cnn_weights(self, input_features: int = 2381, num_classes: int = 2) -> dict:
        """
        Create and initialize SIMPLE CNN weights (original small architecture)
        
        Args:
            input_features: Number of input features (2381 for amber dataset)
            num_classes: Number of output classes (2 for binary classification)
            
        Returns:
            Dictionary containing weights and biases for each layer
        """
        weights = {}
        
        # SIMPLE 4-LAYER CNN architecture - original small version
        # Memory usage: ~181K parameters (memory efficient)
        
        # Layer 1: Conv1d (1 input channel -> 16 output channels, kernel_size=7)
        weights["conv1_weights"] = self._xavier_init_conv1d(1, 16, 7)      # 1 → 16 channels
        weights["conv1_bias"] = np.zeros(16, dtype=np.float32)              # 16 bias terms
        
        # Layer 2: Conv1d (16 input channels -> 32 output channels, kernel_size=5)
        weights["conv2_weights"] = self._xavier_init_conv1d(16, 32, 5)     # 16 → 32 channels
        weights["conv2_bias"] = np.zeros(32, dtype=np.float32)              # 32 bias terms
        
        # Layer 3: Conv1d (32 input channels -> 64 output channels, kernel_size=3)
        weights["conv3_weights"] = self._xavier_init_conv1d(32, 64, 3)     # 32 → 64 channels
        weights["conv3_bias"] = np.zeros(64, dtype=np.float32)              # 64 bias terms
        
        # Layer 4: Conv1d (64 input channels -> 128 output channels, kernel_size=3)
        weights["conv4_weights"] = self._xavier_init_conv1d(64, 128, 3)    # 64 → 128 channels
        weights["conv4_bias"] = np.zeros(128, dtype=np.float32)             # 128 bias terms
        
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
        
        # Simple fully connected layers
        fc_input_size = 128 * 9  # 128 channels × 9 spatial dimensions
        
        # FC Layer: 128*9 -> 128 neurons (simple hidden layer)
        weights["fc1_weights"] = self._xavier_init_linear(fc_input_size, 128)
        weights["fc1_bias"] = np.zeros(128, dtype=np.float32)
        
        # Output Layer: 128 -> num_classes
        weights["fc2_weights"] = self._xavier_init_linear(128, num_classes)
        weights["fc2_bias"] = np.zeros(num_classes, dtype=np.float32)
        
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
                    'conv_channels': [64, 128, 256, 512],  # Very large channels for ~3M params
                    'fc_sizes': [1024, 512, 2],            # Very large FC layers
                    'model_type': 'large_cnn_3m_params'
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
    """Initialize CNN for Amber dataset and store in KeyDB - choose architecture"""
    
    print("CNN Weight Manager - Choose Architecture:")
    print("1. Simple CNN (181K parameters) - Fast training")
    print("2. Large CNN (~3M parameters) - High capacity training")
    
    # For now, let's create the large CNN (~3M parameters)
    choice = 2  # You can change this to 1 for simple CNN
    
    weight_manager = LargeCNNWeightManager()
    
    if choice == 1:
        print("\nCreating SIMPLE CNN weights:")
        print(f"  Input shape: (1, 2381) - 1 channel, 2381 features")
        print(f"  Architecture: 4 Conv1d layers + 2 FC layers")
        print(f"  Conv1: 1→16 channels (kernel=7)")
        print(f"  Conv2: 16→32 channels (kernel=5)")
        print(f"  Conv3: 32→64 channels (kernel=3)")
        print(f"  Conv4: 64→128 channels (kernel=3)")
        print(f"  FC layers: 1,152 → 128 → 2")
        print(f"  Output size: 2 classes (binary classification)")
        
        # Create simple CNN weights
        weights = weight_manager.create_simple_cnn_weights(
            input_features=2381,
            num_classes=2
        )
        
        experiment_name = "amber_simple_cnn"
        
    else:  # choice == 2
        print("\nCreating LARGE CNN weights (~3M parameters):")
        print(f"  Input shape: (1, 2381) - 1 channel, 2381 features")
        print(f"  Architecture: 4 Conv1d layers + 3 FC layers")
        print(f"  Conv1: 1→64 channels (kernel=7)")
        print(f"  Conv2: 64→128 channels (kernel=5)")
        print(f"  Conv3: 128→256 channels (kernel=3)")
        print(f"  Conv4: 256→512 channels (kernel=3)")
        print(f"  FC layers: 4,608 → 1024 → 512 → 2")
        print(f"  Output size: 2 classes (binary classification)")
        
        # Create large CNN weights
        weights = weight_manager.create_large_cnn_weights(
            input_features=2381,
            num_classes=2
        )
        
        experiment_name = "amber_large_cnn"
    
    # Calculate total parameters
    total_params = sum(w.size for w in weights.values())
    print(f"Total parameters: {total_params:,}")
    
    # Show parameter comparison
    if choice == 2:
        mlp_params = 3_128_450
        ratio = total_params / mlp_params
        print(f"MLP parameters: {mlp_params:,}")
        print(f"CNN/MLP ratio: {ratio:.2f}x (CNN is {1/ratio:.1f}x smaller)")
    
    # Store in KeyDB
    print(f"\nStoring {experiment_name} weights in KeyDB...")
    success = weight_manager.store_weights_in_keydb(weights, experiment_name)
    
    if success:
        print(f"✅ {experiment_name} weights successfully stored in KeyDB!")
        
        # Show experiment info
        info = weight_manager.get_experiment_info(experiment_name)
        print(f"\n{experiment_name} Experiment Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print(f"❌ Failed to store {experiment_name} weights")

if __name__ == "__main__":
    main()
