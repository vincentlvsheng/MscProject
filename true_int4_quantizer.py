import torch
import torch.nn as nn
import numpy as np
import time

class TrueINT4Quantizer:
    
    
    def __init__(self, symmetric=True):
        self.symmetric = symmetric
        
        if symmetric:
            # Symmetric quantization: -8 to 7 (4-bit signed)
            self.quant_min = -8
            self.quant_max = 7
        else:
            # Asymmetric quantization: 0 to 15 (4-bit unsigned)
            self.quant_min = 0
            self.quant_max = 15
        
        print(f"🔧 Initialize True INT4 quantizer ({'symmetric' if symmetric else 'asymmetric'}): range [{self.quant_min}, {self.quant_max}]")
    
    def calculate_scale_zero_point(self, tensor):
        
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        if self.symmetric:
            # Symmetric quantization
            max_abs = max(abs(min_val), abs(max_val))
            scale = max_abs / 7  # 7 is the maximum value of the symmetric range
            zero_point = 0
        else:
            # Asymmetric quantization
            scale = (max_val - min_val) / 15  # 15 is the range of the asymmetric range
            if scale == 0:
                scale = 1.0
                zero_point = 0
            else:
                zero_point = -min_val / scale
                zero_point = max(self.quant_min, min(self.quant_max, round(zero_point)))
        
        # Avoid division by zero
        if scale == 0:
            scale = 1.0
        
        return scale, zero_point
    
    def pack_int4_weights(self, quantized_values):
        
        # Ensure quantized values are within valid range
        if self.symmetric:
            # Symmetric quantization: map -8 to 7 to 0 to 15
            packed_values = (quantized_values + 8).to(torch.uint8)
        else:
            # Asymmetric quantization: directly use 0 to 15
            packed_values = quantized_values.to(torch.uint8)
        
        # If the number of elements is odd, a padding element is needed
        original_size = packed_values.numel()
        if original_size % 2 == 1:
            packed_values = torch.cat([packed_values, torch.zeros(1, dtype=torch.uint8)])
        
        # Pack two 4-bit values into a uint8
        packed_values = packed_values.reshape(-1, 2)
        packed_data = (packed_values[:, 0] << 4) | packed_values[:, 1]
        
        return {
            'packed_data': packed_data,
            'original_size': original_size,
            'is_padded': original_size % 2 == 1
        }
    
    def unpack_int4_weights(self, packed_info, original_shape):
        
        packed_data = packed_info['packed_data']
        original_size = packed_info['original_size']
        is_padded = packed_info['is_padded']
        
        # Unpack two 4-bit values
        nibble1 = (packed_data >> 4) & 0xF
        nibble2 = packed_data & 0xF
        
        # Recombine all 4-bit v
        if is_padded:
            unpacked = unpacked[:original_size]
        else:
            unpacked = unpacked[:original_size]
        
        # Convert back to quantized values
        if self.symmetric:
            # Symmetric quantization: map 0 to 15 back to -8 to 7
            quantized_values = unpacked.to(torch.float32) - 8
        else:
            # Asymmetric quantization: directly use 0 to 15
            quantized_values = unpacked.to(torch.float32)
        
        return quantized_values.reshape(original_shape)
    
    def quantize_tensor_int4(self, tensor):
        
        if tensor.numel() == 0:
            return tensor, tensor.shape
        
        original_shape = tensor.shape
        
        # Calculate quantization parameters
        scale, zero_point = self.calculate_scale_zero_point(tensor)
        
        # Quantize: x_q = round(x/scale + zero_point)
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, self.quant_min, self.quant_max)
        
        # Pack for storage
        packed_info = self.pack_int4_weights(quantized)
        
        # Add dequantization parameters
        packed_info['scale'] = scale
        packed_info['zero_point'] = zero_point
        
        return packed_info, original_shape
    
    def dequantize_tensor_int4(self, packed_info, original_shape):
        
        # Unpack quantized values
        quantized_values = self.unpack_int4_weights(packed_info, original_shape)
        
        # Dequantize: x = scale * (x_q - zero_point)
        scale = packed_info['scale']
        zero_point = packed_info['zero_point']
        
        dequantized = scale * (quantized_values - zero_point)
        
        return dequantized

class INT4QuantizedLinear(nn.Module):
    
    
    def __init__(self, in_features, out_features, bias=True, symmetric=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.symmetric = symmetric
        
        # Store quantized weights
        self.weight_data = None
        self.weight_shape = None
        self.bias_data = None
        self.bias_shape = None
        
        self.quantizer = TrueINT4Quantizer(symmetric=symmetric)
    
    def quantize_from_fp32(self, fp32_linear):
        
        # Quantize weights
        self.weight_data, self.weight_shape = \
            self.quantizer.quantize_tensor_int4(fp32_linear.weight)
        
        # Quantize bias
        if fp32_linear.bias is not None:
            self.bias_data, self.bias_shape = \
                self.quantizer.quantize_tensor_int4(fp32_linear.bias)
    
    def forward(self, x):
        
        # Unpack weights
        weight = self.quantizer.dequantize_tensor_int4(self.weight_data, self.weight_shape)
        
        # Unpack bias
        bias = None
        if self.bias_data is not None:
            bias = self.quantizer.dequantize_tensor_int4(self.bias_data, self.bias_shape)
        
        return torch.nn.functional.linear(x, weight, bias)
    
    def get_memory_usage(self):
        
        # 4-bit storage: 0.5 bytes per parameter
        weight_memory = self.weight_data['packed_data'].numel()
        bias_memory = self.bias_data['packed_data'].numel() if self.bias_data else 0
        
        return weight_memory + bias_memory

class INT4QuantizedCNN(nn.Module):
    
    
    def __init__(self, num_classes=10, symmetric=True):
        super().__init__()
        self.num_classes = num_classes
        self.symmetric = symmetric
        
        # Define
        self.conv1_linear = INT4QuantizedLinear(1*3*3, 32, bias=True, symmetric=symmetric)
        self.conv2_linear = INT4QuantizedLinear(32*3*3, 64, bias=True, symmetric=symmetric)
        self.conv3_linear = INT4QuantizedLinear(64*3*3, 128, bias=True, symmetric=symmetric)
        self.conv4_linear = INT4QuantizedLinear(128*3*3, 256, bias=True, symmetric=symmetric)
        self.fc1 = INT4QuantizedLinear(256, 128, bias=True, symmetric=symmetric)
        self.fc2 = INT4QuantizedLinear(128, num_classes, bias=True, symmetric=symmetric)
        
        # Activation functions and other layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def convert_cnn_to_int4_quantized(fp32_model, symmetric=True):
    
    print(f"🔄 Converting FP32 CNN to INT4 quantized model...")
    
    # Create quantizer
    quantizer = TrueINT4Quantizer(symmetric=symmetric)
    
    # Copy model
    quantized_model = type(fp32_model)(num_classes=10)
    quantized_model.load_state_dict(fp32_model.state_dict())
    
    # Quantize all weight parameters
    quantized_params = {}
    with torch.no_grad():
        for name, param in fp32_model.named_parameters():
            if 'weight' in name or 'bias' in name:
                print(f"  🔧 Quantizing {name}: {param.shape}")
                
                # Quantize parameters
                packed_data, original_shape = quantizer.quantize_tensor_int4(param.data)
                quantized_params[name] = (packed_data, original_shape)
                
                # Immediately dequantize to get quantized parameter values
                quantized_param = quantizer.dequantize_tensor_int4(packed_data, original_shape)
                
                # Replace original parameters
                quantized_model.state_dict()[name].copy_(quantized_param)
                
                # Calculate quantization effect
                unique_values = len(torch.unique(quantized_param.view(-1)))
                print(f"    -> {unique_values} unique values")
    
    # Save quantization information to model
    quantized_model.quantized_params = quantized_params
    quantized_model.quantizer = quantizer
    
    print(f"✅ INT4 quantization completed")
    return quantized_model

def calculate_int4_memory_usage(model, symmetric=True):

    total_memory = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            param_count = param.numel()
            # 4位：每个参数0.5字节
            memory_bytes = param_count * 0.5
            total_memory += memory_bytes
    
    return total_memory

def evaluate_int4_quantization_methods():
    
    print("=== INT4 quantizer performance evaluation ===\n")
    
    # Create different size test models
    test_configs = [
        (784, 128),
        (128, 64),
        (64, 10)
    ]
    
    results = {
        'original': [],
        'int4_symmetric': [],
        'int4_asymmetric': []
    }
    
    for in_features, out_features in test_configs:
        print(f"Testing model: {in_features} -> {out_features}")
        
        # Create original model
        fp32_linear = nn.Linear(in_features, out_features)
        
        # Create quantized model
        int4_sym_linear = INT4QuantizedLinear(in_features, out_features, bias=True, symmetric=True)
        int4_asym_linear = INT4QuantizedLinear(in_features, out_features, bias=True, symmetric=False)
        
        # Quantize
        int4_sym_linear.quantize_from_fp32(fp32_linear)
        int4_asym_linear.quantize_from_fp32(fp32_linear)
        
        # Calculate memory usage
        original_memory = fp32_linear.weight.numel() * 4
        if fp32_linear.bias is not None:
            original_memory += fp32_linear.bias.numel() * 4
        
        int4_sym_memory = int4_sym_linear.get_memory_usage()
        int4_asym_memory = int4_asym_linear.get_memory_usage()
        
        # Test accuracy
        test_input = torch.randn(10, in_features)
        
        with torch.no_grad():
            original_output = fp32_linear(test_input)
            int4_sym_output = int4_sym_linear(test_input)
            int4_asym_output = int4_asym_linear(test_input)
            
            int4_sym_error = torch.mean(torch.abs(original_output - int4_sym_output))
            int4_asym_error = torch.mean(torch.abs(original_output - int4_asym_output))
        
        # Test inference speed
        num_trials = 100
        
        # Original model speed
        start_time = time.time()
        for _ in range(num_trials):
            _ = fp32_linear(test_input)
        original_time = time.time() - start_time
        
        # INT4 symmetric model speed
        start_time = time.time()
        for _ in range(num_trials):
            _ = int4_sym_linear(test_input)
        int4_sym_time = time.time() - start_time
        
        # INT4 asymmetric model speed
        start_time = time.time()
        for _ in range(num_trials):
            _ = int4_asym_linear(test_input)
        int4_asym_time = time.time() - start_time
        
        # Store results
        results['original'].append({
            'memory': original_memory,
            'time': original_time,
            'error': 0.0
        })
        
        results['int4_symmetric'].append({
            'memory': int4_sym_memory,
            'time': int4_sym_time,
            'error': int4_sym_error.item()
        })
        
        results['int4_asymmetric'].append({
            'memory': int4_asym_memory,
            'time': int4_asym_time,
            'error': int4_asym_error.item()
        })
        
        print(f"  Memory usage: FP32={original_memory}B, INT4-Sym={int4_sym_memory}B, INT4-Asym={int4_asym_memory}B")
        print(f"  Compression ratio: INT4-Sym={original_memory/int4_sym_memory:.2f}x, INT4-Asym={original_memory/int4_asym_memory:.2f}x")
        print(f"  Error: INT4-Sym={int4_sym_error:.6f}, INT4-Asym={int4_asym_error:.6f}")
        print(f"  Speed: FP32={original_time:.4f}s, INT4-Sym={int4_sym_time:.4f}s, INT4-Asym={int4_asym_time:.4f}s")
        print()
    
    # Summary
    print("=== Summary ===")
    total_original_memory = sum(r['memory'] for r in results['original'])
    total_int4_sym_memory = sum(r['memory'] for r in results['int4_symmetric'])
    total_int4_asym_memory = sum(r['memory'] for r in results['int4_asymmetric'])
    
    avg_int4_sym_error = np.mean([r['error'] for r in results['int4_symmetric']])
    avg_int4_asym_error = np.mean([r['error'] for r in results['int4_asymmetric']])
    
    print(f"Total memory usage: FP32={total_original_memory}B, INT4-Sym={total_int4_sym_memory}B, INT4-Asym={total_int4_asym_memory}B")
    print(f"Total compression ratio: INT4-Sym={total_original_memory/total_int4_sym_memory:.2f}x, INT4-Asym={total_original_memory/total_int4_asym_memory:.2f}x")
    print(f"Average error: INT4-Sym={avg_int4_sym_error:.6f}, INT4-Asym={avg_int4_asym_error:.6f}")

if __name__ == "__main__":
    print("Testing true INT4 quantization...")
    
    # Quick test
    fp32_linear = nn.Linear(784, 128)
    
    # Create quantized version
    int4_sym_linear = INT4QuantizedLinear(784, 128, bias=True, symmetric=True)
    int4_sym_linear.quantize_from_fp32(fp32_linear)
    
    int4_asym_linear = INT4QuantizedLinear(784, 128, bias=True, symmetric=False)
    int4_asym_linear.quantize_from_fp32(fp32_linear)
    
    # Calculate memory usage
    original_memory = fp32_linear.weight.numel() * 4
    if fp32_linear.bias is not None:
        original_memory += fp32_linear.bias.numel() * 4
    
    int4_sym_memory = int4_sym_linear.get_memory_usage()
    int4_asym_memory = int4_asym_linear.get_memory_usage()
    
    print(f"Original FP32 memory: {original_memory} bytes")
    print(f"INT4 symmetric memory: {int4_sym_memory} bytes ({original_memory/int4_sym_memory:.2f}x compression)")
    print(f"INT4 asymmetric memory: {int4_asym_memory} bytes ({original_memory/int4_asym_memory:.2f}x compression)")
    
    # Test accuracy
    test_input = torch.randn(1, 784)
    
    with torch.no_grad():
        original_output = fp32_linear(test_input)
        int4_sym_output = int4_sym_linear(test_input)
        int4_asym_output = int4_asym_linear(test_input)
        
        print(f"\nINT4 symmetric output difference: {torch.mean(torch.abs(original_output - int4_sym_output)):.6f}")
        print(f"INT4 asymmetric output difference: {torch.mean(torch.abs(original_output - int4_asym_output)):.6f}")
    
    print("\n" + "="*50)
    # Run full evaluation
    evaluate_int4_quantization_methods()