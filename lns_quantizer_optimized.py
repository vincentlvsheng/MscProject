import torch
import torch.nn as nn
import numpy as np
import time

class TrueLNSQuantizer:
    
    
    def __init__(self, bits=4, p_scale=3.0):
        self.bits = bits
        self.p_scale = p_scale
        
        # Set quantization range
        if bits == 4:
            self.quant_min = -7  # 3-bit values: -7 to 7
            self.quant_max = 7
        elif bits == 8:
            self.quant_min = -63  # 7-bit values: -63 to 63
            self.quant_max = 63
        else:
            raise ValueError(f"Unsupported bits: {bits}")
    
    def pack_lns_weights_4bit(self, quantized_log, signs):
        
        # Convert sign to 0/1
        sign_bits = (signs > 0).to(torch.uint8)
        
        # Convert log values to unsigned integers (with offset)
        offset = abs(self.quant_min)
        unsigned_log = (quantized_log + offset).to(torch.uint8)
        
        # Combine sign and log values to form 4-bit values
        # 1 sign bit + 3 log bits = 4 bits
        nibbles = (sign_bits << 3) | unsigned_log
        
        # If number of elements is odd, a padding element is needed
        if nibbles.numel() % 2 == 1:
            nibbles = torch.cat([nibbles, torch.zeros(1, dtype=torch.uint8)])
        
        # Pack two 4-bit values into a uint8
        nibbles = nibbles.reshape(-1, 2)
        packed_values = (nibbles[:, 0] << 4) | nibbles[:, 1]
        
        packed_data = {
            'packed_values': packed_values,
            'offset': offset,
            'original_size': quantized_log.numel(),
            'is_padded': quantized_log.numel() % 2 == 1
        }
        
        return packed_data
    
    def unpack_lns_weights_4bit(self, packed_data, original_shape):
        
        packed_values = packed_data['packed_values']
        offset = packed_data['offset']
        original_size = packed_data['original_size']
        is_padded = packed_data['is_padded']
        
        # Unpack two 4-bit values
        nibble1 = (packed_values >> 4) & 0xF
        nibble2 = packed_values & 0xF
        
        # Recombine all 4-bit values
        nibbles = torch.stack([nibble1, nibble2], dim=1).reshape(-1)
        
        # If padding is needed, remove padding elements
        if is_padded:
            nibbles = nibbles[:original_size]
        else:
            nibbles = nibbles[:original_size]
        
        # Extract sign and log values
        signs = ((nibbles >> 3) & 0x1).to(torch.float32) * 2 - 1
        log_values = (nibbles & 0x7).to(torch.float32) - offset
        
        # Reconstruct original values
        abs_values = torch.pow(2.0, log_values)
        abs_values = torch.clamp(abs_values, min=1e-8, max=1e8)
        reconstructed = signs * abs_values
        
        return reconstructed.reshape(original_shape)
    
    def pack_lns_weights(self, quantized_log, signs):
        
        if self.bits == 4:
            return self.pack_lns_weights_4bit(quantized_log, signs)
        elif self.bits == 8:
            # 8-bit storage: 1 sign bit + 7 log bits
            sign_bits = (signs > 0).to(torch.uint8)
            offset = abs(self.quant_min)
            unsigned_log = (quantized_log + offset).to(torch.uint8)
            packed_values = (sign_bits << 7) | unsigned_log
            
            packed_data = {
                'packed_values': packed_values,
                'offset': offset
            }
            return packed_data
    
    def unpack_lns_weights(self, packed_data, original_shape):
        
        if self.bits == 4:
            return self.unpack_lns_weights_4bit(packed_data, original_shape)
        elif self.bits == 8:
            packed_values = packed_data['packed_values']
            signs = ((packed_values >> 7).to(torch.float32) * 2 - 1)
            log_values = (packed_values & 0x7F).to(torch.float32) - packed_data['offset']
            
            abs_values = torch.pow(2.0, log_values)
            abs_values = torch.clamp(abs_values, min=1e-8, max=1e8)
            reconstructed = signs * abs_values
            
            return reconstructed.reshape(original_shape)
    
    def quantize_tensor_lns_madam(self, tensor):
        
        if tensor.numel() == 0:
            return tensor, tensor.shape
        
        original_shape = tensor.shape
        
        # 1. Separate sign and magnitude
        signs = torch.sign(tensor)
        abs_values = torch.abs(tensor)
        
        # 2. Avoid log(0)
        eps = 1e-8
        abs_values = torch.clamp(abs_values, min=eps)
        
        # 3. Log domain operations
        log_values = torch.log2(abs_values)
        
        # 4. Adaptive scaling
        log_mean = torch.mean(log_values)
        log_std = torch.std(log_values)
        
        # Adjust quantization range based on data distribution
        if self.bits == 4:
            adjusted_min = max(self.quant_min, (log_mean - 2 * log_std).item())
            adjusted_max = min(self.quant_max, (log_mean + 2 * log_std).item())
        else:
            adjusted_min = max(self.quant_min, (log_mean - 3 * log_std).item())
            adjusted_max = min(self.quant_max, (log_mean + 3 * log_std).item())
        
        # 5. Quantize log values
        quantized_log = torch.round(log_values).clamp(adjusted_min, adjusted_max)
        
        # 6. Pack for storage
        packed_data = self.pack_lns_weights(quantized_log, signs)
        
        return packed_data, original_shape

class LNSQuantizedLinear(nn.Module):
    
    
    def __init__(self, in_features, out_features, bits=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Store quantized weights
        self.weight_data = None
        self.weight_shape = None
        self.bias_data = None
        self.bias_shape = None
        
        self.quantizer = TrueLNSQuantizer(bits=bits)
    
    def quantize_from_fp32(self, fp32_linear):
        
        # Quantize weights
        self.weight_data, self.weight_shape = \
            self.quantizer.quantize_tensor_lns_madam(fp32_linear.weight)
        
        # Quantize bias
        if fp32_linear.bias is not None:
            self.bias_data, self.bias_shape = \
                self.quantizer.quantize_tensor_lns_madam(fp32_linear.bias)
    
    def forward(self, x):
        
        # Unpack weights
        weight = self.quantizer.unpack_lns_weights(self.weight_data, self.weight_shape)
        
        # Unpack bias
        bias = None
        if self.bias_data is not None:
            bias = self.quantizer.unpack_lns_weights(self.bias_data, self.bias_shape)
        
        return torch.nn.functional.linear(x, weight, bias)
    
    def get_memory_usage(self):
        
        if self.bits == 4:
            # 4-bit storage: 0.5 bytes per parameter
            weight_memory = self.weight_data['packed_values'].numel()
            bias_memory = self.bias_data['packed_values'].numel() if self.bias_data else 0
        elif self.bits == 8:
            # 8-bit storage: 1 byte per parameter
            weight_memory = self.weight_data['packed_values'].numel()
            bias_memory = self.bias_data['packed_values'].numel() if self.bias_data else 0
        
        return weight_memory + bias_memory

def evaluate_quantization_methods():
    
    print("=== LNS quantizer performance evaluation ===\n")
    
    # Create different size test models
    test_configs = [
        (784, 128),
        (128, 64),
        (64, 10)
    ]
    
    results = {
        'original': [],
        'lns4': [],
        'lns8': []
    }
    
    for in_features, out_features in test_configs:
        print(f"Testing model: {in_features} -> {out_features}")
        
        # Create original model
        fp32_linear = nn.Linear(in_features, out_features)
        
        # Create quantized model
        lns4_linear = LNSQuantizedLinear(in_features, out_features, bits=4)
        lns8_linear = LNSQuantizedLinear(in_features, out_features, bits=8)
        
        # Quantize
        lns4_linear.quantize_from_fp32(fp32_linear)
        lns8_linear.quantize_from_fp32(fp32_linear)
        
        # Calculate memory usage
        original_memory = fp32_linear.weight.numel() * 4
        if fp32_linear.bias is not None:
            original_memory += fp32_linear.bias.numel() * 4
        
        lns4_memory = lns4_linear.get_memory_usage()
        lns8_memory = lns8_linear.get_memory_usage()
        
        # Test accuracy
        test_input = torch.randn(10, in_features)
        
        with torch.no_grad():
            original_output = fp32_linear(test_input)
            lns4_output = lns4_linear(test_input)
            lns8_output = lns8_linear(test_input)
            
            lns4_error = torch.mean(torch.abs(original_output - lns4_output))
            lns8_error = torch.mean(torch.abs(original_output - lns8_output))
        
        # Test inference speed
        num_trials = 100
        
        # Original model speed
        start_time = time.time()
        for _ in range(num_trials):
            _ = fp32_linear(test_input)
        original_time = time.time() - start_time
        
        # LNS4 model speed
        start_time = time.time()
        for _ in range(num_trials):
            _ = lns4_linear(test_input)
        lns4_time = time.time() - start_time
        
        # LNS8 model speed
        start_time = time.time()
        for _ in range(num_trials):
            _ = lns8_linear(test_input)
        lns8_time = time.time() - start_time
        
        # Store results
        results['original'].append({
            'memory': original_memory,
            'time': original_time,
            'error': 0.0
        })
        
        results['lns4'].append({
            'memory': lns4_memory,
            'time': lns4_time,
            'error': lns4_error.item()
        })
        
        results['lns8'].append({        
            'memory': lns8_memory,
            'time': lns8_time,
            'error': lns8_error.item()
        })
        
        print(f"  Memory usage: FP32={original_memory}B, LNS4={lns4_memory}B, LNS8={lns8_memory}B")
        print(f"  Compression ratio: LNS4={original_memory/lns4_memory:.2f}x, LNS8={original_memory/lns8_memory:.2f}x")
        print(f"  Error: LNS4={lns4_error:.6f}, LNS8={lns8_error:.6f}")
        print(f"  Speed: FP32={original_time:.4f}s, LNS4={lns4_time:.4f}s, LNS8={lns8_time:.4f}s")
        print()
    
    # Summary
    print("=== Summary ===")
    total_original_memory = sum(r['memory'] for r in results['original'])
    total_lns4_memory = sum(r['memory'] for r in results['lns4'])
    total_lns8_memory = sum(r['memory'] for r in results['lns8'])
    
    avg_lns4_error = np.mean([r['error'] for r in results['lns4']])
    avg_lns8_error = np.mean([r['error'] for r in results['lns8']])
    
    print(f"Total memory usage: FP32={total_original_memory}B, LNS4={total_lns4_memory}B, LNS8={total_lns8_memory}B")
    print(f"Total compression ratio: LNS4={total_original_memory/total_lns4_memory:.2f}x, LNS8={total_original_memory/total_lns8_memory:.2f}x")
    print(f"Average error: LNS4={avg_lns4_error:.6f}, LNS8={avg_lns8_error:.6f}")

# Usage and test
if __name__ == "__main__":
    
    print("Testing true 4-bit LNS quantizer...")
    
    # Quick test
    fp32_linear = nn.Linear(784, 128)
    
    # Create quantized version
    lns4_linear = LNSQuantizedLinear(784, 128, bits=4)
    lns4_linear.quantize_from_fp32(fp32_linear)
    
    lns8_linear = LNSQuantizedLinear(784, 128, bits=8)
    lns8_linear.quantize_from_fp32(fp32_linear)
    
    # Calculate memory usage
    original_memory = fp32_linear.weight.numel() * 4
    if fp32_linear.bias is not None:
        original_memory += fp32_linear.bias.numel() * 4
    
    lns4_memory = lns4_linear.get_memory_usage()
    lns8_memory = lns8_linear.get_memory_usage()
    
    print(f"Original FP32 memory: {original_memory} bytes")
    print(f"LNS4 memory: {lns4_memory} bytes ({original_memory/lns4_memory:.2f}x compression)")
    print(f"LNS8 memory: {lns8_memory} bytes ({original_memory/lns8_memory:.2f}x compression)")
    
    # Test accuracy
    test_input = torch.randn(1, 784)
    
    with torch.no_grad():
        original_output = fp32_linear(test_input)
        lns4_output = lns4_linear(test_input)
        lns8_output = lns8_linear(test_input)
        
        print(f"\nLNS4 output difference: {torch.mean(torch.abs(original_output - lns4_output)):.6f}")
        print(f"LNS8 output difference: {torch.mean(torch.abs(original_output - lns8_output)):.6f}")
    
    print("\n" + "="*50)
    # Run full evaluation
    evaluate_quantization_methods() 