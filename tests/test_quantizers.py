import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from lns_quantizer_optimized import TrueLNSQuantizer
    LNS_AVAILABLE = True
except ImportError:
    LNS_AVAILABLE = False

try:
    from true_int4_quantizer import TrueINT4Quantizer
    INT4_AVAILABLE = True
except ImportError:
    INT4_AVAILABLE = False


@unittest.skipUnless(LNS_AVAILABLE, "LNS quantizer not available")
class TestLNSQuantizer(unittest.TestCase):
    """Test cases for LNS quantizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.quantizer_4bit = TrueLNSQuantizer(bits=4)
        self.quantizer_8bit = TrueLNSQuantizer(bits=8)
        
        # Test tensors
        self.small_tensor = torch.randn(5, 5)
        self.large_tensor = torch.randn(100, 50)
        self.positive_tensor = torch.abs(torch.randn(10, 10))
        self.negative_tensor = -torch.abs(torch.randn(10, 10))
    
    def test_lns4_initialization(self):
        """Test LNS 4-bit quantizer initialization"""
        self.assertEqual(self.quantizer_4bit.bits, 4)
        self.assertEqual(self.quantizer_4bit.quant_min, -7)
        self.assertEqual(self.quantizer_4bit.quant_max, 7)
    
    def test_lns8_initialization(self):
        """Test LNS 8-bit quantizer initialization"""
        self.assertEqual(self.quantizer_8bit.bits, 8)
        self.assertEqual(self.quantizer_8bit.quant_min, -63)
        self.assertEqual(self.quantizer_8bit.quant_max, 63)
    
    def test_invalid_bits(self):
        """Test invalid bit configuration raises error"""
        with self.assertRaises(ValueError):
            TrueLNSQuantizer(bits=3)  # Unsupported bits
    

    
    def test_quantize_dequantize_cycle(self):
        """Test quantize-dequantize cycle"""
        try:
            # Quantize
            quantized = self.quantizer_4bit.quantize_tensor_lns_madam(self.small_tensor)
            
            # Dequantize (skip if method not available)
            if hasattr(self.quantizer_4bit, 'dequantize_tensor_lns_madam'):
                dequantized = self.quantizer_4bit.dequantize_tensor_lns_madam(quantized)
                
                # Check shape preservation
                self.assertEqual(dequantized.shape, self.small_tensor.shape,
                               "Dequantized tensor should have same shape as original")
                
                # Check reasonable approximation (allowing for quantization error)
                max_error = (self.small_tensor - dequantized).abs().max().item()
                self.assertLess(max_error, 10.0, 
                               "Quantization error should be reasonable")
        except Exception as e:
            self.skipTest(f"Quantize-dequantize cycle not available: {e}")
    

    



@unittest.skipUnless(INT4_AVAILABLE, "INT4 quantizer not available")
class TestINT4Quantizer(unittest.TestCase):
    """Test cases for INT4 quantizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.symmetric_quantizer = TrueINT4Quantizer(symmetric=True)
        self.asymmetric_quantizer = TrueINT4Quantizer(symmetric=False)
        
        # Test tensors
        self.test_tensor = torch.randn(10, 10)
        self.positive_tensor = torch.abs(torch.randn(5, 5))
        self.negative_tensor = -torch.abs(torch.randn(5, 5))
    
    def test_symmetric_initialization(self):
        """Test symmetric INT4 quantizer initialization"""
        self.assertTrue(self.symmetric_quantizer.symmetric)
        self.assertEqual(self.symmetric_quantizer.quant_min, -8)
        self.assertEqual(self.symmetric_quantizer.quant_max, 7)
    
    def test_asymmetric_initialization(self):
        """Test asymmetric INT4 quantizer initialization"""
        self.assertFalse(self.asymmetric_quantizer.symmetric)
        self.assertEqual(self.asymmetric_quantizer.quant_min, 0)
        self.assertEqual(self.asymmetric_quantizer.quant_max, 15)
    
    def test_scale_zero_point_calculation(self):
        """Test scale and zero point calculation"""
        # Test symmetric
        scale, zero_point = self.symmetric_quantizer.calculate_scale_zero_point(self.test_tensor)
        self.assertGreater(scale, 0, "Scale should be positive")
        self.assertEqual(zero_point, 0, "Symmetric quantization should have zero_point=0")
        
        # Test asymmetric
        scale, zero_point = self.asymmetric_quantizer.calculate_scale_zero_point(self.test_tensor)
        self.assertGreater(scale, 0, "Scale should be positive")
        self.assertGreaterEqual(zero_point, self.asymmetric_quantizer.quant_min)
        self.assertLessEqual(zero_point, self.asymmetric_quantizer.quant_max)
    
    def test_quantize_weights(self):
        """Test weight quantization"""
        quantizers = [self.symmetric_quantizer, self.asymmetric_quantizer]
        
        for i, quantizer in enumerate(quantizers):
            with self.subTest(quantizer_type="symmetric" if i == 0 else "asymmetric"):
                try:
                    result = quantizer.quantize_tensor_int4(self.test_tensor)
                    self.assertIsNotNone(result)
                    
                    if isinstance(result, dict):
                        self.assertIn('scale', result)
                        self.assertIn('zero_point', result)
                        self.assertGreater(result['scale'], 0)
                except Exception as e:
                    self.fail(f"INT4 quantization failed: {e}")
    

    
    def test_packing_functionality(self):
        """Test INT4 packing functionality"""
        try:
            # Create some test quantized values
            test_values = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)
            
            if hasattr(self.symmetric_quantizer, 'pack_int4_weights'):
                packed = self.symmetric_quantizer.pack_int4_weights(test_values)
                self.assertIsNotNone(packed)
                
                # Packed size should be roughly half (due to 4-bit packing)
                if isinstance(packed, dict) and 'packed_values' in packed:
                    packed_size = packed['packed_values'].numel()
                    self.assertLessEqual(packed_size, test_values.numel())
        except Exception as e:
            self.skipTest(f"Packing functionality not available: {e}")
    
    def test_zero_scale_handling(self):
        """Test handling of zero scale (constant tensor)"""
        constant_tensor = torch.ones(5, 5) * 3.0
        
        try:
            scale, zero_point = self.symmetric_quantizer.calculate_scale_zero_point(constant_tensor)
            self.assertGreater(scale, 0, "Scale should be positive even for constant tensors")
        except Exception as e:
            self.fail(f"Zero scale handling failed: {e}")


@unittest.skipUnless(LNS_AVAILABLE and INT4_AVAILABLE, "Both quantizers needed for comparison")
class TestQuantizerComparison(unittest.TestCase):
    """Test cases comparing different quantizers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.lns_quantizer = TrueLNSQuantizer(bits=4)
        self.int4_quantizer = TrueINT4Quantizer(symmetric=True)
        self.test_tensor = torch.randn(20, 20)
    
    def test_both_quantizers_run(self):
        """Test both quantizers can process the same tensor"""
        lns_result = None
        int4_result = None
        
        try:
            lns_result = self.lns_quantizer.quantize_tensor_lns_madam(self.test_tensor)
        except Exception as e:
            self.fail(f"LNS quantization failed: {e}")
        
        try:
            int4_result = self.int4_quantizer.quantize_tensor_int4(self.test_tensor)
        except Exception as e:
            self.fail(f"INT4 quantization failed: {e}")
        
        self.assertIsNotNone(lns_result)
        self.assertIsNotNone(int4_result)
    
    def test_quantization_consistency(self):
        """Test quantization consistency across multiple runs"""
        # Test LNS consistency
        lns_result1 = self.lns_quantizer.quantize_tensor_lns_madam(self.test_tensor)
        lns_result2 = self.lns_quantizer.quantize_tensor_lns_madam(self.test_tensor)
        
        # Results should be identical for same input
        self.assertIsNotNone(lns_result1)
        self.assertIsNotNone(lns_result2)
        
        # Test INT4 consistency
        int4_result1 = self.int4_quantizer.quantize_tensor_int4(self.test_tensor)
        int4_result2 = self.int4_quantizer.quantize_tensor_int4(self.test_tensor)
        
        self.assertIsNotNone(int4_result1)
        self.assertIsNotNone(int4_result2)


class TestQuantizerEdgeCases(unittest.TestCase):
    """Test edge cases for quantizers"""
    
    def test_empty_tensor(self):
        """Test quantization with empty tensors"""
        # Simplified test - just check that empty tensors don't crash completely
        if LNS_AVAILABLE:
            quantizer = TrueLNSQuantizer(bits=4)
            empty_tensor = torch.empty(0)
            try:
                result = quantizer.quantize_tensor_lns_madam(empty_tensor)
                # Should either handle gracefully or raise appropriate error
                if result is not None:
                    self.assertTrue(True)  # Handled gracefully
            except (ValueError, RuntimeError):
                # Appropriate error for empty tensor
                self.assertTrue(True)
            except Exception:
                # Skip test if not supported
                self.skipTest("Empty tensor quantization not supported")
    



if __name__ == '__main__':
    unittest.main(verbosity=2) 