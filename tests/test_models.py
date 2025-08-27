import unittest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CnnModel import CNNModel, MLPModel


class TestCNNModel(unittest.TestCase):
    """Test cases for CNN model"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = CNNModel(num_classes=10)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 1, 28, 28)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, CNNModel)
        self.assertEqual(self.model.classifier[-1].out_features, 10)
    
    def test_forward_pass_shape(self):
        """Test forward pass output shape"""
        with torch.no_grad():
            output = self.model(self.input_tensor)
        
        # Check output shape
        expected_shape = (self.batch_size, 10)
        self.assertEqual(output.shape, expected_shape, 
                        f"Expected shape {expected_shape}, got {output.shape}")
    
    def test_forward_pass_no_error(self):
        """Test forward pass runs without error"""
        try:
            with torch.no_grad():
                output = self.model(self.input_tensor)
            self.assertTrue(True)  # If we get here, no error occurred
        except Exception as e:
            self.fail(f"Forward pass failed with error: {e}")
    
    def test_parameter_count(self):
        """Test model has reasonable number of parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Should have parameters but not too many
        self.assertGreater(total_params, 1000, "Model should have at least 1000 parameters")
        self.assertLess(total_params, 1000000, "Model should have less than 1M parameters")
    
    def test_gradient_flow(self):
        """Test gradients flow properly through the model"""
        self.model.train()
        
        # Forward pass
        output = self.model(self.input_tensor)
        target = torch.randint(0, 10, (self.batch_size,))
        
        # Backward pass
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for param in self.model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        self.assertTrue(has_grad, "Model should have gradients after backward pass")
    
    def test_different_input_sizes(self):
        """Test model handles different batch sizes"""
        batch_sizes = [1, 2, 8, 16]
        
        for bs in batch_sizes:
            with self.subTest(batch_size=bs):
                input_tensor = torch.randn(bs, 1, 28, 28)
                with torch.no_grad():
                    output = self.model(input_tensor)
                self.assertEqual(output.shape, (bs, 10))
    
    def test_model_eval_mode(self):
        """Test model switches between train and eval modes"""
        # Test train mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Test eval mode
        self.model.eval()
        self.assertFalse(self.model.training)


class TestMLPModel(unittest.TestCase):
    """Test cases for MLP model"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = MLPModel(num_classes=10)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 1, 28, 28)
    
    def test_model_initialization(self):
        """Test MLP model initialization"""
        self.assertIsInstance(self.model, MLPModel)
        self.assertEqual(self.model.fc3.out_features, 10)
    
    def test_forward_pass_shape(self):
        """Test MLP forward pass output shape"""
        with torch.no_grad():
            output = self.model(self.input_tensor)
        
        expected_shape = (self.batch_size, 10)
        self.assertEqual(output.shape, expected_shape)
    
    def test_flattening_works(self):
        """Test input flattening works correctly"""
        # Test with different input shapes
        inputs = [
            torch.randn(2, 1, 28, 28),
            torch.randn(1, 1, 28, 28),
            torch.randn(5, 1, 28, 28)
        ]
        
        for inp in inputs:
            with self.subTest(input_shape=inp.shape):
                with torch.no_grad():
                    output = self.model(inp)
                self.assertEqual(output.shape[1], 10)


class TestModelComparison(unittest.TestCase):
    """Test cases comparing different models"""
    
    def setUp(self):
        self.cnn = CNNModel(num_classes=10)
        self.mlp = MLPModel(num_classes=10)
        self.input_tensor = torch.randn(2, 1, 28, 28)
    
    def test_both_models_same_output_shape(self):
        """Test both models produce same output shape"""
        with torch.no_grad():
            cnn_output = self.cnn(self.input_tensor)
            mlp_output = self.mlp(self.input_tensor)
        
        self.assertEqual(cnn_output.shape, mlp_output.shape)
    
    def test_parameter_count_difference(self):
        """Test CNN has more parameters than MLP"""
        cnn_params = sum(p.numel() for p in self.cnn.parameters())
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        
        self.assertGreater(cnn_params, mlp_params, 
                          "CNN should have more parameters than MLP")


if __name__ == '__main__':
    unittest.main(verbosity=2) 