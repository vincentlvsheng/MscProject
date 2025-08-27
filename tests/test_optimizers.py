import unittest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_optim import SGD, AdamWeg, AdamGD, LNS_Madam


class TestSGDOptimizer(unittest.TestCase):
    """Test cases for custom SGD optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = nn.Linear(10, 1)
        self.x = torch.randn(5, 10)
        self.y = torch.randn(5, 1)
        self.criterion = nn.MSELoss()
    
    def test_sgd_gd_initialization(self):
        """Test SGD with GD update algorithm initialization"""
        optimizer = SGD(self.model.parameters(), lr=0.01, update_alg='gd')
        self.assertEqual(optimizer.defaults['lr'], 0.01)
        self.assertEqual(optimizer.defaults['update_alg'], 'gd')
    
    def test_sgd_eg_initialization(self):
        """Test SGD with EG update algorithm initialization"""
        optimizer = SGD(self.model.parameters(), lr=0.01, update_alg='eg')
        self.assertEqual(optimizer.defaults['update_alg'], 'eg')
    
    def test_invalid_update_algorithm(self):
        """Test invalid update algorithm raises error"""
        with self.assertRaises(ValueError):
            SGD(self.model.parameters(), lr=0.01, update_alg='invalid')
    
    def test_sgd_gd_step(self):
        """Test SGD GD optimizer step"""
        optimizer = SGD(self.model.parameters(), lr=0.01, update_alg='gd')
        
        # Store initial parameters
        initial_params = [p.clone() for p in self.model.parameters()]
        
        # Forward and backward pass
        output = self.model(self.x)
        loss = self.criterion(output, self.y)
        loss.backward()
        optimizer.step()
        
        # Check parameters updated
        for initial, current in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.equal(initial, current), 
                           "Parameters should be updated after optimizer step")
    
    def test_sgd_eg_step(self):
        """Test SGD EG optimizer step"""
        optimizer = SGD(self.model.parameters(), lr=0.01, update_alg='eg')
        
        # Forward and backward pass
        output = self.model(self.x)
        loss = self.criterion(output, self.y)
        loss.backward()
        
        # Should not raise error
        try:
            optimizer.step()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"EG optimizer step failed: {e}")
    
    def test_momentum(self):
        """Test SGD with momentum"""
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9, update_alg='gd')
        
        # Multiple steps to test momentum
        for _ in range(3):
            optimizer.zero_grad()
            output = self.model(self.x)
            loss = self.criterion(output, self.y)
            loss.backward()
            optimizer.step()
        
        # Should complete without error
        self.assertTrue(True)
    
    def test_weight_decay(self):
        """Test SGD with weight decay"""
        optimizer = SGD(self.model.parameters(), lr=0.01, weight_decay=0.01, update_alg='gd')
        
        optimizer.zero_grad()
        output = self.model(self.x)
        loss = self.criterion(output, self.y)
        loss.backward()
        optimizer.step()
        
        # Should complete without error
        self.assertTrue(True)


class TestAdamWegOptimizer(unittest.TestCase):
    """Test cases for AdamWeg optimizer"""
    
    def setUp(self):
        self.model = nn.Linear(10, 1)
        self.x = torch.randn(5, 10)
        self.y = torch.randn(5, 1)
        self.criterion = nn.MSELoss()
    
    def test_adamweg_initialization(self):
        """Test AdamWeg initialization"""
        optimizer = AdamWeg(self.model.parameters(), lr=0.001)
        self.assertEqual(optimizer.defaults['lr'], 0.001)
        self.assertEqual(optimizer.defaults['betas'], (0.9, 0.999))
    
    def test_adamweg_step(self):
        """Test AdamWeg optimizer step"""
        optimizer = AdamWeg(self.model.parameters(), lr=0.001)
        
        initial_params = [p.clone() for p in self.model.parameters()]
        
        output = self.model(self.x)
        loss = self.criterion(output, self.y)
        loss.backward()
        optimizer.step()
        
        # Check parameters updated
        for initial, current in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.equal(initial, current))
    
    def test_adamweg_multiple_steps(self):
        """Test AdamWeg multiple optimization steps"""
        optimizer = AdamWeg(self.model.parameters(), lr=0.001)
        
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            output = self.model(self.x)
            loss = self.criterion(output, self.y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        # Loss should generally decrease (with some tolerance for noise)
        self.assertLess(losses[-1], losses[0] + 0.1, 
                       "Loss should decrease or stay stable during optimization")
    
    def test_adamweg_invalid_params(self):
        """Test AdamWeg with invalid parameters"""
        with self.assertRaises(ValueError):
            AdamWeg(self.model.parameters(), lr=-0.001)  # negative lr
        
        with self.assertRaises(ValueError):
            AdamWeg(self.model.parameters(), lr=0.001, betas=(1.1, 0.999))  # invalid beta


class TestAdamGDOptimizer(unittest.TestCase):
    """Test cases for AdamGD optimizer"""
    
    def setUp(self):
        self.model = nn.Linear(10, 1)
        self.x = torch.randn(5, 10)
        self.y = torch.randn(5, 1)
        self.criterion = nn.MSELoss()
    
    def test_adamgd_initialization(self):
        """Test AdamGD initialization"""
        optimizer = AdamGD(self.model.parameters(), lr=0.001)
        self.assertEqual(optimizer.defaults['lr'], 0.001)
        self.assertEqual(optimizer.defaults['betas'], (0.9, 0.999))
    
    def test_adamgd_step(self):
        """Test AdamGD optimizer step"""
        optimizer = AdamGD(self.model.parameters(), lr=0.001)
        
        output = self.model(self.x)
        loss = self.criterion(output, self.y)
        loss.backward()
        
        try:
            optimizer.step()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"AdamGD step failed: {e}")
    
    def test_adamgd_weight_decay(self):
        """Test AdamGD with weight decay"""
        optimizer = AdamGD(self.model.parameters(), lr=0.001, weight_decay=0.01)
        
        output = self.model(self.x)
        loss = self.criterion(output, self.y)
        loss.backward()
        optimizer.step()
        
        # Should complete without error
        self.assertTrue(True)


class TestLNSMadamOptimizer(unittest.TestCase):
    """Test cases for LNS_Madam optimizer"""
    
    def setUp(self):
        self.model = nn.Linear(10, 1)
        self.x = torch.randn(5, 10)
        self.y = torch.randn(5, 1)
        self.criterion = nn.MSELoss()
    
    def test_lns_madam_initialization(self):
        """Test LNS_Madam initialization"""
        optimizer = LNS_Madam(self.model.parameters(), lr=1/128)
        self.assertEqual(optimizer.defaults['lr'], 1/128)
        self.assertEqual(optimizer.p_scale, 3.0)
        self.assertEqual(optimizer.g_bound, 10.0)
    
    def test_lns_madam_step(self):
        """Test LNS_Madam optimizer step"""
        optimizer = LNS_Madam(self.model.parameters(), lr=1/128)
        
        output = self.model(self.x)
        loss = self.criterion(output, self.y)
        loss.backward()
        
        try:
            optimizer.step()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"LNS_Madam step failed: {e}")
    
    def test_lns_madam_parameter_clamping(self):
        """Test LNS_Madam parameter clamping"""
        optimizer = LNS_Madam(self.model.parameters(), lr=1/128, p_scale=1.0)
        
        # Force large gradients to test clamping
        output = self.model(self.x)
        loss = self.criterion(output, self.y) * 100  # amplify loss
        loss.backward()
        optimizer.step()
        
        # Check parameters are within reasonable bounds
        for param in self.model.parameters():
            max_val = param.abs().max().item()
            self.assertLess(max_val, 100, "Parameters should be clamped to reasonable values")
    
    def test_lns_madam_with_weight_decay(self):
        """Test LNS_Madam with weight decay"""
        optimizer = LNS_Madam(self.model.parameters(), lr=1/128, wd=0.01)
        
        output = self.model(self.x)
        loss = self.criterion(output, self.y)
        loss.backward()
        optimizer.step()
        
        # Should complete without error
        self.assertTrue(True)


class TestOptimizerComparison(unittest.TestCase):
    """Test cases comparing different optimizers"""
    
    def setUp(self):
        self.models = {
            'sgd_gd': nn.Linear(10, 1),
            'sgd_eg': nn.Linear(10, 1),
            'adamweg': nn.Linear(10, 1),
            'adamgd': nn.Linear(10, 1),
            'lns_madam': nn.Linear(10, 1)
        }
        
        # Initialize all models with same weights
        with torch.no_grad():
            for model in self.models.values():
                model.weight.fill_(0.1)
                model.bias.fill_(0.0)
        
        self.x = torch.randn(5, 10)
        self.y = torch.randn(5, 1)
        self.criterion = nn.MSELoss()
    
    def test_all_optimizers_run(self):
        """Test all optimizers can run without error"""
        optimizers = {
            'sgd_gd': SGD(self.models['sgd_gd'].parameters(), lr=0.01, update_alg='gd'),
            'sgd_eg': SGD(self.models['sgd_eg'].parameters(), lr=0.01, update_alg='eg'),
            'adamweg': AdamWeg(self.models['adamweg'].parameters(), lr=0.001),
            'adamgd': AdamGD(self.models['adamgd'].parameters(), lr=0.001),
            'lns_madam': LNS_Madam(self.models['lns_madam'].parameters(), lr=1/128)
        }
        
        for name, optimizer in optimizers.items():
            with self.subTest(optimizer=name):
                model = self.models[name]
                
                optimizer.zero_grad()
                output = model(self.x)
                loss = self.criterion(output, self.y)
                loss.backward()
                
                try:
                    optimizer.step()
                    self.assertTrue(True, f"{name} should run without error")
                except Exception as e:
                    self.fail(f"{name} failed with error: {e}")
    
    def test_optimizers_update_parameters(self):
        """Test all optimizers actually update parameters"""
        optimizers = {
            'sgd_gd': SGD(self.models['sgd_gd'].parameters(), lr=0.01, update_alg='gd'),
            'adamweg': AdamWeg(self.models['adamweg'].parameters(), lr=0.001),
            'adamgd': AdamGD(self.models['adamgd'].parameters(), lr=0.001)
        }
        
        for name, optimizer in optimizers.items():
            with self.subTest(optimizer=name):
                model = self.models[name]
                initial_weight = model.weight.clone()
                
                optimizer.zero_grad()
                output = model(self.x)
                loss = self.criterion(output, self.y)
                loss.backward()
                optimizer.step()
                
                self.assertFalse(torch.equal(initial_weight, model.weight),
                               f"{name} should update model parameters")


if __name__ == '__main__':
    unittest.main(verbosity=2) 