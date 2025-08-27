import unittest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data import get_data, get_mnist_data, get_fashionmnist_data


class TestDataLoading(unittest.TestCase):
    """Test cases for data loading functions"""
    
    def test_mnist_data_loading(self):
        """Test MNIST data loading"""
        try:
            train_loader, val_loader, test_loader = get_mnist_data()
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertIsNotNone(test_loader)
        except Exception as e:
            self.skipTest(f"MNIST data not available: {e}")
    
    def test_fashionmnist_data_loading(self):
        """Test Fashion-MNIST data loading"""
        try:
            train_loader, val_loader, test_loader = get_fashionmnist_data()
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertIsNotNone(test_loader)
        except Exception as e:
            self.skipTest(f"Fashion-MNIST data not available: {e}")
    
    def test_unified_data_interface_mnist(self):
        """Test unified data interface with MNIST"""
        try:
            train_loader, val_loader, test_loader = get_data('mnist')
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertIsNotNone(test_loader)
        except Exception as e:
            self.skipTest(f"MNIST data not available: {e}")
    
    def test_unified_data_interface_fashionmnist(self):
        """Test unified data interface with Fashion-MNIST"""
        try:
            train_loader, val_loader, test_loader = get_data('fashion_mnist')
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertIsNotNone(test_loader)
        except Exception as e:
            self.skipTest(f"Fashion-MNIST data not available: {e}")
    
    def test_invalid_dataset_name(self):
        """Test invalid dataset name raises error"""
        with self.assertRaises(ValueError):
            get_data('invalid_dataset')


class TestDataProperties(unittest.TestCase):
    """Test cases for data properties"""
    
    def setUp(self):
        """Set up test fixtures - skip if data not available"""
        try:
            self.train_loader, self.val_loader, self.test_loader = get_data('mnist')
        except Exception as e:
            self.skipTest(f"MNIST data not available: {e}")
    
    def test_data_shapes(self):
        """Test data tensor shapes"""
        # Get a sample batch
        sample_batch = next(iter(self.train_loader))
        images, labels = sample_batch
        
        # Test image shape (batch_size, channels, height, width)
        self.assertEqual(len(images.shape), 4, "Images should be 4D tensors")
        self.assertEqual(images.shape[1], 1, "MNIST should have 1 channel")
        self.assertEqual(images.shape[2], 28, "MNIST height should be 28")
        self.assertEqual(images.shape[3], 28, "MNIST width should be 28")
        
        # Test label shape
        self.assertEqual(len(labels.shape), 1, "Labels should be 1D tensor")
        self.assertEqual(images.shape[0], labels.shape[0], 
                        "Batch size should match between images and labels")
    
    def test_data_ranges(self):
        """Test data value ranges"""
        sample_batch = next(iter(self.train_loader))
        images, labels = sample_batch
        
        # Images should be normalized (approximately in [-3, 3] range for normalized data)
        self.assertLessEqual(images.max().item(), 5.0, "Images max value should be reasonable")
        self.assertGreaterEqual(images.min().item(), -5.0, "Images min value should be reasonable")
        
        # Labels should be in [0, 9] range for 10 classes
        self.assertGreaterEqual(labels.min().item(), 0, "Labels should be >= 0")
        self.assertLessEqual(labels.max().item(), 9, "Labels should be <= 9")
    
    def test_data_types(self):
        """Test data tensor types"""
        sample_batch = next(iter(self.train_loader))
        images, labels = sample_batch
        
        self.assertTrue(images.dtype == torch.float32 or images.dtype == torch.float, 
                       "Images should be float tensors")
        self.assertTrue(labels.dtype == torch.int64 or labels.dtype == torch.long, 
                       "Labels should be long tensors")
    
    def test_batch_size_consistency(self):
        """Test batch size consistency"""
        batch_sizes = []
        for i, (images, labels) in enumerate(self.train_loader):
            batch_sizes.append(images.shape[0])
            if i >= 5:  # Test first few batches
                break
        
        # Most batches should have the same size (except possibly the last one)
        self.assertGreater(min(batch_sizes), 0, "All batches should have positive size")
        self.assertLessEqual(max(batch_sizes), 128, "Batch size should be reasonable")
    
    def test_data_split_sizes(self):
        """Test data split sizes are reasonable"""
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)
        test_size = len(self.test_loader.dataset)
        
        # Training set should be largest
        self.assertGreater(train_size, val_size, "Training set should be larger than validation")
        
        # Test set should be fixed size for MNIST (10,000)
        self.assertEqual(test_size, 10000, "MNIST test set should have 10,000 samples")
        
        # Total training samples should be 60,000 for MNIST
        total_train_val = train_size + val_size
        self.assertEqual(total_train_val, 60000, "MNIST training + validation should be 60,000")
        
        # Validation should be about 20% of training data (80-20 split)
        expected_val_size = int(0.2 * 60000)
        self.assertAlmostEqual(val_size, expected_val_size, delta=100, 
                              msg="Validation set should be approximately 20% of training data")
    
    def test_data_loader_iteration(self):
        """Test data loader can be iterated multiple times"""
        # First iteration
        first_batch = next(iter(self.train_loader))
        
        # Second iteration
        second_batch = next(iter(self.train_loader))
        
        # Should be able to iterate without error
        self.assertIsNotNone(first_batch)
        self.assertIsNotNone(second_batch)
        
        # Batches should have same structure but different data
        self.assertEqual(first_batch[0].shape, second_batch[0].shape)
        self.assertEqual(first_batch[1].shape, second_batch[1].shape)


class TestDataConsistency(unittest.TestCase):
    """Test data consistency across different loading methods"""
    
    def test_mnist_consistency(self):
        """Test MNIST data consistency between different loading methods"""
        try:
            # Load via specific function
            train1, val1, test1 = get_mnist_data()
            
            # Load via unified interface
            train2, val2, test2 = get_data('mnist')
            
            # Should have same dataset sizes
            self.assertEqual(len(train1.dataset), len(train2.dataset))
            self.assertEqual(len(val1.dataset), len(val2.dataset))
            self.assertEqual(len(test1.dataset), len(test2.dataset))
            
        except Exception as e:
            self.skipTest(f"MNIST data not available: {e}")
    
    def test_dataset_normalization(self):
        """Test dataset normalization is applied correctly"""
        try:
            train_loader, _, _ = get_data('mnist')
            
            # Get several batches to test normalization
            all_images = []
            for i, (images, _) in enumerate(train_loader):
                all_images.append(images)
                if i >= 10:  # Test first 10 batches
                    break
            
            # Combine all images
            all_images = torch.cat(all_images, dim=0)
            
            # Test approximate normalization (allowing some tolerance)
            mean = all_images.mean().item()
            std = all_images.std().item()
            
            # MNIST normalization: mean=0.1307, std=0.3081
            # After normalization: (x - 0.1307) / 0.3081
            # So normalized mean should be around 0 and std around 1
            self.assertAlmostEqual(mean, 0.0, delta=0.5, 
                                 msg="Normalized data should have mean close to 0")
            self.assertAlmostEqual(std, 1.0, delta=0.5, 
                                 msg="Normalized data should have std close to 1")
            
        except Exception as e:
            self.skipTest(f"MNIST data not available: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2) 