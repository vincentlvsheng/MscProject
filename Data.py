# Data loading and preprocessing module
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, split='train_set', transform=None):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.classes = sorted(os.listdir(os.path.join(root_dir, split)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            class_path = os.path.join(root_dir, split, cls)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Data transforms for FashionMNIST
fashion_mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST mean and std
])

# Data transforms for MNIST
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load FashionMNIST dataset
def get_fashionmnist_data():
    # Full training set
    full_train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=fashion_mnist_transform
    )
    
    # Split training set into train and validation sets (80-20 split)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Test set (for final evaluation)
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=fashion_mnist_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# Load MNIST dataset
def get_mnist_data():
    # Full training set
    full_train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=mnist_transform
    )
    
    # Split training set into train and validation sets (80-20 split)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Test set (for final evaluation)
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=mnist_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# Unified data loading interface
def get_data(dataset_name='mnist'):
    """
    Unified data loading interface
    Args:
        dataset_name: 'mnist' or 'fashion_mnist'
    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_name.lower() == 'mnist':
        return get_mnist_data()
    elif dataset_name.lower() == 'fashion_mnist':
        return get_fashionmnist_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

# Test code
if __name__ == '__main__':
    # Test MNIST
    train_loader, val_loader, test_loader = get_data('mnist')
    print("\n=== MNIST Dataset ===")
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    sample_img, sample_label = next(iter(train_loader))
    print(f"Image batch shape: {sample_img.shape}")
    print(f"Label batch shape: {sample_label.shape}")
    
    # Test FashionMNIST
    train_loader, val_loader, test_loader = get_data('fashion_mnist')
    print("\n=== FashionMNIST Dataset ===")
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    sample_img, sample_label = next(iter(train_loader))
    print(f"Image batch shape: {sample_img.shape}")
    print(f"Label batch shape: {sample_label.shape}")
