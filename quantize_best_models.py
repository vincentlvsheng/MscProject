import torch
import torch.nn as nn
import pandas as pd
import pathlib
import os
import warnings
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import datasets, transforms
from CnnModel import CNNModel
from torch.ao.quantization import get_default_qconfig, prepare, convert, fuse_modules
import torch.ao.quantization.quantize_fx as quantize_fx

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# For MPS (Apple Silicon)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

def get_dataloaders(batch_size=128, calib_size=1000):
    """Get MNIST data loaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # Create a subset of training data for calibration
    calib_indices = torch.randperm(len(train_dataset))[:calib_size]
    calib_dataset = torch.utils.data.Subset(train_dataset, calib_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, calib_loader, test_loader

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy

def calibrate_model(model, calib_loader, device='cpu'):
    """Calibrate model using all calibration data"""
    model.eval()
    with torch.no_grad():
        for data, _ in calib_loader:
            data = data.to(device)
            model(data)

def quantize_model_fx(model_fp32, calib_loader, device='cpu'):
    """Quantize model using FX graph mode"""
    model_fp32.eval()
    
    # Prepare quantization config - use qnnpack for Mac
    import platform
    if platform.system() == 'Darwin':  # macOS
        backend = 'qnnpack'
        torch.backends.quantized.engine = 'qnnpack'
    else:
        backend = 'fbgemm'
    
    qconfig_dict = {"": get_default_qconfig(backend)}
    
    # Get example input from calibration loader
    example_inputs = next(iter(calib_loader))[0]
    
    # Prepare model
    model_prepared = quantize_fx.prepare_fx(model_fp32, qconfig_dict, example_inputs)
    
    # Calibrate
    calibrate_model(model_prepared, calib_loader, 'cpu')  # Calibration must be on CPU
    
    # Convert to quantized model
    model_quantized = quantize_fx.convert_fx(model_prepared)
    
    return model_quantized

def get_model_size(model_path):
    """Get model file size in MB"""
    return os.path.getsize(model_path) / (1024 * 1024)

def main():
    # Set device for Mac
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Random seed set to: {SEED}")
    
    # Get data loaders
    train_loader, calib_loader, test_loader = get_dataloaders()
    
    # Read best configurations
    best_cfg = pd.read_csv('best_configurations.csv')
    
    # Optimizer name mapping
    name_map = {
        'GD': 'gd',
        'EG': 'eg', 
        'AdamWeg': 'adamweg',
        'AdamGD': 'adamgd'
    }
    
    # Create quantized results directory
    quantized_dir = pathlib.Path('checkpoints_quantized')
    quantized_dir.mkdir(exist_ok=True)
    
    results = []
    
    # Process the four best configurations (excluding LNS_Madam)
    for _, row in best_cfg.iterrows():
        optimizer = row['Optimizer']
        
        # Only process the specified four optimizers
        if optimizer not in name_map:
            continue
            
        lr = row['Learning Rate']
        wd = row['Weight Decay']
        val_acc = row['Val Acc']
        test_acc = row['Test Acc']
        
        print(f"\nProcessing {optimizer} (lr={lr}, wd={wd})")
        
        # Build checkpoint path
        ckpt_dir = f'checkpoints/lr{lr}_wd{wd}_mnist'
        ckpt_path = f'{ckpt_dir}/{name_map[optimizer]}_best_model.pth'
        
        if not os.path.exists(ckpt_path):
            print(f"Warning: Model file not found {ckpt_path}")
            continue
        
        try:
            # Load original model
            model_fp32 = CNNModel(num_classes=10)
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Extract model state dict from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model_fp32.load_state_dict(state_dict)
            model_fp32.to(device)
            
            # Evaluate original model accuracy
            fp32_accuracy = evaluate_model(model_fp32, test_loader, device)
            print(f"FP32 model accuracy: {fp32_accuracy:.2f}%")
            
            # Quantize model - move to CPU for quantization
            print("Starting quantization...")
            model_fp32_cpu = model_fp32.cpu()  # Quantization needs to be done on CPU
            model_quantized = quantize_model_fx(model_fp32_cpu, calib_loader, 'cpu')
            
            # Evaluate quantized model accuracy
            int8_accuracy = evaluate_model(model_quantized, test_loader, 'cpu')
            print(f"INT8 model accuracy: {int8_accuracy:.2f}%")
            
            # Save quantized model
            quant_subdir = quantized_dir / f'lr{lr}_wd{wd}_mnist'
            quant_subdir.mkdir(exist_ok=True)
            quant_ckpt_path = quant_subdir / f'{name_map[optimizer]}_best_model_int8.pth'
            torch.save(model_quantized.state_dict(), quant_ckpt_path)
            
            # Calculate model size
            fp32_size = get_model_size(ckpt_path)
            int8_size = get_model_size(quant_ckpt_path)
            compression_ratio = fp32_size / int8_size
            
            print(f"FP32 model size: {fp32_size:.2f} MB")
            print(f"INT8 model size: {int8_size:.2f} MB")
            print(f"Compression ratio: {compression_ratio:.2f}x")
            print(f"Accuracy loss: {fp32_accuracy - int8_accuracy:.2f}%")
            
            # Save results
            results.append({
                'Optimizer': optimizer,
                'Learning_Rate': lr,
                'Weight_Decay': wd,
                'Original_Val_Acc': val_acc,
                'Original_Test_Acc': test_acc,
                'FP32_Accuracy': f"{fp32_accuracy:.2f}%",
                'INT8_Accuracy': f"{int8_accuracy:.2f}%",
                'Accuracy_Loss': f"{fp32_accuracy - int8_accuracy:.2f}%",
                'FP32_Size_MB': f"{fp32_size:.2f}",
                'INT8_Size_MB': f"{int8_size:.2f}",
                'Compression_Ratio': f"{compression_ratio:.2f}x"
            })
            
        except Exception as e:
            print(f"Error processing {optimizer}: {str(e)}")
            continue
    
    # Save quantization results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('quantization_results.csv', index=False)
        print(f"\nQuantization results saved to quantization_results.csv")
        print("\nQuantization results summary:")
        print(results_df.to_string(index=False))
    else:
        print("No models were successfully quantized")

if __name__ == '__main__':
    main() 