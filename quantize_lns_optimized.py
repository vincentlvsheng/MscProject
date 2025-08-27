import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from CnnModel import CNNModel
from lns_quantizer_optimized import TrueLNSQuantizer, LNSQuantizedLinear
import platform

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

    
def simple_lns_quantize_model(fp32_model, bits=4):
    """Simple LNS quantization: directly replace model parameters"""
    print(f"🔄 Simple LNS{bits} quantization of model parameters...")
    
    # Create quantizer
    quantizer = TrueLNSQuantizer(bits=bits)
    
    # Copy model
    quantized_model = CNNModel(num_classes=10)
    quantized_model.load_state_dict(fp32_model.state_dict())
    
    # Quantize all weight parameters
    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if 'weight' in name or 'bias' in name:
                print(f"  🔧 Quantizing {name}: {param.shape}")
                
                # Quantize parameters
                packed_data, original_shape = quantizer.quantize_tensor_lns_madam(param.data)
                
                # Immediately unpack to get quantized parameters
                quantized_param = quantizer.unpack_lns_weights(packed_data, original_shape)
                
                # Replace original parameters
                param.data = quantized_param
                
                # Calculate quantization effect
                original_unique = torch.unique(param.data.view(-1)).numel()
                print(f"    -> {original_unique} unique values")
    
    print(f"✅ Simple LNS{bits} quantization completed")
    return quantized_model

def get_dataloaders(batch_size=128):
    """Get MNIST data loader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

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

def get_model_size_mb(model_path):
    """Get model file size (MB)"""
    return os.path.getsize(model_path) / (1024 * 1024)

def calculate_lns_memory_usage(model, bits=4):
    """Calculate theoretical memory usage of LNS quantized model"""
    total_memory = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            param_count = param.numel()
            if bits == 4:
                # 4-bit: 0.5 bytes per parameter
                memory_bytes = param_count * 0.5
            elif bits == 8:
                # 8-bit: 1 byte per parameter
                memory_bytes = param_count * 1.0
            else:
                memory_bytes = param_count * 4  # Default FP32
            
            total_memory += memory_bytes
    
    return total_memory

def analyze_quantization_distribution(model, model_name):
    """Analyze quantization parameter distribution"""
    print(f"\n🔍 {model_name} quantization analysis:")
    
    total_params = 0
    total_unique_values = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            unique_values = len(torch.unique(param))
            param_count = param.numel()
            total_params += param_count
            total_unique_values += unique_values
            
            print(f"  {name}: {param.shape} -> {unique_values} unique values ({param_count} parameters)")
    
    layers_count = len([n for n, p in model.named_parameters() if 'weight' in n or 'bias' in n])
    avg_unique_per_layer = total_unique_values / layers_count if layers_count > 0 else 0
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Average unique values per layer: {avg_unique_per_layer:.1f}")

def main():
    """Main function: LNS quantization of best MNIST models"""
    print("🚀 Starting LNS quantization of best MNIST models")
    print(f"💻 Platform: {platform.system()}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print("=" * 80)
    
    # Set device
    device = torch.device('cpu')  # LNS quantization on CPU
    
    # Get data loader
    test_loader = get_dataloaders()
    
    # Read best configurations
    try:
        best_cfg = pd.read_csv('best_configurations.csv')
        print(f"📄 Loaded best configurations: {len(best_cfg)} optimizers")
        print(best_cfg.to_string(index=False))
    except FileNotFoundError:
        print("❌ best_configurations.csv file not found")
        return
    
    # Optimizer name mapping
    name_map = {
        'GD': 'gd',
        'EG': 'eg', 
        'AdamWeg': 'adamweg',
        'AdamGD': 'adamgd'
    }
    
    # Create result directory
    for bits in [4, 8]:
        os.makedirs(f'checkpoints_quantized_lns{bits}_optimizers', exist_ok=True)
    
    # Quantization bit list
    quantization_bits = [4, 8]
    all_results = []
    
    # Process each best configuration (excluding LNS_Madam)
    for _, row in best_cfg.iterrows():
        optimizer = row['Optimizer']
        
        # Skip LNS_Madam
        if optimizer == 'LNS_Madam':
            print(f"⚠️ Skipping LNS_Madam optimizer as requested")
            continue
        
        if optimizer not in name_map:
            print(f"⚠️ Skipping unsupported optimizer: {optimizer}")
            continue
            
        lr = row['Learning Rate']
        wd = row['Weight Decay']
        
        print(f"\n{'='*80}")
        print(f"🔄 Processing {optimizer} model (lr={lr}, wd={wd})")
        print(f"{'='*80}")
        
        # Build checkpoint path
        ckpt_path = f'checkpoints/lr{lr}_wd{wd}_mnist/{name_map[optimizer]}_best_model.pth'
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint file does not exist: {ckpt_path}")
            continue
        
        try:
            # Load original model
            print("📂 Loading original model...")
            model_fp32 = CNNModel(num_classes=10)
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Extract model state dictionary
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model_fp32.load_state_dict(state_dict)
            model_fp32.eval()
            
            # Evaluate original model
            print("📊 Evaluating original FP32 model...")
            fp32_accuracy = evaluate_model(model_fp32, test_loader, device)
            fp32_size = get_model_size_mb(ckpt_path)
            print(f"✅ FP32 model: {fp32_accuracy:.2f}% accuracy, {fp32_size:.2f} MB")
            
            # Experiment with each quantization bit
            for bits in quantization_bits:
                print(f"\n--- LNS{bits} quantization experiment ---")
                
                # Use simple LNS quantization
                model_lns = simple_lns_quantize_model(model_fp32, bits)
                lns_accuracy = evaluate_model(model_lns, test_loader, device)
                print(f"✅ LNS{bits} accuracy: {lns_accuracy:.2f}%")
                
                # Analyze quantization distribution
                analyze_quantization_distribution(model_lns, f"{optimizer}_LNS{bits}")
                
                # Save quantized model
                save_dir = f'checkpoints_quantized_lns{bits}_optimizers/lr{lr}_wd{wd}_mnist'
                os.makedirs(save_dir, exist_ok=True)
                
                save_path = f'{save_dir}/{name_map[optimizer]}_lns{bits}.pth'
                torch.save(model_lns.state_dict(), save_path)
                print(f"💾 Model saved: {save_path}")
                
                # Calculate model size and compression ratio
                lns_size = get_model_size_mb(save_path)
                theoretical_memory = calculate_lns_memory_usage(model_lns, bits) / (1024 * 1024)  # MB
                compression_ratio = fp32_size / lns_size
                theoretical_compression = (fp32_size * 1024 * 1024) / (theoretical_memory * 1024 * 1024)
                accuracy_loss = fp32_accuracy - lns_accuracy
                
                print(f"📏 Model size: {fp32_size:.2f} MB → {lns_size:.2f} MB")
                print(f"📏 Theoretical memory: {theoretical_memory:.2f} MB")
                print(f"📈 Compression ratio: {compression_ratio:.2f}x")
                print(f"📈 Theoretical compression: {theoretical_compression:.2f}x")
                print(f"📉 Accuracy loss: {accuracy_loss:+.2f}%")
                
                # Save results
                result = {
                    'Optimizer': optimizer,
                    'Learning_Rate': lr,
                    'Weight_Decay': wd,
                    'Quantization_Bits': bits,
                    'Original_Val_Acc': row['Val Acc'],
                    'Original_Test_Acc': row['Test Acc'],
                    'FP32_Accuracy': f"{fp32_accuracy:.2f}%",
                    f'LNS{bits}_Accuracy': f"{lns_accuracy:.2f}%",
                    'Accuracy_Loss': f"{accuracy_loss:+.2f}%",
                    'FP32_Size_MB': f"{fp32_size:.2f}",
                    f'LNS{bits}_Size_MB': f"{lns_size:.2f}",
                    'Theoretical_Memory_MB': f"{theoretical_memory:.2f}",
                    'Compression_Ratio': f"{compression_ratio:.2f}x",
                    'Theoretical_Compression': f"{theoretical_compression:.2f}x"
                }
                
                all_results.append(result)
            
            print(f"✅ {optimizer} processing completed")
            
        except Exception as e:
            print(f"❌ Error processing {optimizer}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv = 'lns_optimizers_quantization_results.csv'
        results_df.to_csv(results_csv, index=False)
        print(f"\n💾 Complete results saved to: {results_csv}")
        
        # Save LNS4 and LNS8 results separately
        for bits in [4, 8]:
            bits_results = results_df[results_df['Quantization_Bits'] == bits].copy()
            if not bits_results.empty:
                bits_csv = f'csv/lns{bits}_optimizers_quantization_results.csv'
                os.makedirs('csv', exist_ok=True)
                bits_results.to_csv(bits_csv, index=False)
                print(f"💾 LNS{bits} results saved to: {bits_csv}")
        
        # Print summary table
        print("\n" + "="*120)
        print("📊 LNS quantization results summary (excluding LNS_Madam)")
        print("="*120)
        
        # Group by bit number and display results
        for bits in [4, 8]:
            bits_results = results_df[results_df['Quantization_Bits'] == bits]
            if not bits_results.empty:
                print(f"\n🔥 LNS{bits} quantization results:")
                print("-" * 100)
                display_cols = ['Optimizer', 'Learning_Rate', 'Weight_Decay', 'FP32_Accuracy', 
                               f'LNS{bits}_Accuracy', 'Accuracy_Loss', 'Compression_Ratio', 'Theoretical_Compression']
                print(bits_results[display_cols].to_string(index=False))
        
        # Statistics
        print(f"\n📈 Statistics:")
        print("-" * 50)
        
        for bits in [4, 8]:
            bits_results = results_df[results_df['Quantization_Bits'] == bits]
            if not bits_results.empty:
                # Calculate average accuracy loss and compression ratio
                acc_losses = []
                compressions = []
                theoretical_compressions = []
                
                for _, row in bits_results.iterrows():
                    acc_loss_str = row['Accuracy_Loss']
                    comp_str = row['Compression_Ratio']
                    theo_comp_str = row['Theoretical_Compression']
                    
                    acc_losses.append(float(acc_loss_str.replace('%', '').replace('+', '')))
                    compressions.append(float(comp_str.replace('x', '')))
                    theoretical_compressions.append(float(theo_comp_str.replace('x', '')))
                
                avg_acc_loss = np.mean(acc_losses)
                avg_compression = np.mean(compressions)
                avg_theoretical_compression = np.mean(theoretical_compressions)
                
                print(f"  LNS{bits}: Average accuracy loss {avg_acc_loss:+.2f}%")
                print(f"         Average compression ratio {avg_compression:.2f}x")
                print(f"         Average theoretical compression {avg_theoretical_compression:.2f}x")
        
    else:
        print("❌ No models were successfully quantized")
    
    print("\n🎉 LNS quantization experiment completed!")
    print("=" * 80)

if __name__ == '__main__':
    main()