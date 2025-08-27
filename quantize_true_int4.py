import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from CnnModel import CNNModel
from true_int4_quantizer import TrueINT4Quantizer, convert_cnn_to_int4_quantized, calculate_int4_memory_usage
import platform

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def simple_int4_quantize_model(fp32_model, symmetric=True):
    
    print(f"🔄 Simple INT4 quantization ({'symmetric' if symmetric else 'asymmetric'}) of model parameters...")
    
    # Create quantizer
    quantizer = TrueINT4Quantizer(symmetric=symmetric)
    
    # Copy model
    quantized_model = CNNModel(num_classes=10)
    quantized_model.load_state_dict(fp32_model.state_dict())
    
    # Store quantization information
    quantized_params = {}
    
    # Quantize all weight parameters
    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if 'weight' in name or 'bias' in name:
                print(f"  🔧 Quantizing {name}: {param.shape}")
                
                # Quantize parameters
                packed_data, original_shape = quantizer.quantize_tensor_int4(param.data)
                quantized_params[name] = (packed_data, original_shape)
                
                # Immediately dequantize to get quantized parameters
                quantized_param = quantizer.dequantize_tensor_int4(packed_data, original_shape)
                
                # Replace original parameters
                param.data = quantized_param
                
                # Calculate quantization effect
                unique_values = len(torch.unique(quantized_param.view(-1)))
                print(f"    -> {unique_values} unique values")
    
    # Save quantization information to model
    quantized_model.quantized_params = quantized_params
    quantized_model.quantizer = quantizer
    
    print(f"✅ Simple INT4 ({'symmetric' if symmetric else 'asymmetric'}) quantization completed")
    return quantized_model

def get_dataloaders(batch_size=128):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

def evaluate_model(model, test_loader, device='cpu'):
    
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
    
    return os.path.getsize(model_path) / (1024 * 1024)

def analyze_quantization_distribution(model, model_name):
    
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

def calculate_true_int4_memory_usage(model):
    
    if not hasattr(model, 'quantized_params'):
        return calculate_int4_memory_usage(model)
    
    total_memory = 0
    
    for name, (packed_data, original_shape) in model.quantized_params.items():
        # Each packed uint8 stores two 4-bit values
        memory_bytes = packed_data['packed_data'].numel()
        total_memory += memory_bytes
        
        # Add memory for quantization parameters (scale and zero_point)
        total_memory += 8  # two float32
    
    return total_memory

def main():
    
    print("🚀 Starting True INT4 quantization of best MNIST models")
    print(f"💻 Platform: {platform.system()}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print("=" * 80)
    
    # Set device
    device = torch.device('cpu')  # INT4 quantization on CPU
    
    # Get data
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
    
    # Create result directories
    for mode in ['symmetric', 'asymmetric']:
        os.makedirs(f'checkpoints_quantized_int4_{mode}', exist_ok=True)
    
    # Quantization mode list
    quantization_modes = ['symmetric', 'asymmetric']
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
            
            # For each quantization mode
            for mode in quantization_modes:
                symmetric = (mode == 'symmetric')
                print(f"\n--- INT4 {mode} quantization experiment ---")
                
                # Use simple INT4 quantization
                model_int4 = simple_int4_quantize_model(model_fp32, symmetric=symmetric)
                int4_accuracy = evaluate_model(model_int4, test_loader, device)
                print(f"✅ INT4 {mode} accuracy: {int4_accuracy:.2f}%")
                
                # Analyze quantization distribution
                analyze_quantization_distribution(model_int4, f"{optimizer}_INT4_{mode}")
                
                # Save quantized model
                save_dir = f'checkpoints_quantized_int4_{mode}/lr{lr}_wd{wd}_mnist'
                os.makedirs(save_dir, exist_ok=True)
                
                save_path = f'{save_dir}/{name_map[optimizer]}_int4_{mode}.pth'
                torch.save(model_int4.state_dict(), save_path)
                print(f"💾 Model saved: {save_path}")
                
                # Calculate model size and compression ratio
                int4_size = get_model_size_mb(save_path)
                true_memory = calculate_true_int4_memory_usage(model_int4) / (1024 * 1024)  # MB
                compression_ratio = fp32_size / int4_size
                true_compression = (fp32_size * 1024 * 1024) / (true_memory * 1024 * 1024)
                accuracy_loss = fp32_accuracy - int4_accuracy
                
                print(f"📏 Model size: {fp32_size:.2f} MB → {int4_size:.2f} MB")
                print(f"📏 True memory usage: {true_memory:.2f} MB")
                print(f"📈 Compression ratio: {compression_ratio:.2f}x")
                print(f"📈 True compression: {true_compression:.2f}x")
                print(f"📉 Accuracy loss: {accuracy_loss:+.2f}%")
                
                # Save results
                result = {
                    'Optimizer': optimizer,
                    'Learning_Rate': lr,
                    'Weight_Decay': wd,
                    'Quantization_Mode': mode,
                    'Original_Val_Acc': row['Val Acc'],
                    'Original_Test_Acc': row['Test Acc'],
                    'FP32_Accuracy': f"{fp32_accuracy:.2f}%",
                    'INT4_Accuracy': f"{int4_accuracy:.2f}%",
                    'Accuracy_Loss': f"{accuracy_loss:+.2f}%",
                    'FP32_Size_MB': f"{fp32_size:.2f}",
                    'INT4_Size_MB': f"{int4_size:.2f}",
                    'True_Memory_MB': f"{true_memory:.2f}",
                    'Compression_Ratio': f"{compression_ratio:.2f}x",
                    'True_Compression': f"{true_compression:.2f}x"
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
        results_csv = 'true_int4_quantization_results.csv'
        results_df.to_csv(results_csv, index=False)
        print(f"\n💾 Complete results saved to: {results_csv}")
        
        # Save symmetric and asymmetric results
        for mode in ['symmetric', 'asymmetric']:
            mode_results = results_df[results_df['Quantization_Mode'] == mode].copy()
            if not mode_results.empty:
                mode_csv = f'csv/int4_{mode}_quantization_results.csv'
                os.makedirs('csv', exist_ok=True)
                mode_results.to_csv(mode_csv, index=False)
                print(f"💾 INT4 {mode} results saved to: {mode_csv}")
        
        # Print summary table
        print("\n" + "="*120)
        print("📊 True INT4 quantization results summary (excluding LNS_Madam)")
        print("="*120)
        
        # Group by mode and display results
        for mode in ['symmetric', 'asymmetric']:
            mode_results = results_df[results_df['Quantization_Mode'] == mode]
            if not mode_results.empty:
                print(f"\n🔥 INT4 {mode} quantization results:")
                print("-" * 100)
                display_cols = ['Optimizer', 'Learning_Rate', 'Weight_Decay', 'FP32_Accuracy', 
                               'INT4_Accuracy', 'Accuracy_Loss', 'Compression_Ratio', 'True_Compression']
                print(mode_results[display_cols].to_string(index=False))
        
        # Statistics
        print(f"\n📈 Statistics:")
        print("-" * 50)
        
        for mode in ['symmetric', 'asymmetric']:
            mode_results = results_df[results_df['Quantization_Mode'] == mode]
            if not mode_results.empty:
                # Calculate average accuracy loss and compression ratio
                acc_losses = []
                compressions = []
                true_compressions = []
                
                for _, row in mode_results.iterrows():
                    acc_loss_str = row['Accuracy_Loss']
                    comp_str = row['Compression_Ratio']
                    true_comp_str = row['True_Compression']
                    
                    acc_losses.append(float(acc_loss_str.replace('%', '').replace('+', '')))
                    compressions.append(float(comp_str.replace('x', '')))
                    true_compressions.append(float(true_comp_str.replace('x', '')))
                
                avg_acc_loss = np.mean(acc_losses)
                avg_compression = np.mean(compressions)
                avg_true_compression = np.mean(true_compressions)
                
                print(f"  INT4 {mode}: Average accuracy loss {avg_acc_loss:+.2f}%")
                print(f"                Average compression ratio {avg_compression:.2f}x")
                print(f"                Average true compression {avg_true_compression:.2f}x")
        
        # Compare symmetric vs asymmetric
        symmetric_results = results_df[results_df['Quantization_Mode'] == 'symmetric']
        asymmetric_results = results_df[results_df['Quantization_Mode'] == 'asymmetric']
        
        if not symmetric_results.empty and not asymmetric_results.empty:
            print(f"\n📊 Symmetric vs Asymmetric comparison:")
            print("-" * 50)
            
            # Calculate symmetric vs asymmetric performance for each optimizer
            for optimizer in symmetric_results['Optimizer'].unique():
                sym_matches = symmetric_results[symmetric_results['Optimizer'] == optimizer]
                asym_matches = asymmetric_results[asymmetric_results['Optimizer'] == optimizer]
                
                if not sym_matches.empty and not asym_matches.empty:
                    sym_row = sym_matches.iloc[0]
                    asym_row = asym_matches.iloc[0]
                    
                    sym_acc = float(sym_row['INT4_Accuracy'].replace('%', ''))
                    asym_acc = float(asym_row['INT4_Accuracy'].replace('%', ''))
                    
                    better_mode = 'Symmetric' if sym_acc > asym_acc else 'Asymmetric'
                    acc_diff = abs(sym_acc - asym_acc)
                    
                    print(f"  {optimizer}: {better_mode} is better by {acc_diff:.2f}%")
        
    else:
        print("❌ No models were successfully quantized")
    
    print("\n🎉 True INT4 quantization experiment completed!")
    print("=" * 80)

if __name__ == '__main__':
    main()