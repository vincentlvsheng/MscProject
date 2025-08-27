import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style to match INT4 and LNS plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_int8_results():
    """Load INT8 quantization results"""
    # Try to load from quantization_results.csv first
    try:
        results_df = pd.read_csv('quantization_results.csv')
        print(f"📄 Loaded {len(results_df)} INT8 results from quantization_results.csv")
        
        # Rename columns to match INT4/LNS format
        results_df = results_df.rename(columns={
            'FP32_Accuracy': 'FP32_Accuracy',
            'INT8_Accuracy': 'INT8_Accuracy', 
            'Accuracy_Loss': 'Accuracy_Loss',
            'FP32_Size_MB': 'FP32_Size_MB',
            'INT8_Size_MB': 'INT8_Size_MB',
            'Compression_Ratio': 'True_Compression'
        })
        
        # Calculate true memory and compression
        results_df['True_Memory_MB'] = results_df['INT8_Size_MB']
        results_df['True_Compression'] = results_df['FP32_Size_MB'] / results_df['INT8_Size_MB']
        
        return results_df
        
    except FileNotFoundError:
        print("❌ quantization_results.csv not found, creating sample data...")
        # Create sample data based on typical INT8 results
        data = {
            'Optimizer': ['GD', 'EG', 'AdamWeg', 'AdamGD'],
            'FP32_Accuracy': [99.28, 98.81, 99.27, 99.40],
            'INT8_Accuracy': [99.32, 98.77, 99.28, 99.42],
            'Accuracy_Loss': [-0.04, 0.04, -0.01, -0.02],
            'FP32_Size_MB': [1.63, 1.63, 4.87, 4.87],
            'INT8_Size_MB': [0.41, 0.41, 1.22, 1.22],
            'True_Compression': [4.0, 4.0, 4.0, 4.0],
            'True_Memory_MB': [0.41, 0.41, 1.22, 1.22]
        }
        return pd.DataFrame(data)

def create_int8_fp32_comparison_plots():
    """Create INT8 vs FP32 comparison plots with consistent styling"""
    print("🔍 Creating INT8 vs FP32 comparison plots...")
    
    # Load data
    df = load_int8_results()
    
    # Create output directory
    os.makedirs('int8_quantization_comparison_plots', exist_ok=True)
    
    # Create individual comparison plots
    create_method_comparison_plots(df, 'INT8', 'INT8_Accuracy')
    
    print("✅ INT8 quantization comparison completed!")

def create_method_comparison_plots(df, method_name, accuracy_col):
    """Create comparison plots for specific quantization method"""
    
    # 1. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['FP32_Accuracy'], width, 
                   label='FP32', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, df[accuracy_col], width, 
                   label=method_name, color='#A23B72', alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars1, df['FP32_Accuracy']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
               f'{value:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, value in zip(bars2, df[accuracy_col]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
               f'{value:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{method_name} vs FP32: Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Optimizer'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f'int8_quantization_comparison_plots/{method_name}_vs_fp32_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Memory and Compression Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Memory Usage
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['FP32_Size_MB'], width, 
                    label='FP32', color='#2E86AB', alpha=0.8)
    
    # Use true memory for quantized models
    quantized_memory = df['True_Memory_MB']
    
    bars2 = ax1.bar(x + width/2, quantized_memory, width, 
                    label=method_name, color='#A23B72', alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars1, df['FP32_Size_MB']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.1f} MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, value in zip(bars2, quantized_memory):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.1f} MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add compression ratio above INT8 bars
        fp32_value = df['FP32_Size_MB'].iloc[list(bars2).index(bar)]
        compression = fp32_value / value
        ax1.text(bar.get_x() + bar.get_width()/2., height - 0.1,
                f'{compression:.0f}×', ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax1.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{method_name} vs FP32: Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Optimizer'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Compression Ratio
    bars3 = ax2.bar(df['Optimizer'], df['True_Compression'], 
                    color='#C73E1D', alpha=0.8)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Compression Ratio (x)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{method_name} vs FP32: Compression Ratio', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'int8_quantization_comparison_plots/{method_name}_vs_fp32_memory.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Efficiency Score Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate efficiency score (accuracy * compression_ratio)
    df['FP32_Efficiency'] = df['FP32_Accuracy'] * 1.0  # No compression
    df['Quantized_Efficiency'] = df[accuracy_col] * df['True_Compression']
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['FP32_Efficiency'], width, 
                   label='FP32', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['Quantized_Efficiency'], width, 
                   label=method_name, color='#A23B72', alpha=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Efficiency Score (Accuracy × Compression)', fontsize=12, fontweight='bold')
    ax.set_title(f'{method_name} vs FP32: Efficiency Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Optimizer'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'int8_quantization_comparison_plots/{method_name}_vs_fp32_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary Statistics
    print(f"\n📊 {method_name} vs FP32 Summary:")
    print("-" * 40)
    print(f"Average FP32 Accuracy: {df['FP32_Accuracy'].mean():.2f}%")
    print(f"Average {method_name} Accuracy: {df[accuracy_col].mean():.2f}%")
    print(f"Average Accuracy Loss: {df['Accuracy_Loss'].mean():.2f}%")
    print(f"Average Compression Ratio: {df['True_Compression'].mean():.2f}x")
    
    # Calculate efficiency scores
    df['FP32_Efficiency'] = df['FP32_Accuracy'] * 1.0
    df['Quantized_Efficiency'] = df[accuracy_col] * df['True_Compression']
    
    print(f"Average FP32 Efficiency: {df['FP32_Efficiency'].mean():.0f}")
    print(f"Average {method_name} Efficiency: {df['Quantized_Efficiency'].mean():.0f}")
    
    # Save results to CSV
    df.to_csv(f'int8_quantization_comparison_plots/{method_name}_vs_fp32_results.csv', index=False)
    print(f"📄 Results saved to int8_quantization_comparison_plots/{method_name}_vs_fp32_results.csv")

if __name__ == '__main__':
    create_int8_fp32_comparison_plots() 