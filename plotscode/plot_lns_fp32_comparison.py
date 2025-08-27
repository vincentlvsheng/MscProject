import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_lns_data():
    """Load LNS4 and LNS8 data from CSV files"""
    
    # Load LNS4 data
    lns4_df = pd.read_csv('csv/lns4_optimizers_quantization_results.csv')
    lns8_df = pd.read_csv('csv/lns8_optimizers_quantization_results.csv')
    
    # Clean percentage columns
    for df in [lns4_df, lns8_df]:
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.contains('%').any():
                df[col] = df[col].str.replace('%', '').astype(float)
    
    # Extract LNS4 data
    lns4_data = []
    for _, row in lns4_df.iterrows():
        if pd.notna(row['LNS4_Accuracy']):
            lns4_data.append({
                'Optimizer': row['Optimizer'],
                'FP32_Accuracy': row['FP32_Accuracy'],
                'LNS4_Accuracy': row['LNS4_Accuracy'],
                'Accuracy_Loss': row['Accuracy_Loss'],
                'FP32_Size_MB': row['FP32_Size_MB'],
                'LNS4_Size_MB': row['LNS4_Size_MB'],
                'Theoretical_Compression': float(row['Theoretical_Compression'].replace('x', ''))
            })
    
    # Extract LNS8 data
    lns8_data = []
    for _, row in lns8_df.iterrows():
        if pd.notna(row['LNS8_Accuracy']):
            lns8_data.append({
                'Optimizer': row['Optimizer'],
                'FP32_Accuracy': row['FP32_Accuracy'],
                'LNS8_Accuracy': row['LNS8_Accuracy'],
                'Accuracy_Loss': row['Accuracy_Loss'],
                'FP32_Size_MB': row['FP32_Size_MB'],
                'LNS8_Size_MB': row['LNS8_Size_MB'],
                'Theoretical_Compression': float(row['Theoretical_Compression'].replace('x', ''))
            })
    
    return pd.DataFrame(lns4_data), pd.DataFrame(lns8_data)

def create_lns_fp32_comparison_plots():
    """Create LNS4, LNS8 vs FP32 comparison plots"""
    
    # Create output directory
    os.makedirs('lns_quantization_comparison_plots', exist_ok=True)
    
    # Load data
    lns4_df, lns8_df = load_lns_data()
    
    # Create comparison plots for LNS4
    create_method_comparison_plots(lns4_df, 'LNS4', 'LNS4_Accuracy')
    
    # Create comparison plots for LNS8
    create_method_comparison_plots(lns8_df, 'LNS8', 'LNS8_Accuracy')

def create_method_comparison_plots(df, method_name, accuracy_col):
    """Create comparison plots for a specific LNS method vs FP32"""
    
    if df.empty:
        print(f"❌ No data available for {method_name}")
        return
    
    # 1. Accuracy Comparison Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: FP32 vs Quantized Accuracy
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['FP32_Accuracy'], width, 
                    label='FP32', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, df[accuracy_col], width, 
                    label=method_name, color='#A23B72', alpha=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{method_name} vs FP32: Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Optimizer'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Right plot: Accuracy Loss
    bars3 = ax2.bar(df['Optimizer'], df['Accuracy_Loss'], 
                    color='#F18F01', alpha=0.8)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:+.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Loss (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{method_name} vs FP32: Accuracy Loss', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'lns_quantization_comparison_plots/{method_name}_vs_fp32_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Memory and Compression Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Memory Usage
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['FP32_Size_MB'], width, 
                    label='FP32', color='#2E86AB', alpha=0.8)
    
    # Use true memory for quantized models
    if method_name == 'LNS4':
        quantized_memory = df['LNS4_Size_MB']
    else:  # LNS8
        quantized_memory = df['LNS8_Size_MB']
    
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
        
        # Add compression ratio above LNS bars
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
    bars3 = ax2.bar(df['Optimizer'], df['Theoretical_Compression'], 
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
    plt.savefig(f'lns_quantization_comparison_plots/{method_name}_vs_fp32_memory.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Efficiency Score Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate efficiency score (accuracy * compression_ratio)
    df['FP32_Efficiency'] = df['FP32_Accuracy'] * 1.0  # No compression
    df['Quantized_Efficiency'] = df[accuracy_col] * df['Theoretical_Compression']
    
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
    plt.savefig(f'lns_quantization_comparison_plots/{method_name}_vs_fp32_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary Statistics
    print(f"\n📊 {method_name} vs FP32 Summary:")
    print("-" * 40)
    print(f"Average FP32 Accuracy: {df['FP32_Accuracy'].mean():.2f}%")
    print(f"Average {method_name} Accuracy: {df[accuracy_col].mean():.2f}%")
    print(f"Average Accuracy Loss: {df['Accuracy_Loss'].mean():.2f}%")
    print(f"Average Compression Ratio: {df['Theoretical_Compression'].mean():.2f}x")
    print(f"Average FP32 Efficiency: {df['FP32_Efficiency'].mean():.0f}")
    print(f"Average {method_name} Efficiency: {df['Quantized_Efficiency'].mean():.0f}")
    
    # Save results to CSV
    results_df = df.copy()
    results_df[f'{method_name}_Memory_MB'] = quantized_memory
    results_df.to_csv(f'lns_quantization_comparison_plots/{method_name}_vs_fp32_results.csv', index=False)

if __name__ == '__main__':
    print("🔍 Creating LNS4 and LNS8 vs FP32 comparison plots...")
    create_lns_fp32_comparison_plots()
    print("✅ LNS quantization comparison completed!") 