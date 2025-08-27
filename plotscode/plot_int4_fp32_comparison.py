import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_int4_data():
    """Load INT4 symmetric quantization data from CSV file"""
    
    # Load INT4 symmetric data
    int4_df = pd.read_csv('csv/int4_symmetric_quantization_results.csv')
    
    # Clean percentage columns
    for col in int4_df.columns:
        if int4_df[col].dtype == 'object' and int4_df[col].str.contains('%').any():
            int4_df[col] = int4_df[col].str.replace('%', '').astype(float)
    
    # Extract INT4 data
    int4_data = []
    for _, row in int4_df.iterrows():
        if pd.notna(row['INT4_Accuracy']):
            int4_data.append({
                'Optimizer': row['Optimizer'],
                'FP32_Accuracy': row['FP32_Accuracy'],
                'INT4_Accuracy': row['INT4_Accuracy'],
                'Accuracy_Loss': row['Accuracy_Loss'],
                'FP32_Size_MB': row['FP32_Size_MB'],
                'INT4_Size_MB': row['INT4_Size_MB'],
                'True_Memory_MB': row['True_Memory_MB'],
                'True_Compression': float(row['True_Compression'].replace('x', ''))
            })
    
    return pd.DataFrame(int4_data)

def create_int4_fp32_comparison_plots():
    """Create INT4 symmetric vs FP32 comparison plots"""
    
    # Create output directory
    os.makedirs('int4_quantization_comparison_plots', exist_ok=True)
    
    # Load data
    int4_df = load_int4_data()
    
    # Create comparison plots for INT4
    create_method_comparison_plots(int4_df, 'INT4', 'INT4_Accuracy')

def create_method_comparison_plots(df, method_name, accuracy_col):
    """Create comparison plots for INT4 symmetric vs FP32"""
    
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
    plt.savefig(f'int4_quantization_comparison_plots/{method_name}_vs_fp32_accuracy.png', dpi=300, bbox_inches='tight')
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
        
        # Add compression ratio above INT4 bars
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
    plt.savefig(f'int4_quantization_comparison_plots/{method_name}_vs_fp32_memory.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'int4_quantization_comparison_plots/{method_name}_vs_fp32_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary Statistics
    print(f"\n📊 {method_name} vs FP32 Summary:")
    print("-" * 40)
    print(f"Average FP32 Accuracy: {df['FP32_Accuracy'].mean():.2f}%")
    print(f"Average {method_name} Accuracy: {df[accuracy_col].mean():.2f}%")
    print(f"Average Accuracy Loss: {df['Accuracy_Loss'].mean():.2f}%")
    print(f"Average Compression Ratio: {df['True_Compression'].mean():.2f}x")
    print(f"Average FP32 Efficiency: {df['FP32_Efficiency'].mean():.0f}")
    print(f"Average {method_name} Efficiency: {df['Quantized_Efficiency'].mean():.0f}")
    
    # Save results to CSV
    results_df = df.copy()
    results_df[f'{method_name}_Memory_MB'] = quantized_memory
    results_df.to_csv(f'int4_quantization_comparison_plots/{method_name}_vs_fp32_results.csv', index=False)

if __name__ == '__main__':
    print("🔍 Creating INT4 symmetric vs FP32 comparison plots...")
    create_int4_fp32_comparison_plots()
    print("✅ INT4 quantization comparison completed!") 