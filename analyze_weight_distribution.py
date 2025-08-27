import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from CnnModel import CNNModel
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_model_and_data():
    """Load model and data"""   
    # Read best configurations
    best_cfg = pd.read_csv('best_configurations.csv')
    
    # Optimizer name mapping
    name_map = {
        'GD': 'gd',
        'EG': 'eg', 
        'AdamWeg': 'adamweg',
        'AdamGD': 'adamgd'
    }
    
    models_data = {}
    
    for _, row in best_cfg.iterrows():
        optimizer = row['Optimizer']
        if optimizer not in name_map:
            continue
            
        lr = row['Learning Rate']
        wd = row['Weight Decay']
        
        # Build checkpoint path
        ckpt_path = f'checkpoints/lr{lr}_wd{wd}_mnist/{name_map[optimizer]}_best_model.pth'
        
        if os.path.exists(ckpt_path):
            try:
                # Load model
                model = CNNModel(num_classes=10)
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                model.load_state_dict(state_dict)
                model.eval()
                
                models_data[optimizer] = {
                    'model': model,
                    'lr': lr,
                    'wd': wd,
                    'path': ckpt_path
                }
                
                print(f"✅ Loaded {optimizer} model (lr={lr}, wd={wd})")
                
            except Exception as e:
                print(f"❌ Error loading {optimizer}: {str(e)}")
    
    return models_data

def analyze_layer_weights(model, layer_name, weights):
    """Analyze distribution of single layer weights"""
    weights_flat = weights.flatten().detach().numpy()
    
    # Basic statistics
    stats = {
        'layer': layer_name,
        'shape': list(weights.shape),
        'total_params': weights.numel(),
        'mean': float(np.mean(weights_flat)),
        'std': float(np.std(weights_flat)),
        'min': float(np.min(weights_flat)),
        'max': float(np.max(weights_flat)),
        'median': float(np.median(weights_flat)),
        'q25': float(np.percentile(weights_flat, 25)),
        'q75': float(np.percentile(weights_flat, 75)),
        'iqr': float(np.percentile(weights_flat, 75) - np.percentile(weights_flat, 25)),
        'skewness': float(pd.Series(weights_flat).skew()),
        'kurtosis': float(pd.Series(weights_flat).kurtosis()),
        'zero_ratio': float(np.sum(weights_flat == 0) / len(weights_flat)),
        'near_zero_ratio': float(np.sum(np.abs(weights_flat) < 1e-6) / len(weights_flat)),
        'large_weights_ratio': float(np.sum(np.abs(weights_flat) > 1.0) / len(weights_flat))
    }
    
    return stats, weights_flat

def create_weight_distribution_plots(models_data):
    """Create weight distribution plots"""
    os.makedirs('weight_distribution_analysis', exist_ok=True)
    
    all_stats = []
    
    for optimizer, data in models_data.items():
        model = data['model']
        print(f"\n📊 Analyzing {optimizer} weight distributions...")
        
        # Collect statistics for all layers
        optimizer_stats = []
        
        # Count total weight layers first
        weight_layers = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_layers.append(name)
        
        total_weight_layers = len(weight_layers)
        print(f"Found {total_weight_layers} weight layers")
        
        # Create subplots: 3 rows x 4 columns for up to 12 layers
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'{optimizer} Weight Distribution Analysis (All Layers)', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        layer_idx = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                stats, weights_flat = analyze_layer_weights(model, name, param.data)
                optimizer_stats.append(stats)
                
                # Visualize all weight layers
                if layer_idx < len(axes_flat):
                    ax = axes_flat[layer_idx]
                    
                    # Histogram
                    ax.hist(weights_flat, bins=50, alpha=0.7, density=True, 
                           label=f'{name}\nμ={stats["mean"]:.3f}, σ={stats["std"]:.3f}')
                    ax.set_title(f'{name} Weight Distribution', fontsize=10)
                    ax.set_xlabel('Weight Value', fontsize=8)
                    ax.set_ylabel('Density', fontsize=8)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics text box
                    textstr = f'Min: {stats["min"]:.3f}\nMax: {stats["max"]:.3f}\n'
                    textstr += f'Q25: {stats["q25"]:.3f}\nQ75: {stats["q75"]:.3f}\n'
                    textstr += f'Zero ratio: {stats["zero_ratio"]:.3f}'
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=7,
                           verticalalignment='top', bbox=props)
                
                layer_idx += 1
        
        # Hide unused subplots
        for i in range(layer_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'weight_distribution_analysis/{optimizer}_weight_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        all_stats.extend(optimizer_stats)
    
    # Save statistics
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv('weight_distribution_analysis/weight_statistics.csv', index=False)
    
    return stats_df

def create_comparison_plots(stats_df):
    """Create comparison plots between optimizers"""
    
    # 1. Weight range comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Group by optimizer
    optimizers = stats_df['layer'].str.extract(r'(\w+)\.')[0].unique()
    
    for i, optimizer in enumerate(optimizers):
        opt_stats = stats_df[stats_df['layer'].str.contains(optimizer)]
        
        # Weight range
        ax1.scatter(opt_stats['std'], opt_stats['max'] - opt_stats['min'], 
                   label=optimizer, alpha=0.7, s=100)
        
        # Skewness and kurtosis
        ax2.scatter(opt_stats['skewness'], opt_stats['kurtosis'], 
                   label=optimizer, alpha=0.7, s=100)
        
        # Zero ratio
        ax3.scatter(opt_stats['zero_ratio'], opt_stats['near_zero_ratio'], 
                   label=optimizer, alpha=0.7, s=100)
        
        # Large weights ratio
        ax4.scatter(opt_stats['large_weights_ratio'], opt_stats['iqr'], 
                   label=optimizer, alpha=0.7, s=100)
    
    ax1.set_xlabel('Standard Deviation')
    ax1.set_ylabel('Weight Range (Max - Min)')
    ax1.set_title('Weight Range vs Standard Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Skewness')
    ax2.set_ylabel('Kurtosis')
    ax2.set_title('Distribution Shape (Skewness vs Kurtosis)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Zero Ratio')
    ax3.set_ylabel('Near Zero Ratio (< 1e-6)')
    ax3.set_title('Sparsity Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Large Weights Ratio (> 1.0)')
    ax4.set_ylabel('IQR')
    ax4.set_title('Weight Magnitude Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weight_distribution_analysis/optimizer_comparisons.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap: statistics for each layer
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select key statistics
    key_stats = ['mean', 'std', 'skewness', 'kurtosis', 'zero_ratio', 'large_weights_ratio']
    
    # Create pivot table - using simpler method
    pivot_data = stats_df.set_index('layer')[key_stats]
    
    # Standardize data for heatmap
    normalized_data = (pivot_data - pivot_data.mean()) / pivot_data.std()
    
    sns.heatmap(normalized_data, annot=True, cmap='RdYlBu_r', center=0, 
                ax=ax, cbar_kws={'label': 'Normalized Value'})
    
    ax.set_title('Normalized Weight Statistics Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Statistics')
    ax.set_ylabel('Layers')
    
    plt.tight_layout()
    plt.savefig('weight_distribution_analysis/statistics_heatmap.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

def analyze_quantization_sensitivity(stats_df):
    """Analyze quantization sensitivity"""
    
    print("\n🔍 Quantization Sensitivity Analysis:")
    print("=" * 60)
    
    # Group by optimizer
    optimizers = stats_df['layer'].str.extract(r'(\w+)\.')[0].unique()
    
    for optimizer in optimizers:
        opt_stats = stats_df[stats_df['layer'].str.contains(optimizer)]
        
        print(f"\n📊 {optimizer} Optimization:")
        print("-" * 40)
        
        # Calculate quantization sensitivity metrics
        avg_std = opt_stats['std'].mean()
        avg_range = (opt_stats['max'] - opt_stats['min']).mean()
        avg_skewness = opt_stats['skewness'].mean()
        avg_zero_ratio = opt_stats['zero_ratio'].mean()
        
        print(f"Average Standard Deviation: {avg_std:.4f}")
        print(f"Average Weight Range: {avg_range:.4f}")
        print(f"Average Skewness: {avg_skewness:.4f}")
        print(f"Average Zero Ratio: {avg_zero_ratio:.4f}")
        
        # Quantization sensitivity evaluation
        sensitivity_score = 0
        
        if avg_std > 0.5:
            sensitivity_score += 1
            print("⚠️  High weight variance - may be sensitive to quantization")
        
        if avg_range > 4.0:
            sensitivity_score += 1
            print("⚠️  Large weight range - may need wide quantization range")
        
        if abs(avg_skewness) > 1.0:
            sensitivity_score += 1
            print("⚠️  Highly skewed distribution - asymmetric quantization may help")
        
        if avg_zero_ratio > 0.1:
            sensitivity_score += 1
            print("⚠️  High sparsity - may benefit from sparse quantization")
        
        if sensitivity_score == 0:
            print("✅ Low quantization sensitivity - should quantize well")
        elif sensitivity_score <= 2:
            print("⚠️  Moderate quantization sensitivity")
        else:
            print("❌ High quantization sensitivity - may need special handling")
        
        print(f"Quantization Sensitivity Score: {sensitivity_score}/4")

def generate_summary_report(stats_df):
    """Generate summary report"""
    
    report = []
    report.append("# Weight Distribution Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    # Overall statistics
    report.append("## Overall Statistics")
    report.append("")
    report.append(f"- Total layers analyzed: {len(stats_df)}")
    report.append(f"- Average parameters per layer: {stats_df['total_params'].mean():.0f}")
    report.append(f"- Average weight standard deviation: {stats_df['std'].mean():.4f}")
    report.append(f"- Average weight range: {(stats_df['max'] - stats_df['min']).mean():.4f}")
    report.append("")
    
    # Group by optimizer
    optimizers = stats_df['layer'].str.extract(r'(\w+)\.')[0].unique()
    
    for optimizer in optimizers:
        opt_stats = stats_df[stats_df['layer'].str.contains(optimizer)]
        
        report.append(f"## {optimizer} Optimization")
        report.append("")
        report.append(f"- Layers: {len(opt_stats)}")
        report.append(f"- Average std: {opt_stats['std'].mean():.4f}")
        report.append(f"- Average range: {(opt_stats['max'] - opt_stats['min']).mean():.4f}")
        report.append(f"- Average skewness: {opt_stats['skewness'].mean():.4f}")
        report.append(f"- Average zero ratio: {opt_stats['zero_ratio'].mean():.4f}")
        report.append("")
    
    # Quantization recommendations
    report.append("## Quantization Recommendations")
    report.append("")
    
    # Based on statistics, give recommendations
    avg_std = stats_df['std'].mean()
    avg_skewness = stats_df['skewness'].mean()
    avg_zero_ratio = stats_df['zero_ratio'].mean()
    
    if avg_std < 0.3:
        report.append("- ✅ Low variance suggests good quantization potential")
    else:
        report.append("- ⚠️  High variance may require careful quantization range selection")
    
    if abs(avg_skewness) < 0.5:
        report.append("- ✅ Symmetric distribution suggests symmetric quantization may work well")
    else:
        report.append("- ⚠️  Skewed distribution suggests asymmetric quantization may be beneficial")
    
    if avg_zero_ratio > 0.05:
        report.append("- ✅ High sparsity suggests potential for sparse quantization")
    else:
        report.append("- ℹ️  Low sparsity - standard quantization should work")
    
    # Save report
    with open('weight_distribution_analysis/analysis_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("\n📋 Analysis report saved to: weight_distribution_analysis/analysis_report.md")

def main():
    """Main function"""
    print("🔍 Starting weight distribution analysis...")
    print("=" * 60)
    
    # Load model and data
    models_data = load_model_and_data()
    
    if not models_data:
        print("❌ No models loaded successfully")
        return
    
    print(f"\n✅ Loaded {len(models_data)} models for analysis")
    
    # Analyze weight distribution
    stats_df = create_weight_distribution_plots(models_data)
    
    # Create comparison plots
    create_comparison_plots(stats_df)
    
    # Analyze quantization sensitivity
    analyze_quantization_sensitivity(stats_df)
    
    # Generate summary report
    generate_summary_report(stats_df)
    
    print("\n🎉 Weight distribution analysis completed!")
    print("📁 Results saved in: weight_distribution_analysis/")
    print("=" * 60)

if __name__ == '__main__':
    main() 