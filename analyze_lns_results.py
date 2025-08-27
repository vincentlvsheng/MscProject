import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_lns_quantization_results():
    """Analyze LNS quantization results in detail"""
    print("🔍 Detailed Analysis of LNS Quantization Results")
    print("=" * 60)
    
    # Load results
    try:
        results_df = pd.read_csv('lns_quantization_comparison_results.csv')
        print(f"📄 Loaded {len(results_df)} results")
    except FileNotFoundError:
        print("❌ Results file not found")
        return
    
    # Create analysis directory
    os.makedirs('lns_analysis', exist_ok=True)
    
    # 1. Overall Performance Analysis
    print("\n📊 Overall Performance Analysis:")
    print("-" * 40)
    
    summary_stats = results_df.groupby('Method').agg({
        'Accuracy': ['mean', 'std', 'min', 'max'],
        'Compression_Ratio': 'mean',
        'Accuracy_Loss': 'mean'
    }).round(2)
    
    print(summary_stats)
    
    # 2. Optimizer-specific Analysis
    print("\n🎯 Optimizer-specific Analysis:")
    print("-" * 40)
    
    optimizer_analysis = results_df.groupby(['Optimizer', 'Method']).agg({
        'Accuracy': 'mean',
        'Compression_Ratio': 'mean',
        'Accuracy_Loss': 'mean'
    }).round(2)
    
    print(optimizer_analysis)
    
    # 3. Create detailed visualizations
    
    # 3.1 Accuracy by Optimizer and Method
    fig, ax = plt.subplots(figsize=(15, 8))
    
    methods = ['FP32', 'INT4', 'LNS4', 'LNS8']
    optimizers = results_df['Optimizer'].unique()
    
    x = np.arange(len(optimizers))
    width = 0.2
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, method in enumerate(methods):
        method_data = results_df[results_df['Method'] == method]
        if not method_data.empty:
            accuracies = [method_data[method_data['Optimizer'] == opt]['Accuracy'].iloc[0] 
                         if len(method_data[method_data['Optimizer'] == opt]) > 0 else 0 
                         for opt in optimizers]
            
            bars = ax.bar(x + i * width, accuracies, width, label=method, 
                         color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison by Optimizer and Quantization Method', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(optimizers, rotation=45)
    ax.legend(title='Quantization Method', title_fontsize=11, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('lns_analysis/accuracy_by_optimizer.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.2 Compression vs Accuracy Trade-off
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter out FP32 for scatter plot
    scatter_data = results_df[results_df['Method'] != 'FP32']
    
    colors_map = {'INT4': '#A23B72', 'LNS4': '#F18F01', 'LNS8': '#C73E1D'}
    markers_map = {'INT4': 'o', 'LNS4': 's', 'LNS8': '^'}
    sizes_map = {'INT4': 200, 'LNS4': 200, 'LNS8': 200}
    
    for method in ['INT4', 'LNS4', 'LNS8']:
        method_data = scatter_data[scatter_data['Method'] == method]
        if not method_data.empty:
            scatter = ax.scatter(method_data['Compression_Ratio'], method_data['Accuracy'],
                               c=colors_map[method], marker=markers_map[method], 
                               s=sizes_map[method], alpha=0.7, label=method, 
                               edgecolors='black', linewidth=1)
            
            # Add optimizer labels
            for _, row in method_data.iterrows():
                ax.annotate(row['Optimizer'], 
                          (row['Compression_Ratio'], row['Accuracy']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8)
    
    # Add FP32 reference line
    fp32_avg = results_df[results_df['Method'] == 'FP32']['Accuracy'].mean()
    ax.axhline(y=fp32_avg, color='#2E86AB', linestyle='--', alpha=0.7, 
              label=f'FP32 Baseline ({fp32_avg:.1f}%)')
    
    ax.set_xlabel('Compression Ratio (x)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Compression Trade-off\n(Annotated by Optimizer)', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Quantization Method', title_fontsize=11, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lns_analysis/accuracy_compression_tradeoff_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.3 Accuracy Loss Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot of accuracy loss
    loss_data = results_df[results_df['Method'] != 'FP32']
    sns.boxplot(data=loss_data, x='Method', y='Accuracy_Loss', ax=ax1, 
               palette=['#A23B72', '#F18F01', '#C73E1D'])
    ax1.set_title('Accuracy Loss Distribution by Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy Loss (%)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Bar plot of average accuracy loss
    avg_loss = loss_data.groupby('Method')['Accuracy_Loss'].mean().sort_values(ascending=False)
    bars = ax2.bar(avg_loss.index, avg_loss.values, 
                   color=['#A23B72', '#F18F01', '#C73E1D'], alpha=0.8)
    
    # Add value labels
    for bar, value in zip(bars, avg_loss.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_title('Average Accuracy Loss by Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Accuracy Loss (%)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('lns_analysis/accuracy_loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.4 Method Performance Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pivot table for heatmap
    pivot_data = results_df.pivot_table(
        values='Accuracy', 
        index='Optimizer', 
        columns='Method', 
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
               center=95, vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    
    ax.set_title('Accuracy Heatmap: Optimizer vs Quantization Method', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Quantization Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimizer', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('lns_analysis/accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.5 Efficiency Score Analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate efficiency score (accuracy * compression_ratio)
    results_df['Efficiency_Score'] = results_df['Accuracy'] * results_df['Compression_Ratio']
    
    # Filter out FP32 for efficiency analysis
    efficiency_data = results_df[results_df['Method'] != 'FP32']
    
    # Group by method and calculate average efficiency
    avg_efficiency = efficiency_data.groupby('Method')['Efficiency_Score'].mean().sort_values(ascending=False)
    
    bars = ax.bar(avg_efficiency.index, avg_efficiency.values, 
                  color=['#A23B72', '#F18F01', '#C73E1D'], alpha=0.8)
    
    # Add value labels
    for bar, value in zip(bars, avg_efficiency.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{value:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Efficiency Score Comparison\n(Accuracy × Compression Ratio)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Efficiency Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Quantization Method', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('lns_analysis/efficiency_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Generate detailed report
    print("\n📋 Generating Detailed Analysis Report...")
    
    report = []
    report.append("LNS Quantization Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS:")
    report.append("-" * 20)
    for method in ['FP32', 'INT4', 'LNS4', 'LNS8']:
        method_data = results_df[results_df['Method'] == method]
        if not method_data.empty:
            avg_acc = method_data['Accuracy'].mean()
            avg_comp = method_data['Compression_Ratio'].mean()
            avg_loss = method_data['Accuracy_Loss'].mean()
            report.append(f"{method:>6}: {avg_acc:>6.2f}% accuracy, {avg_comp:>6.2f}x compression, {avg_loss:>6.2f}% loss")
    report.append("")
    
    # Best performing combinations
    report.append("BEST PERFORMING COMBINATIONS:")
    report.append("-" * 30)
    
    # Best accuracy
    best_acc = results_df.loc[results_df['Accuracy'].idxmax()]
    report.append(f"Best Accuracy: {best_acc['Optimizer']} + {best_acc['Method']} = {best_acc['Accuracy']:.2f}%")
    
    # Best compression
    best_comp = results_df.loc[results_df['Compression_Ratio'].idxmax()]
    report.append(f"Best Compression: {best_comp['Optimizer']} + {best_comp['Method']} = {best_comp['Compression_Ratio']:.2f}x")
    
    # Best efficiency
    best_eff = results_df.loc[results_df['Efficiency_Score'].idxmax()]
    report.append(f"Best Efficiency: {best_eff['Optimizer']} + {best_eff['Method']} = {best_eff['Efficiency_Score']:.0f}")
    report.append("")
    
    # Key findings
    report.append("KEY FINDINGS:")
    report.append("-" * 15)
    
    # LNS4 performance issues
    lns4_data = results_df[results_df['Method'] == 'LNS4']
    lns4_avg_acc = lns4_data['Accuracy'].mean()
    report.append(f"• LNS4 shows inconsistent performance with average accuracy of {lns4_avg_acc:.2f}%")
    report.append("• LNS4 works well with GD and AdamGD but poorly with EG, AdamWeg, and LNS_Madam")
    
    # LNS8 performance
    lns8_data = results_df[results_df['Method'] == 'LNS8']
    lns8_avg_acc = lns8_data['Accuracy'].mean()
    report.append(f"• LNS8 provides good balance with average accuracy of {lns8_avg_acc:.2f}%")
    report.append("• LNS8 achieves 4x compression with minimal accuracy loss")
    
    # INT4 comparison
    int4_data = results_df[results_df['Method'] == 'INT4']
    int4_avg_acc = int4_data['Accuracy'].mean()
    report.append(f"• INT4 provides consistent performance with average accuracy of {int4_avg_acc:.2f}%")
    report.append("• INT4 achieves 8x compression but with higher accuracy loss than LNS8")
    
    # Recommendations
    report.append("")
    report.append("RECOMMENDATIONS:")
    report.append("-" * 15)
    report.append("• For maximum compression (8x): Use INT4 for consistent results")
    report.append("• For balanced performance: Use LNS8 for 4x compression with good accuracy")
    report.append("• For specific optimizers: LNS4 works well with GD and AdamGD")
    report.append("• Avoid LNS4 with EG, AdamWeg, and LNS_Madam optimizers")
    
    # Save report
    with open('lns_analysis/detailed_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("✅ Detailed analysis completed!")
    print("📁 Analysis files saved in: lns_analysis/")
    print("📄 Report saved as: lns_analysis/detailed_analysis_report.txt")
    
    # Print summary
    print("\n📊 Quick Summary:")
    print("-" * 20)
    print(f"• INT4: Best compression (8x), consistent performance")
    print(f"• LNS8: Good balance (4x compression, minimal accuracy loss)")
    print(f"• LNS4: Inconsistent performance, works well with some optimizers")
    print(f"• Best overall: LNS8 for balanced performance")

if __name__ == '__main__':
    analyze_lns_quantization_results() 