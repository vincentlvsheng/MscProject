#!/usr/bin/env python3
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def parse_training_curves(log_path):
    """Parse training curve data from log files"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract learning rate and weight decay
    lr_match = re.search(r'Learning rate: ([\d.]+)', content)
    wd_match = re.search(r'Weight decay: ([\d.]+)', content)
    
    if not lr_match or not wd_match:
        return None
        
    lr = float(lr_match.group(1))
    wd = float(wd_match.group(1))
    
    training_data = {}
    
    # Extract training curves for each optimizer
    optimizers = ['gd', 'eg', 'adamweg', 'adamgd', 'lnsmadam']
    
    for opt in optimizers:
        # Find all epoch data for this optimizer
        pattern = rf'\[{opt}\] Epoch (\d+)/\d+.*?\[{opt}\] Train Loss: ([\d.]+), Train Acc: ([\d.]+)%.*?\[{opt}\] Val Loss: ([\d.]+), Val Acc: ([\d.]+)%'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if matches:
            epochs = []
            train_losses = []
            train_accs = []
            val_losses = []
            val_accs = []
            
            for match in matches:
                epoch, train_loss, train_acc, val_loss, val_acc = match
                epochs.append(int(epoch))
                train_losses.append(float(train_loss))
                train_accs.append(float(train_acc))
                val_losses.append(float(val_loss))
                val_accs.append(float(val_acc))
            
            training_data[opt.upper()] = {
                'lr': lr,
                'weight_decay': wd,
                'epochs': epochs,
                'train_loss': train_losses,
                'train_acc': train_accs,
                'val_loss': val_losses,
                'val_acc': val_accs
            }
    
    return training_data

def parse_log_file(log_path):
    """Parse a single log file to extract experiment results"""
    results = {}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract learning rate and weight decay
    lr_match = re.search(r'Learning rate: ([\d.]+)', content)
    wd_match = re.search(r'Weight decay: ([\d.]+)', content)
    
    if not lr_match or not wd_match:
        return None
        
    lr = float(lr_match.group(1))
    wd = float(wd_match.group(1))
    
    # Extract final results for each optimizer
    optimizers = ['GD', 'EG', 'AdamWeg', 'AdamGD', 'LNS_Madam']
    
    for opt in optimizers:
        # Find final results
        pattern = rf'{opt} - Val Acc: ([\d.]+)%, Test Acc: ([\d.]+)%'
        match = re.search(pattern, content)
        
        if match:
            val_acc = float(match.group(1))
            test_acc = float(match.group(2))
            
            results[opt] = {
                'lr': lr,
                'weight_decay': wd,
                'val_acc': val_acc,
                'test_acc': test_acc
            }
    
    return results

def collect_all_results(log_dir):
    """Collect results from all log files"""
    all_results = []
    all_training_curves = []
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.out')]
    print(f"Found {len(log_files)} log files")
    
    for log_file in log_files:
        log_path = os.path.join(log_dir, log_file)
        
        # Collect final results
        results = parse_log_file(log_path)
        if results:
            for optimizer, data in results.items():
                all_results.append({
                    'file': log_file,
                    'optimizer': optimizer,
                    'lr': data['lr'],
                    'weight_decay': data['weight_decay'],
                    'val_acc': data['val_acc'],
                    'test_acc': data['test_acc']
                })
        
        # Collect training curves
        training_curves = parse_training_curves(log_path)
        if training_curves:
            for optimizer, data in training_curves.items():
                for i, epoch in enumerate(data['epochs']):
                    all_training_curves.append({
                        'file': log_file,
                        'optimizer': optimizer,
                        'lr': data['lr'],
                        'weight_decay': data['weight_decay'],
                        'epoch': epoch,
                        'train_loss': data['train_loss'][i],
                        'train_acc': data['train_acc'][i],
                        'val_loss': data['val_loss'][i],
                        'val_acc': data['val_acc'][i]
                    })
    
    return pd.DataFrame(all_results), pd.DataFrame(all_training_curves)

def check_data_completeness(curves_df):
    """Check data completeness and find missing combinations"""
    print("\n" + "="*80)
    print("DATA COMPLETENESS CHECK")
    print("="*80)
    
    optimizers = curves_df['optimizer'].unique()
    all_lr_wd_combos = curves_df[['lr', 'weight_decay']].drop_duplicates().sort_values(['lr', 'weight_decay'])
    
    print(f"Total expected lr/wd combinations: {len(all_lr_wd_combos)}")
    print("All combinations:")
    for _, combo in all_lr_wd_combos.iterrows():
        print(f"  lr={combo['lr']}, wd={combo['weight_decay']}")
    
    for optimizer in optimizers:
        opt_data = curves_df[curves_df['optimizer'] == optimizer]
        opt_combos = opt_data[['lr', 'weight_decay']].drop_duplicates().sort_values(['lr', 'weight_decay'])
        
        print(f"\n{optimizer} optimizer:")
        print(f"  Actual combinations with data: {len(opt_combos)}")
        
        # Find missing combinations
        missing_combos = []
        for _, expected_combo in all_lr_wd_combos.iterrows():
            lr, wd = expected_combo['lr'], expected_combo['weight_decay']
            if not ((opt_combos['lr'] == lr) & (opt_combos['weight_decay'] == wd)).any():
                missing_combos.append((lr, wd))
        
        if missing_combos:
            print(f"  Missing combinations: {missing_combos}")
        else:
            print("  ✅ All combinations have data")
            
        # Check data points for each combination
        for _, combo in opt_combos.iterrows():
            lr, wd = combo['lr'], combo['weight_decay']
            combo_data = opt_data[(opt_data['lr'] == lr) & (opt_data['weight_decay'] == wd)]
            epochs = sorted(combo_data['epoch'].unique())
            print(f"  lr={lr}, wd={wd}: {len(combo_data)} data points, epochs: {epochs}")

def plot_training_curves_by_optimizer(curves_df):
    """Plot training curves for each optimizer"""
    optimizers = curves_df['optimizer'].unique()
    
    for optimizer in optimizers:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        opt_data = curves_df[curves_df['optimizer'] == optimizer]
        
        # Group by parameter combinations and sort them
        unique_configs = opt_data[['lr', 'weight_decay']].drop_duplicates().sort_values(['lr', 'weight_decay'])
        
        # Use diverse color combinations to ensure 12 combinations have different colors
        if len(unique_configs) <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_configs)))
        elif len(unique_configs) <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_configs)))
        else:
            # If more than 20, combine multiple color palettes
            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
            colors2 = plt.cm.Set3(np.linspace(0, 1, len(unique_configs) - 20))
            colors = np.concatenate([colors1, colors2])
        
        print(f"\n{optimizer} configuration statistics:")
        print(f"Total configurations: {len(unique_configs)}")
        
        # Create a dictionary to store complete data for each configuration
        config_data_dict = {}
        
        # First collect data for all configurations
        for i, (_, config) in enumerate(unique_configs.iterrows()):
            lr, wd = config['lr'], config['weight_decay']
            config_data = opt_data[(opt_data['lr'] == lr) & (opt_data['weight_decay'] == wd)]
            
            if not config_data.empty:
                config_data_dict[(lr, wd)] = {
                    'data': config_data.sort_values('epoch'),
                    'color': colors[i]
                }
                print(f"Configuration lr={lr}, wd={wd}: {len(config_data)} data points")
        
        print(f"Configurations with data: {len(config_data_dict)}")
        
        # Plot data for all configurations
        for (lr, wd), config_info in config_data_dict.items():
            config_data = config_info['data']
            color = config_info['color']
            label = f'lr={lr}, wd={wd}'
            
            # Ensure data is sorted by epoch
            config_data = config_data.sort_values('epoch')
            
            # Training loss
            ax1.plot(config_data['epoch'], config_data['train_loss'], 
                    color=color, label=label, marker='o', markersize=3)
            
            # Validation loss
            ax2.plot(config_data['epoch'], config_data['val_loss'], 
                    color=color, label=label, marker='s', markersize=3)
            
            # Training accuracy
            ax3.plot(config_data['epoch'], config_data['train_acc'], 
                    color=color, label=label, marker='o', markersize=3)
            
            # Validation accuracy
            ax4.plot(config_data['epoch'], config_data['val_acc'], 
                    color=color, label=label, marker='s', markersize=3)
        
        # Set chart properties
        ax1.set_title(f'{optimizer} - Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        ax2.set_title(f'{optimizer} - Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        ax3.set_title(f'{optimizer} - Training Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        ax4.set_title(f'{optimizer} - Validation Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.grid(True, alpha=0.3)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        plt.savefig(f'analysis_plots/{optimizer}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_learning_rate_comparison(curves_df):
    """Compare training effects under different learning rates"""
    optimizers = curves_df['optimizer'].unique()
    
    # Calculate the number of rows and columns needed
    n_optimizers = len(optimizers)
    n_cols = 3  # Use 3 columns to better accommodate 5 optimizers
    n_rows = (n_optimizers + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, optimizer in enumerate(optimizers):
        ax = axes[i]
        opt_data = curves_df[curves_df['optimizer'] == optimizer]
        
        # Only compare weight_decay=0 cases
        wd_zero_data = opt_data[opt_data['weight_decay'] == 0.0]
        
        learning_rates = sorted(wd_zero_data['lr'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(learning_rates)))
        
        for lr, color in zip(learning_rates, colors):
            lr_data = wd_zero_data[wd_zero_data['lr'] == lr]
            
            # Plot validation accuracy curves
            ax.plot(lr_data['epoch'], lr_data['val_acc'], 
                   color=color, label=f'lr={lr}', marker='o', markersize=4, linewidth=2)
        
        ax.set_title(f'{optimizer} - Learning Rate Comparison (wd=0)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(len(optimizers), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('analysis_plots/learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_weight_decay_comparison(curves_df):
    """Compare training effects under different weight decay values"""
    optimizers = curves_df['optimizer'].unique()
    
    # Calculate the number of rows and columns needed
    n_optimizers = len(optimizers)
    n_cols = 3  # Use 3 columns to better accommodate 5 optimizers
    n_rows = (n_optimizers + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, optimizer in enumerate(optimizers):
        ax = axes[i]
        opt_data = curves_df[curves_df['optimizer'] == optimizer]
        
        # Only compare lr=0.01 cases
        lr_fixed_data = opt_data[opt_data['lr'] == 0.01]
        
        if lr_fixed_data.empty:
            ax.text(0.5, 0.5, 'No data for lr=0.01', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{optimizer} - Weight Decay Comparison (lr=0.01)')
            continue
        
        weight_decays = sorted(lr_fixed_data['weight_decay'].unique())
        colors = plt.cm.plasma(np.linspace(0, 1, len(weight_decays)))
        
        for wd, color in zip(weight_decays, colors):
            wd_data = lr_fixed_data[lr_fixed_data['weight_decay'] == wd]
            
            # Plot validation accuracy curves
            ax.plot(wd_data['epoch'], wd_data['val_acc'], 
                   color=color, label=f'wd={wd}', marker='s', markersize=4, linewidth=2)
        
        ax.set_title(f'{optimizer} - Weight Decay Comparison (lr=0.01)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(len(optimizers), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('analysis_plots/weight_decay_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_bar(df):
    """Create optimizer comparison bar chart"""
    # Calculate average performance for each optimizer
    avg_performance = df.groupby('optimizer').agg({
        'val_acc': ['mean', 'std'],
        'test_acc': ['mean', 'std']
    }).round(2)
    
    avg_performance.columns = ['val_acc_mean', 'val_acc_std', 'test_acc_mean', 'test_acc_std']
    avg_performance = avg_performance.reset_index()
    
    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(avg_performance))
    width = 0.35
    
    # Validation accuracy
    bars1 = ax1.bar(x, avg_performance['val_acc_mean'], width, 
                   yerr=avg_performance['val_acc_std'], 
                   label='Validation Accuracy', capsize=5)
    ax1.set_xlabel('Optimizer')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Average Validation Accuracy by Optimizer')
    ax1.set_xticks(x)
    ax1.set_xticklabels(avg_performance['optimizer'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val, std in zip(bars1, avg_performance['val_acc_mean'], avg_performance['val_acc_std']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{val:.1f}±{std:.1f}', ha='center', va='bottom')
    
    # Test accuracy
    bars2 = ax2.bar(x, avg_performance['test_acc_mean'], width,
                   yerr=avg_performance['test_acc_std'],
                   label='Test Accuracy', capsize=5, color='orange')
    ax2.set_xlabel('Optimizer')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Average Test Accuracy by Optimizer')
    ax2.set_xticks(x)
    ax2.set_xticklabels(avg_performance['optimizer'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val, std in zip(bars2, avg_performance['test_acc_mean'], avg_performance['test_acc_std']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{val:.1f}±{std:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return avg_performance

def find_best_configurations(df):
    """Find best configuration for each optimizer"""
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS FOR EACH OPTIMIZER")
    print("="*80)
    
    best_configs = []
    
    for optimizer in df['optimizer'].unique():
        opt_data = df[df['optimizer'] == optimizer]
        
        # Sort by test accuracy
        best = opt_data.loc[opt_data['test_acc'].idxmax()]
        
        best_configs.append({
            'Optimizer': optimizer,
            'Learning Rate': best['lr'],
            'Weight Decay': best['weight_decay'],
            'Val Acc': f"{best['val_acc']:.2f}%",
            'Test Acc': f"{best['test_acc']:.2f}%"
        })
        
        print(f"\n{optimizer}:")
        print(f"  Best Configuration: lr={best['lr']}, wd={best['weight_decay']}")
        print(f"  Validation Accuracy: {best['val_acc']:.2f}%")
        print(f"  Test Accuracy: {best['test_acc']:.2f}%")
    
    # Create summary table
    best_df = pd.DataFrame(best_configs)
    print(f"\n{best_df.to_string(index=False)}")
    
    return best_df

def main():
    log_dir = 'grid_logs1'  # Updated to use new log directory
    
    if not os.path.exists(log_dir):
        print(f"Error: Directory '{log_dir}' not found!")
        return
    
    print("Collecting experiment results and training curves...")
    results_df, curves_df = collect_all_results(log_dir)
    
    if results_df.empty:
        print("No results found!")
        return
    
    print(f"Successfully collected {len(results_df)} experiment results")
    print(f"Successfully collected {len(curves_df)} training data points")
    print(f"Optimizers: {results_df['optimizer'].unique()}")
    print(f"Learning rate range: {results_df['lr'].min()} - {results_df['lr'].max()}")
    print(f"Weight decay range: {results_df['weight_decay'].min()} - {results_df['weight_decay'].max()}")
    
    # Save raw data
    results_df.to_csv('grid_search_results.csv', index=False)
    curves_df.to_csv('training_curves.csv', index=False)
    print(f"\nRaw data saved to: grid_search_results.csv and training_curves.csv")
    
    # Create output folder
    os.makedirs('analysis_plots', exist_ok=True)
    
    # 0. Check data completeness
    check_data_completeness(curves_df)
    
    # 1. Plot detailed training curves for each optimizer
    print("\nGenerating training curve plots...")
    plot_training_curves_by_optimizer(curves_df)
    
    # 2. Learning rate comparison
    print("Generating learning rate comparison plots...")
    plot_learning_rate_comparison(curves_df)
    
    # 3. Weight decay comparison
    print("Generating weight decay comparison plots...")
    plot_weight_decay_comparison(curves_df)
    
    # 4. Optimizer comparison
    print("Generating comparison plots...")
    avg_perf = plot_comparison_bar(results_df)
    plt.savefig('analysis_plots/optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Find best configurations
    best_configs = find_best_configurations(results_df)
    best_configs.to_csv('best_configurations.csv', index=False)
    
    print(f"\n✅ Analysis completed!")
    print(f"📊 Charts saved in: analysis_plots/ folder")
    print(f"🎯 Key line plots to focus on:")
    print(f"   - Training curves by optimizer: *_training_curves.png")
    print(f"   - Learning rate comparison: learning_rate_comparison.png")
    print(f"   - Weight decay comparison: weight_decay_comparison.png")
    print(f"   - Optimizer comparison: optimizer_comparison.png")
    print(f"📋 Best configurations saved in: best_configurations.csv")
    print(f"📈 Complete data saved in: grid_search_results.csv and training_curves.csv")

if __name__ == "__main__":
    main() 