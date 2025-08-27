#!/usr/bin/env python3
"""
Upload existing training results to WandB for visualization and analysis
Use existing CSV data and charts without retraining
"""

import pandas as pd
import numpy as np
import wandb
import os
import argparse
from pathlib import Path

def upload_historical_data(project_name="optimizer-comparison-analysis", entity=None):
    """Upload historical training data to WandB"""
    
    print("🚀 Starting upload of historical data to WandB...")
    
    # Check if data files exist
    required_files = [
        "grid_search_results.csv",
        "training_curves.csv", 
        "best_configurations.csv"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing file: {file}")
            return False
    
    # Read data
    print("📊 Reading data files...")
    results_df = pd.read_csv("grid_search_results.csv")
    curves_df = pd.read_csv("training_curves.csv")
    best_configs_df = pd.read_csv("best_configurations.csv")
    
    print(f"✅ Loaded {len(results_df)} experiment results")
    print(f"✅ Loaded {len(curves_df)} training data points")
    
    # Get unique experiment configurations
    unique_configs = results_df[['lr', 'weight_decay']].drop_duplicates()
    
    # Create a WandB run for each configuration
    for _, config in unique_configs.iterrows():
        lr, wd = config['lr'], config['weight_decay']
        
        # Filter data for this configuration
        config_results = results_df[(results_df['lr'] == lr) & (results_df['weight_decay'] == wd)]
        config_curves = curves_df[(curves_df['lr'] == lr) & (curves_df['weight_decay'] == wd)]
        
        if config_results.empty or config_curves.empty:
            continue
        
        # Create WandB run
        run_name = f"lr{lr}_wd{wd}_historical"
        
        with wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            config={
                "learning_rate": lr,
                "weight_decay": wd,
                "dataset": "mnist",  # inferred from data
                "method": "grid_search",
                "uploaded_from": "historical_data"
            },
            tags=["historical", "grid_search", f"lr_{lr}", f"wd_{wd}"]
        ) as run:
            
            print(f"\n📝 Processing configuration: lr={lr}, wd={wd}")
            
            # Upload training curve data
            for optimizer in ['GD', 'EG', 'ADAMWEG', 'LNSMADAM']:
                opt_curves = config_curves[config_curves['optimizer'] == optimizer]
                
                if opt_curves.empty:
                    continue
                
                # Sort by epoch order
                opt_curves = opt_curves.sort_values('epoch')
                
                # Log data for each epoch
                for _, row in opt_curves.iterrows():
                    wandb.log({
                        f"{optimizer}/train_loss": row['train_loss'],
                        f"{optimizer}/train_acc": row['train_acc'],
                        f"{optimizer}/val_loss": row['val_loss'],
                        f"{optimizer}/val_acc": row['val_acc'],
                        f"{optimizer}/epoch": row['epoch'],
                        "global_step": row['epoch']
                    })
            
            # Log final results
            final_metrics = {}
            for _, row in config_results.iterrows():
                optimizer = row['optimizer']
                final_metrics[f"{optimizer}/final_val_acc"] = row['val_acc']
                final_metrics[f"{optimizer}/final_test_acc"] = row['test_acc']
            
            wandb.log(final_metrics)
            
            # Create results table
            results_table = wandb.Table(columns=["Optimizer", "Val Acc (%)", "Test Acc (%)"])
            for _, row in config_results.iterrows():
                results_table.add_data(row['optimizer'], row['val_acc'], row['test_acc'])
            
            wandb.log({"results_table": results_table})
            
            print(f"✅ Upload completed: {run_name}")
    
    return True

def upload_summary_analysis(project_name="optimizer-comparison-summary", entity=None):
    """Upload summary analysis to WandB"""
    
    print("\n📊 Uploading summary analysis...")
    
    # Read data
    results_df = pd.read_csv("grid_search_results.csv")
    best_configs_df = pd.read_csv("best_configurations.csv")
    
    with wandb.init(
        project=project_name,
        entity=entity,
        name="optimizer_comparison_summary",
        config={
            "analysis_type": "summary",
            "total_experiments": len(results_df),
            "optimizers": results_df['optimizer'].unique().tolist(),
            "learning_rates": results_df['lr'].unique().tolist(),
            "weight_decays": results_df['weight_decay'].unique().tolist()
        },
        tags=["summary", "comparison", "analysis"]
    ) as run:
        
        # 1. Optimizer performance comparison
        optimizer_stats = results_df.groupby('optimizer').agg({
            'val_acc': ['mean', 'std', 'max'],
            'test_acc': ['mean', 'std', 'max']
        }).round(2)
        
        # Flatten column names
        optimizer_stats.columns = ['_'.join(col).strip() for col in optimizer_stats.columns]
        optimizer_stats = optimizer_stats.reset_index()
        
        # Create optimizer comparison table
        comparison_table = wandb.Table(columns=[
            "Optimizer", "Avg Val Acc", "Std Val Acc", "Max Val Acc",
            "Avg Test Acc", "Std Test Acc", "Max Test Acc"
        ])
        
        for _, row in optimizer_stats.iterrows():
            comparison_table.add_data(
                row['optimizer'],
                row['val_acc_mean'], row['val_acc_std'], row['val_acc_max'],
                row['test_acc_mean'], row['test_acc_std'], row['test_acc_max']
            )
        
        wandb.log({"optimizer_comparison": comparison_table})
        
        # 2. Best configurations table
        best_configs_table = wandb.Table(columns=["Optimizer", "Learning Rate", "Weight Decay", "Val Acc", "Test Acc"])
        for _, row in best_configs_df.iterrows():
            best_configs_table.add_data(
                row['Optimizer'], row['Learning Rate'], row['Weight Decay'],
                row['Val Acc'], row['Test Acc']
            )
        
        wandb.log({"best_configurations": best_configs_table})
        
        # 3. Log key metrics
        best_overall = results_df.loc[results_df['test_acc'].idxmax()]
        wandb.log({
            "best_test_accuracy": best_overall['test_acc'],
            "best_optimizer": best_overall['optimizer'],
            "best_lr": best_overall['lr'],
            "best_weight_decay": best_overall['weight_decay']
        })
        
        # 4. Upload analysis plots as artifacts
        if os.path.exists("analysis_plots"):
            artifact = wandb.Artifact("analysis_plots", type="plots")
            artifact.add_dir("analysis_plots")
            run.log_artifact(artifact)
            print("✅ Uploaded analysis plots")
        
        # 5. Upload statistical report
        if os.path.exists("statistical_report.txt"):
            artifact = wandb.Artifact("statistical_report", type="report")
            artifact.add_file("statistical_report.txt")
            run.log_artifact(artifact)
            print("✅ Uploaded statistical report")
        
        print("✅ Summary analysis upload completed")

def create_wandb_dashboard_config():
    """Create WandB dashboard configuration recommendations"""
    
    dashboard_config = """
# WandB Dashboard Configuration Recommendations

## Recommended Visualization Panels:

1. **Training Curve Comparison** (Line Plot)
   - X axis: epoch
   - Y axis: val_acc
   - Group by: optimizer
   - Filter: by learning_rate and weight_decay

2. **Final Accuracy Comparison** (Bar Chart)
   - X axis: optimizer
   - Y axis: final_test_acc
   - Group by: learning_rate

3. **Parameter Heatmap** (Heatmap)
   - X axis: learning_rate
   - Y axis: weight_decay
   - Color: test_acc
   - Facet: optimizer

4. **Convergence Speed Analysis** (Scatter Plot)
   - X axis: epoch (convergence point)
   - Y axis: final_test_acc
   - Color: optimizer
   - Size: learning_rate

5. **Loss Function Comparison** (Line Plot)
   - X axis: epoch
   - Y axis: train_loss, val_loss
   - Group by: optimizer

## Useful Filters:
- learning_rate: [0.0001, 0.001, 0.01, 0.1]
- weight_decay: [0.0, 0.0001, 0.001, 0.01]
- optimizer: [GD, EG, ADAMWEG, LNSMADAM]

## Suggested Tags:
- Dataset: mnist
- Method type: grid_search
- Experiment type: optimizer_comparison
"""
    
    with open("wandb_dashboard_guide.md", "w", encoding="utf-8") as f:
        f.write(dashboard_config)

def main():
    parser = argparse.ArgumentParser(description='Upload historical training results to WandB')
    parser.add_argument('--project', type=str, default='optimizer-comparison', help='WandB project name')
    parser.add_argument('--entity', type=str, default=None, help='WandB entity name (username or team name)')
    parser.add_argument('--summary-only', action='store_true', help='Upload only summary analysis')
    
    args = parser.parse_args()
    
    print("🎯 WandB Historical Data Upload Tool")
    print("="*50)
    
    try:
        # Check wandb installation
        import wandb
        print("✅ WandB is installed")
    except ImportError:
        print("❌ Please install WandB first: pip install wandb")
        print("Then run: wandb login")
        return
    
    # Check login status
    try:
        wandb.api.viewer()
        print("✅ WandB is logged in")
    except:
        print("❌ Please login to WandB first: wandb login")
        return
    
    success = True
    
    # Upload detailed data (unless summary only)
    if not args.summary_only:
        success = upload_historical_data(
            project_name=f"{args.project}-detailed",
            entity=args.entity
        )
    
    # Upload summary analysis
    if success:
        upload_summary_analysis(
            project_name=f"{args.project}-summary", 
            entity=args.entity
        )
    
    # Create dashboard guide
    create_wandb_dashboard_config()
    
    print("\n🎉 Upload completed!")
    print("\n📊 Next steps:")
    print(f"1. Visit https://wandb.ai to view projects: {args.project}-detailed and {args.project}-summary")
    print("2. Create custom dashboards to visualize results")
    print("3. Check wandb_dashboard_guide.md for dashboard configuration recommendations")
    print("4. Use WandB's report feature to share results")

if __name__ == "__main__":
    main() 