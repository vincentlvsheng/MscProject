#!/usr/bin/env python3
"""
Improved WandB data upload script
Upload training curve data with optimized structure for visualization
"""

import pandas as pd
import numpy as np
import wandb
import os
import argparse

def upload_training_curves_improved(project_name="optimizer-comparison-viz", entity=None):
    """Upload training curves with improved structure for visualization"""
    
    print("🚀 Uploading training curves to WandB...")
    
    # Read data
    curves_df = pd.read_csv("training_curves.csv")
    results_df = pd.read_csv("grid_search_results.csv")
    
    print(f"✅ Loaded training curve data: {len(curves_df)} records")
    
    # Group by experiment configuration
    unique_configs = curves_df[['lr', 'weight_decay']].drop_duplicates()
    
    for _, config in unique_configs.iterrows():
        lr, wd = config['lr'], config['weight_decay']
        
        print(f"\n📊 Processing configuration: lr={lr}, wd={wd}")
        
        # Get all data for this configuration
        config_curves = curves_df[
            (curves_df['lr'] == lr) & 
            (curves_df['weight_decay'] == wd)
        ].copy()
        
        config_results = results_df[
            (results_df['lr'] == lr) & 
            (results_df['weight_decay'] == wd)
        ]
        
        if config_curves.empty:
            continue
        
        # Create WandB run
        run_name = f"lr{lr}_wd{wd}_curves"
        
        with wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            config={
                "learning_rate": lr,
                "weight_decay": wd,
                "dataset": "mnist",
                "method": "grid_search"
            },
            tags=["training_curves", f"lr_{lr}", f"wd_{wd}"],
            reinit=True
        ) as run:
            
            # Organize data by epoch, synchronously record all optimizer data
            max_epoch = config_curves['epoch'].max()
            
            for epoch in range(1, max_epoch + 1):
                epoch_data = config_curves[config_curves['epoch'] == epoch]
                
                log_dict = {"epoch": epoch}
                
                # Record data for each optimizer at this epoch
                for optimizer in ['GD', 'EG', 'ADAMWEG', 'LNSMADAM']:
                    opt_data = epoch_data[epoch_data['optimizer'] == optimizer]
                    
                    if not opt_data.empty:
                        row = opt_data.iloc[0]
                        log_dict.update({
                            f"{optimizer}/train_loss": row['train_loss'],
                            f"{optimizer}/train_acc": row['train_acc'],
                            f"{optimizer}/val_loss": row['val_loss'],
                            f"{optimizer}/val_acc": row['val_acc']
                        })
                
                # Log all data for this epoch
                if len(log_dict) > 1:  # Only log if there's data
                    wandb.log(log_dict)
            
            # Record final results
            final_results = {}
            for _, row in config_results.iterrows():
                optimizer = row['optimizer']
                final_results.update({
                    f"{optimizer}/final_val_acc": row['val_acc'],
                    f"{optimizer}/final_test_acc": row['test_acc']
                })
            
            if final_results:
                wandb.log(final_results)
            
            print(f"✅ Completed: {run_name}")

def create_dashboard_templates(project_name):
    """Create dashboard template guide"""
    
    template = f"""
# WandB Dashboard Configuration Guide - {project_name}

## 🎯 Charts you can now create:

### 1. **Training Accuracy Comparison** (Line Plot)
```
- Chart Type: Line Plot
- X Axis: epoch  
- Y Axis: Select multiple:
  * GD/val_acc
  * EG/val_acc  
  * ADAMWEG/val_acc
  * LNSMADAM/val_acc
- Group by: run (each lr/wd configuration will be a different colored line)
- Title: "Validation Accuracy Comparison"
```

### 2. **Training Loss Comparison** (Line Plot)  
```
- Chart Type: Line Plot
- X Axis: epoch
- Y Axis: Select multiple:
  * GD/train_loss
  * EG/train_loss
  * ADAMWEG/train_loss  
  * LNSMADAM/train_loss
- Group by: run
- Title: "Training Loss Comparison"
```

### 3. **Validation vs Training Loss** (Line Plot)
```
- Chart Type: Line Plot
- X Axis: epoch
- Y Axis: Select:
  * GD/train_loss
  * GD/val_loss
- Group by: run
- Title: "GD: Training vs Validation Loss"
```

### 4. **Final Performance Comparison** (Bar Chart)
```
- Chart Type: Bar Chart  
- X Axis: Not set (use default)
- Y Axis: Select multiple:
  * GD/final_test_acc
  * EG/final_test_acc
  * ADAMWEG/final_test_acc
  * LNSMADAM/final_test_acc
- Group by: run
- Title: "Final Test Accuracy Comparison"
```

## 🔧 **Creation Steps:**

1. Go to project: {project_name}
2. Click the "+" button in top right
3. Select "Line plot" or "Bar chart"
4. Configure X axis and Y axis as above
5. Click "Save"

## 📈 **Advanced Tips:**

- **Filters**: Use config.learning_rate and config.weight_decay to filter specific configurations
- **Faceting**: Can create faceted plots by config.learning_rate
- **Custom Colors**: You can customize colors for each optimizer in chart settings

## 🎨 **Recommended Dashboard Layout:**

```
┌─────────────────┬─────────────────┐
│   Training      │   Validation    │
│   Accuracy      │   Accuracy      │
│   Comparison    │   Comparison    │
├─────────────────┼─────────────────┤
│   Training      │   Final         │
│   Loss          │   Performance   │
│   Comparison    │   Bar Chart     │
└─────────────────┴─────────────────┘
```
"""
    
    with open("wandb_dashboard_guide_detailed.md", "w", encoding="utf-8") as f:
        f.write(template)
    
    print(f"✅ Created detailed configuration guide: wandb_dashboard_guide_detailed.md")

def main():
    parser = argparse.ArgumentParser(description='Improved WandB data upload')
    parser.add_argument('--project', type=str, default='optimizer-comparison-viz', help='WandB project name')
    parser.add_argument('--entity', type=str, default=None, help='WandB entity name')
    
    args = parser.parse_args()
    
    print("🎯 Improved WandB Visualization Data Upload")
    print("="*50)
    
    # Check files
    if not os.path.exists("training_curves.csv"):
        print("❌ Cannot find training_curves.csv file")
        return
    
    try:
        import wandb
        wandb.api.viewer()
        print("✅ WandB logged in")
    except:
        print("❌ Please login to WandB first: wandb login")
        return
    
    # Upload data
    upload_training_curves_improved(
        project_name=args.project,
        entity=args.entity
    )
    
    # Create configuration guide
    create_dashboard_templates(args.project)
    
    print(f"\n🎉 Upload completed!")
    print(f"📊 Visit: https://wandb.ai/{args.entity or 'your-username'}/{args.project}")
    print(f"📋 Check configuration guide: wandb_dashboard_guide_detailed.md")

if __name__ == "__main__":
    main() 