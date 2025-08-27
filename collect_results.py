#!/usr/bin/env python3
import os
import glob
import pandas as pd

def collect_results():
    results = []
    
    # Find all result files
    result_files = glob.glob("results/*/results.txt")
    print(f"Found {len(result_files)} result files")
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Parse experiment parameters
            exp_line = lines[0].strip()
            if "lr=" in exp_line:
                parts = exp_line.split(", ")
                lr = float(parts[0].split("lr=")[1])
                wd = float(parts[1].split("weight_decay=")[1])
                dataset = parts[2].split("dataset=")[1]
                
                # Parse results
                for line in lines[2:]:
                    if ":" in line and "Val Acc:" in line:
                        optimizer = line.split(":")[0].strip()
                        val_acc = float(line.split("Val Acc: ")[1].split("%")[0])
                        test_acc = float(line.split("Test Acc: ")[1].split("%")[0])
                        
                        results.append({
                            'dataset': dataset,
                            'lr': lr,
                            'weight_decay': wd,
                            'optimizer': optimizer,
                            'val_acc': val_acc,
                            'test_acc': test_acc
                        })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if results:
        df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv('grid_search_results.csv', index=False)
        print(f"Collected {len(results)} results")
        print("Results saved to grid_search_results.csv")
        
        # Print summary table
        print("\n" + "="*80)
        print("GRID SEARCH RESULTS SUMMARY")
        print("="*80)
        
        # Group by optimizer and show best results
        for opt in sorted(df['optimizer'].unique()):
            print(f"\n{opt} Optimizer:")
            print("-" * 50)
            opt_df = df[df['optimizer'] == opt]
            
            # Sort by test accuracy (descending)
            opt_df_sorted = opt_df.sort_values('test_acc', ascending=False)
            
            print(f"{'Dataset':<12} {'LR':<8} {'WD':<8} {'Val Acc':<8} {'Test Acc':<8}")
            print("-" * 50)
            
            for _, row in opt_df_sorted.head(5).iterrows():  # Show top 5
                print(f"{row['dataset']:<12} {row['lr']:<8.4f} {row['weight_decay']:<8.4f} "
                      f"{row['val_acc']:<8.2f} {row['test_acc']:<8.2f}")
        
        # Overall best result
        best_idx = df['test_acc'].idxmax()
        best_row = df.loc[best_idx]
        
        print(f"\n" + "="*80)
        print("OVERALL BEST RESULT:")
        print(f"Optimizer: {best_row['optimizer']}")
        print(f"Dataset: {best_row['dataset']}")
        print(f"Learning Rate: {best_row['lr']}")
        print(f"Weight Decay: {best_row['weight_decay']}")
        print(f"Validation Accuracy: {best_row['val_acc']:.2f}%")
        print(f"Test Accuracy: {best_row['test_acc']:.2f}%")
        print("="*80)
        
    else:
        print("No results found")

if __name__ == "__main__":
    collect_results() 