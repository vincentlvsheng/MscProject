import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Data import get_data
from CnnModel import CNNModel, MLPModel
from custom_optim import SGD, AdamWeg, LNS_Madam, AdamGD

import os
import copy
import argparse


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(train_loader)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # print every 10% progress
        if (batch_idx + 1) % max(1, total_batches // 10) == 0:
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            progress = (batch_idx + 1) / total_batches * 100
            print(f'\rProgress: {progress:5.1f}% [{batch_idx+1:>4}/{total_batches}] Loss: {current_loss:.4f} Acc: {current_acc:5.2f}%', end='')
    
    print()  # new line
    return running_loss/len(train_loader), 100.*correct/total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss/len(val_loader), 100.*correct/total

def save_checkpoint(model, optimizer, epoch, val_acc, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    torch.save(checkpoint, filename)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_and_eval(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs, save_dir, tag):
    best_val_acc = 0
    best_model_state = None
    for epoch in range(num_epochs):
        print(f'\n[{tag}] Epoch {epoch+1}/{num_epochs}')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'[{tag}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'[{tag}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_model_path = os.path.join(save_dir, f'{tag}_best_model.pth')
            save_checkpoint(model, optimizer, epoch+1, val_acc, best_model_path)
            print(f'[{tag}] New best model saved with validation accuracy: {val_acc:.2f}%')
    
    print(f'[{tag}] Training completed. Best validation accuracy: {best_val_acc:.2f}%')
    return best_val_acc, best_model_state


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train models with different optimizers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    
    # Create save directory with experiment info
    exp_name = f"lr{args.lr}_wd{args.weight_decay}_{args.dataset}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader = get_data(args.dataset)
    print(f"\nUsing dataset: {args.dataset}")
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    criterion = nn.CrossEntropyLoss()
    
    # 1. GD
    model_gd = CNNModel(num_classes=10).to(device)
    optimizer_gd = SGD(model_gd.parameters(), lr=args.lr, update_alg='gd', weight_decay=args.weight_decay)
    scheduler_gd = optim.lr_scheduler.ReduceLROnPlateau(optimizer_gd, mode='max', factor=0.5, patience=2, verbose=True)
    best_val_acc_gd, best_model_state_gd = train_and_eval(model_gd, train_loader, val_loader, optimizer_gd, scheduler_gd, criterion, device, args.epochs, save_dir, tag='gd')
    
    # 2. EG
    model_eg = CNNModel(num_classes=10).to(device)
    optimizer_eg = SGD(model_eg.parameters(), lr=args.lr, update_alg='eg', weight_decay=args.weight_decay)
    scheduler_eg = optim.lr_scheduler.ReduceLROnPlateau(optimizer_eg, mode='max', factor=0.5, patience=2, verbose=True)
    best_val_acc_eg, best_model_state_eg = train_and_eval(model_eg, train_loader, val_loader, optimizer_eg, scheduler_eg, criterion, device, args.epochs, save_dir, tag='eg')
    
    # 3. AdamWeg
    model_adamweg = CNNModel(num_classes=10).to(device)
    optimizer_adamweg = AdamWeg(model_adamweg.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_adamweg = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adamweg, mode='max', factor=0.5, patience=2, verbose=True)
    best_val_acc_adamweg, best_model_state_adamweg = train_and_eval(model_adamweg, train_loader, val_loader, optimizer_adamweg, scheduler_adamweg, criterion, device, args.epochs, save_dir, tag='adamweg')
    
    # 4. AdamGD
    model_adamgd = CNNModel(num_classes=10).to(device)
    optimizer_adamgd = AdamGD(model_adamgd.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_adamgd = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adamgd, mode='max', factor=0.5, patience=2, verbose=True)
    best_val_acc_adamgd, best_model_state_adamgd = train_and_eval(model_adamgd, train_loader, val_loader, optimizer_adamgd, scheduler_adamgd, criterion, device, args.epochs, save_dir, tag='adamgd')
    
    # 5. LNS_Madam
    model_lnsmadam = CNNModel(num_classes=10).to(device)
    optimizer_lnsmadam = LNS_Madam(model_lnsmadam.parameters(), lr=args.lr/128, p_scale=3.0, g_bound=10.0, wd=args.weight_decay)
    scheduler_lnsmadam = optim.lr_scheduler.ReduceLROnPlateau(optimizer_lnsmadam, mode='max', factor=0.5, patience=2, verbose=True)
    best_val_acc_lnsmadam, best_model_state_lnsmadam = train_and_eval(model_lnsmadam, train_loader, val_loader, optimizer_lnsmadam, scheduler_lnsmadam, criterion, device, args.epochs, save_dir, tag='lnsmadam')
    
    # Evaluate on test set with best models
    print("\n=== Final Test Set Evaluation ===")
    
    # Load best models and evaluate on test set
    models_info = [
        ('GD', model_gd, best_model_state_gd, best_val_acc_gd),
        ('EG', model_eg, best_model_state_eg, best_val_acc_eg),
        ('AdamWeg', model_adamweg, best_model_state_adamweg, best_val_acc_adamweg),
        ('AdamGD', model_adamgd, best_model_state_adamgd, best_val_acc_adamgd),
        ('LNS_Madam', model_lnsmadam, best_model_state_lnsmadam, best_val_acc_lnsmadam)
    ]
    
    test_results = []
    for name, model, best_state, val_acc in models_info:
        model.load_state_dict(best_state)
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        test_results.append((name, val_acc, test_acc))
        print(f"{name} - Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    print("\n=== Benchmark Results ===")
    print("+-----------+--------+------------+------------+")
    print("| Optimizer | Model  | Val Acc(%) | Test Acc(%) |")
    print("+-----------+--------+------------+------------+")
    for name, val_acc, test_acc in test_results:
        print(f"| {name:9} | {'CNN':6} | {val_acc:10.2f} | {test_acc:11.2f} |")
    print("+-----------+--------+------------+------------+")
    
    # Save results to file
    results_file = os.path.join(save_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Experiment: lr={args.lr}, weight_decay={args.weight_decay}, dataset={args.dataset}\n")
        f.write("Results:\n")
        for name, val_acc, test_acc in test_results:
            f.write(f"{name}: Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main() 