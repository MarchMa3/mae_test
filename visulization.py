"""
visualize_training.py - Visualize training results from saved checkpoints
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

def extract_losses_from_checkpoints(checkpoint_dir='checkpoints'):
    """
    Extract train and val losses from all checkpoint files
    
    Returns:
        epochs: list of epoch numbers
        train_losses: list of training losses
        val_losses: list of validation losses
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pth'))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by epoch number
    def get_epoch_num(filename):
        match = re.search(r'epoch_(\d+)\.pth', filename)
        return int(match.group(1)) if match else 0
    
    checkpoint_files.sort(key=get_epoch_num)
    
    epochs = []
    train_losses = []
    val_losses = []
    
    for ckpt_file in checkpoint_files:
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            epoch = checkpoint['epoch']
            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
            
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        except Exception as e:
            print(f"Warning: Could not load {ckpt_file}: {e}")
            continue
    
    return epochs, train_losses, val_losses

def plot_loss_curves(epochs, train_losses, val_losses, save_path='loss_curves.png'):
    """
    Plot training and validation loss curves
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_losses, linewidth=2.5, color='#2E86AB', 
             label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs, val_losses, linewidth=2.5, color='#A23B72', 
             label='Val Loss', marker='s', markersize=3)
    
    # Mark best validation loss
    best_val_idx = np.argmin(val_losses)
    best_epoch = epochs[best_val_idx]
    best_val_loss = val_losses[best_val_idx]
    plt.scatter([best_epoch], [best_val_loss], color='red', s=100, 
                zorder=5, label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})')
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Reconstruction Loss', fontsize=14, fontweight='bold')
    plt.title('MAE Training on Lab Events Data', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss curves saved to {save_path}")
    plt.show()

def print_training_summary(epochs, train_losses, val_losses):
    """
    Print summary statistics
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs: {len(epochs)}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"\nBest validation loss: {min(val_losses):.4f} (Epoch {epochs[np.argmin(val_losses)]})")
    print(f"Initial train loss: {train_losses[0]:.4f}")
    print(f"Loss reduction: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%")
    
    # Check for overfitting
    gap = train_losses[-1] - val_losses[-1]
    if gap > 0.1:
        print(f"\n⚠ Warning: Large train-val gap ({gap:.4f}) - possible overfitting")
    elif val_losses[-1] > train_losses[-1] + 0.05:
        print(f"\n✓ Model generalizes well (val slightly higher than train)")
    else:
        print(f"\n✓ Good convergence")
    print("="*60)

def plot_loss_comparison(epochs, train_losses, val_losses, save_path='loss_comparison.png'):
    """
    Plot train-val loss difference over time
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, linewidth=2.5, color='#2E86AB', label='Train Loss')
    ax1.plot(epochs, val_losses, linewidth=2.5, color='#A23B72', label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Train-val gap
    gap = np.array(val_losses) - np.array(train_losses)
    ax2.plot(epochs, gap, linewidth=2.5, color='#F18F01', marker='o', markersize=3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Val Loss - Train Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Generalization Gap', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss comparison saved to {save_path}")
    plt.show()

def main():
    checkpoint_dir = 'checkpoints'
    
    print("Loading checkpoint data...")
    try:
        epochs, train_losses, val_losses = extract_losses_from_checkpoints(checkpoint_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"Loaded {len(epochs)} checkpoints")
    
    # Print summary
    print_training_summary(epochs, train_losses, val_losses)
    
    # Plot loss curves
    plot_loss_curves(epochs, train_losses, val_losses)
    
    # Plot comparison
    plot_loss_comparison(epochs, train_losses, val_losses)

if __name__ == '__main__':
    main()