"""
Simple script to check if sequence length correlates with loss
Usage: python verify_length_bias.py --checkpoint_path your_checkpoint.pth
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pickle

from MAE_v2 import MAE
from utils_v2 import collate_fn_dynamic, LabDataset


def analyze_length_vs_loss(model, data_loader, device, max_batches=200):
    """Check correlation between sequence length and loss"""
    model.eval()
    
    all_lengths = []
    all_losses = []
    
    print("Analyzing length bias...")
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break
                
            token_ids = batch['token_ids'].to(device)
            value_ids = batch['value_ids'].to(device)
            
            # Get actual sequence lengths (non-padding)
            valid_mask = (token_ids != 0) & (value_ids != 0)
            lengths = valid_mask.sum(dim=1).cpu().numpy()
            
            # Compute per-sample loss
            loss, pred, target = model(token_ids, value_ids, mask_ratio=0.5)
            sample_losses = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2)).cpu().numpy()
            
            all_lengths.extend(lengths.tolist())
            all_losses.extend(sample_losses.tolist())
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1} batches...")
    
    # Compute correlation
    correlation = np.corrcoef(all_lengths, all_losses)[0, 1]
    
    print("\n" + "="*60)
    print(f"Pearson Correlation: {correlation:.4f}")
    print("="*60)
    
    if abs(correlation) > 0.3:
        print("❌ WARNING: Strong length bias detected!")
        print("   Model is using sequence length as a shortcut.")
    elif abs(correlation) > 0.15:
        print("⚠️  Moderate length bias.")
    else:
        print("✅ OK: Minimal length bias.")
    print("="*60 + "\n")
    
    return all_lengths, all_losses, correlation


def plot_results(all_lengths, all_losses, save_path='length_bias.png'):
    """Plot scatter and trend"""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(all_lengths, all_losses, alpha=0.3, s=10)
    
    # Trend line
    z = np.polyfit(all_lengths, all_losses, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(all_lengths), max(all_lengths), 100)
    plt.plot(x_trend, p(x_trend), "r--", linewidth=2, 
             label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
    
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss vs. Sequence Length', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to: {save_path}\n")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data/labevents_mimic.pkl')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_batches', type=int, default=200)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    model = MAE(
        vocab_size=768,
        value_vocab_size=10,
        input_dim=256,
        encoder_depth=6,
        decoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        max_seq_len=512
    ).to(device)
    
    # Handle DDP checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print("Model loaded.\n")
    
    # Load data
    print("Loading data...")
    val_dataset = LabDataset(args.data_path, split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=dynamic_collate_fn,
        shuffle=False
    )
    print(f"Data loaded ({len(val_dataset)} samples).\n")
    
    # Analyze
    all_lengths, all_losses, correlation = analyze_length_vs_loss(
        model, val_loader, device, args.max_batches
    )
    
    # Plot
    plot_results(all_lengths, all_losses)
    
    print("="*60)
    print(f"Final Correlation: {correlation:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()