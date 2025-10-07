"""
train_mae.py - Train MAE on lab data
"""
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from MAE_v2 import mae_large, mae_base, mae_small
from utils_v2 import (
    set_seed,
    count_parameters,
    LabDataset,
    save_checkpoint,
    load_checkpoint,
    adjust_learning_rate
) 

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        tokens = batch['embedding'].to(device)
        missing_mask = batch['missing_mask'].to(device)
        
        optimizer.zero_grad()
        loss, pred, mask = model(tokens, missing_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """Validate on test set"""
    model.eval()  # Keep eval mode for no dropout, etc.
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['embedding'].to(device)
            missing_mask = batch['missing_mask'].to(device)
            
            # Force mask_ratio during validation
            loss, pred, mask = model(tokens, missing_mask, mask_ratio=0.75)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def load_sequences_from_h5(filepath):
    """
    Load all sequences from HDF5 file structure
    
    Args:
        filepath: path to HDF5 file
    
    Returns:
        tokens: (N, max_seq_len) padded token sequences
        missing_masks: (N, max_seq_len) padding masks (1=real data, 0=padding)
        seq_len: maximum sequence length
        vocab_size: vocabulary size from file attributes
    """
    with h5py.File(filepath, 'r') as f:
        # Read metadata
        vocab_size = f.attrs['vocab_size']
        total_sequences = f.attrs['total_sequences']
        
        print(f"File contains {total_sequences} sequences")
        print(f"Vocab size: {vocab_size}")
        
        # Read all sequences
        sequences_group = f['sequences']
        all_loinc_tokens = []
        all_value_tokens = []
        all_masks = []
        seq_lengths = []
        
        for seq_id in sequences_group.keys():
            seq_data = sequences_group[seq_id]
            loinc_tokens = seq_data['loinc_tokens'][:]  
            value_tokens = seq_data['value_tokens'][:] 
            mask = seq_data['missing_mask'][:]  
            
            all_loinc_tokens.append(loinc_tokens)
            all_value_tokens.append(value_tokens)
            all_masks.append(mask)
            seq_lengths.append(len(tokens))
        
        # Find max length
        max_seq_len = max(seq_lengths)
        print(f"Sequence lengths - Min: {min(seq_lengths)}, Max: {max_seq_len}, Mean: {np.mean(seq_lengths):.1f}")
        
        # Pad all sequences to max_seq_len
        padded_loinc = np.zeros((total_sequences, max_seq_len), dtype=np.int64)   
        padded_value = np.zeros((total_sequences, max_seq_len), dtype=np.float32)
        padded_masks = np.zeros((total_sequences, max_seq_len), dtype=np.float32)
        
        for i, (tokens, mask) in enumerate(zip(all_tokens, all_masks)):
            seq_len = len(tokens)
            padded_loinc[i, :seq_len] = loinc
            padded_value[i, :seq_len] = value
            padded_masks[i, :seq_len] = mask  
        
        return padded_loinc, padded_value, padded_masks, max_seq_len, int(vocab_size)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data from labevents.h5
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    tokens, missing_mask, seq_len, vocab_size = load_sequences_from_h5('./data/labevents.h5')
    
    print(f"\nProcessed data shape:")
    print(f"  LOINC tokens: {loinc_tokens.shape}")  
    print(f"  Value tokens: {value_tokens.shape}") 
    print(f"  Missing mask: {missing_mask.shape}")
    
    print(f"\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    print(f"Total samples: {tokens.shape[0]}")
    print(f"Sequence length (seq_len): {seq_len}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"LOINC token range: [{loinc_tokens.min()}, {loinc_tokens.max()}]")  
    print(f"Value token range: [{value_tokens.min():.1f}, {value_tokens.max():.1f}]")  
    print(f"Missing ratio: {(missing_mask == 0).sum() / missing_mask.size * 100:.2f}%")
    
    # Train/val split
    n_samples = loinc_tokens.shape[0]
    n_train = int(0.8 * n_samples)
    
    train_dataset = LabDataset(
        loinc_tokens[:n_train],   
        value_tokens[:n_train],   
        missing_mask[:n_train]
    )
    val_dataset = LabDataset(
        loinc_tokens[n_train:],   
        value_tokens[n_train:],   
        missing_mask[n_train:]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create model with dynamic parameters
    print(f"\n" + "="*60)
    print("MODEL INITIALIZATION")
    print("="*60)
    
    model = mae_small(
        seq_len=seq_len,
        vocab_size=vocab_size,
        input_dim=64,
        use_cls_token=True,
        mask_ratio=0.75,
        exclude_columns=[0]  # Exclude CLS token from masking
    )
    model = model.to(device)
    count_parameters(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    num_epochs = 100
    best_val_loss = float('inf')
    
    print(f"\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  New best validation loss!")
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'seq_len': seq_len,
            'vocab_size': vocab_size,
        }, f'checkpoints_v2/epoch_{epoch+1}.pth', is_best=is_best)
    
    print(f"\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()