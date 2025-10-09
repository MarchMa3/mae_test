"""
train_mae.py - Train MAE on lab data (Multi-GPU Optimized)
"""
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DataParallel
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
        loinc_tokens = batch['loinc_tokens'].to(device) 
        value_tokens = batch['value_tokens'].to(device)
        missing_mask = batch['missing_mask'].to(device)
        
        optimizer.zero_grad()
        loss, pred, mask = model(loinc_tokens, value_tokens, missing_mask)
        
        # Handle multi-GPU (DataParallel returns mean loss automatically)
        if isinstance(loss, tuple):
            loss = loss[0]
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """Validate on test set"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            loinc_tokens = batch['loinc_tokens'].to(device)
            value_tokens = batch['value_tokens'].to(device)
            missing_mask = batch['missing_mask'].to(device)
            
            # Force mask_ratio during validation
            loss, pred, mask = model(loinc_tokens, value_tokens, missing_mask, mask_ratio=0.75)
            
            # Handle multi-GPU
            if isinstance(loss, tuple):
                loss = loss[0]
                
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def load_sequences_from_pickle(filepath):
    """
    Load all sequences from pickle file
    
    Args:
        filepath: path to pickle file
    
    Returns:
        loinc_tokens: (N, max_seq_len) padded LOINC token sequences
        value_tokens: (N, max_seq_len) padded value token sequences
        missing_masks: (N, max_seq_len) padding masks (1=real data, 0=padding)
        seq_len: maximum sequence length
        vocab_size: vocabulary size from file attributes
    """
    print("Loading pickle file...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Read metadata from vocab_info
    vocab_info = data['vocab_info']
    vocab_size = vocab_info['vocab_size']
    
    # Read patient sequences
    patient_sequences = data['patient_sequences']
    total_sequences = len(patient_sequences)
    
    print(f"File contains {total_sequences} sequences")
    print(f"Vocab size: {vocab_size}")
    
    # Extract sequences
    all_loinc_tokens = []
    all_value_tokens = []
    all_masks = []
    seq_lengths = []
    
    print("Processing sequences...")
    for i, seq_data in enumerate(patient_sequences):
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i+1}/{total_sequences} sequences...")
        
        loinc_tokens = seq_data['loinc_tokens']
        value_tokens = seq_data['value_tokens']
        mask = seq_data['missing_mask']
        
        all_loinc_tokens.append(loinc_tokens)
        all_value_tokens.append(value_tokens)
        all_masks.append(mask)
        seq_lengths.append(len(loinc_tokens))
    
    # Find max length
    max_seq_len = max(seq_lengths)
    print(f"\nSequence lengths - Min: {min(seq_lengths)}, Max: {max_seq_len}, Mean: {np.mean(seq_lengths):.1f}")
    
    # Pad all sequences to max_seq_len
    print("Padding sequences...")
    padded_loinc = np.zeros((total_sequences, max_seq_len), dtype=np.int64)   
    padded_value = np.zeros((total_sequences, max_seq_len), dtype=np.float32)
    padded_masks = np.zeros((total_sequences, max_seq_len), dtype=np.float32)
    
    for i, (loinc, value, mask) in enumerate(zip(all_loinc_tokens, all_value_tokens, all_masks)):
        if (i + 1) % 100000 == 0:
            print(f"  Padded {i+1}/{total_sequences} sequences...")
        seq_len = len(loinc)
        padded_loinc[i, :seq_len] = loinc
        padded_value[i, :seq_len] = value
        padded_masks[i, :seq_len] = mask  
    
    print("✓ Data loading complete!")
    return padded_loinc, padded_value, padded_masks, max_seq_len, int(vocab_size)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load data from pickle file
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    loinc_tokens, value_tokens, missing_mask, seq_len, vocab_size = load_sequences_from_pickle('./data/labevents_mimic.pkl')
    
    print(f"\nProcessed data shape:")
    print(f"  LOINC tokens: {loinc_tokens.shape}")  
    print(f"  Value tokens: {value_tokens.shape}") 
    print(f"  Missing mask: {missing_mask.shape}")
    
    print(f"\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    print(f"Total samples: {loinc_tokens.shape[0]}")
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
    
    # OPTIMIZED: Larger batch size for multi-GPU, more workers
    batch_size = 64  # Increased from 16
    num_workers = 8  # Increased from 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create model with dynamic parameters
    print(f"\n" + "="*60)
    print("MODEL INITIALIZATION")
    print("="*60)
    
    model = mae_small(  # Using small model for faster training
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_bins=10,
        input_dim=64,
        use_cls_token=True,
        mask_ratio=0.75,
        exclude_columns=[0]  # Exclude CLS token from masking
    )
    
    # OPTIMIZED: Use all available GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = DataParallel(model)
    
    model = model.to(device)
    count_parameters(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    # OPTIMIZED: Reduced epochs with early stopping
    num_epochs = 50  # Reduced from 100
    best_val_loss = float('inf')
    patience = 5  # Early stopping patience
    patience_counter = 0
    
    print(f"\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    print(f"Training for up to {num_epochs} epochs with early stopping (patience={patience})")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ✓ New best validation loss!")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Save checkpoint (extract model from DataParallel if needed)
        model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'seq_len': seq_len,
            'vocab_size': vocab_size,
        }, f'checkpoints_mimic/epoch_{epoch+1}.pth', is_best=is_best)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()