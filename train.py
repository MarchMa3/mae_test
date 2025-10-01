"""
train_mae.py - Train MAE on lab data
"""
import torch
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader
from MAE import mae_base
from utils import (
    LabDataset, set_seed, adjust_learning_rate, 
    get_grad_norm_, save_checkpoint, count_parameters
)

def train_mae(
    model,
    train_loader,
    val_loader,
    max_epochs=50,
    base_lr=1e-3,
    device='cuda'
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
    
    best_val_loss = float('inf')
    
    for epoch in range(max_epochs):
        # Adjust learning rate
        current_lr = adjust_learning_rate(
            optimizer, epoch, 
            lr=base_lr, min_lr=1e-6,
            max_epochs=max_epochs, warmup_epochs=5
        )
        
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch['embedding'].to(device)
            missing_mask = batch['missing_mask'].to(device)
            
            # Forward
            loss, pred, mask = model(embeddings, missing_mask, mask_ratio=0.75)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            grad_norm = get_grad_norm_(model.parameters())
            if grad_norm > 10.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}, Grad: {grad_norm:.4f}")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embedding'].to(device)
                missing_mask = batch['missing_mask'].to(device)
                
                loss, pred, mask = model(embeddings, missing_mask, mask_ratio=0.75)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"\nEpoch {epoch}/{max_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"LR: {current_lr:.6f}\n")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if epoch % 10 == 0 or is_best:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, filepath=f'checkpoints/mae_epoch_{epoch}.pth', is_best=is_best)

if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    with h5py.File('./data/simulate/mae_medical_data.h5', 'r') as f:
        tokens = torch.from_numpy(f['final_tokens'][:]).float()
        missing_mask = torch.from_numpy(f['missing_mask'][:]).float()
    
    # Split
    n_train = int(0.8 * len(tokens))
    train_dataset = LabDataset(tokens[:n_train], missing_mask[:n_train])
    val_dataset = LabDataset(tokens[n_train:], missing_mask[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = mae_base(
        seq_len=tokens.shape[1],
        hdf5_embed_dim=tokens.shape[2],
        use_cls_token=True
    )
    count_parameters(model)
    
    # Train
    train_mae(model, train_loader, val_loader, max_epochs=50, device=device)