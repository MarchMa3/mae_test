"""
train_mae.py - Train MAE on lab data
"""
import torch
import h5py
from torch.utils.data import DataLoader
import torch.optim as optim
from MAE import mae_large, mae_base, mae_small
from utils import (
    set_seed,
    count_parameters,
    LabDataset,
    save_checkpoint,
    load_checkpoint,
    adjust_learning_rate
) 

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        features = batch['embedding'].to(device)
        missing_mask = batch['missing_mask'].to(device)
        optimizer.zero_grad()
        loss, pred, mask = model(features, missing_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            features = batch['embedding'].to(device)
            missing_mask = batch['missing_mask'].to(device)
            
            loss, pred, mask = model(features, missing_mask)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with h5py.File('./data/simulate/mae_medical_data.h5', 'r') as f:
        features = f['final_tokens'][:]
        missing_mask = f['missing_mask'][:]
    
    n_samples = features.shape[0]
    n_train = int(0.8 * n_samples)
    
    train_dataset = LabDataset(features[:n_train], missing_mask[:n_train])
    val_dataset = LabDataset(features[n_train:], missing_mask[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = mas_small(
        seq_len=features.shape[1],
        input_dim=features.shape[2],
        use_cls_token=True,
        mask_ratio=0.75,
        exclude_columns=[0]
    )
    model = model.to(device)
    count_parameters(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    num_epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'checkpoints/epoch_{epoch+1}.pth', is_best=is_best)

if __name__ == '__main__':
    main()