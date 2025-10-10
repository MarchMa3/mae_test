# train_v2_ddp.py  — DDP + Mixed Precision
import os
import argparse
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler  

from MAE_v2 import mae_large, mae_base, mae_small
from utils_v2 import (
    set_seed,
    count_parameters,
    LabDataset,
    save_checkpoint,
    adjust_learning_rate,
    collate_fn_dynamic,
)

# Data utils
def load_sequences_from_pickle(filepath):
    print("Loading pickle file...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    vocab_info = data['vocab_info']
    vocab_size = int(vocab_info['vocab_size'])

    patient_sequences = data['patient_sequences']
    total_sequences = len(patient_sequences)
    print(f"File contains {total_sequences} sequences")
    print(f"Vocab size: {vocab_size}")

    all_loinc_tokens, all_value_tokens, all_masks, seq_lengths = [], [], [], []
    print("Processing sequences...")
    for i, seq_data in enumerate(patient_sequences.values()):
        loinc_tokens = seq_data['loinc_tokens']
        value_tokens = seq_data['value_tokens']
        mask = seq_data['missing_mask']
        all_loinc_tokens.append(loinc_tokens)
        all_value_tokens.append(value_tokens)
        all_masks.append(mask)
        seq_lengths.append(len(loinc_tokens))

    max_seq_len = max(seq_lengths)
    print(f"\nSequence lengths - Min: {min(seq_lengths)}, Max: {max_seq_len}, Mean: {np.mean(seq_lengths):.1f}")

    print("Padding sequences...")
    padded_loinc = np.zeros((total_sequences, max_seq_len), dtype=np.int64)
    padded_value = np.zeros((total_sequences, max_seq_len), dtype=np.int64)
    padded_masks = np.zeros((total_sequences, max_seq_len), dtype=np.float32)

    for i, (loinc, value, mask) in enumerate(zip(all_loinc_tokens, all_value_tokens, all_masks)):
        L = len(loinc)
        padded_loinc[i, :L] = loinc
        padded_value[i, :L] = value
        padded_masks[i, :L] = mask

    print("✓ Data loading complete!")
    return padded_loinc, padded_value, padded_masks, max_seq_len, vocab_size

# Train / Val
def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, world_size, rank):  
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        loinc_tokens = batch['loinc_tokens'].to(device)
        value_tokens = batch['value_tokens'].to(device)
        missing_mask = batch['missing_mask'].to(device)
        actual_lengths = batch['actual_lengths'].to(device)

        optimizer.zero_grad()
        
        with autocast():
            loss, _, _ = model(loinc_tokens, value_tokens, missing_mask, mask_ratio=None, actual_lengths=actual_lengths)
            if isinstance(loss, tuple):
                loss = loss[0]
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())

        if rank == 0 and (batch_idx + 1) % 100 == 0:
            print(f"  [Epoch {epoch}] Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    loss_tensor = torch.tensor(total_loss / max(len(dataloader), 1), device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    return float(loss_tensor.item())

@torch.no_grad()
def validate(model, dataloader, device, world_size):
    model.eval()
    sum_loss, num_batches = 0.0, 0
    for batch in dataloader:
        loinc_tokens = batch['loinc_tokens'].to(device)
        value_tokens = batch['value_tokens'].to(device)
        missing_mask = batch['missing_mask'].to(device)
        actual_lengths = batch['actual_lengths'].to(device)

        with autocast():
            loss, _, _ = model(loinc_tokens, value_tokens, missing_mask, mask_ratio=None, actual_lengths=actual_lengths)
            if isinstance(loss, tuple):
                loss = loss[0]
        
        sum_loss += float(loss.item())
        num_batches += 1

    sum_loss_t = torch.tensor(sum_loss, device=device)
    num_batches_t = torch.tensor(num_batches, device=device, dtype=torch.float32)
    dist.all_reduce(sum_loss_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches_t, op=dist.ReduceOp.SUM)
    val_loss = (sum_loss_t / torch.clamp(num_batches_t, min=1.0)).item()
    return val_loss

# Main (DDP)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/labevents_mimic.pkl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)  
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "base", "large"])
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=2e-4)  
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    # DDP init
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    set_seed(42 + rank)

    if rank == 0:
        print(f"DDP world_size={world_size}, local_rank={local_rank}, global_rank={rank}")
        print("Loading data...")

    loinc_tokens, value_tokens, missing_mask, seq_len, vocab_size = load_sequences_from_pickle(args.data)

    n_samples = loinc_tokens.shape[0]
    n_train = int(0.8 * n_samples)

    train_dataset = LabDataset(loinc_tokens[:n_train], value_tokens[:n_train], missing_mask[:n_train])
    val_dataset   = LabDataset(loinc_tokens[n_train:], value_tokens[n_train:], missing_mask[n_train:])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_dynamic,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_dynamic,
        drop_last=False,
    )

    # Model
    if args.model_size == "small":
        model = mae_small(
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_bins=10,
            input_dim=64,
            use_cls_token=True,
            mask_ratio=args.mask_ratio,
            exclude_columns=[0] 
        )
    elif args.model_size == "base":
        model = mae_base(
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_bins=10,
            input_dim=64,
            use_cls_token=True,
            mask_ratio=args.mask_ratio,
            exclude_columns=[0]
        )
    else:
        model = mae_large(
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_bins=10,
            input_dim=64,
            use_cls_token=True,
            mask_ratio=args.mask_ratio,
            exclude_columns=[0]
        )

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    if rank == 0:
        count_parameters(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()  

    best_val_loss = float("inf")
    patience_counter = 0

    if rank == 0:
        print("\n" + "=" * 60)
        print("TRAINING START (DDP + Mixed Precision)")  
        print("=" * 60)
        print(f"epochs={args.epochs}, batch_size_per_gpu={args.batch_size}, num_workers={args.num_workers}")
        print(f"Total batch size: {args.batch_size * world_size}")

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch-1, lr=args.lr, min_lr=args.lr*0.1,
                             max_epochs=args.epochs, warmup_epochs=max(1, args.epochs//20))

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, world_size, rank)  
        val_loss = validate(model, val_loader, device, world_size)

        if rank == 0:
            print(f"\nEpoch [{epoch}/{args.epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                print("  ✓ New best validation loss!")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{args.patience})")

            os.makedirs("checkpoints_mimic", exist_ok=True)
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'seq_len': seq_len,
                'vocab_size': vocab_size,
            }, f'checkpoints_mimic/epoch_{epoch}.pth', is_best=is_best)

        stop_t = torch.tensor(1 if patience_counter >= args.patience else 0, device=device)
        dist.broadcast(stop_t, src=0)
        if int(stop_t.item()) == 1:
            if rank == 0:
                print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    if rank == 0:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best validation loss: {best_val_loss:.4f}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()