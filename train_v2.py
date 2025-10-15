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
import wandb

from MAE_v2 import mae_large, mae_base, mae_small
from utils_v2 import (
    set_seed,
    count_parameters,
    LabDataset,
    save_checkpoint,
    adjust_learning_rate,
    collate_fn_fixed_length,
)

# Data utils
def load_sequences_from_pickle(filepath):
    """Load sequences and return percentiles for fixed-length padding"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    vocab_size = int(data['vocab_info']['vocab_size'])
    patient_sequences = data['patient_sequences']

    all_loinc_tokens, all_value_tokens, all_masks, seq_lengths = [], [], [], []
    for seq_data in patient_sequences.values():
        all_loinc_tokens.append(seq_data['loinc_tokens'])
        all_value_tokens.append(seq_data['value_tokens'])
        all_masks.append(seq_data['missing_mask'])
        seq_lengths.append(len(seq_data['loinc_tokens']))

    max_seq_len = max(seq_lengths)
    seq_lengths_array = np.array(seq_lengths)
    
    # Calculate percentiles
    percentiles = {
        'p95': int(np.percentile(seq_lengths_array, 95)),
        'p99': int(np.percentile(seq_lengths_array, 99)),
    }

    # Pad to max length
    total_sequences = len(all_loinc_tokens)
    padded_loinc = np.zeros((total_sequences, max_seq_len), dtype=np.int64)
    padded_value = np.zeros((total_sequences, max_seq_len), dtype=np.int64)
    padded_masks = np.zeros((total_sequences, max_seq_len), dtype=np.float32)

    for i, (loinc, value, mask) in enumerate(zip(all_loinc_tokens, all_value_tokens, all_masks)):
        L = len(loinc)
        padded_loinc[i, :L] = loinc
        padded_value[i, :L] = value
        padded_masks[i, :L] = mask

    return padded_loinc, padded_value, padded_masks, max_seq_len, vocab_size, percentiles

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

        if (batch_idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

        if rank == 0 and (batch_idx + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']  
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/learning_rate": current_lr,
                "train/epoch": epoch,
                "train/batch": batch_idx + 1,
            })
            print(f"  [Epoch {epoch}] Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        del loinc_tokens, value_tokens, missing_mask, actual_lengths, loss 

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
    parser.add_argument("--data", type=str, default="data/labevents_mimic_trainval.pkl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)  
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "base", "large"])
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=2e-4)  
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--drop_ratio", type=float, default=0.1)
    parser.add_argument("--attn_drop_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--fixed_length", type=int, default=120, help="Fixed sequence length for padding")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
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
        print(f"DDP: world_size={world_size}, rank={rank}")
    
    # WandB init
    if rank == 0:
        os.environ['WANDB_MODE'] = 'online'
        os.environ['WANDB_DISABLED'] = 'false'
        os.environ['WANDB_API_KEY'] = '6cf65e599a667502406f2f26fd010cf12ac94b99'
        
        try:
            wandb.login(key='6cf65e599a667502406f2f26fd010cf12ac94b99', relogin=True, force=True)
            wandb.init(
                project="mae-mimic",
                name=f"{args.model_size}_bs{args.batch_size * world_size}_lr{args.lr}_fixlen{args.fixed_length}",
                config={
                    "model_size": args.model_size,
                    "batch_size_per_gpu": args.batch_size,
                    "total_batch_size": args.batch_size * world_size,
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "mask_ratio": args.mask_ratio,
                    "weight_decay": args.weight_decay,
                    "patience": args.patience,
                    "num_gpus": world_size,
                    "fixed_length": args.fixed_length,
                },
                mode="online",
                settings=wandb.Settings(start_method="thread", _disable_stats=False, _disable_meta=False)
            )
            if rank == 0:
                print(f"WandB initialized: {wandb.run.get_url()}")
        except Exception as e:
            print(f"WandB init failed: {e}")

    # Load data
    loinc_tokens, value_tokens, missing_mask, max_seq_len, vocab_size, percentiles = load_sequences_from_pickle(args.data)
    
    FIXED_LENGTH = args.fixed_length
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Fixed Length: {FIXED_LENGTH} | Max: {max_seq_len} | 95th: {percentiles['p95']}")
        print(f"{'='*60}\n")

    # Split data
    n_samples = loinc_tokens.shape[0]
    n_train = int(0.8 * n_samples)

    train_dataset = LabDataset(loinc_tokens[:n_train], value_tokens[:n_train], missing_mask[:n_train])
    val_dataset = LabDataset(loinc_tokens[n_train:], value_tokens[n_train:], missing_mask[n_train:])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn_fixed_length(batch, fixed_length=FIXED_LENGTH),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn_fixed_length(batch, fixed_length=FIXED_LENGTH),
        drop_last=False,
    )

    # Model
    if args.model_size == "small":
        model = mae_small(
            seq_len=FIXED_LENGTH,
            vocab_size=vocab_size,
            num_bins=10,
            input_dim=64,
            use_cls_token=True,
            mask_ratio=args.mask_ratio,
            drop_ratio=args.drop_ratio,
            attn_drop_ratio=args.attn_drop_ratio,
            exclude_columns=[0]
        )
    elif args.model_size == "base":
        model = mae_base(
            seq_len=FIXED_LENGTH,
            vocab_size=vocab_size,
            num_bins=10,
            input_dim=128,
            use_cls_token=True,
            mask_ratio=args.mask_ratio,
            drop_ratio=args.drop_ratio,
            attn_drop_ratio=args.attn_drop_ratio,
            exclude_columns=[0]
        )
    else:
        model = mae_large(
            seq_len=FIXED_LENGTH,
            vocab_size=vocab_size,
            num_bins=10,
            input_dim=256,
            use_cls_token=True,
            mask_ratio=args.mask_ratio,
            drop_ratio=args.drop_ratio,
            attn_drop_ratio=args.attn_drop_ratio,
            exclude_columns=[0]
        )

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    if rank == 0:
        count_parameters(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    
    if args.resume:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Loading checkpoint from: {args.resume}")
            print(f"{'='*60}\n")
        
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float("inf"))
        
        if rank == 0:
            print(f"✓ Resumed from epoch {checkpoint['epoch']}")
            print(f"✓ Best val loss so far: {best_val_loss:.4f}")
            print(f"✓ Starting from epoch {start_epoch}\n")

    if rank == 0:
        print(f"\nTraining starts: epochs={args.epochs}, batch_size={args.batch_size * world_size}\n")

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch-1, lr=args.lr, min_lr=args.min_lr,
                             max_epochs=args.epochs, warmup_epochs=args.warmup_epochs)

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, world_size, rank)
        val_loss = validate(model, val_loader, device, world_size)

        if rank == 0:
            wandb.log({
                "epoch/train_loss": train_loss,
                "epoch/val_loss": val_loss,
                "epoch/epoch": epoch,
            })
            print(f"Epoch [{epoch}/{args.epochs}]  Train: {train_loss:.4f}  Val: {val_loss:.4f}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                print("  ✓ New best!")
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
                'seq_len': FIXED_LENGTH,
                'vocab_size': vocab_size,
            }, f'checkpoints_mimic/epoch_{epoch}.pth', is_best=is_best)

        stop_t = torch.tensor(1 if patience_counter >= args.patience else 0, device=device)
        dist.broadcast(stop_t, src=0)
        if int(stop_t.item()) == 1:
            if rank == 0:
                print(f"\nEarly stopping at epoch {epoch}")
            break

    if rank == 0:
        print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
        wandb.finish()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
