"""
utils functions
"""
import torch
import numpy as np
import torch.nn as nn
import math
import torch.utils.data as data
import os
from typing import List, Dict, Any

PAD_LOINC_ID: int = 0
PAD_VALUE_ID: int = 0
PAD_MASK_VAL: float = 0.0

# Helper function
def _to_1d_long(x) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.ndim != 1:
        t = t.view(-1)
    return t.long()

def _to_1d_float(x) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.ndim != 1:
        t = t.view(-1)
    return t.float()

def get_1d_sincos_pos_embed(embed_dim, pos, cls_token=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed

def adjust_learning_rate(optimizer, epoch, lr, min_lr, max_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        tmp_lr = min_lr + (lr - min_lr) * (epoch + 1) / warmup_epochs 
    else:
        tmp_lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = tmp_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = tmp_lr
    return tmp_lr


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class LabDataset(data.Dataset):
    def __init__(self, loinc_tokens, value_tokens, missing_mask):
        """
        Args:
            loinc_tokens: (N, seq_len) numpy array
            value_tokens: (N, seq_len) numpy array
            missing_mask: (N, seq_len) from 'missing_mask'
        """
        self.loinc_tokens: List[torch.Tensor] = []
        self.value_tokens: List[torch.Tensor] = []
        self.missing_mask: List[torch.Tensor] = []
        self.actual_lengths: List[int] = []
        
        for lt, vt, mm in zip(loinc_tokens, value_tokens, missing_mask):
            lt_t = _to_1d_long(lt)
            vt_t = _to_1d_long(vt)
            mm_t = _to_1d_float(mm)

            self.loinc_tokens.append(lt_t)
            self.value_tokens.append(vt_t)
            self.missing_mask.append(mm_t)

            actual_len = int((lt_t != PAD_LOINC_ID).sum().item())
            self.actual_lengths.append(actual_len)
    
    def __len__(self):
        return len(self.loinc_tokens)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        return {
            'loinc_tokens': self.loinc_tokens[idx],
            'value_tokens': self.value_tokens[idx],
            'missing_mask': self.missing_mask[idx],
            'actual_len': self.actual_lengths[idx]
        }

def save_checkpoint(state, filepath, is_best=False):
    """
    Save training checkpoint
    
    Args:
        state: dict containing model, optimizer, epoch, etc.
        filepath: where to save (e.g., 'checkpoints/epoch_10.pth')
        is_best: if True, also save as 'best_model.pth'
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"✓ Checkpoint saved: {filepath}")
    
    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), 'best_model.pth')
        torch.save(state, best_path)
        print(f"✓ Best model saved: {best_path}")

def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load checkpoint and restore training state
    
    Args:
        filepath: path to checkpoint file
        model: model to load weights into
        optimizer: optimizer to load state into (optional)
        device: device to map tensors to
        
    Returns:
        checkpoint dict with epoch, loss, etc.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded: {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return checkpoint

def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")

def count_parameters(model):
    """
    Count total and trainable parameters
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {total - trainable:,}")
    
    return total, trainable

def init_weights(module):
    """
    Initialize model weights following MAE conventions
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.1)

def collate_fn_fixed_length(batch: List[Dict[str, Any]], fixed_length: int = 120) -> Dict[str, torch.Tensor]:
    """
    Fixed-length collate function (NO dynamic padding).
    All sequences are padded/truncated to the same fixed length.
    
    Args:
        batch: List of samples
        fixed_length: Fixed sequence length for ALL batches (default: 80)
    
    Returns:
        Batched tensors with uniform fixed length
    """
    B = len(batch)
    
    loinc_padded = torch.full((B, fixed_length), fill_value=PAD_LOINC_ID, dtype=torch.long)
    value_padded = torch.full((B, fixed_length), fill_value=PAD_VALUE_ID, dtype=torch.long)
    mask_padded = torch.full((B, fixed_length), fill_value=PAD_MASK_VAL, dtype=torch.float32)
    actual_lengths = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        actual_len = int(item['actual_len'])
        # Cap length at fixed_length
        L = min(actual_len, fixed_length)
        actual_lengths[i] = L
        
        loinc = item['loinc_tokens']
        value = item['value_tokens']
        mask = item['missing_mask']

        loinc_padded[i, :L] = loinc[:L].long()
        value_padded[i, :L] = value[:L].long()
        mask_padded[i, :L] = mask[:L].float()

    return {
        'loinc_tokens': loinc_padded,
        'value_tokens': value_padded,
        'missing_mask': mask_padded,
        'actual_lengths': actual_lengths
    }
