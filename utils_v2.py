"""
utils functions
"""
import torch
import numpy as np
import torch.nn as nn
import math
import torch.utils.data as data
import os

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
        tmp_lr = lr * epoch / warmup_epochs 
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
        self.loinc_tokens = torch.from_numpy(loinc_tokens) 
        self.value_tokens = torch.from_numpy(value_tokens)
        self.missing_mask = torch.from_numpy(missing_mask).float() if isinstance(missing_mask, np.ndarray) else missing_mask
        self.targets = targets

        assert self.tokens.shape[:2] == self.missing_mask.shape, \
            f"Shape mismatch: tokens {self.tokens.shape[:2]} vs missing_mask {self.missing_mask.shape}"
        
    def __len__(self):
        return len(self.loinc_tokens)
    
    def __getitem__(self, idx):
        return {
            'loinc_tokens': self.loinc_tokens[idx],  
            'value_tokens': self.value_tokens[idx],  
            'missing_mask': self.missing_mask[idx]
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
        nn.init.normal_(module.weight, std=0.02)