"""
Tokenization includes 1 tokens for 1 lab test (seperately).
Sequence format: [CLS, test1, test2, test3, ...]
  - loinc_tokens: [CLS, loinc1, loinc2, ...]
  - value_tokens: [0.0, value1, value2, ...]
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from utils import get_1d_sincos_pos_embed, init_weights

eps = 1e-6

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            # attn_mask: [B, N] -> [B, 1, 1, N] for broadcasting
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            # Mask out padding positions (where attn_mask == 0)
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask) 
        x = x + self.mlp(self.norm2(x))

        return x

class MAE(nn.Module):
    """
    Masked Autoencoder for pre-computed lab embeddings
    
    Args:
        seq_len: Sequence length 
        encoder_embed_dim: Encoder hidden dimension
        decoder_embed_dim: Decoder hidden dimension
        encoder_depth: Number of encoder layers
        decoder_depth: Number of decoder layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        use_cls_token: Whether to use CLS token
        mask_ratio: Default masking ratio for training
    """
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        num_bins: int=10,
        input_dim: int=64,
        encoder_embed_dim: int=768,
        decoder_embed_dim: int=512,
        encoder_depth: int=12,
        decoder_depth: int=8,
        num_heads: int=12,
        mlp_ratio: float=4.0,
        use_cls_token: bool=True,
        mask_ratio: float=0.75,
        drop_ratio: float=0.,
        attn_drop_ratio: float=0.,
        exclude_columns: List[int] = []
    ):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_bins = num_bins
        self.input_dim = input_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.use_cls_token = use_cls_token
        self.mask_ratio = mask_ratio
        self.exclude_columns = exclude_columns

        # tokens to embeddings
        self.token_embedding = nn.Embedding(vocab_size, input_dim)
        # 0-3 means special tokens; 4-13 means bins
        self.value_embedding = nn.Embedding(num_bins+4, input_dim)

        # Patch layer
        self.patch_embed = nn.Sequential(
            nn.Linear(input_dim, encoder_embed_dim),
            nn.GELU(),
            nn.Linear(encoder_embed_dim, encoder_embed_dim)
        )
        # Encoder position embeddings 
        encoder_pos_embed = get_1d_sincos_pos_embed(
            embed_dim=encoder_embed_dim,
            pos=seq_len,
            cls_token=use_cls_token
        )
        self.encoder_pos_embed = nn.Parameter(
            torch.from_numpy(encoder_pos_embed).float().unsqueeze(0),
            requires_grad=False
        )

        # CLS tokens
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
            torch.nn.init.normal_(self.cls_token, std=0.02)
        
        # Encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(
                dim=encoder_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_ratio,
                attn_drop=attn_drop_ratio
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)
        
        # Encoder-to-Decoder projection
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)

        # Mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Decoder position embeddings
        decoder_pos_embed = get_1d_sincos_pos_embed(
            embed_dim=decoder_embed_dim,
            pos=seq_len,
            cls_token=use_cls_token
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0),
            requires_grad=False
        )

        # Decoder
        self.decoder = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_ratio,
                attn_drop=attn_drop_ratio
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        # Reconstruction head
        self.decoder_pred = nn.Linear(decoder_embed_dim, input_dim, bias=True)
        torch.nn.init.xavier_uniform_(self.decoder_pred.weight, gain=0.01)
        torch.nn.init.zeros_(self.decoder_pred.bias) 

        # Weight initialization
        self.apply(init_weights)
    
    def embed_tokens(
        self,
        loinc_tokens: torch.Tensor,
        value_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Deal with loinc and value tokens seperately.
        Args:
            loinc_tokens: (B, N), CLS + loinc codes
            value_tokens: (B, N), bin numbers (0-9) or MISSING token
        
        Returns:
            embeddings: (B, N, input_dim)
        """
        B, N = loinc_tokens.shape

        # loinc code embedding
        loinc_emb = self.token_embedding(loinc_tokens.long())

        # value embedding
        value_emb = self.value_embedding(value_tokens.long())

        embeddings = loinc_emb + value_emb
        embeddings[:, 0, :] = loinc_emb[:, 0, :]

        return embeddings

    def random_masking(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        mask_ratio: float,
        actual_lengths: torch.Tensor,
        exclude_columns: List[int] = []
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        
        Args:
            x: [N, L, D], sequence embeddings with positional encoding
            m: [N, L], missing mask (1=observed, 0=missing)
            mask_ratio: float, ratio of observed tokens to mask
            exclude_columns: list of column indices to exclude from masking
        
        Returns:
            x_masked: [N, max_len_keep, D], visible tokens only (physically removed)
            mask: [N, L], binary mask (1=masked by MAE, 0=kept)
            nask: [N, L], inverse mask (1=kept, 0=masked)
            ids_restore: [N, L], indices to restore original order
            attn_mask: [N, max_len_keep], attention mask for encoder (1=valid, 0=padding)
        """
        N, L, D = x.shape 
        
        effective_lengths = actual_lengths.float()
        len_keep = torch.ceil(effective_lengths * (1 - mask_ratio)).long()
        
        # Generate random noise for shuffling
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        for i in range(N):
            actual_len = actual_lengths[i].item()
            noise[i, actual_len:] = 2.0  # padding positions get highest noise
        
        # Missing values get high noise (but lower than padding)
        noise[m < eps] = 1.5

        # Exclude specific columns from masking 
        if exclude_columns is not None and len(exclude_columns) > 0:
            noise[:, exclude_columns] = 0  
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # indices to restore original order
        
        # Keep the first subset as per len_keep
        ids_keep_list = [ids_shuffle[i, :len_keep[i]] for i in range(N)]
        
        # Pad ids_keep to max_len_keep for batching
        max_len_keep = len_keep.max().item()
        ids_keep_padded = [
            torch.cat([
                ids_keep_list[i], 
                torch.full((max_len_keep - len_keep[i],), L, device=x.device)  # pad with L
            ], dim=0) 
            for i in range(N)
        ]
        ids_keep = torch.stack(ids_keep_padded, dim=0)  # [N, max_len_keep]
        
        # Add padding to x, then gather visible tokens
        x_padded = torch.cat([x, torch.zeros((N, 1, D), device=x.device)], dim=1)  # [N, L+1, D]
        x_masked = torch.gather(
            x_padded, dim=1, 
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )  # [N, max_len_keep, D]
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        for i in range(N):
            mask[i, :len_keep[i]] = 0  # first len_keep[i] are kept
        
        mask = torch.gather(mask, dim=1, index=ids_restore)  # restore to original order
        nask = torch.ones([N, L], device=x.device) - mask  # inverse mask
        
        # Create attention mask for variable-length sequences
        attn_mask = torch.zeros(N, max_len_keep, device=x.device)
        for i in range(N):
            attn_mask[i, :len_keep[i]] = 1  # 1=valid token, 0=padding
        
        return x_masked, mask, nask, ids_restore, attn_mask
    
    def forward_encoder(
        self,
        loinc_tokens: torch.Tensor,   
        value_tokens: torch.Tensor,
        missing_mask: torch.Tensor,
        mask_ratio: float,
        actual_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encoder forward pass
        
        Args:
            loinc_tokens: (B, N), CLS + loinc codes
            value_tokens: (B, N), lab test values
            missing_mask: (B, N) 
            mask_ratio: masking ratio
            actual_lengths: (B,) actual sequence length for each sample (excluding padding)
        
        Returns:
            latent: (B, max_len_keep+1, encoder_embed_dim) encoded features (if use_cls_token)
            mask: (B, N) MAE mask (1=masked, 0=kept)
            nask: (B, N) inverse mask (1=kept, 0=masked)
            ids_restore: (B, N) restore indices
            attn_mask: (B, max_len_keep+1) attention mask for encoder
        """
        B, N = loinc_tokens.shape
        
        # If actual_lengths not provided, compute from loinc_tokens (backward compatibility)
        if actual_lengths is None:
            actual_lengths = (loinc_tokens != 0).sum(dim=1)
        
        # Tokens -> embeddings
        features = self.embed_tokens(loinc_tokens, value_tokens) 
        
        # Project to encoder dimension
        x = self.patch_embed(features)  
        
        # Add positional embeddings
        if self.use_cls_token:
            x = x + self.encoder_pos_embed[:, 1:, :]  # skip cls token position
        else:
            x = x + self.encoder_pos_embed
        
        # Random masking with attention mask (pass actual_lengths)
        x_masked, mask, nask, ids_restore, attn_mask = self.random_masking(
            x, missing_mask, mask_ratio, actual_lengths, self.exclude_columns
        )  # x_masked: (B, max_len_keep, encoder_embed_dim)
        
        # Add CLS token if needed
        if self.use_cls_token:
            cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)
            x_masked = torch.cat([cls_tokens, x_masked], dim=1)  # (B, max_len_keep+1, D)
            
            # CLS token can attend to everything (add 1 to attn_mask)
            cls_mask = torch.ones(B, 1, device=x_masked.device)
            attn_mask = torch.cat([cls_mask, attn_mask], dim=1)  # (B, max_len_keep+1)
        
        # Pass attention mask through encoder blocks
        for block in self.encoder:
            x_masked = block(x_masked, attn_mask=attn_mask)
        
        x_masked = self.encoder_norm(x_masked)
        
        return x_masked, mask, nask, ids_restore, attn_mask
    
    def forward_decoder(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None 
    ) -> torch.Tensor:
        """
        Decoder forward pass
        
        Args:
            latent: (B, max_len_keep+1, encoder_embed_dim) encoder output
            ids_restore: (B, L) restore indices
            attn_mask: (B, max_len_keep+1) attention mask from encoder (optional)
        
        Returns:
            pred: (B, L, input_dim) reconstructed features
        """
        B = latent.shape[0]
        
        # Project to decoder dimension
        x = self.decoder_embed(latent)  # (B, max_len_keep+1, decoder_embed_dim)
        
        # Remove CLS token before inserting mask tokens
        if self.use_cls_token:
            x_no_cls = x[:, 1:, :]  # (B, max_len_keep, decoder_embed_dim)
        else:
            x_no_cls = x
        
        # Calculate actual number of visible tokens per sample (remove padding)
        if attn_mask is not None and self.use_cls_token:
            # attn_mask: (B, max_len_keep+1), remove cls position
            valid_mask = attn_mask[:, 1:]  # (B, max_len_keep)
            num_visible = valid_mask.sum(dim=1).long()  # (B,)
        else:
            num_visible = torch.full((B,), x_no_cls.shape[1], device=x_no_cls.device)
        
        # Create mask tokens for masked positions
        L = ids_restore.shape[1]
        max_num_visible = x_no_cls.shape[1]
        
        # For each sample, replace padding with mask tokens
        x_full_list = []
        for i in range(B):
            n_visible = num_visible[i].item()
            n_masked = L - n_visible
            
            # Get valid visible tokens (remove padding)
            x_visible_i = x_no_cls[i, :n_visible, :]  # (n_visible, D)
            
            # Create mask tokens
            mask_tokens_i = self.mask_token.squeeze(0).repeat(n_masked, 1)  # (n_masked, D)
            
            # Concatenate
            x_full_i = torch.cat([x_visible_i, mask_tokens_i], dim=0)  # (L, D)
            
            # Unshuffle to restore original order
            x_full_i = x_full_i[ids_restore[i]]  # (L, D)
            
            x_full_list.append(x_full_i)
        
        x_full = torch.stack(x_full_list, dim=0)  # (B, L, decoder_embed_dim)
        
        # Add CLS token back
        if self.use_cls_token:
            x_full = torch.cat([x[:, :1, :], x_full], dim=1)  # (B, L+1, D)
        
        # Add decoder position embeddings
        x_full = x_full + self.decoder_pos_embed
        
        # Apply decoder blocks (no attention mask needed in decoder)
        for block in self.decoder:
            x_full = block(x_full, attn_mask=None)
        
        x_full = self.decoder_norm(x_full)
        
        # Prediction head
        pred = self.decoder_pred(x_full)  
        
        # Remove CLS token from predictions
        if self.use_cls_token:
            pred = pred[:, 1:, :] 
        
        return pred
    
    def forward_loss(
        self,
        loinc_tokens: torch.Tensor,   
        value_tokens: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        missing_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss on masked positions only
        
        Args:
            target: (B, N, input_dim) original features
            pred: (B, N, input_dim) predicted features
            mask: (B, N) - 1=masked by MAE, 0=visible
            missing_mask: (B, N) - 1=observed, 0=originally missing
        
        Returns:
            loss: MSE
        """
        with torch.no_grad():
            target = self.embed_tokens(loinc_tokens, value_tokens)  # (B, L, input_dim)
            valid_mask = (mask * missing_mask) > 0.5
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        loss = torch.nn.functional.mse_loss(
            pred[valid_mask], 
            target[valid_mask],
            reduction='mean'
        )
        
        return loss

    def forward(
        self,
        loinc_tokens: torch.Tensor,   
        value_tokens: torch.Tensor,
        missing_mask: torch.Tensor,
        mask_ratio: Optional[float] = None,
        actual_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            loinc_tokens: (B, N), CLS + loinc codes
            value_tokens: (B, N), lab test values
            missing_mask: (B, N)
            mask_ratio: masking ratio (if None, use self.mask_ratio)
            actual_lengths: (B,) actual sequence length for each sample

        Returns:
            loss: scalar reconstruction loss
            pred: (B, L, input_dim) predicted features
            mask: (B, L) MAE mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Encoder: embed and mask
        latent, mask, nask, ids_restore, attn_mask = self.forward_encoder(
            loinc_tokens, value_tokens, missing_mask, mask_ratio, actual_lengths
        )

        # Decoder: reconstruct
        pred = self.forward_decoder(latent, ids_restore, attn_mask)

        # Loss
        loss = self.forward_loss(loinc_tokens, value_tokens, pred, mask, missing_mask)

        return loss, pred, mask

    def encode_only(
        self,
        loinc_tokens: torch.Tensor,   
        value_tokens: torch.Tensor,
        missing_mask: torch.Tensor,
        mask_ratio: float = 0.0,
        actual_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Extract encoder features without reconstruction
        
        Args:
            loinc_tokens: (B, N), CLS + loinc codes
            value_tokens: (B, N), lab test values
            missing_mask: (B, L)
            mask_ratio: 0.0 for no masking, >0 for masked encoding
            actual_lengths: (B,) actual sequence length for each sample
        
        Returns:
            features: (B, num_tokens, encoder_embed_dim) encoder output
        """
        latent, _, _, _, _ = self.forward_encoder(
            loinc_tokens, value_tokens, missing_mask, mask_ratio, actual_lengths
        )
        return latent

    def reconstruct(
        self,
        loinc_tokens: torch.Tensor,   
        value_tokens: torch.Tensor,
        missing_mask: torch.Tensor,
        mask_ratio: float = 0.75,
        actual_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct embeddings without computing loss
        
        Args:
            loinc_tokens: (B, N), CLS + loinc codes
            value_tokens: (B, N), lab test values
            missing_mask: (B, L)
            mask_ratio: masking ratio
            actual_lengths: (B,) actual sequence length for each sample
        
        Returns:
            pred: (B, L, input_dim) predictions
            mask: (B, L) which tokens were masked
        """
        latent, mask, nask, ids_restore, attn_mask = self.forward_encoder(
            loinc_tokens, value_tokens, missing_mask, mask_ratio, actual_lengths
        )
        pred = self.forward_decoder(latent, ids_restore, attn_mask)
        return pred, mask

    
def mae_small(**kwargs):
    """Small MAE model for quick experiments"""
    if 'input_dim' not in kwargs or 'seq_len' not in kwargs:
        raise ValueError("Must provide 'input_dim' and 'seq_len'")
    
    if 'num_bins' not in kwargs:
        kwargs['num_bins'] = 10

    model = MAE(
        encoder_embed_dim=256,
        decoder_embed_dim=128,
        encoder_depth=6,
        decoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        **kwargs
    )
    return model    


def mae_base(**kwargs):
    """Base MAE model (balanced)"""
    if 'input_dim' not in kwargs or 'seq_len' not in kwargs:
        raise ValueError("Must provide 'input_dim' and 'seq_len'")

    if 'num_bins' not in kwargs:
        kwargs['num_bins'] = 10

    model = MAE(
        encoder_embed_dim=512,
        decoder_embed_dim=256,
        encoder_depth=8,
        decoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        **kwargs
    )
    return model


def mae_large(**kwargs):
    """Large MAE model (best performance)"""
    if 'input_dim' not in kwargs or 'seq_len' not in kwargs:
        raise ValueError("Must provide 'input_dim' and 'seq_len'")
    
    if 'num_bins' not in kwargs:
        kwargs['num_bins'] = 10
        
    model = MAE(
        encoder_embed_dim=1024,
        decoder_embed_dim=512,
        encoder_depth=12,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs
    )
    return model


    