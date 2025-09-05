"""
Scaled Rotary Position Embedding (RoPE) implementation for wav2vec2_2d
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class ScaledRoPE(nn.Module):
    """
    Scaled Rotary Position Embedding (RoPE) for neural data
    
    This implementation applies rotary position embeddings to neural data,
    where positions correspond to different channels or time steps.
    """
    
    def __init__(self, 
                 dim: int, 
                 max_seq_len: int = 4096,
                 base_freq: float = 10000.0,
                 scale_factor: float = 1.0,
                 theta: float = 10000.0):
        """
        Args:
            dim: Embedding dimension (must be even)
            max_seq_len: Maximum sequence length
            base_freq: Base frequency for position encoding
            scale_factor: Scaling factor for extrapolation to longer sequences
            theta: Theta parameter for frequency computation
        """
        super().__init__()
        
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor
        self.theta = theta
        
        # Precompute frequency matrix
        self.register_buffer('freqs', self._compute_freqs())
        
    def _compute_freqs(self) -> torch.Tensor:
        """Compute frequency matrix for RoPE"""
        # Create position indices
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        
        # Create dimension indices
        dim_indices = torch.arange(0, self.dim, 2, dtype=torch.float32)
        
        # Compute frequencies: 1 / (theta^(2i/d))
        freqs = 1.0 / (self.theta ** (dim_indices / self.dim))
        
        # Create frequency matrix: [max_seq_len, dim//2]
        freqs = positions.unsqueeze(1) * freqs.unsqueeze(0)
        
        # Apply scaling factor for extrapolation
        freqs = freqs * self.scale_factor
        
        return freqs
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply RoPE to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            positions: Optional position indices of shape [seq_len]
            
        Returns:
            Tensor with RoPE applied
        """
        batch_size, seq_len, dim = x.shape
        
        # Use provided positions or default to sequential
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        
        # Ensure positions are within bounds
        positions = positions.clamp(0, self.max_seq_len - 1)
        
        # Get frequencies for current positions
        freqs = self.freqs[positions]  # [seq_len, dim//2]
        
        # Expand to full dimension
        freqs = freqs.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, dim//2]
        
        # Create cos and sin values
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        
        # Interleave cos and sin to create rotation matrix
        # [batch_size, seq_len, dim] -> [batch_size, seq_len, dim//2, 2]
        x_reshaped = x.view(batch_size, seq_len, dim // 2, 2)
        
        # Apply rotation
        x_rotated = torch.stack([
            x_reshaped[..., 0] * cos_freqs - x_reshaped[..., 1] * sin_freqs,
            x_reshaped[..., 0] * sin_freqs + x_reshaped[..., 1] * cos_freqs
        ], dim=-1)
        
        # Reshape back to original shape
        x_rotated = x_rotated.view(batch_size, seq_len, dim)
        
        return x_rotated


class ScaledRoPEAttention(nn.Module):
    """
    Attention mechanism with Scaled RoPE integration
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 max_seq_len: int = 4096,
                 scale_factor: float = 1.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # RoPE for queries and keys
        self.rope_q = ScaledRoPE(self.head_dim, max_seq_len, scale_factor=scale_factor)
        self.rope_k = ScaledRoPE(self.head_dim, max_seq_len, scale_factor=scale_factor)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                positions: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with RoPE
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            positions: Position indices [seq_len]
            attn_mask: Attention mask [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE to queries and keys
        q = self.rope_q(q, positions)
        k = self.rope_k(k, positions)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_proj(out)
        
        return out


def create_channel_positions(num_channels: int, 
                           max_depth: float = 3.8,
                           device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Create position indices for neural channels based on depth
    
    Args:
        num_channels: Number of neural channels
        max_depth: Maximum depth in mm
        device: Device for tensor creation
        
    Returns:
        Position indices [num_channels]
    """
    # Create depth-based positions
    depths = torch.linspace(0, max_depth, num_channels, device=device)
    
    # Convert to position indices (can be scaled as needed)
    positions = torch.arange(num_channels, device=device, dtype=torch.long)
    
    return positions
