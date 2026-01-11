"""
WanControlNet - Uni3C ControlNet for Wan video generation.

Ported from Kijai's ComfyUI-WanVideoWrapper implementation.
https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/uni3c/controlnet.py

This provides video-to-video structural guidance through a ControlNet that
extracts features from a guide/reference video and injects them into the
main transformer's generation process.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Generate sinusoidal positional embeddings (1D).
    
    Args:
        dim: Embedding dimension (must be even)
        position: Position tensor
        
    Returns:
        Sinusoidal embeddings of shape [len(position), dim]
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.float()
    
    sinusoid = torch.outer(
        position, 
        torch.pow(10000.0, -torch.arange(half, device=position.device, dtype=torch.float32) / half)
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class WanRMSNorm(nn.Module):
    """RMS Normalization layer."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, L, C]
        """
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Args:
        q: Query tensor [B, S, H, D]
        k: Key tensor [B, S, H, D]
        freqs_cos: Cosine frequencies [S, D]
        freqs_sin: Sine frequencies [S, D]
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Expand freqs for broadcasting: [S, D] -> [1, S, 1, D]
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    
    q_embed = (q * freqs_cos) + (rotate_half(q) * freqs_sin)
    k_embed = (k * freqs_cos) + (rotate_half(k) * freqs_sin)
    
    return q_embed, k_embed


class WanControlnetSelfAttention(nn.Module):
    """
    Self-attention module for ControlNet transformer blocks.
    Simplified version focused on the ControlNet use case.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        attention_mode: str = "sdpa",
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.attention_mode = attention_mode

        # Projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
        # Normalization
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, C]
            freqs: Optional tuple of (freqs_cos, freqs_sin) for RoPE
            
        Returns:
            Output tensor [B, L, C]
        """
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        
        # Compute Q, K, V
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        
        # Apply RoPE if provided
        if freqs is not None:
            freqs_cos, freqs_sin = freqs
            q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        
        # Attention - use scaled_dot_product_attention (SDPA)
        # Transpose for attention: [B, S, H, D] -> [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # SDPA
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        # Transpose back and reshape
        x = x.transpose(1, 2).reshape(b, s, -1)
        
        # Output projection
        x = self.o(x)
        
        return x


class WanControlnetTransformerBlock(nn.Module):
    """
    Transformer block for the ControlNet.
    
    Architecture:
    - Self attention with RoPE
    - Feed-forward network (MLP with GELU)
    - AdaLN-style modulation from time embedding
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        attention_mode: str = "sdpa",
        eps: float = 1e-6,
    ):
        super().__init__()
        
        self.dim = dim
        self.ffn_dim = ffn_dim
        
        # Self attention
        self.self_attn = WanControlnetSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=True,
            eps=eps,
            attention_mode=attention_mode,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        
        # Normalization layers
        self.norm1 = WanRMSNorm(dim, eps=eps)
        self.norm2 = WanRMSNorm(dim, eps=eps)
        
        # AdaLN modulation - projects time embedding to scale/shift params
        # 6 params per block: (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(5120, 6 * dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, C]
            temb: Time embedding [B, 5120]
            freqs: Optional RoPE frequencies
            
        Returns:
            Output tensor [B, L, C]
        """
        # Get modulation parameters from time embedding
        modulation = self.adaLN_modulation(temb)  # [B, 6*dim]
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
        
        # Self attention block with modulation
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + gate1.unsqueeze(1) * self.self_attn(x_norm, freqs=freqs)
        
        # FFN block with modulation
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.ffn(x_norm)
        
        return x


class WanControlNet(nn.Module):
    """
    Uni3C ControlNet for Wan video generation.
    
    Takes a guide video (encoded to latents) and produces per-block
    control signals to inject into the main transformer.
    
    Architecture:
    - Patch embedding (Conv3D with stride (1,2,2))
    - Time embedding MLP
    - 20 transformer blocks
    - Per-block output projections to match main model dimension
    
    The forward pass returns a list of 20 tensors, one per block,
    each of shape [B, seq_len, 5120].
    """

    def __init__(
        self,
        in_channels: int,
        conv_out_dim: int = 5120,
        time_embed_dim: int = 5120,
        dim: int = 1024,
        ffn_dim: int = 8192,
        num_heads: int = 16,
        num_layers: int = 20,
        add_channels: int = 7,
        mid_channels: int = 256,
        attention_mode: str = "sdpa",
        quantized: bool = False,
        base_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.dim = dim
        self.num_layers = num_layers
        self.patch_size = (1, 2, 2)  # Temporal, Height, Width
        self.dtype = base_dtype
        
        # Patch embedding: Conv3D
        self.controlnet_patch_embedding = nn.Conv3d(
            in_channels, 
            conv_out_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        
        # Project patch embeddings down to transformer dim
        self.proj_in = nn.Linear(conv_out_dim, dim)
        
        # Time embedding MLP
        self.time_embedding = nn.Sequential(
            nn.Linear(256, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Transformer blocks
        self.controlnet_blocks = nn.ModuleList([
            WanControlnetTransformerBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                attention_mode=attention_mode,
            )
            for _ in range(num_layers)
        ])
        
        # Per-block output projections to main model dimension (5120)
        self.proj_out = nn.ModuleList([
            nn.Linear(dim, 5120)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        render_latent: torch.Tensor,
        render_mask: Optional[torch.Tensor] = None,
        camera_embedding: Optional[torch.Tensor] = None,
        temb: torch.Tensor = None,
        freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        out_device: Optional[torch.device] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass through the ControlNet.
        
        Args:
            render_latent: Guide video latents [B, C, F, H, W]
            render_mask: Not implemented (for future masking support)
            camera_embedding: Not implemented (for future camera control)
            temb: Timestep embedding [B, 5120] - pre-projection from main model
            freqs: Optional RoPE frequencies tuple (cos, sin)
            out_device: Device to move outputs to (for memory offloading)
            
        Returns:
            List of 20 tensors, one per block, each shape [B, seq_len, 5120]
        """
        # Patch embedding: [B, C, F, H, W] -> [B, conv_out_dim, F, H/2, W/2]
        hidden_states = self.controlnet_patch_embedding(render_latent)
        
        # Reshape: [B, C, F, H, W] -> [B, F*H*W, C] -> project to dim
        b, c, f, h, w = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)  # [B, F, H, W, C]
        hidden_states = hidden_states.reshape(b, f * h * w, c)  # [B, seq_len, C]
        hidden_states = self.proj_in(hidden_states)  # [B, seq_len, dim]
        
        # Add time embedding if provided
        if temb is not None:
            # Create sinusoidal embedding for the timestep (assumes temb is raw timestep or pre-embedded)
            # If temb is already 5120-dim, we use it directly for modulation in blocks
            # For the initial injection, we use the time_embedding MLP
            if temb.shape[-1] == 256:
                # Raw timestep needs embedding
                time_emb = self.time_embedding(temb)  # [B, 5120]
            else:
                # Already embedded, use directly
                time_emb = temb
        else:
            time_emb = torch.zeros(b, 5120, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Run through transformer blocks and collect outputs
        controlnet_states = []
        for i, block in enumerate(self.controlnet_blocks):
            hidden_states = block(hidden_states, temb=time_emb, freqs=freqs)
            
            # Project to output dimension and collect
            state = self.proj_out[i](hidden_states)
            
            if out_device is not None:
                state = state.to(out_device)
                
            controlnet_states.append(state)
        
        return controlnet_states

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

