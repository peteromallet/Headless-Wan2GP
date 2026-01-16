"""
Uni3C ControlNet checkpoint loader with auto-download.

This module handles downloading the Uni3C checkpoint from HuggingFace
and loading it with automatic config inference.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from safetensors.torch import load_file

# HuggingFace repo and filename
UNI3C_REPO_ID = "Kijai/WanVideo_comfy"
UNI3C_CHECKPOINT_FILENAME = "Wan21_Uni3C_controlnet_fp16.safetensors"


def get_uni3c_checkpoint_path(ckpts_dir: str = "ckpts") -> Path:
    """Get the path where Uni3C checkpoint should be stored."""
    return Path(ckpts_dir) / "controlnets" / UNI3C_CHECKPOINT_FILENAME


def download_uni3c_checkpoint_if_missing(
    ckpts_dir: str = "ckpts",
    force_download: bool = False
) -> str:
    """
    Download Uni3C checkpoint from HuggingFace if not present.
    
    Args:
        ckpts_dir: Base checkpoints directory
        force_download: If True, download even if file exists
        
    Returns:
        Path to the checkpoint file
    """
    from huggingface_hub import hf_hub_download
    
    target_path = get_uni3c_checkpoint_path(ckpts_dir)
    target_dir = target_path.parent
    
    if target_path.exists() and not force_download:
        print(f"[UNI3C] Checkpoint already exists: {target_path}")
        return str(target_path)
    
    print(f"[UNI3C] Downloading checkpoint from {UNI3C_REPO_ID}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Download using HuggingFace hub
    downloaded_path = hf_hub_download(
        repo_id=UNI3C_REPO_ID,
        filename=UNI3C_CHECKPOINT_FILENAME,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False
    )
    
    print(f"[UNI3C] Downloaded checkpoint to: {downloaded_path}")
    return downloaded_path


def infer_config_from_checkpoint(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Infer Uni3C config from checkpoint tensor shapes.
    
    This matches Kijai's approach in nodes.py for automatic config detection.
    
    Args:
        state_dict: Loaded checkpoint state dict
        
    Returns:
        Config dict with architecture parameters
        
    Raises:
        ValueError: If checkpoint is invalid/missing required keys
    """
    # Required keys to verify checkpoint validity
    if "controlnet_patch_embedding.weight" not in state_dict:
        raise ValueError("[UNI3C] Invalid checkpoint: missing controlnet_patch_embedding.weight")
    
    if "controlnet_blocks.0.ffn.0.bias" not in state_dict:
        raise ValueError("[UNI3C] Invalid checkpoint: missing controlnet_blocks.0.ffn.0.bias")
    
    # Infer from tensor shapes (matching Kijai's approach)
    in_channels = state_dict["controlnet_patch_embedding.weight"].shape[1]
    ffn_dim = state_dict["controlnet_blocks.0.ffn.0.bias"].shape[0]
    
    return {
        "in_channels": in_channels,
        "conv_out_dim": 5120,
        "time_embed_dim": 5120,
        "dim": 1024,
        "ffn_dim": ffn_dim,
        "num_heads": 16,
        "num_layers": 20,
        "add_channels": 7,
        "mid_channels": 256,
    }


def load_uni3c_checkpoint(
    checkpoint_path: Optional[str] = None,
    ckpts_dir: str = "ckpts",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load Uni3C checkpoint state dict.

    Args:
        checkpoint_path: Path to checkpoint (auto-downloads if None)
        ckpts_dir: Base checkpoints directory for auto-download
        device: Device to load tensors to
        dtype: Optional dtype conversion (e.g. torch.float16)

    Returns:
        Tuple of (state_dict, config)

    Raises:
        FileNotFoundError: If checkpoint doesn't exist and can't be downloaded
    """
    if checkpoint_path is None:
        checkpoint_path = download_uni3c_checkpoint_if_missing(ckpts_dir)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[UNI3C] Checkpoint not found: {checkpoint_path}")

    print(f"[UNI3C] Loading checkpoint from: {checkpoint_path}")

    # Load safetensors directly to target device
    state_dict = load_file(checkpoint_path, device=device)

    # Convert dtype if specified (do this in-place to save memory)
    if dtype is not None:
        sample_dtype = next(iter(state_dict.values())).dtype
        if sample_dtype != dtype:
            print(f"[UNI3C] Converting checkpoint from {sample_dtype} to {dtype}")
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

    # Log checkpoint stats for debugging
    total_params = sum(t.numel() for t in state_dict.values())
    total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1024 / 1024
    sample_dtype = next(iter(state_dict.values())).dtype
    print(f"[UNI3C] Checkpoint loaded: {total_params:,} params, {total_size_mb:.1f} MB, dtype={sample_dtype}")

    # Infer config from checkpoint
    config = infer_config_from_checkpoint(state_dict)
    print(f"[UNI3C] Inferred config: in_channels={config['in_channels']}, ffn_dim={config['ffn_dim']}")

    return state_dict, config


def load_uni3c_controlnet(
    ckpts_dir: str = "ckpts",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
):
    """
    Load and initialize Uni3C ControlNet model in one step.

    This is the optimized loading path that:
    - Loads checkpoint directly to GPU
    - Converts to fp16 during load (not after)
    - Creates model on target device

    Args:
        ckpts_dir: Base checkpoints directory
        device: Target device (default: cuda)
        dtype: Target dtype (default: float16)

    Returns:
        Initialized WanControlNet model ready for inference
    """
    from .controlnet import WanControlNet

    # Load checkpoint directly to GPU with target dtype
    state_dict, config = load_uni3c_checkpoint(
        ckpts_dir=ckpts_dir,
        device=device,
        dtype=dtype
    )

    # Add dtype to config so model knows its precision
    config["base_dtype"] = dtype

    # Create model and move to device FIRST (with random weights)
    controlnet = WanControlNet(config)
    controlnet = controlnet.to(device=device, dtype=dtype)

    # Now load state_dict - GPU→GPU copy since both are on same device
    controlnet.load_state_dict(state_dict, strict=False)
    controlnet.eval()

    # Free the state dict
    del state_dict

    print(f"[UNI3C] ControlNet ready on {device} with dtype {dtype}")
    return controlnet

