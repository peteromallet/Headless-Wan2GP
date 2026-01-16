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

# Module-level cache for controlnet (avoids reloading from disk each generation)
_controlnet_cache = {
    "model": None,
    "device": None,
    "dtype": None,
    "used_this_task": False,  # Track if cache was used in current task
}

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
    dtype: torch.dtype = torch.float16,
    use_cache: bool = True
):
    """
    Load and initialize Uni3C ControlNet model in one step.

    This is the optimized loading path (matching Kijai's approach):
    - Creates model with empty weights (no random init allocation)
    - Loads checkpoint directly to target device
    - Per-layer dtype control (patch embeddings stay float32)
    - Minimal peak memory usage
    - Caches model between calls to avoid reloading from disk

    Args:
        ckpts_dir: Base checkpoints directory
        device: Target device (default: cuda)
        dtype: Target dtype (default: float16)
        use_cache: If True, return cached model if available (default: True)

    Returns:
        Initialized WanControlNet model ready for inference
    """
    global _controlnet_cache

    # Check cache first
    if use_cache and _controlnet_cache["model"] is not None:
        cached = _controlnet_cache["model"]
        # Move to requested device if needed (handles offload/reload)
        current_device = next(cached.parameters()).device
        if str(current_device) != str(device):
            print(f"[UNI3C] Moving cached controlnet from {current_device} to {device}")
            cached = cached.to(device)
        _controlnet_cache["used_this_task"] = True
        print(f"[UNI3C] Using cached controlnet (skipping disk load)")
        return cached

    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from .controlnet import WanControlNet

    # Load checkpoint to target device (don't convert dtype yet - we'll do per-layer)
    state_dict, config = load_uni3c_checkpoint(
        ckpts_dir=ckpts_dir,
        device=device,
        dtype=None  # Keep original dtype, we'll convert per-layer
    )

    # Add dtype to config so model knows its precision
    config["base_dtype"] = dtype

    # Create model with empty weights - no memory allocated for parameters
    print(f"[UNI3C] Creating model with empty weights...")
    with init_empty_weights():
        controlnet = WanControlNet(config)

    # Load each parameter individually with per-layer dtype control
    # Matching Kijai's strategy: patch embeddings stay float32, others use base dtype
    for name, param in state_dict.items():
        # Patch embeddings need float32 for precision
        if "patch_embedding" in name:
            param_dtype = torch.float32
        # Normalization layers benefit from higher precision
        elif "norm" in name:
            param_dtype = dtype
        else:
            param_dtype = dtype

        set_module_tensor_to_device(
            controlnet,
            name,
            device=device,
            dtype=param_dtype,
            value=param
        )

    # Free the state dict
    del state_dict

    controlnet.eval()

    # Cache the model
    if use_cache:
        _controlnet_cache["model"] = controlnet
        _controlnet_cache["device"] = device
        _controlnet_cache["dtype"] = dtype
        _controlnet_cache["used_this_task"] = True
        print(f"[UNI3C] ControlNet cached for future use")

    print(f"[UNI3C] ControlNet ready on {device} with base dtype {dtype}")
    return controlnet


def clear_uni3c_cache():
    """Clear the cached controlnet to free memory."""
    global _controlnet_cache
    if _controlnet_cache["model"] is not None:
        del _controlnet_cache["model"]
        _controlnet_cache["model"] = None
        _controlnet_cache["device"] = None
        _controlnet_cache["dtype"] = None
        _controlnet_cache["used_this_task"] = False
        torch.cuda.empty_cache()
        print("[UNI3C] Cache cleared")


def reset_uni3c_task_flag():
    """Reset the used_this_task flag at the start of a new task."""
    global _controlnet_cache
    _controlnet_cache["used_this_task"] = False


def clear_uni3c_cache_if_unused():
    """Clear cache only if it wasn't used in the current task.

    Call this during task cleanup to free memory when uni3c wasn't used,
    while preserving cache for consecutive uni3c tasks.
    """
    global _controlnet_cache
    if _controlnet_cache["model"] is not None and not _controlnet_cache["used_this_task"]:
        print("[UNI3C] Cache not used this task, clearing to free memory")
        clear_uni3c_cache()
    # Reset flag for next task
    _controlnet_cache["used_this_task"] = False

