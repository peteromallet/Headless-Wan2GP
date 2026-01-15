"""
LoRA download and cleanup utilities.

This module provides:
- _download_lora_from_url: Download LoRAs from URLs (HuggingFace or direct)
- cleanup_legacy_lora_collisions: Remove collision-prone generic LoRA filenames

Note: LoRA format handling and URL detection are now in source/params/lora.py (LoRAConfig).
"""

import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import unquote

from source.logging_utils import headless_logger


def _download_lora_from_url(url: str, task_id: str, dprint=None, model_type: str = None) -> str:
    """
    Download a LoRA from URL to appropriate local directory.

    Args:
        url: LoRA download URL
        task_id: Task ID for logging
        dprint: Optional debug print function
        model_type: Model type to determine correct LoRA directory (e.g., "wan_2_2_vace_lightning_baseline_2_2_2")

    Returns:
        Local filename of downloaded LoRA

    Raises:
        Exception: If download fails
    """
    # Use absolute paths based on this file's location to avoid working directory issues
    repo_root = Path(__file__).parent.parent
    wan_dir = repo_root / "Wan2GP"

    # Extract filename from URL and decode URL-encoded characters
    # e.g., "%E5%BB%B6%E6%97%B6%E6%91%84%E5%BD%B1-high.safetensors" → "延时摄影-high.safetensors"
    url_filename = url.split("/")[-1]
    generic_filename = url_filename  # Save original before modification

    # Handle Wan2.2 Lightning LoRA collisions by prefixing parent folder
    if url_filename in ["high_noise_model.safetensors", "low_noise_model.safetensors"]:
        parts = url.split("/")
        if len(parts) > 2:
            parent = parts[-2].replace("%20", "_")
            url_filename = f"{parent}_{url_filename}"

    local_filename = unquote(url_filename)

    # If we derived a unique filename (collision detected), clean up old generic file
    if local_filename != generic_filename:
        if dprint:
            dprint(f"[LORA_DOWNLOAD] Task {task_id}: Collision-prone LoRA detected: {generic_filename} → {local_filename}")

        # Check ALL standard lora directories (using absolute paths) and delete old generic versions
        lora_search_dirs = [
            # Wan2GP LoRA directories
            wan_dir / "loras",
            wan_dir / "loras" / "wan",
            wan_dir / "loras_i2v",
            wan_dir / "loras_hunyuan",
            wan_dir / "loras_hunyuan" / "1.5",
            wan_dir / "loras_hunyuan_i2v",
            wan_dir / "loras_flux",
            wan_dir / "loras_qwen",
            wan_dir / "loras_ltxv",
            wan_dir / "loras_kandinsky5",
            # Also check repo root for any stray files from previous bugs
            repo_root / "loras",
            repo_root / "loras" / "wan",
        ]

        for search_dir in lora_search_dirs:
            if search_dir.is_dir():
                old_path = search_dir / generic_filename
                if old_path.is_file():
                    if dprint:
                        dprint(f"[LORA_DOWNLOAD] Task {task_id}: 🗑️  Removing legacy LoRA file: {old_path}")
                    try:
                        old_path.unlink()
                        if dprint:
                            dprint(f"[LORA_DOWNLOAD] Task {task_id}: ✅ Successfully deleted legacy file")
                    except Exception as e:
                        if dprint:
                            dprint(f"[LORA_DOWNLOAD] Task {task_id}: ⚠️  Failed to delete old LoRA {old_path}: {e}")

    # Determine LoRA directory based on model type
    # Each model family has its own LoRA directory in Wan2GP
    # Always use absolute paths under Wan2GP directory
    lora_dir = wan_dir / "loras"  # Default fallback

    if model_type:
        model_lower = model_type.lower()

        # Wan 2.x / VACE models → loras/wan
        if "wan" in model_lower or "vace" in model_lower:
            lora_dir = wan_dir / "loras" / "wan"

        # Hunyuan models - check I2V first (more specific), then 1.5, then general
        elif "hunyuan" in model_lower:
            if "i2v" in model_lower:
                lora_dir = wan_dir / "loras_hunyuan_i2v"
            elif "1_5" in model_lower or "1.5" in model_lower:
                lora_dir = wan_dir / "loras_hunyuan" / "1.5"
            else:
                lora_dir = wan_dir / "loras_hunyuan"

        # Flux models → loras_flux
        elif "flux" in model_lower:
            lora_dir = wan_dir / "loras_flux"

        # Qwen models → loras_qwen
        elif "qwen" in model_lower:
            lora_dir = wan_dir / "loras_qwen"

        # LTX-Video models → loras_ltxv
        elif "ltx" in model_lower:
            lora_dir = wan_dir / "loras_ltxv"

        # Kandinsky models → loras_kandinsky5
        elif "kandinsky" in model_lower:
            lora_dir = wan_dir / "loras_kandinsky5"

    local_path = lora_dir / local_filename
    
    if dprint:
        dprint(f"[LORA_DOWNLOAD] Task {task_id}: Downloading {local_filename} to {lora_dir} from {url}")

    # Normalize HuggingFace URLs: convert /blob/ to /resolve/ for direct downloads
    if "huggingface.co/" in url and "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
        if dprint:
            dprint(f"[LORA_DOWNLOAD] Task {task_id}: Normalized HuggingFace URL from /blob/ to /resolve/")

    # Check if file already exists
    if not local_path.is_file():
        if url.startswith("https://huggingface.co/") and "/resolve/main/" in url:
            # Use HuggingFace hub for HF URLs
            from huggingface_hub import hf_hub_download

            # Parse HuggingFace URL
            url_path = url[len("https://huggingface.co/"):]
            url_parts = url_path.split("/resolve/main/")
            repo_id = url_parts[0]
            rel_path_encoded = url_parts[-1]
            # Decode URL-encoded path components (e.g., Chinese characters)
            rel_path = unquote(rel_path_encoded)
            filename = Path(rel_path).name
            subfolder = str(Path(rel_path).parent) if Path(rel_path).parent != Path(".") else ""

            # Ensure LoRA directory exists
            lora_dir.mkdir(parents=True, exist_ok=True)

            # Download using HuggingFace hub. Some hubs require `subfolder` to locate
            # the file, but we want the final artifact at `loras/<filename>` because
            # WGP expects LoRAs in the root loras directory.
            if len(subfolder) > 0:
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(lora_dir), subfolder=subfolder)
                # If the file landed under a nested path, move it up to lora_dir
                nested_path = lora_dir / subfolder / filename
                if nested_path.exists() and not local_path.exists():
                    try:
                        lora_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(nested_path), str(local_path))
                        # Clean up empty subfolder tree if any
                        try:
                            # Remove empty dirs going up from the deepest
                            cur = lora_dir / subfolder
                            while cur.is_relative_to(lora_dir) and cur != lora_dir:
                                if not any(cur.iterdir()):
                                    cur.rmdir()
                                cur = cur.parent
                        except Exception:
                            pass
                    except Exception:
                        # If move fails, leave as-is; higher-level checks may still find it
                        pass
            else:
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(lora_dir))
        else:
            # Use urllib for other URLs
            lora_dir.mkdir(parents=True, exist_ok=True)
            urlretrieve(url, str(local_path))
        
        if dprint:
            dprint(f"[LORA_DOWNLOAD] Task {task_id}: Successfully downloaded {local_filename}")
    else:
        if dprint:
            dprint(f"[LORA_DOWNLOAD] Task {task_id}: {local_filename} already exists")
    
    return local_filename


def cleanup_legacy_lora_collisions():
    """
    Remove legacy generic LoRA filenames that collide with new uniquely-named versions.
    
    This runs at worker startup to ensure old collision-prone files like
    'high_noise_model.safetensors' and 'low_noise_model.safetensors' are removed
    before WGP loads models with updated LoRA URLs.
    
    Checks ALL possible LoRA directories to ensure comprehensive cleanup.
    """
    repo_root = Path(__file__).parent.parent
    wan_dir = repo_root / "Wan2GP"
    
    # Comprehensive list of all possible LoRA directories
    lora_dirs = [
        # Wan2GP subdirectories (standard)
        wan_dir / "loras",
        wan_dir / "loras" / "wan",
        wan_dir / "loras_i2v",
        wan_dir / "loras_hunyuan",
        wan_dir / "loras_hunyuan" / "1.5",
        wan_dir / "loras_hunyuan_i2v",
        wan_dir / "loras_flux",
        wan_dir / "loras_qwen",
        wan_dir / "loras_ltxv",
        wan_dir / "loras_kandinsky5",
        # Parent directory (for stray files from previous bugs)
        repo_root / "loras",
        repo_root / "loras" / "wan",
        repo_root / "loras_qwen",
    ]
    
    # Generic filenames that are collision-prone
    collision_prone_files = [
        "high_noise_model.safetensors",
        "low_noise_model.safetensors",
    ]
    
    cleaned_files = []
    for lora_dir in lora_dirs:
        if not lora_dir.exists():
            continue
        
        for filename in collision_prone_files:
            file_path = lora_dir / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    cleaned_files.append(str(file_path))
                    headless_logger.info(f"🗑️  Removed legacy LoRA file: {file_path}")
                except Exception as e:
                    headless_logger.warning(f"⚠️  Failed to remove legacy LoRA {file_path}: {e}")
    
    if cleaned_files:
        headless_logger.info(f"✅ Cleanup complete: removed {len(cleaned_files)} legacy LoRA file(s)")
    else:
        headless_logger.debug("No legacy LoRA files found to clean up")
