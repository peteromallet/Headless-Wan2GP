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


def _download_lora_from_url(url: str, task_id: str, dprint=None) -> str:
    """
    Download a LoRA from URL to appropriate local directory.
    
    Args:
        url: LoRA download URL
        task_id: Task ID for logging
        dprint: Optional debug print function
        
    Returns:
        Local filename of downloaded LoRA
        
    Raises:
        Exception: If download fails
    """
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
        
        # Check ALL standard lora directories and delete old generic versions
        lora_search_dirs = [
            "loras",
            "Wan2GP/loras",
            "loras_i2v",
            "Wan2GP/loras_i2v",
            "loras_hunyuan_i2v",
            "Wan2GP/loras_hunyuan_i2v",
            "loras_qwen",
            "Wan2GP/loras_qwen",
        ]
        
        for search_dir in lora_search_dirs:
            if os.path.isdir(search_dir):
                old_path = os.path.join(search_dir, generic_filename)
                if os.path.isfile(old_path):
                    if dprint:
                        dprint(f"[LORA_DOWNLOAD] Task {task_id}: 🗑️  Removing legacy LoRA file: {old_path}")
                    try:
                        os.remove(old_path)
                        if dprint:
                            dprint(f"[LORA_DOWNLOAD] Task {task_id}: ✅ Successfully deleted legacy file")
                    except Exception as e:
                        if dprint:
                            dprint(f"[LORA_DOWNLOAD] Task {task_id}: ⚠️  Failed to delete old LoRA {old_path}: {e}")
    
    # Determine LoRA directory: prefer the WGP-visible root 'loras'
    lora_dir = "loras"
    
    local_path = os.path.join(lora_dir, local_filename)
    
    if dprint:
        dprint(f"[LORA_DOWNLOAD] Task {task_id}: Downloading {local_filename} to {lora_dir} from {url}")

    # Normalize HuggingFace URLs: convert /blob/ to /resolve/ for direct downloads
    if "huggingface.co/" in url and "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
        if dprint:
            dprint(f"[LORA_DOWNLOAD] Task {task_id}: Normalized HuggingFace URL from /blob/ to /resolve/")

    # Check if file already exists
    if not os.path.isfile(local_path):
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
            filename = os.path.basename(rel_path)
            subfolder = os.path.dirname(rel_path)

            # Ensure LoRA directory exists
            os.makedirs(lora_dir, exist_ok=True)

            # Download using HuggingFace hub. Some hubs require `subfolder` to locate
            # the file, but we want the final artifact at `loras/<filename>` because
            # WGP expects LoRAs in the root loras directory.
            if len(subfolder) > 0:
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=lora_dir, subfolder=subfolder)
                # If the file landed under a nested path, move it up to lora_dir
                nested_path = os.path.join(lora_dir, subfolder, filename)
                if os.path.exists(nested_path) and not os.path.exists(local_path):
                    try:
                        os.makedirs(lora_dir, exist_ok=True)
                        shutil.move(nested_path, local_path)
                        # Clean up empty subfolder tree if any
                        try:
                            # Remove empty dirs going up from the deepest
                            cur = os.path.join(lora_dir, subfolder)
                            while os.path.normpath(cur).startswith(os.path.normpath(lora_dir)) and cur != lora_dir:
                                if not os.listdir(cur):
                                    os.rmdir(cur)
                                cur = os.path.dirname(cur)
                        except Exception:
                            pass
                    except Exception:
                        # If move fails, leave as-is; higher-level checks may still find it
                        pass
            else:
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=lora_dir)
        else:
            # Use urllib for other URLs
            os.makedirs(lora_dir, exist_ok=True)
            urlretrieve(url, local_path)
        
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
        wan_dir / "loras_i2v",
        wan_dir / "loras_hunyuan_i2v",
        wan_dir / "loras_qwen",
        wan_dir / "loras_flux",
        wan_dir / "loras_hunyuan",
        wan_dir / "loras_ltxv",
        # Parent directory (for dev setups)
        repo_root / "loras",
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
