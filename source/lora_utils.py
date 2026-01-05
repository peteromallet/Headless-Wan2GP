"""
Shared utilities for LoRA detection and parameter optimization.

This module provides consistent CausVid and LightI2X LoRA detection across
headless_model_management.py and travel_between_images.py, ensuring both
use the same sophisticated parameter precedence logic.

═══════════════════════════════════════════════════════════════════════════
                         LoRA Processing Pipeline
═══════════════════════════════════════════════════════════════════════════

DATA FLOW:
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. INPUT SOURCES (multiple entry points)                                │
│    - Model preset JSON: model.loras, model.loras_multipliers            │
│    - Task params: activated_loras, loras_multipliers                    │
│    - Phase config: phases[].loras (URLs with multipliers)               │
│    - Direct task setup: qwen_image_edit, qwen_image_style, etc.         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. PARSE STAGE (worker.py)                                              │
│    - parse_phase_config() → additional_loras dict {url: multiplier}     │
│    - Task handlers → lora_names list (local filenames or URLs)          │
│    OUTPUT: lora_names=[], additional_loras={urls}                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. NORMALIZE STAGE (lora_utils.normalize_lora_format)                   │
│    - Convert all formats → standard internal format                     │
│    - activated_loras → lora_names (list of filenames)                   │
│    - loras_multipliers → lora_multipliers (list)                        │
│    OUTPUT: lora_names=[], lora_multipliers=[], additional_loras={}      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. DOWNLOAD STAGE (lora_utils._process_additional_loras)                │
│    - Download URLs → local filenames                                    │
│    - Merge downloaded files into lora_names                             │
│    OUTPUT: lora_names=[filenames], additional_loras deleted             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. WGP FORMAT STAGE (headless_wgp.py)                                   │
│    - lora_names → activated_loras (list for WGP)                        │
│    - lora_multipliers → loras_multipliers (space-separated string)      │
│    OUTPUT: WGP-compatible format                                        │
└─────────────────────────────────────────────────────────────────────────┘

CRITICAL INVARIANTS:
- After stage 2: lora_names should be EMPTY, additional_loras has URLs
- After stage 4: lora_names has ONLY local filenames, NO URLs
- After stage 5: activated_loras is WGP format (list), loras_multipliers is string

MAIN ENTRY POINT: process_all_loras() - coordinates stages 3-4

TERMINOLOGY:
- lora_names: Internal list format (local filenames)
- activated_loras: WGP format (list, may contain URLs before normalize)
- additional_loras: Download queue dict {url: multiplier}
- lora_multipliers: Internal format (list of floats or phase strings)
- loras_multipliers: WGP format (space-separated string, with 's')
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from source.logging_utils import headless_logger


def detect_loras_in_model_config(model_name: str, dprint=None) -> Tuple[bool, bool]:
    """
    Detect if CausVid or LightI2X LoRAs are present in model configuration.
    
    Args:
        model_name: Name of the model to check
        dprint: Optional debug print function
        
    Returns:
        Tuple of (model_has_causvid, model_has_lighti2x)
    """
    model_has_causvid = False
    model_has_lighti2x = False
    
    try:
        # Import WGP to access model definitions
        import sys
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        # Protect sys.argv before importing wgp to prevent argparse errors
        # wgp.py uses argparse and will fail if sys.argv contains database arguments
        _saved_argv = sys.argv[:]
        if dprint:
            dprint(f"[LORA_UTILS_DEBUG] About to import wgp, sys.argv before protection: {sys.argv}")
        try:
            sys.argv = ["lora_utils.py"]
            if dprint:
                dprint(f"[LORA_UTILS_DEBUG] sys.argv during import: {sys.argv}")
            import wgp
            if dprint:
                dprint(f"[LORA_UTILS_DEBUG] wgp imported successfully")
        finally:
            sys.argv = _saved_argv
            if dprint:
                dprint(f"[LORA_UTILS_DEBUG] sys.argv restored to: {sys.argv}")

        model_def = wgp.get_model_def(model_name)
        
        if model_def and "model" in model_def and "loras" in model_def["model"]:
            model_loras = model_def["model"]["loras"]
            if dprint:
                dprint(f"[LORA_DETECT] Model {model_name} has LoRAs: {[os.path.basename(lora) for lora in model_loras if isinstance(lora, str)]}")
            
            # Check for CausVid LoRA in model config
            for lora in model_loras:
                if isinstance(lora, str) and "CausVid" in lora:
                    model_has_causvid = True
                    if dprint:
                        dprint(f"[LORA_DETECT] CausVid LoRA detected in model config: {os.path.basename(lora)}")
                    break
            
            # Check for LightI2X LoRA in model config  
            for lora in model_loras:
                if isinstance(lora, str) and ("lightx2v" in lora.lower() or "lighti2x" in lora.lower()):
                    model_has_lighti2x = True
                    if dprint:
                        dprint(f"[LORA_DETECT] LightI2X LoRA detected in model config: {os.path.basename(lora)}")
                    break
        else:
            if dprint:
                dprint(f"[LORA_DETECT] No LoRAs found in model config for {model_name}")
                
    except Exception as e:
        if dprint:
            dprint(f"[LORA_DETECT] Failed to check model config LoRAs: {e}")
    
    return model_has_causvid, model_has_lighti2x


def normalize_lora_format(params: Dict[str, Any], task_id: str = "unknown", dprint=None) -> Dict[str, Any]:
    """
    Normalize LoRA parameters from various input formats to standard format.

    STAGE: 3 - NORMALIZE (see module docstring for pipeline overview)

    Handles:
    - activated_loras (string/list) → lora_names (list)
    - loras_multipliers (string) → lora_multipliers (list of floats)
    - additional_loras (dict) → processed and downloaded

    Args:
        params: Parameters dict to normalize
        task_id: Task ID for logging
        dprint: Optional debug print function

    Returns:
        Updated parameters dict with normalized LoRA format

    POSTCONDITION: lora_names contains only local filenames (no URLs)
    """
    if dprint:
        additional = params.get('additional_loras', {})
        additional_keys = list(additional.keys()) if isinstance(additional, dict) else f"<non-dict: {type(additional).__name__}>"
        dprint(f"[LORA_NORMALIZE] Task {task_id}: INPUT - lora_names={params.get('lora_names', [])}, additional_loras={additional_keys}")
    # Convert activated_loras to lora_names
    if "activated_loras" in params:
        loras = params["activated_loras"]
        if isinstance(loras, str):
            # Convert comma-separated string to list
            params["lora_names"] = [lora.strip() for lora in loras.split(",") if lora.strip()]
        elif isinstance(loras, list):
            params["lora_names"] = loras.copy()
        del params["activated_loras"]  # Remove old format
        
        if dprint:
            dprint(f"[LORA_NORM] Task {task_id}: Converted activated_loras to lora_names: {params.get('lora_names', [])}")
    
    # Convert loras_multipliers string to list
    if "loras_multipliers" in params:
        multipliers = params["loras_multipliers"]
        if isinstance(multipliers, str):
            # Check if this is phase-config format (contains semicolons)
            # Phase-config: "0.9;0 0;0.9" (space-separated) or "0.9;0,0;0.9" (comma-separated)
            # Regular: "0.9,1.0" (comma-separated floats)
            if ";" in multipliers:
                # Phase-config format: split by spaces first, fallback to commas
                if " " in multipliers:
                    params["lora_multipliers"] = [x.strip() for x in multipliers.split(" ") if x.strip()]
                else:
                    params["lora_multipliers"] = [x.strip() for x in multipliers.split(",") if x.strip()]
            else:
                # Regular format: convert comma-separated string to list of floats
                params["lora_multipliers"] = [float(x.strip()) for x in multipliers.split(",") if x.strip()]
        # Keep as-is if already a list

        if dprint:
            dprint(f"[LORA_NORM] Task {task_id}: Normalized loras_multipliers: {params.get('lora_multipliers', [])}")

    # CRITICAL: Also normalize lora_multipliers (without 's') from orchestrator
    # This is needed when phase_config parsing sets lora_multipliers directly
    if "lora_multipliers" in params:
        multipliers = params["lora_multipliers"]
        if dprint:
            dprint(f"[LORA_NORM] Task {task_id}: Processing lora_multipliers (without 's'): type={type(multipliers)}, value={multipliers}")
        if isinstance(multipliers, list):
            # Check if this is phase-config format (any element contains semicolon)
            is_phase_config = any(";" in str(m) for m in multipliers)
            if dprint:
                dprint(f"[LORA_NORM] Task {task_id}: is_phase_config check: {is_phase_config}, elements: {[str(m) for m in multipliers]}")
            if not is_phase_config:
                # Regular format: ensure all elements are floats
                try:
                    params["lora_multipliers"] = [float(m) for m in multipliers]
                    if dprint:
                        dprint(f"[LORA_NORM] Task {task_id}: Converted regular lora_multipliers to floats: {params['lora_multipliers']}")
                except (ValueError, TypeError):
                    # Keep as-is if conversion fails
                    if dprint:
                        dprint(f"[LORA_NORM] Task {task_id}: Keeping lora_multipliers as-is (conversion failed): {multipliers}")
            else:
                # Phase-config format: keep strings intact
                if dprint:
                    dprint(f"[LORA_NORM] Task {task_id}: Keeping phase-config lora_multipliers as strings: {multipliers}")
        elif isinstance(multipliers, str):
            # Same logic as above for string format
            if ";" in multipliers:
                # Phase-config format
                if " " in multipliers:
                    params["lora_multipliers"] = [x.strip() for x in multipliers.split(" ") if x.strip()]
                else:
                    params["lora_multipliers"] = [x.strip() for x in multipliers.split(",") if x.strip()]
                if dprint:
                    dprint(f"[LORA_NORM] Task {task_id}: Parsed phase-config lora_multipliers from string: {params['lora_multipliers']}")
            else:
                # Regular format
                params["lora_multipliers"] = [float(x.strip()) for x in multipliers.split(",") if x.strip()]
                if dprint:
                    dprint(f"[LORA_NORM] Task {task_id}: Parsed regular lora_multipliers from string: {params['lora_multipliers']}")
    
    # Process additional_loras dict format
    additional_loras_dict = params.get("additional_loras", {})
    if additional_loras_dict and isinstance(additional_loras_dict, dict):
        if dprint:
            dprint(f"[LORA_NORM] Task {task_id}: Processing {len(additional_loras_dict)} additional LoRAs")
        
        # Process URLs and download if needed
        processed_names, processed_multipliers = _process_additional_loras(
            additional_loras_dict, task_id, dprint
        )
        
        # Merge with existing LoRA lists
        current_loras = params.get("lora_names", [])
        current_multipliers = params.get("lora_multipliers", [])
        
        for i, lora_name in enumerate(processed_names):
            if lora_name not in current_loras:
                current_loras.append(lora_name)
                multiplier = processed_multipliers[i] if i < len(processed_multipliers) else 1.0
                current_multipliers.append(multiplier)
                
                if dprint:
                    dprint(f"[LORA_NORM] Task {task_id}: Added additional LoRA: {lora_name} (multiplier: {multiplier})")
        
        params["lora_names"] = current_loras
        params["lora_multipliers"] = current_multipliers

        # Remove processed additional_loras
        del params["additional_loras"]

    # VALIDATION: Check postconditions
    final_lora_names = params.get("lora_names", [])
    if dprint and final_lora_names:
        urls_found = [name for name in final_lora_names if isinstance(name, str) and name.startswith(("http://", "https://", "ftp://"))]
        if urls_found:
            dprint(f"[LORA_NORMALIZE] ⚠️  Task {task_id}: Note - URLs in lora_names after normalize (may indicate download failure or WGP-handled URLs): {urls_found}")
        # Truncate long lists to avoid log spam
        if len(final_lora_names) > 10:
            dprint(f"[LORA_NORMALIZE] Task {task_id}: OUTPUT - lora_names=[{len(final_lora_names)} LoRAs, showing first 10: {final_lora_names[:10]}]")
        else:
            dprint(f"[LORA_NORMALIZE] Task {task_id}: OUTPUT - lora_names={final_lora_names}")

    return params


def _process_additional_loras(additional_loras_dict: Dict[str, float], task_id: str, dprint=None) -> Tuple[list, list]:
    """
    Process additional LoRAs dict, downloading URLs if needed.
    
    Args:
        additional_loras_dict: Dict mapping LoRA names/URLs to multipliers
        task_id: Task ID for logging
        dprint: Optional debug print function
        
    Returns:
        Tuple of (processed_names, multipliers) - only includes successfully processed LoRAs
    """
    processed_names = []
    processed_multipliers = []
    
    for idx, (lora_name_or_url, multiplier) in enumerate(additional_loras_dict.items()):
        if lora_name_or_url.startswith("http"):
            # It's a URL - download it
            try:
                local_filename = _download_lora_from_url(lora_name_or_url, task_id, dprint)
                processed_names.append(local_filename)
                processed_multipliers.append(multiplier)
                if dprint:
                    dprint(f"[LORA_DOWNLOAD] Task {task_id}: Successfully processed URL → {local_filename}")
            except Exception as e:
                if dprint:
                    dprint(f"[LORA_DOWNLOAD] Task {task_id}: ⚠️  FAILED to download {lora_name_or_url}: {e}")
                    dprint(f"[LORA_DOWNLOAD] Task {task_id}: ⚠️  Skipping this LoRA - it will not be applied to generation")
                # Don't add URL to processed list - WGP can't handle URLs
                # This LoRA will be skipped, but generation will continue
        else:
            # Already a local filename
            processed_names.append(lora_name_or_url)
            processed_multipliers.append(multiplier)
    
    return processed_names, processed_multipliers


def _check_lora_exists(lora_filename: str) -> bool:
    """
    Check if a LoRA file exists in any of the standard LoRA directories.
    
    Args:
        lora_filename: Name of the LoRA file to check
        
    Returns:
        True if file exists, False otherwise
    """
    import os
    
    # Standard LoRA directories (include Qwen-specific dir)
    lora_dirs = [
        "loras",
        "Wan2GP/loras",
        "loras_i2v",
        "Wan2GP/loras_i2v",
        "loras_hunyuan_i2v",
        "Wan2GP/loras_hunyuan_i2v",
        "loras_qwen",
        "Wan2GP/loras_qwen",
    ]
    
    for lora_dir in lora_dirs:
        lora_path = os.path.join(lora_dir, lora_filename)
        if os.path.isfile(lora_path):
            return True
    
    return False


def _download_lora_auto(lora_filename: str, lora_type: str, dprint=None) -> bool:
    """
    Attempt to auto-download a LoRA based on known filename patterns.
    
    Args:
        lora_filename: Name of the LoRA file to download
        lora_type: Type of LoRA (for logging and URL determination)
        dprint: Optional debug print function
        
    Returns:
        True if download succeeded, False otherwise
    """
    # Known LoRA download URLs
    lora_urls = {
        # LightI2X LoRA (14B). Correct HF location is under the Wan2.1 loras_accelerators folder.
        "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors":
            "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
        # CausVid LoRA (14B)
        "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors":
            "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
        # Lightning LoRAs (Wan2.2 T2V 14B 4-steps)
        "high_noise_model.safetensors":
            "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
        "low_noise_model.safetensors":
            "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",
        # Fractal Concept LoRA
        "246838-wan22_14B-high-fractal_concept-e459.safetensors":
            "https://huggingface.co/Cseti/wan2.2-14B-Kinestasis_concept-lora-v1/resolve/main/246838-wan22_14B-high-fractal_concept-e459.safetensors",
        # Banostasis Concept LoRA
        "246839-wan22_14B-high-banostasis_concept-e459.safetensors":
            "https://huggingface.co/Cseti/wan2.2-14B-Kinestasis_concept-lora-v1/resolve/main/246839-wan22_14B-high-banostasis_concept-e459.safetensors"
    }

    # Provide fallbacks for known mirrors when primary URL is missing/gated
    lora_fallback_urls = {
        "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors": [
            # Kijai mirror
            "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
        ]
    }
    
    if lora_filename in lora_urls:
        try:
            url = lora_urls[lora_filename]
            try:
                downloaded_filename = _download_lora_from_url(url, "auto-download", dprint)
                return downloaded_filename == lora_filename
            except Exception:
                # Try fallbacks if any
                for fb_url in lora_fallback_urls.get(lora_filename, []):
                    try:
                        downloaded_filename = _download_lora_from_url(fb_url, "auto-download (fallback)", dprint)
                        return downloaded_filename == lora_filename
                    except Exception:
                        continue
                raise
        except Exception as e:
            if dprint:
                dprint(f"[LORA_DOWNLOAD] Auto-download failed for {lora_filename}: {e}")
            return False
    else:
        if dprint:
            dprint(f"[LORA_DOWNLOAD] No auto-download URL available for {lora_filename}")
        return False


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
    import os
    import shutil
    from urllib.request import urlretrieve
    from urllib.parse import unquote
    
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
            # WGP and `_check_lora_exists` look in the root directory.
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


def ensure_lora_in_list(params: Dict[str, Any], lora_filename: str, lora_type: str, task_id: str = "unknown", dprint=None) -> Dict[str, Any]:
    """
    Ensure a specific LoRA is included in the activated LoRA list.
    Auto-downloads the LoRA if it doesn't exist locally.
    
    Args:
        params: Parameters dict to modify
        lora_filename: Filename of the LoRA to ensure is present
        lora_type: Type of LoRA (for logging)
        task_id: Task ID for logging
        dprint: Optional debug print function
        
    Returns:
        Updated parameters dict
    """
    current_loras = params.get("lora_names", [])
    
    if dprint:
        dprint(f"[LORA_LIST] Task {task_id}: Current LoRAs before {lora_type}: {current_loras}")
    
    # Auto-download LoRA if needed
    if lora_filename not in current_loras:
        # Check if LoRA exists locally, if not try to download
        if not _check_lora_exists(lora_filename):
            if dprint:
                dprint(f"[LORA_DOWNLOAD] Task {task_id}: {lora_type} LoRA not found locally, attempting auto-download: {lora_filename}")
            
            download_success = _download_lora_auto(lora_filename, lora_type, dprint)
            if not download_success and dprint:
                dprint(f"[LORA_DOWNLOAD] Task {task_id}: Warning - Failed to auto-download {lora_type} LoRA: {lora_filename}")
        
        current_loras.append(lora_filename)
        params["lora_names"] = current_loras
        
        # Add multiplier for the new LoRA
        current_multipliers = params.get("lora_multipliers", [])
        while len(current_multipliers) < len(current_loras):
            current_multipliers.append(1.0)
        params["lora_multipliers"] = current_multipliers
        
        if dprint:
            dprint(f"[LORA_LIST] Task {task_id}: Added {lora_type} LoRA to list: {current_loras}")
            dprint(f"[LORA_LIST] Task {task_id}: Updated LoRA multipliers: {current_multipliers}")
    else:
        if dprint:
            dprint(f"[LORA_LIST] Task {task_id}: {lora_type} LoRA already in list at index {current_loras.index(lora_filename)}")
    
    return params


def process_all_loras(params: Dict[str, Any], task_params: Dict[str, Any], model_name: str, 
                     orchestrator_payload: Optional[Dict[str, Any]] = None, task_id: str = "unknown", dprint=None) -> Dict[str, Any]:
    """
    Complete LoRA processing pipeline - handles detection, optimization, download, and formatting.
    
    This is the main entry point that replaces scattered LoRA logic across multiple files.
    
    Args:
        params: Parameters dict to modify
        task_params: Task-specific parameters
        model_name: Name of the model being used
        orchestrator_payload: Optional orchestrator parameters
        task_id: Task ID for logging
        dprint: Optional debug print function
        
    Returns:
        Updated parameters dict with all LoRA processing complete
    """
    if dprint:
        dprint(f"[LORA_PROCESS] Task {task_id}: Starting comprehensive LoRA processing for model {model_name}")
    
    # Step 1: Normalize LoRA formats (activated_loras, loras_multipliers, additional_loras)
    params = normalize_lora_format(params, task_id, dprint)
    
    # Step 2: Model-specific optimizations removed - all parameters from model JSON config
    
    # Step 3: Process additional_loras from orchestrator payload
    if orchestrator_payload and "additional_loras" in orchestrator_payload:
        params["additional_loras"] = orchestrator_payload["additional_loras"]
        params = normalize_lora_format(params, task_id, dprint)  # Re-normalize after adding additional

        if dprint:
            dprint(f"[LORA_PROCESS] Task {task_id}: Added {len(orchestrator_payload['additional_loras'])} additional LoRAs from orchestrator")

    # Step 3.5: Parse phase_config if present in orchestrator_payload OR task_params
    # For orchestrator tasks: phase_config is in orchestrator_payload
    # For segment tasks: phase_config is at top level of task_params
    phase_config = None
    phase_config_source = None
    model_name_for_phase = None
    
    if orchestrator_payload and "phase_config" in orchestrator_payload:
        phase_config = orchestrator_payload["phase_config"]
        phase_config_source = "orchestrator_payload"
        model_name_for_phase = orchestrator_payload.get("model_name")
    elif task_params and "phase_config" in task_params:
        phase_config = task_params["phase_config"]
        phase_config_source = "task_params"
        model_name_for_phase = task_params.get("model_name")
    
    if phase_config:
        if dprint:
            dprint(f"[LORA_PROCESS] Task {task_id}: phase_config detected in {phase_config_source} - parsing LoRA configuration")

        try:
            from source.task_conversion import parse_phase_config

            # Get num_inference_steps for parsing
            steps_per_phase = phase_config.get("steps_per_phase", [2, 2, 2])
            total_steps = sum(steps_per_phase)

            # Parse phase_config to get lora_names and lora_multipliers
            parsed = parse_phase_config(
                phase_config=phase_config,
                num_inference_steps=total_steps,
                task_id=task_id,
                model_name=model_name_for_phase
            )

            # Override with parsed values
            if "lora_names" in parsed:
                params["lora_names"] = parsed["lora_names"]
                if dprint:
                    dprint(f"[LORA_PROCESS] Task {task_id}: Using phase_config lora_names: {parsed['lora_names']}")

            if "lora_multipliers" in parsed:
                params["lora_multipliers"] = parsed["lora_multipliers"]
                if dprint:
                    dprint(f"[LORA_PROCESS] Task {task_id}: Using phase_config lora_multipliers: {parsed['lora_multipliers']}")

            if "additional_loras" in parsed:
                params["additional_loras"] = parsed["additional_loras"]
                if dprint:
                    dprint(f"[LORA_PROCESS] Task {task_id}: Using phase_config additional_loras: {len(parsed['additional_loras'])} LoRAs")
                # Re-normalize to download URLs from phase_config additional_loras
                params = normalize_lora_format(params, task_id, dprint)

        except Exception as e:
            if dprint:
                dprint(f"[LORA_PROCESS] Task {task_id}: ERROR parsing phase_config: {e}")
            import traceback
            traceback.print_exc()

    # Step 4: Ensure all LoRAs in the list exist (auto-download if needed)
    lora_names = params.get("lora_names", [])
    if lora_names:
        if dprint:
            dprint(f"[LORA_PROCESS] Task {task_id}: Verifying {len(lora_names)} LoRAs exist: {lora_names}")

        for lora_name in lora_names[:]:  # Copy list to avoid modification during iteration
            if not _check_lora_exists(lora_name):
                if dprint:
                    dprint(f"[LORA_DOWNLOAD] Task {task_id}: LoRA not found locally, attempting auto-download: {lora_name}")

                download_success = _download_lora_auto(lora_name, "required", dprint)
                if not download_success:
                    if dprint:
                        dprint(f"[LORA_DOWNLOAD] Task {task_id}: Auto-download not available for '{lora_name}'")
                        dprint(f"[LORA_PROCESS] Task {task_id}: Passing through '{lora_name}' - may be URL or model JSON entry that WGP will handle")
                    # ✅ DON'T DROP IT - let WGP handle URLs, model JSON entries, or error appropriately
        # No pruning - keep all LoRAs in the list
        params["lora_names"] = lora_names
    
    # Step 5: Ensure multipliers list matches LoRA names list
    lora_names = params.get("lora_names", [])
    lora_multipliers = params.get("lora_multipliers", [])
    
    # DEBUG: Log multipliers before extension
    if dprint:
        dprint(f"[LORA_DEBUG] Task {task_id}: Before extension - lora_multipliers={lora_multipliers}, len={len(lora_multipliers)}, lora_names count={len(lora_names)}")

    # Extend multipliers list if needed
    # IMPORTANT: Detect if we're in phase-config mode to extend with proper format
    is_phase_config = any(";" in str(m) for m in lora_multipliers) if lora_multipliers else False
    while len(lora_multipliers) < len(lora_names):
        if is_phase_config:
            # Phase-config: need to determine number of phases and append proper format
            # Parse first multiplier to count phases
            first_mult_str = str(lora_multipliers[0])
            num_phases = first_mult_str.count(";") + 1
            # Append "1.0;1.0;..." for the number of phases
            default_mult = ";".join(["1.0"] * num_phases)
            lora_multipliers.append(default_mult)
            if dprint:
                dprint(f"[LORA_DEBUG] Task {task_id}: Extended phase-config with {default_mult}")
        else:
            lora_multipliers.append(1.0)

    # Truncate multipliers list if too long
    if len(lora_multipliers) > len(lora_names):
        lora_multipliers = lora_multipliers[:len(lora_names)]

    # DEBUG: Log multipliers after extension
    if dprint:
        dprint(f"[LORA_DEBUG] Task {task_id}: After extension - lora_multipliers={lora_multipliers}")

    params["lora_multipliers"] = lora_multipliers

    # Step 6: Convert to final WGP format
    if lora_names:
        params["activated_loras"] = lora_names  # WGP expects this format

        # Detect if multipliers are in phase-config format (contain semicolons)
        # Phase-config format: ["0.9;0", "0;0.9"] → space-separated "0.9;0 0;0.9"
        # Regular format: [0.9, 1.0] → comma-separated "0.9,1.0"
        if any(";" in str(m) for m in lora_multipliers):
            result = " ".join(map(str, lora_multipliers))
            params["loras_multipliers"] = result  # Space-separated for phase-config
            if dprint:
                dprint(f"[LORA_DEBUG] Task {task_id}: Phase-config format detected, output: '{result}'")
        else:
            result = ",".join(map(str, lora_multipliers))
            params["loras_multipliers"] = result  # Comma-separated for regular
            if dprint:
                dprint(f"[LORA_DEBUG] Task {task_id}: Regular format detected, output: '{result}'")
        
        if dprint:
            dprint(f"[LORA_PROCESS] Task {task_id}: Final LoRA processing complete")
            dprint(f"[LORA_PROCESS] Task {task_id}: activated_loras: {params['activated_loras']}")
            dprint(f"[LORA_PROCESS] Task {task_id}: loras_multipliers: {params['loras_multipliers']}")
    else:
        # Ensure empty lists for no LoRAs
        params["activated_loras"] = []
        params["loras_multipliers"] = ""
        
        if dprint:
            dprint(f"[LORA_PROCESS] Task {task_id}: No LoRAs to process")
    
    return params


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
