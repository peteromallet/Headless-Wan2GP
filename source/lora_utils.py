"""
Shared utilities for LoRA detection and parameter optimization.

This module provides consistent CausVid and LightI2X LoRA detection across
headless_model_management.py and travel_between_images.py, ensuring both
use the same sophisticated parameter precedence logic.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


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


def detect_lora_optimization_flags(task_params: Dict[str, Any], orchestrator_payload: Optional[Dict[str, Any]] = None, 
                                   model_name: Optional[str] = None, dprint=None) -> Tuple[bool, bool]:
    """
    Detect if CausVid or LightI2X optimization should be enabled based on explicit flags and model config.
    
    Args:
        task_params: Task parameters dict
        orchestrator_payload: Optional orchestrator payload for travel tasks
        model_name: Optional model name for auto-detection
        dprint: Optional debug print function
        
    Returns:
        Tuple of (causvid_enabled, lighti2x_enabled)
    """
    # Check explicit flags first
    causvid_enabled = bool(task_params.get("use_causvid_lora", False))
    lighti2x_enabled = bool(task_params.get("use_lighti2x_lora", False))
    
    # For travel tasks, also check orchestrator payload
    if orchestrator_payload:
        causvid_enabled = causvid_enabled or bool(orchestrator_payload.get("apply_causvid", False) or orchestrator_payload.get("use_causvid_lora", False))
        lighti2x_enabled = lighti2x_enabled or bool(orchestrator_payload.get("use_lighti2x_lora", False))
    
    # Auto-detect from model config if not explicitly enabled and model name provided
    if model_name and (not causvid_enabled or not lighti2x_enabled):
        model_has_causvid, model_has_lighti2x = detect_loras_in_model_config(model_name, dprint)
        
        if not causvid_enabled and model_has_causvid:
            causvid_enabled = True
            if dprint:
                dprint(f"[LORA_DETECT] Auto-enabled CausVid optimization from model config")
                
        if not lighti2x_enabled and model_has_lighti2x:
            lighti2x_enabled = True
            if dprint:
                dprint(f"[LORA_DETECT] Auto-enabled LightI2X optimization from model config")
    
    return causvid_enabled, lighti2x_enabled


def apply_lora_parameter_optimization(
    params: Dict[str, Any], 
    causvid_enabled: bool, 
    lighti2x_enabled: bool,
    model_name: str,
    task_params: Optional[Dict[str, Any]] = None,
    orchestrator_payload: Optional[Dict[str, Any]] = None,
    task_id: str = "unknown",
    dprint=None
) -> Dict[str, Any]:
    """
    Apply CausVid/LightI2X parameter optimizations with sophisticated precedence logic.
    
    This follows the pattern from travel_between_images.py where:
    1. Models with built-in LoRAs use their JSON config settings
    2. Explicit LoRA requests (not from model config) use hardcoded optimizations
    3. Task parameters always take highest precedence
    
    Args:
        params: Parameters dict to modify
        causvid_enabled: Whether CausVid optimization is enabled
        lighti2x_enabled: Whether LightI2X optimization is enabled
        model_name: Name of the model
        task_params: Optional task parameters for precedence
        orchestrator_payload: Optional orchestrator payload for precedence
        task_id: Task ID for logging
        dprint: Optional debug print function
        
    Returns:
        Updated parameters dict
    """
    if not (causvid_enabled or lighti2x_enabled):
        return params
    
    # Detect if LoRAs are built into model config vs explicit requests
    model_has_causvid, model_has_lighti2x = detect_loras_in_model_config(model_name, dprint)
    
    if dprint:
        dprint(f"[LORA_OPT] Task {task_id}: Parameter optimization analysis:")
        dprint(f"[LORA_OPT]   causvid_enabled: {causvid_enabled}")
        dprint(f"[LORA_OPT]   lighti2x_enabled: {lighti2x_enabled}")
        dprint(f"[LORA_OPT]   model_has_causvid: {model_has_causvid}")
        dprint(f"[LORA_OPT]   model_has_lighti2x: {model_has_lighti2x}")
    
    # Determine parameter defaults based on LoRA type
    if causvid_enabled and not model_has_causvid:
        # Explicit CausVid request (not from model config) - use hardcoded optimization
        num_inference_steps_default = 9   # CausVid optimized steps
        guidance_scale_default = 1.0      # CausVid optimized guidance
        flow_shift_default = 1.0          # CausVid optimized flow_shift
        optimization_type = "explicit_causvid"
        if dprint:
            dprint(f"[LORA_OPT] Task {task_id}: Using explicit CausVid optimization")
            
    elif lighti2x_enabled and not model_has_lighti2x:
        # Explicit LightI2X request (not from model config) - use hardcoded optimization
        num_inference_steps_default = 6   # LightI2X optimized steps
        guidance_scale_default = 1.0      # LightI2X optimized guidance
        flow_shift_default = 5.0          # LightI2X optimized flow_shift
        optimization_type = "explicit_lighti2x"
        if dprint:
            dprint(f"[LORA_OPT] Task {task_id}: Using explicit LightI2X optimization")
    else:
        # Model has built-in LoRAs - use JSON config settings (already loaded in params)
        num_inference_steps_default = 30  # Will be overridden by JSON config if available
        guidance_scale_default = 5.0      # Will be overridden by JSON config if available  
        flow_shift_default = 3.0          # Will be overridden by JSON config if available
        optimization_type = "json_config"
        if dprint:
            dprint(f"[LORA_OPT] Task {task_id}: Using model's JSON config settings")
    
    # Apply parameter precedence chain
    # Priority: task_params > orchestrator_payload > current params > optimization defaults
    
    # Number of inference steps (check both 'num_inference_steps' and 'steps')
    # Check if steps are explicitly set anywhere in the precedence chain
    explicit_steps = (
        (task_params.get("num_inference_steps") if task_params else None) or
        (task_params.get("steps") if task_params else None) or
        (orchestrator_payload.get("num_inference_steps") if orchestrator_payload else None) or
        (orchestrator_payload.get("steps") if orchestrator_payload else None) or
        params.get("num_inference_steps") or
        params.get("steps")
    )
    
    # Only use LoRA step optimization if no explicit steps are set
    if explicit_steps is not None:
        final_steps = explicit_steps
        step_source = "explicit"
    else:
        final_steps = num_inference_steps_default
        step_source = "lora_optimization"
    
    # Guidance scale
    final_guidance = (
        (task_params.get("guidance_scale") if task_params else None) or
        (orchestrator_payload.get("guidance_scale") if orchestrator_payload else None) or
        params.get("guidance_scale") or
        guidance_scale_default
    )
    
    # Flow shift
    final_flow_shift = (
        (task_params.get("flow_shift") if task_params else None) or
        (orchestrator_payload.get("flow_shift") if orchestrator_payload else None) or
        params.get("flow_shift") or
        flow_shift_default
    )
    
    # Update parameters
    params["num_inference_steps"] = final_steps
    params["guidance_scale"] = final_guidance
    params["flow_shift"] = final_flow_shift
    
    if dprint:
        dprint(f"[LORA_OPT] Task {task_id}: Applied {optimization_type} optimization:")
        dprint(f"[LORA_OPT]   num_inference_steps: {final_steps} (source: {step_source})")
        dprint(f"[LORA_OPT]   guidance_scale: {final_guidance}")
        dprint(f"[LORA_OPT]   flow_shift: {final_flow_shift}")
        
        # Validation feedback - only warn about step optimization when it's actually being applied
        if step_source == "lora_optimization":
            if causvid_enabled and final_steps == 9:
                dprint(f"[LORA_OPT] ✅ CausVid optimization SUCCESS: Using optimized 9 steps!")
            elif causvid_enabled and final_steps != 9:
                dprint(f"[LORA_OPT] ⚠️  WARNING: CausVid enabled but using {final_steps} steps instead of optimized 9 steps!")
                
            if lighti2x_enabled and final_steps == 6:
                dprint(f"[LORA_OPT] ✅ LightI2X optimization SUCCESS: Using optimized 6 steps!")
            elif lighti2x_enabled and final_steps != 6:
                dprint(f"[LORA_OPT] ⚠️  WARNING: LightI2X enabled but using {final_steps} steps instead of optimized 6 steps!")
        else:
            # Steps were explicitly set, so just note that LoRA optimization is respecting explicit values
            if causvid_enabled or lighti2x_enabled:
                dprint(f"[LORA_OPT] ℹ️  Using explicit step count ({final_steps}), LoRA step optimization bypassed")
    
    return params


def normalize_lora_format(params: Dict[str, Any], task_id: str = "unknown", dprint=None) -> Dict[str, Any]:
    """
    Normalize LoRA parameters from various input formats to standard format.
    
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
    """
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
            # Convert comma-separated string to list of floats
            params["lora_multipliers"] = [float(x.strip()) for x in multipliers.split(",") if x.strip()]
        # Keep as-is if already a list
        
        if dprint:
            dprint(f"[LORA_NORM] Task {task_id}: Normalized lora_multipliers: {params.get('lora_multipliers', [])}")
    
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
    
    return params


def _process_additional_loras(additional_loras_dict: Dict[str, float], task_id: str, dprint=None) -> Tuple[list, list]:
    """
    Process additional LoRAs dict, downloading URLs if needed.
    
    Args:
        additional_loras_dict: Dict mapping LoRA names/URLs to multipliers
        task_id: Task ID for logging
        dprint: Optional debug print function
        
    Returns:
        Tuple of (processed_names, multipliers)
    """
    processed_names = []
    multipliers = list(additional_loras_dict.values())
    
    for lora_name_or_url in additional_loras_dict.keys():
        if lora_name_or_url.startswith("http"):
            # It's a URL - download it
            try:
                local_filename = _download_lora_from_url(lora_name_or_url, task_id, dprint)
                processed_names.append(local_filename)
            except Exception as e:
                if dprint:
                    dprint(f"[LORA_DOWNLOAD] Task {task_id}: Failed to download {lora_name_or_url}: {e}")
                processed_names.append(lora_name_or_url)  # Keep URL as fallback
        else:
            # Already a local filename
            processed_names.append(lora_name_or_url)
    
    return processed_names, multipliers


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
    
    # Extract filename from URL
    local_filename = url.split("/")[-1]
    
    # Determine LoRA directory: prefer the WGP-visible root 'loras'
    lora_dir = "loras"
    
    local_path = os.path.join(lora_dir, local_filename)
    
    if dprint:
        dprint(f"[LORA_DOWNLOAD] Task {task_id}: Downloading {local_filename} to {lora_dir} from {url}")
    
    # Check if file already exists
    if not os.path.isfile(local_path):
        if url.startswith("https://huggingface.co/") and "/resolve/main/" in url:
            # Use HuggingFace hub for HF URLs
            from huggingface_hub import hf_hub_download

            # Parse HuggingFace URL
            url_path = url[len("https://huggingface.co/"):]
            url_parts = url_path.split("/resolve/main/")
            repo_id = url_parts[0]
            rel_path = url_parts[-1]
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
    
    # Step 2: Apply model-specific optimizations (LightI2X, CausVid)
    use_causvid, use_lighti2x = detect_lora_optimization_flags(
        task_params=task_params,
        model_name=model_name,
        dprint=dprint
    )
    
    if use_causvid or use_lighti2x:
        params = apply_lora_parameter_optimization(
            params=params,
            causvid_enabled=use_causvid,
            lighti2x_enabled=use_lighti2x,
            model_name=model_name,
            task_params=task_params,
            orchestrator_payload=orchestrator_payload,
            task_id=task_id,
            dprint=dprint
        )
    
    # Add required LoRAs to the list
    if use_causvid:
        params = ensure_lora_in_list(
            params, "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors", 
            "CausVid", task_id, dprint
        )
    
    if use_lighti2x:
        params = ensure_lora_in_list(
            params, "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
            "LightI2X", task_id, dprint
        )
        # LightI2X-specific settings
        params["sample_solver"] = "unipc"
        params["denoise_strength"] = 1.0
    
    # Step 3: Process additional_loras from orchestrator payload
    if orchestrator_payload and "additional_loras" in orchestrator_payload:
        params["additional_loras"] = orchestrator_payload["additional_loras"]
        params = normalize_lora_format(params, task_id, dprint)  # Re-normalize after adding additional
        
        if dprint:
            dprint(f"[LORA_PROCESS] Task {task_id}: Added {len(orchestrator_payload['additional_loras'])} additional LoRAs from orchestrator")
    
    # Step 4: Ensure all LoRAs in the list exist (auto-download if needed)
    lora_names = params.get("lora_names", [])
    if lora_names:
        if dprint:
            dprint(f"[LORA_PROCESS] Task {task_id}: Verifying {len(lora_names)} LoRAs exist: {lora_names}")

        for lora_name in lora_names[:]:  # Copy list to avoid modification during iteration
            if not _check_lora_exists(lora_name):
                if dprint:
                    dprint(f"[LORA_DOWNLOAD] Task {task_id}: LoRA not found, attempting auto-download: {lora_name}")

                download_success = _download_lora_auto(lora_name, "required", dprint)
                if not download_success:
                    if dprint:
                        dprint(f"[LORA_DOWNLOAD] Task {task_id}: Warning - Could not download required LoRA: {lora_name}")
                        dprint(f"[LORA_PROCESS] Task {task_id}: Dropping missing LoRA '{lora_name}' to avoid generation failure")
                    # Remove missing LoRA to avoid downstream loader errors
                    try:
                        idx = lora_names.index(lora_name)
                        lora_names.pop(idx)
                        # Keep multipliers list in sync if already present
                        lora_multipliers = params.get("lora_multipliers", [])
                        if isinstance(lora_multipliers, list) and idx < len(lora_multipliers):
                            lora_multipliers.pop(idx)
                            params["lora_multipliers"] = lora_multipliers
                    except Exception:
                        pass
        # Store potentially pruned list back
        params["lora_names"] = lora_names
    
    # Step 5: Ensure multipliers list matches LoRA names list
    lora_names = params.get("lora_names", [])
    lora_multipliers = params.get("lora_multipliers", [])
    
    # Extend multipliers list if needed
    while len(lora_multipliers) < len(lora_names):
        lora_multipliers.append(1.0)
    
    # Truncate multipliers list if too long
    if len(lora_multipliers) > len(lora_names):
        lora_multipliers = lora_multipliers[:len(lora_names)]
    
    params["lora_multipliers"] = lora_multipliers
    
    # Step 6: Convert to final WGP format
    if lora_names:
        params["activated_loras"] = lora_names  # WGP expects this format
        params["loras_multipliers"] = ",".join(map(str, lora_multipliers))  # WGP expects comma-separated string
        
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
