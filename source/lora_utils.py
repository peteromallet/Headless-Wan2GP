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
        
        import wgp
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
    final_steps = (
        (task_params.get("num_inference_steps") if task_params else None) or
        (task_params.get("steps") if task_params else None) or
        (orchestrator_payload.get("num_inference_steps") if orchestrator_payload else None) or
        (orchestrator_payload.get("steps") if orchestrator_payload else None) or
        params.get("num_inference_steps") or
        params.get("steps") or
        num_inference_steps_default
    )
    
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
        dprint(f"[LORA_OPT]   num_inference_steps: {final_steps}")
        dprint(f"[LORA_OPT]   guidance_scale: {final_guidance}")
        dprint(f"[LORA_OPT]   flow_shift: {final_flow_shift}")
        
        # Validation warnings
        if causvid_enabled and final_steps == 9:
            dprint(f"[LORA_OPT] ✅ CausVid optimization SUCCESS: Using optimized 9 steps!")
        elif causvid_enabled and final_steps != 9:
            dprint(f"[LORA_OPT] ⚠️  WARNING: CausVid enabled but using {final_steps} steps instead of optimized 9 steps!")
            
        if lighti2x_enabled and final_steps == 6:
            dprint(f"[LORA_OPT] ✅ LightI2X optimization SUCCESS: Using optimized 6 steps!")
        elif lighti2x_enabled and final_steps != 6:
            dprint(f"[LORA_OPT] ⚠️  WARNING: LightI2X enabled but using {final_steps} steps instead of optimized 6 steps!")
    
    return params


def ensure_lora_in_list(params: Dict[str, Any], lora_filename: str, lora_type: str, task_id: str = "unknown", dprint=None) -> Dict[str, Any]:
    """
    Ensure a specific LoRA is included in the activated LoRA list.
    
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
    
    if lora_filename not in current_loras:
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
