"""Utility helpers around Wan2GP.wgp API used by headless tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Tuple, Dict, Optional, Union, Callable
from PIL import Image


def _ensure_lora_in_lists(lora_name: str, multiplier: str, activated_loras: list, loras_multipliers: Union[list, str]) -> Tuple[list, Union[list, str]]:
    """
    Helper function to ensure a LoRA is present in activated_loras with the correct multiplier.
    Returns updated (activated_loras, loras_multipliers) tuple.
    """
    if lora_name not in activated_loras:
        activated_loras.insert(0, lora_name)
        if isinstance(loras_multipliers, list):
            loras_multipliers.insert(0, multiplier)
        elif isinstance(loras_multipliers, str):
            mult_list = [m.strip() for m in loras_multipliers.split(",") if m.strip()] if loras_multipliers else []
            mult_list.insert(0, multiplier)
            loras_multipliers = ",".join(mult_list)
        else:
            loras_multipliers = [multiplier]
    return activated_loras, loras_multipliers


def _set_param_if_different(params: dict, key: str, target_value: Union[int, float], task_id: str, lora_name: str, dprint: Callable):
    """
    Helper function to set a parameter value if it differs from the target, with logging.
    """
    current_value = params.get(key)
    if isinstance(target_value, int):
        try:
            current_int = int(current_value) if current_value is not None else 0
        except (TypeError, ValueError):
            current_int = 0
        if current_int != target_value:
            dprint(f"{task_id}: {lora_name} active – overriding {key} {current_int} → {target_value}")
            params[key] = target_value
    elif isinstance(target_value, float):
        current_float = float(current_value) if current_value is not None else 0.0
        if current_float != target_value:
            dprint(f"{task_id}: {lora_name} active – setting {key} → {target_value}")
            params[key] = target_value


def _normalize_loras_multipliers_format(loras_multipliers: Union[list, str]) -> str:
    """
    Helper function to normalize loras_multipliers to string format expected by WGP.
    """
    if isinstance(loras_multipliers, list):
        return ",".join(loras_multipliers) if loras_multipliers else ""
    else:
        return loras_multipliers if loras_multipliers else ""


def generate_single_video(*args, **kwargs) -> Tuple[bool, Optional[str]]:
    """
    Centralized wrapper for wgp_mod.generate_video.

    Uses flexible keyword-style API only:
    generate_single_video(
        wgp_mod=wgp_mod,
        task_id="abc",
        prompt="...",
        negative_prompt="...",
        resolution="832x512",
        video_length=81,
        seed=123,
        model_filename=model_filename,
        dprint=print,  # optional
        **extra_params  # forwarded verbatim to wgp_mod.generate_video
    )
    
    This is the ONLY way to generate videos - all handlers must use this interface.
    
    Args:
        wgp_mod: WGP module instance
        task_id: Unique task identifier
        prompt: Text prompt for generation
        negative_prompt: Negative prompt (defaults to space)
        resolution: Resolution in "WIDTHxHEIGHT" format
        video_length: Number of frames to generate
        seed: Random seed for generation
        model_filename: Model filename to use
        dprint: Debug print function (optional)
        **kwargs: Additional parameters for wgp_mod.generate_video
        
    Returns:
        Tuple of (success: bool, output_path_or_None: str | None)
    """
    # Extract required parameters
    wgp_mod = kwargs.pop("wgp_mod", None)
    if wgp_mod is None and args:
        wgp_mod = args[0]
    if wgp_mod is None:
        raise ValueError("wgp_mod must be provided")

    task_id: str = kwargs.pop("task_id")
    prompt: str = kwargs.pop("prompt")
    resolution: str = kwargs.pop("resolution")
    video_length: int = kwargs.pop("video_length")
    seed: int = kwargs.pop("seed")
    model_filename: str = kwargs.pop("model_filename")
    negative_prompt: str = kwargs.pop("negative_prompt", " ")  # Default to space
    dprint: Callable = kwargs.pop("dprint", (lambda *_: None))

    # Validate required parameters
    if not prompt or not prompt.strip():
        prompt = " "  # Ensure valid prompt
    if not negative_prompt or not negative_prompt.strip():
        negative_prompt = " "  # Ensure valid negative prompt

    # Translate convenience keys to the names expected by wgp.py
    if "video_guide_path" in kwargs and "video_guide" not in kwargs:
        kwargs["video_guide"] = kwargs.pop("video_guide_path")
    if "video_mask_path" in kwargs and "video_mask" not in kwargs:
        kwargs["video_mask"] = kwargs.pop("video_mask_path")
    if "image_refs_paths" in kwargs and "image_refs" not in kwargs:
        kwargs["image_refs"] = kwargs.pop("image_refs_paths")

    # ------------------------------------------------------------------
    # Special flags (popped early so they don't pollute **kwargs later)
    # ------------------------------------------------------------------
    use_causvid_lora: bool = bool(kwargs.pop("use_causvid_lora", False))
    # New: Lightweight I2X LoRA (self-forcing) support
    use_lighti2x_lora: bool = bool(kwargs.pop("use_lighti2x_lora", False))
    apply_reward_lora: bool = bool(kwargs.pop("apply_reward_lora", False))

    # Handle additional_loras parameter
    processed_additional_loras: Dict[str, str] = kwargs.pop("additional_loras", {})
    
    # Convert to WGP's expected LoRA format
    activated_loras = []
    loras_multipliers = []
    
    if processed_additional_loras:
        for lora_filename, strength_str in processed_additional_loras.items():
            activated_loras.append(lora_filename)
            loras_multipliers.append(str(strength_str))
    
    # Defaults for the enormous parameter list
    defaults = dict(
        num_inference_steps=30,
        guidance_scale=5.0,
        audio_guidance_scale=5.0,
        flow_shift=wgp_mod.get_default_flow(model_filename, wgp_mod.test_class_i2v(model_filename)),
        embedded_guidance_scale=6.0,
        repeat_generation=1,
        multi_images_gen_type=0,
        tea_cache_setting=0.0,
        tea_cache_start_step_perc=0,
        activated_loras=activated_loras,
        loras_multipliers=','.join(loras_multipliers) if loras_multipliers else "",
        image_prompt_type="T",
        image_start=[],
        image_end=[],
        model_mode=0,
        video_source=None,
        keep_frames_video_source="",
        video_prompt_type=kwargs.get("video_prompt_type", ""),
        keep_frames_video_guide="",
        sliding_window_size=81,
        sliding_window_overlap=5,
        sliding_window_overlap_noise=20,
        sliding_window_discard_last_frames=0,
        remove_background_images_ref=False,
        temporal_upsampling="",
        spatial_upsampling="",
        RIFLEx_setting=0,
        slg_switch=0,
        slg_layers=[9],
        slg_start_perc=10,
        slg_end_perc=90,
        cfg_star_switch=0,
        cfg_zero_step=-1,
        prompt_enhancer="",
        state={},
    )
    params = {**defaults, **kwargs}

    # ------------------------------------------------------------
    #  CausVid LoRA upstream fix – keep WGP happy
    # ------------------------------------------------------------
    if use_causvid_lora:
        causvid_lora_name = "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"
        activated_loras, loras_multipliers = _ensure_lora_in_lists(causvid_lora_name, "1.0", activated_loras, loras_multipliers)
        _set_param_if_different(params, "num_inference_steps", 9, task_id, "CausVid", dprint)
        _set_param_if_different(params, "guidance_scale", 1.0, task_id, "CausVid", dprint)
        _set_param_if_different(params, "flow_shift", 1.0, task_id, "CausVid", dprint)

    # ------------------------------------------------------------
    #  Light I2X LoRA tweaks – fast 4-step schedule with unipc sampler
    # ------------------------------------------------------------
    if use_lighti2x_lora:
        lighti2x_lora_name = "wan_lcm_r16_fp32_comfy.safetensors"
        activated_loras, loras_multipliers = _ensure_lora_in_lists(lighti2x_lora_name, "1.0", activated_loras, loras_multipliers)
        _set_param_if_different(params, "num_inference_steps", 4, task_id, "LightI2X", dprint)
        _set_param_if_different(params, "guidance_scale", 1.0, task_id, "LightI2X", dprint)
        _set_param_if_different(params, "flow_shift", 5.0, task_id, "LightI2X", dprint)
        _set_param_if_different(params, "tea_cache_setting", 0.0, task_id, "LightI2X", dprint)
        _set_param_if_different(params, "sample_solver", "unipc", task_id, "LightI2X", dprint)
        _set_param_if_different(params, "denoise_strength", 1.0, task_id, "LightI2X", dprint)

    # Apply LoRA changes to params if any special LoRAs were configured
    if use_causvid_lora or use_lighti2x_lora:
        params["activated_loras"] = activated_loras
        params["loras_multipliers"] = _normalize_loras_multipliers_format(loras_multipliers)

    # Expose the flags to downstream logic (build_task_state & wgp)
    params["use_causvid_lora"] = use_causvid_lora
    params["use_lighti2x_lora"] = use_lighti2x_lora
    params["apply_reward_lora"] = apply_reward_lora

    # Convert PIL images in image_start/image_end if any
    for key in ("image_start", "image_end"):
        if key in params and isinstance(params[key], list):
            params[key] = [wgp_mod.convert_image(img) for img in params[key]]

    # Convert image_refs paths to PIL images if provided as paths/strings
    if "image_refs" in params and isinstance(params["image_refs"], list):
        refs_list = params["image_refs"]

        # If the list is empty → treat as no refs at all to avoid VAE index errors.
        if not refs_list:
            params["image_refs"] = None
        # Non-empty list: convert string/Path entries to PIL then to tensor
        elif isinstance(refs_list[0], (str, Path)):
            converted_refs = []
            for ref_path in refs_list:
                try:
                    img_pil = Image.open(str(ref_path)).convert("RGB")
                    converted_refs.append(wgp_mod.convert_image(img_pil))
                except Exception as e_img:
                    dprint(f"{task_id}: WARNING – failed to load reference image '{ref_path}': {e_img}")
            params["image_refs"] = converted_refs if converted_refs else None

    # Ensure we have a valid `state` structure expected by wgp.py.  When callers
    # (e.g. single-image or travel_segment helpers) invoke this wrapper they may
    # omit the `state` argument entirely which causes wgp.py to raise a
    # KeyError or, later on, a mysterious `list index out of range` when it
    # tries to access nested lists inside the state dict.  We therefore attempt
    # to build a **fully-featured** state using the same helper that headless.py
    # relies on.  If that fails for any reason we gracefully fall back to the
    # minimal stub so that the previous behaviour is preserved.
    if "state" not in params or not params["state"]:
        try:
            # --- Prefer the robust builder used by headless.py ---
            from source.common_utils import build_task_state  # Local import to avoid circular deps

            # Obtain the full list of LoRAs available for this model so that
            # build_task_state can resolve `activated_loras` correctly.
            lora_dir_for_model = wgp_mod.get_lora_dir(model_filename)
            all_loras_for_model, *_ = wgp_mod.setup_loras(
                model_filename, None, lora_dir_for_model, "", None
            )

            task_params_for_state = {
                "task_id": task_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "resolution": resolution,
                "video_length": video_length,
                "seed": seed,
                "num_inference_steps": params["num_inference_steps"],
                "guidance_scale": params["guidance_scale"],
                "flow_shift": params["flow_shift"],
                "activated_loras": params["activated_loras"],
                "loras_multipliers": params["loras_multipliers"],
                # Expose the CausVid flag so that builder can apply its tweaks
                "use_causvid_lora": use_causvid_lora,
                "use_lighti2x_lora": use_lighti2x_lora,
            }
            # Merge any miscellaneous keys the caller already supplied – this
            # lets advanced features (video_guide, image_refs, etc.) propagate.
            task_params_for_state.update({k: v for k, v in kwargs.items()})

            state_built, _ = build_task_state(
                wgp_mod,
                model_filename,
                task_params_for_state,
                all_loras_for_model,
                image_download_dir=None,
                apply_reward_lora=apply_reward_lora,
            )
            params["state"] = state_built
        except Exception as e_state:
            # Fall back to the lightweight stub built below
            dprint(f"{task_id}: Failed to build full task state ({e_state}); falling back to minimal stub.")
            params["state"] = None  # Will be replaced a few lines later

    if not params["state"]:
        # --- Minimal fallback (previous behaviour) ---
        minimal_state = {
            "model_filename": model_filename,
            "validate_success": 1,
            "advanced": True,
            # The UI logic inside wgp.py expects this nested structure to exist.
            "gen": {
                "queue": [],
                "file_list": [],
                "file_settings_list": [],
                "prompt_no": 1,
                "prompts_max": 1,
            },
            "loras": [],
        }
        # Add a per-model settings bucket so that wgp.py can read/write into it.
        model_type_key = wgp_mod.get_model_type(model_filename)
        minimal_state[model_type_key] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "resolution": resolution,
            "video_length": video_length,
            "seed": seed,
            "num_inference_steps": params["num_inference_steps"],
            "guidance_scale": params["guidance_scale"],
            "flow_shift": params["flow_shift"],
            "activated_loras": params["activated_loras"],
            "loras_multipliers": params["loras_multipliers"],
        }
        params["state"] = minimal_state

    # At this point we are guaranteed to have a usable state dict.

    # Build send_cmd for progress reporting
    def _send(cmd: str, data: Any = None) -> None:
        prefix = f"[Task ID: {task_id}]"
        if cmd == "progress":
            if isinstance(data, list) and len(data) >= 2:
                prog, txt = data[0], data[1]
                if isinstance(prog, tuple) and len(prog) == 2:
                    step, total = prog
                    print(f"{prefix}[Progress] {step}/{total} – {txt}")
                else:
                    print(f"{prefix}[Progress] {txt}")
        elif cmd == "status":
            print(f"{prefix}[Status] {data}")
        elif cmd == "info":
            print(f"{prefix}[INFO] {data}")
        elif cmd == "error":
            print(f"{prefix}[ERROR] {data}")
            raise RuntimeError(f"wgp.py error for {task_id}: {data}")
        elif cmd == "output":
            print(f"{prefix}[Output] video written.")

    # Preserve true single-frame requests.  Only bump **other** tiny values (<4 and ≠1).
    if video_length < 4 and video_length != 1:
        dprint(f"{task_id}: Requested video_length={video_length} too small; bumping to 4 to satisfy WGP minimum (except when exactly 1 frame is desired).")
        video_length = 4

    try:
        dprint(f"{task_id}: Calling wgp_mod.generate_video (centralized API)")
        wgp_mod.generate_video(
            task={"id": 1, "prompt": prompt,
                  "params": {"model_filename_from_gui_state": model_filename}},
            send_cmd=_send,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            video_length=video_length,
            seed=seed,
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
            audio_guidance_scale=params["audio_guidance_scale"],
            flow_shift=params["flow_shift"],
            embedded_guidance_scale=params["embedded_guidance_scale"],
            repeat_generation=params["repeat_generation"],
            multi_images_gen_type=params["multi_images_gen_type"],
            tea_cache_setting=params["tea_cache_setting"],
            tea_cache_start_step_perc=params["tea_cache_start_step_perc"],
            activated_loras=params["activated_loras"],
            loras_multipliers=params["loras_multipliers"],
            image_prompt_type=params["image_prompt_type"],
            image_start=params["image_start"],
            image_end=params["image_end"],
            model_mode=params["model_mode"],
            video_source=params["video_source"],
            keep_frames_video_source=params["keep_frames_video_source"],
            video_prompt_type=params["video_prompt_type"],
            image_refs=params.get("image_refs"),
            video_guide=params.get("video_guide"),
            keep_frames_video_guide=params["keep_frames_video_guide"],
            video_mask=params.get("video_mask"),
            audio_guide=params.get("audio_guide"),
            sliding_window_size=params["sliding_window_size"],
            sliding_window_overlap=params["sliding_window_overlap"],
            sliding_window_overlap_noise=params["sliding_window_overlap_noise"],
            sliding_window_discard_last_frames=params["sliding_window_discard_last_frames"],
            remove_background_images_ref=params["remove_background_images_ref"],
            temporal_upsampling=params["temporal_upsampling"],
            spatial_upsampling=params["spatial_upsampling"],
            RIFLEx_setting=params["RIFLEx_setting"],
            slg_switch=params["slg_switch"],
            slg_layers=params["slg_layers"],
            slg_start_perc=params["slg_start_perc"],
            slg_end_perc=params["slg_end_perc"],
            cfg_star_switch=params["cfg_star_switch"],
            cfg_zero_step=params["cfg_zero_step"],
            prompt_enhancer=params["prompt_enhancer"],
            state=params["state"],
            model_filename=model_filename,
        )
        out_candidates = sorted(Path(wgp_mod.save_path).glob("*.mp4"))
        return True, str(out_candidates[-1]) if out_candidates else None
    except Exception as e:
        import traceback
        print(f"\n[FULL TRACEBACK] {task_id}: generate_single_video failed:")
        traceback.print_exc()
        dprint(f"{task_id}: generate_single_video failed: {e}")
        return False, None 