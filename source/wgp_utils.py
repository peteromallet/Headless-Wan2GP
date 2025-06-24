"""Utility helpers around Wan2GP.wgp API used by headless tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Tuple, Dict, Optional, Union, Callable


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

    # Override LoRA settings from processed_additional_loras
    params["activated_loras"] = activated_loras
    params["loras_multipliers"] = ','.join(loras_multipliers) if loras_multipliers else ""

    # Convert PIL images in image_start/image_end if any
    for key in ("image_start", "image_end"):
        if key in params and isinstance(params[key], list):
            params[key] = [wgp_mod.convert_image(img) for img in params[key]]

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
        dprint(f"{task_id}: generate_single_video failed: {e}")
        return False, None 