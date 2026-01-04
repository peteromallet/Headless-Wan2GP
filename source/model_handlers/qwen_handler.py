"""
Qwen Image Edit Model Handler

Handles all Qwen-specific task types with proper LoRA configuration,
system prompts, and preprocessing logic.

Extracted from worker.py (687 lines of Qwen code).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from io import BytesIO
from PIL import Image  # type: ignore
from huggingface_hub import hf_hub_download  # type: ignore

from source.logging_utils import headless_logger
from source.common_utils import download_image_if_url as sm_download_image_if_url


class QwenHandler:
    """Handles all Qwen image editing task types."""
    
    def __init__(self, wan_root: str, task_id: str, dprint_func):
        """
        Initialize Qwen handler.
        
        Args:
            wan_root: Path to Wan2GP root directory
            task_id: Task ID for logging
            dprint_func: Debug print function
        """
        self.wan_root = Path(wan_root).resolve()
        self.task_id = task_id
        self.dprint = dprint_func
        self.qwen_lora_dir = self.wan_root / "loras_qwen"
        self.qwen_lora_dir.mkdir(parents=True, exist_ok=True)

    def _log_debug(self, message: str):
        """Log debug message."""
        if self.dprint:
            self.dprint(f"[QWEN_HANDLER] Task {self.task_id}: {message}")

    def _log_info(self, message: str):
        """Log info message."""
        headless_logger.info(f"[QWEN_HANDLER] Task {self.task_id}: {message}", task_id=self.task_id)

    def _log_warning(self, message: str):
        """Log warning message."""
        headless_logger.warning(f"[QWEN_HANDLER] Task {self.task_id}: {message}", task_id=self.task_id)

    def _log_error(self, message: str):
        """Log error message."""
        headless_logger.error(f"[QWEN_HANDLER] Task {self.task_id}: {message}", task_id=self.task_id)

    def cap_qwen_resolution(self, resolution_str: str) -> Optional[str]:
        """
        Cap resolution to 1200px max dimension while maintaining aspect ratio.
        """
        max_dimension = 1200
        if not resolution_str or 'x' not in resolution_str:
            return None
        try:
            width, height = map(int, resolution_str.split('x'))
        except ValueError:
            self._log_warning(f"Invalid resolution format: {resolution_str}")
            return None
        
        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension / width, max_dimension / height)
            width = int(width * ratio)
            height = int(height * ratio)
            capped = f"{width}x{height}"
            self._log_info(f"Resolution capped from {resolution_str} to {capped}")
            return capped
        return resolution_str

    def create_qwen_masked_composite(
        self,
        image_url: str,
        mask_url: str,
        output_dir: Path
    ) -> str:
        """
        Create composite image with green overlay for Qwen inpainting/annotation.
        """
        try:
            self._log_debug(f"Downloading image from {image_url}")
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()
            image = Image.open(BytesIO(img_response.content)).convert('RGB')
            
            max_dimension = 1200
            width, height = image.size
            if width > max_dimension or height > max_dimension:
                ratio = min(max_dimension / width, max_dimension / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self._log_debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
                width, height = new_width, new_height
            
            self._log_debug(f"Downloading mask from {mask_url}")
            mask_response = requests.get(mask_url, timeout=30)
            mask_response.raise_for_status()
            mask = Image.open(BytesIO(mask_response.content)).convert('L')
            
            if mask.size != (width, height):
                mask = mask.resize((width, height), Image.Resampling.LANCZOS)
                self._log_debug(f"Resized mask to match image: {width}x{height}")
            
            mask = mask.point(lambda x: 0 if x < 128 else 255)
            green_overlay = Image.new('RGB', (width, height), (0, 255, 0))
            composite = Image.composite(green_overlay, image, mask)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            composite_filename = f"inpaint_composite_{self.task_id}.jpg"
            composite_path = output_dir / composite_filename
            composite.save(composite_path, 'JPEG', quality=95)
            
            self._log_info(f"Created green mask composite: {composite_path}")
            return str(composite_path)
        except Exception as e:
            self._log_error(f"Failed to create masked composite: {e}")
            raise ValueError(f"Composite image creation failed: {e}")

    def _download_lora_if_missing(self, repo_id: str, filename: str) -> Optional[Path]:
        """Helper to download a LoRA if it doesn't exist locally."""
        target_path = self.qwen_lora_dir / filename
        if target_path.exists():
            self._log_debug(f"LoRA already present: {target_path}")
            return target_path.resolve()
        
        self._log_info(f"Downloading LoRA '{filename}' from '{repo_id}'")
        try:
            dl_path = hf_hub_download(
                repo_id=repo_id, 
                filename=filename, 
                revision="main", 
                local_dir=str(self.qwen_lora_dir)
            )
            actual_path = Path(dl_path)
            if actual_path.exists() and actual_path.resolve() != target_path.resolve():
                actual_path.replace(target_path)
            self._log_info(f"Successfully downloaded {filename}")
            return target_path.resolve()
        except Exception as e:
            self._log_warning(f"LoRA download failed for {filename}: {e}")
            return None

    def handle_qwen_image_edit(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """Handle qwen_image_edit task type."""
        self._log_info("Processing qwen_image_edit task")
        
        image_url = db_task_params.get("image") or db_task_params.get("image_url")
        if not image_url:
            raise ValueError(f"Task {self.task_id}: 'image' or 'image_url' required for qwen_image_edit")
        
        downloads_dir = Path("outputs/qwen_edit_images")
        downloads_dir.mkdir(parents=True, exist_ok=True)
        local_image_path = sm_download_image_if_url(
            image_url, downloads_dir, 
            task_id_for_logging=self.task_id, 
            debug_mode=False,
            descriptive_name="edit_image"
        )
        generation_params["image_guide"] = str(local_image_path)
        self._log_info(f"Using image_guide: {local_image_path}")
        
        if "resolution" in db_task_params:
            capped_res = self.cap_qwen_resolution(db_task_params["resolution"])
            if capped_res:
                generation_params["resolution"] = capped_res
        
        generation_params.setdefault("video_prompt_type", "KI")
        generation_params.setdefault("guidance_scale", 1)
        generation_params.setdefault("num_inference_steps", 12)
        generation_params.setdefault("video_length", 1)

        if "system_prompt" in db_task_params and db_task_params["system_prompt"]:
            generation_params["system_prompt"] = db_task_params["system_prompt"]
        else:
            generation_params["system_prompt"] = "You are a professional image editor. Analyze the input image carefully, then apply the requested modifications precisely while maintaining visual coherence and image quality."

        # Ensure Lightning LoRA
        lightning_fname = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
        selected_lightning_path = self.qwen_lora_dir / lightning_fname
        if not selected_lightning_path.exists():
            selected_lightning_path = self._download_lora_if_missing("lightx2v/Qwen-Image-Lightning", lightning_fname)
        
        if selected_lightning_path:
            if "lora_names" not in generation_params:
                generation_params["lora_names"] = []
            if "lora_multipliers" not in generation_params:
                generation_params["lora_multipliers"] = []
            
            if selected_lightning_path.name not in generation_params["lora_names"]:
                generation_params["lora_names"].append(selected_lightning_path.name)
                generation_params["lora_multipliers"].append(0.75)
                self._log_info(f"Added Lightning LoRA with strength 0.75")

        # Optional hires fix - can be enabled on any qwen_image_edit task
        self._maybe_add_hires_config(db_task_params, generation_params)

    def _maybe_add_hires_config(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """Add hires_config if hires params are present in task params."""
        hires_scale = db_task_params.get("hires_scale")
        if hires_scale is None:
            return  # No hires fix requested
        
        hires_config = {
            "enabled": True,
            "scale": float(hires_scale),
            "hires_steps": int(db_task_params.get("hires_steps", 6)),
            "denoising_strength": float(db_task_params.get("hires_denoise", 0.5)),
            "upscale_method": db_task_params.get("hires_upscale_method", "bicubic"),
        }
        generation_params["hires_config"] = hires_config
        
        self._log_info(
            f"Hires fix enabled: {hires_config['scale']}x scale, "
            f"{hires_config['hires_steps']} steps @ {hires_config['denoising_strength']} denoise"
        )

    def handle_image_inpaint(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """Handle image_inpaint task type."""
        self._log_info("Processing image_inpaint task")
        
        image_url = db_task_params.get("image_url") or db_task_params.get("image")
        mask_url = db_task_params.get("mask_url")
        
        if not image_url or not mask_url:
            raise ValueError(f"Task {self.task_id}: 'image_url' and 'mask_url' required")
        
        composite_dir = Path("outputs/qwen_inpaint_composites")
        composite_path = self.create_qwen_masked_composite(image_url, mask_url, composite_dir)
        
        generation_params["image_guide"] = str(composite_path)
        
        if "resolution" in db_task_params:
            capped_res = self.cap_qwen_resolution(db_task_params["resolution"])
            if capped_res:
                generation_params["resolution"] = capped_res
        
        generation_params.setdefault("video_prompt_type", "KI")
        generation_params.setdefault("guidance_scale", 1)
        generation_params.setdefault("num_inference_steps", 12)
        generation_params.setdefault("video_length", 1)

        if "system_prompt" in db_task_params and db_task_params["system_prompt"]:
            generation_params["system_prompt"] = db_task_params["system_prompt"]
        else:
            generation_params["system_prompt"] = "You are an expert at inpainting. The green areas indicate regions to fill. Analyze the context and generate natural content based on the description."

        # Download LoRAs
        inpaint_fname = "qwen_image_edit_inpainting.safetensors"
        self._download_lora_if_missing("ostris/qwen_image_edit_inpainting", inpaint_fname)
        
        lightning_fname = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
        if not (self.qwen_lora_dir / lightning_fname).exists():
            self._download_lora_if_missing("lightx2v/Qwen-Image-Lightning", lightning_fname)
        selected_lightning = lightning_fname

        if "lora_names" not in generation_params:
            generation_params["lora_names"] = []
        if "lora_multipliers" not in generation_params:
            generation_params["lora_multipliers"] = []

        if selected_lightning not in generation_params["lora_names"]:
            generation_params["lora_names"].append(selected_lightning)
            generation_params["lora_multipliers"].append(0.75)

        if inpaint_fname not in generation_params["lora_names"]:
            generation_params["lora_names"].append(inpaint_fname)
            generation_params["lora_multipliers"].append(1.0)

        # Optional hires fix
        self._maybe_add_hires_config(db_task_params, generation_params)

    def handle_annotated_image_edit(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """Handle annotated_image_edit task type."""
        self._log_info("Processing annotated_image_edit task")
        
        image_url = db_task_params.get("image_url") or db_task_params.get("image")
        mask_url = db_task_params.get("mask_url")
        
        if not image_url or not mask_url:
            raise ValueError(f"Task {self.task_id}: Both 'image_url' and 'mask_url' required")
        
        composite_dir = Path("outputs/qwen_annotate_composites")
        composite_path = self.create_qwen_masked_composite(image_url, mask_url, composite_dir)
        
        generation_params["image_guide"] = str(composite_path)
        
        if "resolution" in db_task_params:
            capped_res = self.cap_qwen_resolution(db_task_params["resolution"])
            if capped_res:
                generation_params["resolution"] = capped_res
        
        generation_params.setdefault("video_prompt_type", "KI")
        generation_params.setdefault("guidance_scale", 1)
        generation_params.setdefault("num_inference_steps", 12)
        generation_params.setdefault("video_length", 1)

        if "system_prompt" in db_task_params and db_task_params["system_prompt"]:
            generation_params["system_prompt"] = db_task_params["system_prompt"]
        else:
            generation_params["system_prompt"] = "You are an expert at interpreting visual annotations. Analyze the green annotations and modify the marked areas according to instructions."

        # Download LoRAs
        annotate_fname = "in_scene_pure_squares_flipped_450_lr_000006700.safetensors"
        self._download_lora_if_missing("peteromallet/random_junk", annotate_fname)
        
        lightning_fname = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
        if not (self.qwen_lora_dir / lightning_fname).exists():
            self._download_lora_if_missing("lightx2v/Qwen-Image-Lightning", lightning_fname)
        selected_lightning = lightning_fname

        if "lora_names" not in generation_params:
            generation_params["lora_names"] = []
        if "lora_multipliers" not in generation_params:
            generation_params["lora_multipliers"] = []

        if selected_lightning not in generation_params["lora_names"]:
            generation_params["lora_names"].append(selected_lightning)
            generation_params["lora_multipliers"].append(0.75)

        if annotate_fname not in generation_params["lora_names"]:
            generation_params["lora_names"].append(annotate_fname)
            generation_params["lora_multipliers"].append(1.0)

        # Optional hires fix
        self._maybe_add_hires_config(db_task_params, generation_params)

    def handle_qwen_image_style(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """Handle qwen_image_style task type."""
        self._log_info("Processing qwen_image_style task")
        
        original_prompt = generation_params.get("prompt", db_task_params.get("prompt", ""))
        
        style_strength = float(db_task_params.get("style_reference_strength", 0.0) or 0.0)
        subject_strength = float(db_task_params.get("subject_strength", 0.0) or 0.0)
        scene_strength = float(db_task_params.get("scene_reference_strength", 0.0) or 0.0)
        subject_description = db_task_params.get("subject_description", "")
        in_this_scene = db_task_params.get("in_this_scene", False)

        prompt_parts = []
        has_style_prefix = False

        if style_strength > 0.0:
            prompt_parts.append("In the style of this image,")
            has_style_prefix = True

        if subject_strength > 0.0 and subject_description:
            make_word = "make" if has_style_prefix else "Make"
            if in_this_scene:
                prompt_parts.append(f"{make_word} an image of this {subject_description} in this scene:")
            else:
                prompt_parts.append(f"{make_word} an image of this {subject_description}:")

        if prompt_parts:
            modified_prompt = " ".join(prompt_parts) + " " + original_prompt
            generation_params["prompt"] = modified_prompt
            self._log_info(f"Modified prompt to: {modified_prompt}")

        reference_image = db_task_params.get("style_reference_image") or db_task_params.get("subject_reference_image")
        if reference_image:
            try:
                downloads_dir = Path("outputs/style_refs")
                downloads_dir.mkdir(parents=True, exist_ok=True)
                local_ref_path = sm_download_image_if_url(
                    reference_image, downloads_dir, 
                    task_id_for_logging=self.task_id, 
                    debug_mode=False, 
                    descriptive_name="reference_image"
                )
                generation_params["image_guide"] = str(local_ref_path)
            except Exception as e:
                self._log_warning(f"Failed to download reference image: {e}")

        if "resolution" in db_task_params:
            capped_res = self.cap_qwen_resolution(db_task_params["resolution"])
            if capped_res:
                generation_params["resolution"] = capped_res

        generation_params.setdefault("video_prompt_type", "KI")
        generation_params.setdefault("guidance_scale", 1)
        generation_params.setdefault("num_inference_steps", 12)
        generation_params.setdefault("video_length", 1)

        # Set system prompt based on parameters
        if "system_prompt" in db_task_params and db_task_params["system_prompt"]:
            generation_params["system_prompt"] = db_task_params["system_prompt"]
        else:
            has_subject = subject_strength > 0
            has_style = style_strength > 0
            has_scene = scene_strength > 0

            if has_subject and has_style and has_scene:
                generation_params["system_prompt"] = "You are an expert at creating images with consistent subjects, styles, and scenes."
            elif has_subject and has_style:
                generation_params["system_prompt"] = "You are an expert at creating images with consistent subjects and styles."
            elif has_style:
                generation_params["system_prompt"] = "You are an expert at applying artistic styles consistently."
            else:
                generation_params["system_prompt"] = "You are an expert at image-to-image generation."

        # Download LoRAs
        lightning_fname = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
        if not (self.qwen_lora_dir / lightning_fname).exists():
            self._download_lora_if_missing("lightx2v/Qwen-Image-Lightning", lightning_fname)

        style_fname = "style_transfer_qwen_edit_2_000011250.safetensors"
        if style_strength > 0.0:
            self._download_lora_if_missing("peteromallet/ad_motion_loras", style_fname)

        subject_fname = "in_subject_qwen_edit_2_000006750.safetensors"
        if subject_strength > 0.0:
            self._download_lora_if_missing("peteromallet/mystery_models", subject_fname)

        scene_fname = "in_scene_different_object_000010500.safetensors"
        if scene_strength > 0.0:
            self._download_lora_if_missing("peteromallet/random_junk", scene_fname)

        # Build LoRA lists
        if "lora_names" not in generation_params:
            generation_params["lora_names"] = []
        if "lora_multipliers" not in generation_params:
            generation_params["lora_multipliers"] = []

        if lightning_fname not in generation_params["lora_names"]:
            generation_params["lora_names"].append(lightning_fname)
            generation_params["lora_multipliers"].append(0.85)

        if style_strength > 0.0 and style_fname not in generation_params["lora_names"]:
            generation_params["lora_names"].append(style_fname)
            generation_params["lora_multipliers"].append(style_strength)

        if subject_strength > 0.0 and subject_fname not in generation_params["lora_names"]:
            generation_params["lora_names"].append(subject_fname)
            generation_params["lora_multipliers"].append(subject_strength)

        if scene_strength > 0.0 and scene_fname not in generation_params["lora_names"]:
            generation_params["lora_names"].append(scene_fname)
            generation_params["lora_multipliers"].append(scene_strength)

        # Optional hires fix
        self._maybe_add_hires_config(db_task_params, generation_params)

    def handle_qwen_image_hires(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """
        Handle qwen_image_hires task type - two-pass generation with latent upscaling.
        
        Workflow:
        - Pass 1: Generate at base resolution (default 1328x1328)
        - Latent upscale: Bicubic interpolation in latent space
        - Pass 2: Refine at higher resolution with partial denoising
        
        This replicates the ComfyUI two-pass hires fix workflow.
        """
        self._log_info("Processing qwen_image_hires task (two-pass generation)")
        
        # Default base resolutions matching ComfyUI workflow
        BASE_RESOLUTIONS = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }
        
        # Determine base resolution
        resolution_input = db_task_params.get("resolution", "1328x1328")
        if resolution_input in BASE_RESOLUTIONS:
            # Aspect ratio key provided
            w, h = BASE_RESOLUTIONS[resolution_input]
            resolution_str = f"{w}x{h}"
        else:
            resolution_str = resolution_input
        
        generation_params["resolution"] = resolution_str
        
        # Build hires config
        hires_config = {
            "enabled": True,
            "scale": float(db_task_params.get("hires_scale", 2.0)),
            "hires_steps": int(db_task_params.get("hires_steps", 6)),
            "denoising_strength": float(db_task_params.get("hires_denoise", 0.5)),
            "upscale_method": db_task_params.get("hires_upscale_method", "bicubic"),
        }
        generation_params["hires_config"] = hires_config
        
        # Base generation params
        generation_params.setdefault("video_prompt_type", "KI")
        generation_params.setdefault("guidance_scale", 1)
        generation_params.setdefault("num_inference_steps", int(db_task_params.get("num_inference_steps", 10)))
        generation_params.setdefault("video_length", 1)
        
        # System prompt
        if "system_prompt" in db_task_params and db_task_params["system_prompt"]:
            generation_params["system_prompt"] = db_task_params["system_prompt"]
        else:
            generation_params["system_prompt"] = "You are a professional image generator. Create high-quality, detailed images based on the description."
        
        # Lightning LoRA setup
        lightning_fname = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
        if not (self.qwen_lora_dir / lightning_fname).exists():
            self._download_lora_if_missing("lightx2v/Qwen-Image-Lightning", lightning_fname)
        selected_lightning = lightning_fname
        
        # LoRA strength from task params or default 0.45 (ComfyUI default)
        lora_strength = float(db_task_params.get("lightning_lora_strength", 0.45))
        
        if "lora_names" not in generation_params:
            generation_params["lora_names"] = []
        if "lora_multipliers" not in generation_params:
            generation_params["lora_multipliers"] = []
        
        if selected_lightning not in generation_params["lora_names"]:
            generation_params["lora_names"].append(selected_lightning)
            generation_params["lora_multipliers"].append(lora_strength)
            self._log_info(f"Added Lightning LoRA '{selected_lightning}' @ {lora_strength}")
        
        # Log final config
        base_w, base_h = map(int, resolution_str.split("x"))
        final_w = int(base_w * hires_config["scale"])
        final_h = int(base_h * hires_config["scale"])
        self._log_info(
            f"Hires workflow: {resolution_str} → {final_w}x{final_h}, "
            f"base {generation_params['num_inference_steps']} steps, "
            f"hires {hires_config['hires_steps']} steps @ {hires_config['denoising_strength']} denoise"
        )
