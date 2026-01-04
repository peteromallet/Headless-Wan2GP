
from mmgp import offload
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import sys
from pathlib import Path

import numpy as np
import torch, json, os
import math

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.phase_multiplier_utils import get_phase_loras

from diffusers.image_processor import VaeImageProcessor
from .transformer_qwenimage import QwenImageTransformer2DModel

from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, AutoTokenizer, Qwen2VLProcessor
from .autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers import FlowMatchEulerDiscreteScheduler
from .pipeline_qwenimage import QwenImagePipeline, calculate_shift, retrieve_timesteps
from PIL import Image
from shared.utils.utils import calculate_new_dimensions, convert_tensor_to_image
from shared.utils import files_locator as fl 

_QWEN_FUSED_SPLIT_MAP = {
    "attn.to_qkv": {"mapped_modules": ("attn.to_q", "attn.to_k", "attn.to_v")},
    "attn.add_qkv_proj": {"mapped_modules": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")},
}


from shared.qtypes import nunchaku_int4 as _nunchaku_int4
_split_nunchaku_fused_qkv = _nunchaku_int4.make_nunchaku_splitter(_QWEN_FUSED_SPLIT_MAP)


def stitch_images(img1, img2):
    # Resize img2 to match img1's height
    width1, height1 = img1.size
    width2, height2 = img2.size
    new_width2 = int(width2 * height1 / height2)
    img2_resized = img2.resize((new_width2, height1), Image.Resampling.LANCZOS)
    
    stitched = Image.new('RGB', (width1 + new_width2, height1))
    stitched.paste(img1, (0, 0))
    stitched.paste(img2_resized, (width1, 0))
    return stitched

class model_factory():
    def __init__(
        self,
        checkpoint_dir,
        model_filename = None,
        model_type = None, 
        model_def = None,
        base_model_type = None,
        text_encoder_filename = None,
        quantizeTransformer = False,
        save_quantized = False,
        dtype = torch.bfloat16,
        VAE_dtype = torch.float32,
        mixed_precision_transformer = False,
        VAE_upsampling = None,
    ):
    

        transformer_filename = model_filename[0]
        processor = None
        tokenizer = None
        if base_model_type in ["qwen_image_edit_20B", "qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B", "qwen_image_layered_20B"]:
            processor = Qwen2VLProcessor.from_pretrained(fl.locate_folder("Qwen2.5-VL-7B-Instruct"))
        tokenizer = AutoTokenizer.from_pretrained(fl.locate_folder("Qwen2.5-VL-7B-Instruct"))
        self.base_model_type = base_model_type

        if base_model_type == "qwen_image_layered_20B":
            base_config_file = "models/qwen/configs/qwen_image_layered_20B.json"
        elif base_model_type == "qwen_image_edit_plus2_20B":
            base_config_file = "models/qwen/configs/qwen_image_edit_plus2_20B.json"
        else:
            base_config_file = "models/qwen/configs/qwen_image_20B.json"
        with open(base_config_file, 'r', encoding='utf-8') as f:
            transformer_config = json.load(f)
        transformer_config.pop("_diffusers_version", None)
        transformer_config.pop("_class_name", None)
        transformer_config.pop("pooled_projection_dim", None)
        
        from accelerate import init_empty_weights
        with init_empty_weights():
            transformer = QwenImageTransformer2DModel(**transformer_config)
        source =  model_def.get("source", None)

        if source is not None:
            offload.load_model_data(transformer, source, preprocess_sd=_split_nunchaku_fused_qkv)
        else:
            offload.load_model_data(transformer, transformer_filename, preprocess_sd=_split_nunchaku_fused_qkv)
        # transformer = offload.fast_load_transformers_model("transformer_quanto.safetensors", writable_tensors= True , modelClass=QwenImageTransformer2DModel, defaultConfigPath="transformer_config.json")

        if not source is None:
            from wgp import save_model
            save_model(transformer, model_type, dtype, None)

        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, model_filename[0], dtype, base_config_file)

        text_encoder = offload.fast_load_transformers_model(text_encoder_filename,  writable_tensors= True , modelClass=Qwen2_5_VLForConditionalGeneration,  defaultConfigPath= fl.locate_file(os.path.join("Qwen2.5-VL-7B-Instruct", "config.json")) )
        # text_encoder = offload.fast_load_transformers_model(text_encoder_filename, do_quantize=True,  writable_tensors= True , modelClass=Qwen2_5_VLForConditionalGeneration, defaultConfigPath="text_encoder_config.json", verboseLevel=2)
        # text_encoder.to(torch.float16)
        # offload.save_model(text_encoder, "text_encoder_quanto_fp16.safetensors", do_quantize= True)
        use_Wan_VAE = False
        if base_model_type == "qwen_image_layered_20B":
            VAE_upsampling = None
            VAE_upsampler_factor = 1
        else:
            VAE_upsampler_factor = 2 if VAE_upsampling is not None else 1

        if use_Wan_VAE:
            vae_checkpoint = "Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors" if VAE_upsampler_factor == 2 else "Wan2.1_VAE.safetensors"
            from ..wan.modules.vae import WanVAE
            vae = WanVAE( vae_pth=fl.locate_file(vae_checkpoint), dtype= VAE_dtype, upsampler_factor = VAE_upsampler_factor, device="cpu")
            vae.device = "cuda" #self.device # need to set to cuda so that vae buffers are properly moved (although the rest will stay in the CPU)
        else:
            if base_model_type == "qwen_image_layered_20B":
                convert_state_dict = None
                vae_checkpoint = "qwen_image_layered_vae_bf16.safetensors"
                vae_config_file = "models/qwen/configs/qwen_image_layered_vae_config.json"
                vae_override = model_def.get("vae_URL", None) or model_def.get("vae_URLs", None)
                if isinstance(vae_override, list):
                    vae_override = vae_override[0] if len(vae_override) > 0 else None
                if isinstance(vae_override, dict):
                    vae_override = vae_override.get("URLs", None)
                if vae_override:
                    vae_checkpoint = vae_override
            elif VAE_upsampler_factor ==2 :
                from .convert_diffusers_qwen_vae import convert_state_dict
                vae_checkpoint = "Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors"
                vae_config_file = "qwen_vae_config.json"
            else:
                convert_state_dict = None
                vae_checkpoint = "qwen_vae.safetensors"
                vae_config_file = "qwen_vae_config.json"
            vae = offload.fast_load_transformers_model( fl.locate_file(vae_checkpoint), writable_tensors= True , modelClass=AutoencoderKLQwenImage, defaultConfigPath= fl.locate_file(vae_config_file), configKwargs={"upsampler_factor": VAE_upsampler_factor}, preprocess_sd=convert_state_dict)
        vae.upsampling_set = VAE_upsampling
        self.pipeline = QwenImagePipeline(vae, text_encoder, tokenizer, transformer, processor)
        self.pipeline.use_Wan_VAE = use_Wan_VAE
        self.vae=vae
        self.text_encoder=text_encoder
        self.tokenizer=tokenizer
        self.transformer=transformer
        self.processor = processor

    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "replace the logo with the text 'Black Forest Labs'",
        n_prompt = None,
        sampling_steps: int = 20,
        input_ref_images = None,
        input_frames= None,
        input_masks= None,
        width= 832,
        height=480,
        guide_scale: float = 4,
        fit_into_canvas = None,
        callback = None,
        loras_slists = None,
        batch_size = 1,
        video_prompt_type = "",
        VAE_tile_size = None, 
        joint_pass = True,
        sample_solver='default',
        denoising_strength = 1.,
        masking_strength = 1.,
        model_mode = 0,
        outpainting_dims = None,
        **bbargs
    ):
        # Generate with different aspect ratios
        aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472)
        }
        

        if sample_solver =='lightning':
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),  # We use shift=3 in distillation
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),  # We use shift=3 in distillation
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,  # set shift_terminal to None
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
        else:
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": 0.5,
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": 0.9,
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": 0.02,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False
            }

        self.scheduler=FlowMatchEulerDiscreteScheduler(**scheduler_config)
        self.pipeline.scheduler = self.scheduler 
        if VAE_tile_size is not None:
            if isinstance(VAE_tile_size, int):
                tiling_type = VAE_tile_size
                VAE_tile_size = [False, 0] if tiling_type == 0 else [True, 256]  
            self.vae.use_tiling  = VAE_tile_size[0] 
            self.vae.tile_latent_min_height  = VAE_tile_size[1] 
            self.vae.tile_latent_min_width  = VAE_tile_size[1]
            tile_size  = VAE_tile_size[1]
        # tile_size = 256
        qwen_edit_plus = self.base_model_type in ["qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B"]
        qwen_layered = self.base_model_type in ["qwen_image_layered_20B"]
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()
        # width, height = aspect_ratios["16:9"]

        if n_prompt is None or len(n_prompt) == 0:
            if qwen_layered:
                n_prompt = " "
            else:
                n_prompt=  "text, watermark, copyright, blurry, low resolution"

        image_mask = None if input_masks is None else convert_tensor_to_image(input_masks, mask_levels= True)
        if input_frames is not None:
            input_ref_images = [convert_tensor_to_image(input_frames) ] +  ([] if input_ref_images  is None else input_ref_images )

        if input_ref_images is not None:
            # image stiching method
            if qwen_layered:
                input_ref_images = [input_ref_images[0]]
            else:
                stiched = input_ref_images[0]
                if "K" in video_prompt_type :
                    w, h = input_ref_images[0].size
                    height, width = calculate_new_dimensions(height, width, h, w, fit_into_canvas)

                if not qwen_edit_plus:
                    for new_img in input_ref_images[1:]:
                        stiched = stitch_images(stiched, new_img)
                    input_ref_images  = [stiched]

        num_images_per_prompt = 1 if qwen_layered else batch_size
        layers = batch_size if qwen_layered else 1

        # Extract hires config and system prompt from bbargs
        hires_config = bbargs.get("hires_config", None)
        system_prompt = bbargs.get("system_prompt", None)

        # ========== TWO-PASS HIRES FIX ==========
        if hires_config and hires_config.get("enabled"):
            hires_scale = hires_config.get("scale", 2.0)
            hires_steps = hires_config.get("hires_steps", 6)
            hires_denoise = hires_config.get("denoising_strength", 0.5)
            hires_upscale_method = hires_config.get("upscale_method", "bicubic")
            hires_seed_offset = int(hires_config.get("seed_offset", 1))

            print(f"🔍 Hires fix enabled: {width}x{height} → {int(width*hires_scale)}x{int(height*hires_scale)}")
            print(f"   Pass 1: {sampling_steps} steps | Pass 2: {hires_steps} steps @ {hires_denoise} denoise")

            # === PASS 1: Generate at base resolution, output latents ===
            latents = self.pipeline(
                prompt=input_prompt,
                negative_prompt=n_prompt,
                image=input_ref_images,
                image_mask=image_mask,
                width=width,
                height=height,
                num_inference_steps=sampling_steps,
                num_images_per_prompt=num_images_per_prompt,
                layers=layers,
                cfg_normalize=True,
                true_cfg_scale=guide_scale,
                callback=callback,
                pipeline=self,
                loras_slists=loras_slists,
                joint_pass=joint_pass,
                denoising_strength=denoising_strength,
                masking_strength=masking_strength,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                lora_inpaint=image_mask is not None and model_mode == 1,
                outpainting_dims=outpainting_dims,
                qwen_edit_plus=qwen_edit_plus,
                VAE_tile_size=tile_size,
                output_type="latent",  # Return latents, don't VAE decode yet
            )
            if latents is None:
                return None

            print(f"✅ Pass 1 complete: latents shape {latents.shape}")

            # === DEBUG: Save Pass 1 output before upscaling ===
            if os.getenv("DEBUG_HIRES_SAVE_PASS1", "0") == "1":
                # convert_tensor_to_image already imported at module level
                from shared.config import model_config
                print("📸 Saving Pass 1 output for debugging...")
                # Decode latents to image
                latents_decoded = self.pipeline._unpack_latents(latents.clone(), height, width, self.pipeline.vae_scale_factor)
                latents_decoded = latents_decoded.to(self.pipeline.vae.dtype)
                latents_mean = (
                    torch.tensor(self.pipeline.vae.config.latents_mean)
                    .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
                    .to(latents_decoded.device, latents_decoded.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(
                    1, self.pipeline.vae.config.z_dim, 1, 1, 1
                ).to(latents_decoded.device, latents_decoded.dtype)
                latents_decoded = latents_decoded / latents_std + latents_mean
                pass1_image = self.pipeline.vae.decode(latents_decoded, return_dict=False)[0][:, :, 0]

                # Save to file using the same output directory as final images
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = model_config.get("image_save_path", "outputs/")
                debug_path = os.path.join(output_dir, f"debug_pass1_{timestamp}.png")
                os.makedirs(output_dir, exist_ok=True)
                convert_tensor_to_image(pass1_image[0]).save(debug_path)
                print(f"   Saved Pass 1 output to: {debug_path}")

            # === UPSCALE LATENTS ===
            latents, new_height, new_width = QwenImagePipeline._upscale_latents(
                latents, height, width,
                self.pipeline.vae_scale_factor,
                hires_scale,
                hires_upscale_method
            )
            print(f"📐 Latents upscaled: {width}x{height} → {new_width}x{new_height}")

            # === DEBUG: Save upscaled latents (before adding noise) ===
            if os.getenv("DEBUG_HIRES_SAVE_PASS1", "0") == "1":
                # convert_tensor_to_image already imported at module level
                from shared.config import model_config
                print("📸 Saving upscaled latents (pre-noise) for debugging...")
                # Decode upscaled latents to image
                latents_upscaled_decoded = self.pipeline._unpack_latents(latents.clone(), new_height, new_width, self.pipeline.vae_scale_factor)
                latents_upscaled_decoded = latents_upscaled_decoded.to(self.pipeline.vae.dtype)
                latents_mean = (
                    torch.tensor(self.pipeline.vae.config.latents_mean)
                    .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
                    .to(latents_upscaled_decoded.device, latents_upscaled_decoded.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(
                    1, self.pipeline.vae.config.z_dim, 1, 1, 1
                ).to(latents_upscaled_decoded.device, latents_upscaled_decoded.dtype)
                latents_upscaled_decoded = latents_upscaled_decoded / latents_std + latents_mean
                upscaled_image = self.pipeline.vae.decode(latents_upscaled_decoded, return_dict=False)[0][:, :, 0]

                # Save to file using the same output directory as final images
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = model_config.get("image_save_path", "outputs/")
                debug_path = os.path.join(output_dir, f"debug_upscaled_prenoise_{timestamp}.png")
                os.makedirs(output_dir, exist_ok=True)
                convert_tensor_to_image(upscaled_image[0]).save(debug_path)
                print(f"   Saved upscaled latents (pre-noise) to: {debug_path}")

            # === PASS 2: Refine at higher resolution ===
            # Reset scheduler for second pass
            self.scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)
            self.pipeline.scheduler = self.scheduler

            # Build sigma/timestep schedule so pass 2 starts at the correct noise level.
            # NOTE: In this codebase, timestep is on a 0..1000 scale, so sigma ≈ timestep/1000.
            device = "cuda"

            # FIX: Create a full schedule with enough steps to support the denoising strength
            # e.g., if hires_steps=1 and denoise=0.5, we need at least 2 total steps to start halfway
            total_steps_needed = max(hires_steps, math.ceil(hires_steps / max(hires_denoise, 0.01)))
            sigmas_full = np.linspace(1.0, 1 / total_steps_needed, total_steps_needed).tolist()

            hires_image_seq_len = int(latents.shape[1])
            mu = calculate_shift(
                hires_image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )

            timesteps_full, _ = retrieve_timesteps(
                self.scheduler,
                num_inference_steps=None,
                device=device,
                sigmas=sigmas_full,
                mu=mu,
            )

            # Calculate where to start based on denoising strength
            start_idx = int(len(timesteps_full) * (1.0 - float(hires_denoise)))
            start_idx = max(0, min(start_idx, len(timesteps_full) - 1))

            # Ensure we get exactly hires_steps by trimming from the start
            # If total_steps_needed > hires_steps, take the last hires_steps
            if len(sigmas_full) - start_idx > hires_steps:
                start_idx = len(sigmas_full) - hires_steps

            sigma0 = float(timesteps_full[start_idx].item()) / 1000.0
            sigmas_trimmed = sigmas_full[start_idx:]

            # === DEBUG: Log sigma schedule details ===
            if os.getenv("DEBUG_HIRES_SAVE_PASS1", "0") == "1":
                print(f"🔍 SIGMA SCHEDULE DEBUG:")
                print(f"   total_steps_needed: {total_steps_needed}")
                print(f"   hires_steps requested: {hires_steps}")
                print(f"   hires_denoise: {hires_denoise}")
                print(f"   sigmas_full: {sigmas_full}")
                print(f"   timesteps_full: {[f'{t.item():.2f}' for t in timesteps_full]}")
                print(f"   start_idx: {start_idx} / {len(timesteps_full)}")
                print(f"   sigma0: {sigma0:.4f}")
                print(f"   sigmas_trimmed: {sigmas_trimmed}")
                print(f"   actual steps in pass 2: {len(sigmas_trimmed)}")
                print(f"   noise mixing: latent * {1.0-sigma0:.4f} + noise * {sigma0:.4f}")

            # Add noise to the (clean) upscaled latents so the scheduler starts at sigma0.
            noise_gen = torch.Generator(device="cuda").manual_seed(int(seed) + hires_seed_offset)
            noise = torch.randn(latents.shape, generator=noise_gen, device=latents.device, dtype=latents.dtype)
            latents = latents * (1.0 - sigma0) + noise * sigma0
            print(f"🌀 Pass 2 init noise: start_idx={start_idx}/{len(timesteps_full)} sigma0={sigma0:.4f} steps={len(sigmas_trimmed)}")

            # === Apply Phase-Based LoRA Multipliers for Pass 2 ===
            # Save original activated_loras and loras_multipliers from pass 1
            original_activated_loras = bbargs.get("activated_loras", []).copy() if isinstance(bbargs.get("activated_loras", []), list) else []
            original_loras_multipliers = bbargs.get("loras_multipliers", "")

            # Get phase-based multipliers from hires_config (set by qwen_handler.py)
            phase_multipliers = hires_config.get("phase_lora_multipliers", [])

            if phase_multipliers:
                # Use stored phase multipliers to filter for pass 2
                try:
                    pass2_loras, pass2_multipliers = get_phase_loras(
                        lora_names=original_activated_loras,
                        multipliers=phase_multipliers,
                        phase_index=1,  # Pass 2 (0-indexed)
                        num_phases=2
                    )

                    bbargs["activated_loras"] = pass2_loras
                    bbargs["loras_multipliers"] = " ".join(pass2_multipliers)  # Space-separated for WGP
                    print(f"🔧 Pass 2: Using {len(pass2_loras)}/{len(original_activated_loras)} LoRAs (phase-based filtering)")

                except Exception as e:
                    # Fallback: keep all LoRAs if parsing fails
                    print(f"⚠️ Phase multiplier parsing failed: {e}")
                    print(f"⚠️ Falling back to using all LoRAs in pass 2")
                    bbargs["activated_loras"] = original_activated_loras
                    bbargs["loras_multipliers"] = original_loras_multipliers
            else:
                # No phase multipliers provided, keep all LoRAs for pass 2
                print(f"🔧 Pass 2: No phase multipliers found, using all {len(original_activated_loras)} LoRAs")

            image = self.pipeline(
                prompt=input_prompt,
                negative_prompt=n_prompt,
                image=None,  # No input image for pass 2 - we have latents
                image_mask=None,
                width=new_width,
                height=new_height,
                num_inference_steps=hires_steps,
                sigmas=sigmas_trimmed,
                num_images_per_prompt=num_images_per_prompt,
                layers=layers,
                cfg_normalize=True,
                true_cfg_scale=guide_scale,
                callback=callback,
                pipeline=self,
                loras_slists=loras_slists,
                joint_pass=joint_pass,
                denoising_strength=hires_denoise,  # (kept for compatibility; sigma schedule controls actual start)
                masking_strength=masking_strength,
                latents=latents,  # Pre-computed upscaled latents
                generator=torch.Generator(device="cuda").manual_seed(int(seed) + hires_seed_offset),
                lora_inpaint=False,
                outpainting_dims=None,
                qwen_edit_plus=qwen_edit_plus,
                VAE_tile_size=tile_size,
                output_type="pil",  # Now VAE decode to final image
            )
            if image is None:
                return None

            # === Restore original LoRAs ===
            bbargs["activated_loras"] = original_activated_loras
            bbargs["loras_multipliers"] = original_loras_multipliers

            print(f"✅ Hires fix complete: {new_width}x{new_height}")
            return image.transpose(0, 1)

        # ========== STANDARD SINGLE-PASS (existing behavior) ==========
        image = self.pipeline(
            prompt=input_prompt,
            negative_prompt=n_prompt,
            image = input_ref_images,
            image_mask = image_mask,
            width=width,
            height=height,
            num_inference_steps=sampling_steps,
            num_images_per_prompt = num_images_per_prompt,
            layers = layers,
            cfg_normalize = True,
            true_cfg_scale=guide_scale,
            callback = callback,
            pipeline=self,
            loras_slists=loras_slists,
            joint_pass = joint_pass,
            denoising_strength=denoising_strength,
            masking_strength=masking_strength,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            lora_inpaint = image_mask is not None and model_mode == 1,
            outpainting_dims = outpainting_dims,
            qwen_edit_plus = qwen_edit_plus,
            VAE_tile_size = tile_size,
        )      
        if image is None: return None
        return image.transpose(0, 1)

    def get_loras_transformer(self, get_model_recursive_prop, model_type, model_mode, image_mode, **kwargs):
        if image_mode !=2 or model_mode != 1: return [], []
        preloadURLs = get_model_recursive_prop(model_type,  "preload_URLs")
        if len(preloadURLs) == 0: return [], []
        return [ fl.locate_file(os.path.basename(preloadURLs[0]))] , [1]


