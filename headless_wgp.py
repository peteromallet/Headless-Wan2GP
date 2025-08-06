"""
WanGP Content Generation Orchestrator

A thin wrapper around wgp.py's generate_video() function for programmatic use.
Supports T2V, VACE, and Flux generation without running the Gradio UI.
"""

import os
import sys
from typing import Optional, List, Union


class WanOrchestrator:
    """Thin adapter around `wgp.generate_video` for easier programmatic use."""

    def __init__(self, wan_root: str):
        """Initialize orchestrator with WanGP directory.
        
        Args:
            wan_root: Path to WanGP repository root directory
        """
        self.wan_root = os.path.abspath(wan_root)
        
        # Add to Python path and change directory (wgp.py expects relative paths)
        sys.path.insert(0, self.wan_root)
        os.chdir(self.wan_root)
        
        # Import WGP components after path setup
        try:
            from wgp import (
                generate_video, get_base_model_type, get_model_family,
                test_vace_module, apply_changes
            )
            # Apply VACE fix wrapper to generate_video
            self._generate_video = self._create_vace_fixed_generate_video(generate_video)
            self._get_base_model_type = get_base_model_type
            self._get_model_family = get_model_family
            self._test_vace_module = test_vace_module
            self._apply_changes = apply_changes
            
            # Initialize WGP global state (normally done by UI)
            import wgp
            for attr, default in {
                'wan_model': None, 'offloadobj': None, 'reload_needed': True,
                'transformer_type': None, 'server_config': {}
            }.items():
                if not hasattr(wgp, attr):
                    setattr(wgp, attr, default)
                    
        except ImportError as e:
            raise ImportError(f"Failed to import wgp module. Ensure {wan_root} contains wgp.py: {e}")

        # Initialize state object (mimics UI state)
        self.state = {
            "generation": {},
            "model_type": None,
            "model_filename": "",
            "advanced": False,
            "last_model_per_family": {},
            "last_resolution_per_group": {},
            "gen": {
                "queue": [],
                "file_list": [],
                "file_settings_list": [],
                "selected": 0,
                "prompt_no": 1,
                "prompts_max": 1
            },
            "loras": [],
            "loras_names": [],
            "loras_presets": {},
            # Additional state properties to prevent future KeyErrors
            "validate_success": 1,      # Required for validation checks
            "apply_success": 1,         # Required for settings application  
            "refresh": None,            # Required for UI refresh operations
            "all_settings": {},        # Required for settings persistence
            "image_mode_tab": 0,       # Required for image/video mode switching
            "prompt": ""               # Required for prompt handling
        }
        
        # Apply sensible defaults (mirrors typical UI defaults)
        self._apply_changes(
            self.state,
            transformer_types_choices=["t2v"],   # default to T2V model family
            transformer_dtype_policy_choice="auto",
            text_encoder_quantization_choice="bf16",
            VAE_precision_choice="fp32",
            mixed_precision_choice=0,
            save_path_choice="outputs/",
            attention_choice="auto",
            compile_choice=0,
            profile_choice=4,  # Profile 4: LowRAM_LowVRAM (Default)
            vae_config_choice="default",
            metadata_choice="none",
            quantization_choice="int8"
        )
        self.current_model = None
        self.offloadobj = None  # Store WGP's offload object
        print(f"✅ WanOrchestrator initialized with WGP at {wan_root}")

    def load_model(self, model_key: str):
        """Load and validate a model type using WGP's load_models function.
        
        Args:
            model_key: Model identifier (e.g., "t2v", "vace_14B", "flux")
            
        Raises:
            ValueError: If model_key is not recognized
        """
        if self._get_base_model_type(model_key) is None:
            raise ValueError(f"Unknown model: {model_key}")
        
        # Import WGP to call load_models
        import wgp
        import os
        import shutil
        
        # VACE models require a config.json next to the module file for mmgp
        if self._test_vace_module(model_key):
            print(f"🔧 [CONFIG_DETECTION] VACE model '{model_key}' detected - setting up config.json for mmgp")
            
            # Get absolute paths to handle different working directories
            wgp_dir = os.path.dirname(os.path.abspath(wgp.__file__))
            print(f"🔧 [CONFIG_PATHS] WGP directory: {wgp_dir}")
            
            # Create config.json directly in ckpts/ for VACE modules  
            # Note: We don't call get_model_filename() here because that only returns
            # the base model file. load_models() handles building the complete file list.
            ckpts_dir = os.path.join(wgp_dir, "ckpts")
            config_target = os.path.join(ckpts_dir, "config.json")
            
            if not os.path.exists(config_target):
                print(f"🔧 [CONFIG_CREATE] Creating config.json for VACE modules at: {config_target}")
                
                # Embedded VACE config to avoid dependency on external files
                vace_config = {
                    "_class_name": "VaceWanModel",
                    "_diffusers_version": "0.30.0",
                    "dim": 5120,
                    "eps": 1e-06,
                    "ffn_dim": 13824,
                    "freq_dim": 256,
                    "in_dim": 16,
                    "model_type": "t2v",
                    "num_heads": 40,
                    "num_layers": 40,
                    "out_dim": 16,
                    "text_len": 512,
                    "vace_layers": [0, 5, 10, 15, 20, 25, 30, 35],
                    "vace_in_dim": 96
                }
                
                import json
                os.makedirs(ckpts_dir, exist_ok=True)
                with open(config_target, 'w') as f:
                    json.dump(vace_config, f, indent=2)
                print(f"🔧 [CONFIG_SUCCESS] ✅ Created {config_target}")
            else:
                print(f"🔧 [CONFIG_EXISTS] Config already exists: {config_target}")
                
            # Show where we expect mmgp to look for config.json
            print(f"🔧 [MMGP_PATHS] Our config: {config_target}")
            print(f"🔧 [MMGP_PATHS] VACE module: ckpts/wan2.1_Vace_14B_module_quanto_mbf16_int8.safetensors")
            print(f"🔧 [MMGP_PATHS] mmgp might expect config.json next to the VACE module file")
            
            # Create config.json next to the VACE module as well
            vace_module_dir = ckpts_dir  # VACE module is in ckpts/ too
            vace_module_config = os.path.join(vace_module_dir, "wan2.1_Vace_14B_module_quanto_mbf16_int8.json") 
            if not os.path.exists(vace_module_config):
                vace_config = {
                    "_class_name": "VaceWanModel",
                    "_diffusers_version": "0.30.0",
                    "dim": 5120,
                    "eps": 1e-06,
                    "ffn_dim": 13824,
                    "freq_dim": 256,
                    "in_dim": 16,
                    "model_type": "t2v",
                    "num_heads": 40,
                    "num_layers": 40,
                    "out_dim": 16,
                    "text_len": 512,
                    "vace_layers": [0, 5, 10, 15, 20, 25, 30, 35],
                    "vace_in_dim": 96
                }
                
                with open(vace_module_config, 'w') as f:
                    json.dump(vace_config, f, indent=2)
                print(f"🔧 [CONFIG_ALT] ✅ Also created config next to VACE module: {vace_module_config}")
            else:
                print(f"🔧 [CONFIG_ALT] Config next to VACE module already exists: {vace_module_config}")
        
        # Debug: Show that module discovery is working correctly
        modules = wgp.get_model_recursive_prop(model_key, "modules", return_list=True)
        print(f"🔧 [MODULE_DISCOVERY] Modules for '{model_key}': {modules}")
        
        model_def = wgp.get_model_def(model_key)
        print(f"🔧 [MODEL_DEF] Architecture: {model_def.get('architecture')} | Modules: {model_def.get('modules')}")
        
        # Actually load the model using WGP's proper loading flow
        # This handles VACE modules, LoRA discovery, etc.
        loras, loras_names, _, _, _, _, _ = wgp.setup_loras(model_key, None, wgp.get_lora_dir(model_key), '', None)
        print(f"🎨 Discovered {len(loras)} LoRAs for {model_key}: {loras_names}")
        wan_model, self.offloadobj = wgp.load_models(model_key)
        
        self.current_model = model_key
        self.state["model_type"] = model_key
        
        # Store the loaded model globally for WGP to use
        wgp.wan_model = wan_model
        wgp.offloadobj = self.offloadobj
        
        family = self._get_model_family(model_key, for_ui=True)
        print(f"📋 Loaded model: {model_key} ({family})")
    
    def _setup_loras_for_model(self, model_type: str):
        """Initialize LoRA discovery for a model type.
        
        This matches WGP's exact setup_loras call pattern from generate_video_tab.
        Scans the LoRA directory and populates state with available LoRAs.
        The actual loading/activation happens during generation.
        """
        try:
            # Import WGP functions
            import wgp
            setup_loras = wgp.setup_loras
            get_lora_dir = wgp.get_lora_dir
            
            # Use exact same call pattern as WGP's generate_video_tab (line 6941)
            # setup_loras(model_type, transformer, lora_dir, lora_preselected_preset, split_linear_modules_map)
            preset_to_load = ""  # No preset in headless mode (equivalent to lora_preselected_preset)
            
            loras, loras_names, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset = setup_loras(
                model_type,           # Same as WGP
                None,                 # transformer=None for discovery phase (same as WGP)
                get_lora_dir(model_type),  # lora_dir (same as WGP)
                preset_to_load,       # lora_preselected_preset="" (same as WGP)
                None                  # split_linear_modules_map=None (same as WGP)
            )
            
            # Update state with discovered LoRAs - exact same pattern as WGP (lines 6943-6945)
            self.state["loras"] = loras
            self.state["loras_presets"] = loras_presets
            self.state["loras_names"] = loras_names
            
            if loras:
                print(f"🎨 Discovered {len(loras)} LoRAs for {model_type}: {[os.path.basename(l) for l in loras[:3]]}{'...' if len(loras) > 3 else ''}")
            else:
                print(f"🎨 No LoRAs found for {model_type}")
                
        except Exception as e:
            print(f"⚠️  LoRA discovery failed for {model_type}: {e}")
            # Keep empty defaults to prevent crashes
            self.state["loras"] = []
            self.state["loras_names"] = []
            self.state["loras_presets"] = {}

    def _create_vace_fixed_generate_video(self, original_generate_video):
        """Create a wrapper around generate_video for VACE models.
        
        VACE models are properly loaded via load_models() now, so no special handling needed.
        """
        def vace_fixed_generate_video(*args, **kwargs):
            # VACE modules are now properly loaded via load_models() - no patching needed
            return original_generate_video(*args, **kwargs)
        
        return vace_fixed_generate_video

    def _is_vace(self) -> bool:
        """Check if current model is a VACE model."""
        return self._test_vace_module(self.current_model)

    def _is_flux(self) -> bool:
        """Check if current model is a Flux model."""
        return self._get_base_model_type(self.current_model) == "flux"

    def _is_t2v(self) -> bool:
        """Check if current model is a T2V model."""
        base_type = self._get_base_model_type(self.current_model)
        return base_type in ["t2v", "t2v_1.3B", "hunyuan", "ltxv_13B"]

    def generate(self, 
                prompt: str,
                # Common parameters
                resolution: str = "1280x720",
                video_length: int = 49,
                num_inference_steps: int = 25,
                guidance_scale: float = 7.5,
                seed: int = 42,
                # VACE parameters
                video_guide: Optional[str] = None,
                video_mask: Optional[str] = None,
                video_guide2: Optional[str] = None,  # NEW: Secondary guide for dual encoding
                video_mask2: Optional[str] = None,   # NEW: Secondary mask for dual encoding
                video_prompt_type: str = "VP",
                control_net_weight: float = 1.0,
                control_net_weight2: float = 1.0,
                # Flux parameters
                embedded_guidance_scale: float = 3.0,
                # LoRA parameters
                lora_names: Optional[List[str]] = None,
                lora_multipliers: Optional[List[float]] = None,
                # Other parameters
                negative_prompt: str = "",
                batch_size: int = 1,
                **kwargs) -> str:
        """Generate content using the loaded model.
        
        Args:
            prompt: Text prompt for generation
            resolution: Output resolution (e.g., "1280x720", "1024x1024")
            video_length: Number of frames for video or images for Flux
            num_inference_steps: Denoising steps
            guidance_scale: CFG guidance strength
            seed: Random seed for reproducibility
            video_guide: Path to control video (required for VACE)
            video_mask: Path to mask video (optional)
            video_guide2: Path to secondary control video for dual encoding (NEW)
            video_mask2: Path to secondary mask video for dual encoding (NEW)
            video_prompt_type: VACE encoding type (e.g., "VP", "VPD", "VPDA")
            control_net_weight: Strength for first VACE encoding
            control_net_weight2: Strength for second VACE encoding
            embedded_guidance_scale: Flux-specific guidance
            lora_names: List of LoRA filenames
            lora_multipliers: List of LoRA strength multipliers
            negative_prompt: Negative prompt text
            batch_size: Batch size for generation
            **kwargs: Additional parameters passed to generate_video
            
        Returns:
            Path to generated output file(s)
            
        Raises:
            RuntimeError: If no model is loaded
            ValueError: If required parameters are missing for model type
        """
        if not self.current_model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Test VACE module detection with detailed logging
        print(f"[VACE_MODULE_DEBUG] === VACE Module Detection Test ===")
        print(f"[VACE_MODULE_DEBUG] current_model: '{self.current_model}'")
        
        # Test the actual function calls we use
        base_model_type = self._get_base_model_type(self.current_model)
        print(f"[VACE_MODULE_DEBUG] _get_base_model_type('{self.current_model}') → '{base_model_type}'")
        
        vace_test_result = self._test_vace_module(self.current_model)
        print(f"[VACE_MODULE_DEBUG] _test_vace_module('{self.current_model}') → {vace_test_result}")
        
        is_vace = self._is_vace()
        is_flux = self._is_flux()
        is_t2v = self._is_t2v()
        
        print(f"[VACE_MODULE_DEBUG] Final detection results:")
        print(f"[VACE_MODULE_DEBUG]   is_vace: {is_vace} {'✅' if is_vace else '❌'}")
        print(f"[VACE_MODULE_DEBUG]   is_flux: {is_flux}")
        print(f"[VACE_MODULE_DEBUG]   is_t2v: {is_t2v}")
        
        # Log model type detection and VACE parameters  
        print(f"[HEADLESS_WGP_DEBUG] === Generate Call Analysis ===")
        print(f"[HEADLESS_WGP_DEBUG] current_model: {self.current_model}")
        print(f"[HEADLESS_WGP_DEBUG] is_vace: {is_vace}, is_flux: {is_flux}, is_t2v: {is_t2v}")
        print(f"[HEADLESS_WGP_DEBUG] prompt: '{prompt}'")
        print(f"[HEADLESS_WGP_DEBUG] resolution: {resolution}, video_length: {video_length}")
        print(f"[HEADLESS_WGP_DEBUG] === VACE Input Parameters ===")
        print(f"[HEADLESS_WGP_DEBUG] video_guide: {video_guide}")
        print(f"[HEADLESS_WGP_DEBUG] video_mask: {video_mask}")
        print(f"[HEADLESS_WGP_DEBUG] video_guide2: {video_guide2}")
        print(f"[HEADLESS_WGP_DEBUG] video_mask2: {video_mask2}")
        print(f"[HEADLESS_WGP_DEBUG] video_prompt_type: '{video_prompt_type}'")
        print(f"[HEADLESS_WGP_DEBUG] control_net_weight: {control_net_weight}")
        print(f"[HEADLESS_WGP_DEBUG] control_net_weight2: {control_net_weight2}")

        # Validate model-specific requirements
        if is_vace and not video_guide:
            raise ValueError("VACE models require video_guide parameter")

        # Configure model-specific parameters
        if is_flux:
            image_mode = 1
            # For Flux, video_length means number of images
            actual_video_length = 1
            actual_batch_size = video_length
            # Use embedded guidance for Flux
            actual_guidance = embedded_guidance_scale
        else:
            image_mode = 0
            actual_video_length = video_length
            actual_batch_size = batch_size
            actual_guidance = guidance_scale

        # Set up VACE parameters
        if not is_vace:
            video_guide = None
            video_mask = None
            video_guide2 = None
            video_mask2 = None
            video_prompt_type = "disabled"
            control_net_weight = 0.0
            control_net_weight2 = 0.0

        # Prepare LoRA parameters  
        activated_loras = lora_names if lora_names else []
        # WGP expects loras_multipliers as string, not list
        if lora_multipliers:
            # Convert list of floats to space-separated string
            loras_multipliers_str = " ".join(str(m) for m in lora_multipliers)
        else:
            loras_multipliers_str = ""

        # Define missing variables with defaults
        flow_shift = 7.0
        sample_solver = "euler"
        image_prompt_type = "disabled"
        image_start = None
        image_end = None
        model_mode = "generate"
        video_source = ""
        image_refs = []
        denoising_strength = 1.0

        # Create minimal task and callback objects
        task = {"id": 1, "params": {}, "repeats": 1}
        
        def send_cmd(cmd: str, data=None):
            if cmd == "status":
                print(f"📊 Status: {data}")
            elif cmd == "progress":
                if isinstance(data, list) and len(data) >= 2:
                    progress, status = data[0], data[1]
                    print(f"⏳ Progress: {progress}% - {status}")
                else:
                    print(f"⏳ Progress: {data}")
            elif cmd == "output":
                print("📤 Output generated")
            elif cmd == "exit":
                print("🏁 Generation completed")
            elif cmd == "error":
                print(f"❌ Error: {data}")
            elif cmd == "info":
                print(f"ℹ️  Info: {data}")
            elif cmd == "preview":
                print("🖼️  Preview updated")

        # Generate content type description
        content_type = "images" if is_flux else "video"
        model_type_desc = "Flux" if is_flux else ("VACE" if is_vace else "T2V")
        count_desc = f"{video_length} {'images' if is_flux else 'frames'}"
        
        print(f"🎬 Generating {model_type_desc} {content_type}: {resolution}, {count_desc}")
        if is_vace:
            encodings = [c for c in video_prompt_type if c in "PDSLCMUA"]
            print(f"🎯 VACE encodings: {encodings}")
            if video_guide2:
                print(f"🎯 Using secondary guide: {video_guide2}")
        if activated_loras:
            print(f"🎨 LoRAs: {activated_loras}")

        try:
            print(f"[HEADLESS_WGP_DEBUG] === Calling WGP.generate_video with VACE fix ===")
            print(f"[HEADLESS_WGP_DEBUG] Applying VACE load_models override for proper module loading")
            
            # Log critical VACE parameters being passed to WGP
            if is_vace:
                print(f"[WGP_CALL_DEBUG] === VACE Parameters to WGP ===")
                print(f"[WGP_CALL_DEBUG] model_type: '{self.current_model}'")
                print(f"[WGP_CALL_DEBUG] video_guide: {video_guide}")
                print(f"[WGP_CALL_DEBUG] video_mask: {video_mask}")
                print(f"[WGP_CALL_DEBUG] video_guide2: {video_guide2}")
                print(f"[WGP_CALL_DEBUG] video_mask2: {video_mask2}")
                print(f"[WGP_CALL_DEBUG] video_prompt_type: '{video_prompt_type}'")
                print(f"[WGP_CALL_DEBUG] control_net_weight: {control_net_weight}")
                print(f"[WGP_CALL_DEBUG] control_net_weight2: {control_net_weight2}")
                print(f"[WGP_CALL_DEBUG] About to call _generate_video with VACE patching...")
            else:
                print(f"[WGP_CALL_DEBUG] === Non-VACE Parameters to WGP ===")
                print(f"[WGP_CALL_DEBUG] model_type: '{self.current_model}'")
                print(f"[WGP_CALL_DEBUG] About to call _generate_video directly (no patching)...")
            
            # Call the VACE-fixed generate_video (patching is handled in wrapper)
            result = self._generate_video(
                        task=task,
                        send_cmd=send_cmd,
                        state=self.state,
                        
                        # Core parameters
                        model_type=self.current_model,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        resolution=resolution,
                        video_length=actual_video_length,
                        batch_size=actual_batch_size,
                        seed=seed,
                        force_fps="24",  # Must be string, not int
                        num_inference_steps=num_inference_steps,
                        guidance_scale=actual_guidance,
                        guidance2_scale=actual_guidance,
                        switch_threshold=0.5,
                        embedded_guidance_scale=embedded_guidance_scale if is_flux else 0.0,
                        image_mode=image_mode,
                        
                        # VACE control parameters
                        video_guide=video_guide,
                        video_mask=video_mask,
                        video_guide2=video_guide2,  # NEW: Secondary guide
                        video_mask2=video_mask2,    # NEW: Secondary mask
                        video_prompt_type=video_prompt_type,
                        control_net_weight=control_net_weight,
                        control_net_weight2=control_net_weight2,
                        denoising_strength=1.0,
                        
                        # LoRA parameters
                        activated_loras=activated_loras,
                        loras_multipliers=loras_multipliers_str,
                        
                        # Standard defaults for other parameters
                        audio_guidance_scale=1.0,
                        flow_shift=7.0,
                        sample_solver="euler",
                        repeat_generation=1,
                        multi_prompts_gen_type=0,  # 0: new video, 1: sliding window
                        multi_images_gen_type=0,  # 0: every combination, 1: match prompts
                        skip_steps_cache_type="",  # Empty string disables caching
                        skip_steps_multiplier=1.0,
                        skip_steps_start_step_perc=0.0,
                        
                        # Image parameters
                        image_prompt_type="disabled",
                        image_start=None,
                        image_end=None,
                        image_refs=[],
                        frames_positions="",  # Must be string, not list
                        image_guide=None,
                        image_mask=None,
                        
                        # Video parameters
                        model_mode="generate",
                        video_source=None,
                        keep_frames_video_source="",
                        keep_frames_video_guide="",
                        video_guide_outpainting="0 0 0 0",  # Must be space-separated, not comma-separated
                        mask_expand=0,
                        
                        # Audio parameters (disabled)
                        audio_guide=None,
                        audio_guide2=None,
                        audio_source=None,
                        audio_prompt_type="",  # Empty string disables audio prompts
                        speakers_locations="",
                        
                        # Sliding window (disabled for short videos)  
                        sliding_window_size=129,  # Default to 129 frames (matches WGP UI default)
                        sliding_window_overlap=0,
                        sliding_window_color_correction_strength=0.0,
                        sliding_window_overlap_noise=0.1,
                        sliding_window_discard_last_frames=0,
                        
                        # Post-processing (disabled)
                        remove_background_images_ref=0,
                        temporal_upsampling="",  # Must be string, not float
                        spatial_upsampling="",   # Must be string, not float
                        film_grain_intensity=0.0,
                        film_grain_saturation=0.0,
                        MMAudio_setting=0,       # Must be int, not string
                        MMAudio_prompt="",
                        MMAudio_neg_prompt="",
                        
                        # Advanced parameters (defaults)
                        RIFLEx_setting=0,
                        NAG_scale=0.0,
                        NAG_tau=1.0,
                        NAG_alpha=0.0,
                        slg_switch=0,
                        slg_layers="",
                        slg_start_perc=0.0,
                        slg_end_perc=100.0,
                        apg_switch=0,
                        cfg_star_switch=0,
                        cfg_zero_step=0,
                        prompt_enhancer=0,
                        min_frames_if_references=9,
                        
                        # Mode and filename
                        mode="generate",
                        model_filename="",
                        
                        # Additional kwargs
                        **kwargs
                    )

            # WGP doesn't return the path, but stores it in state["gen"]["file_list"]
            output_path = None
            try:
                file_list = self.state["gen"]["file_list"]
                if file_list:
                    output_path = file_list[-1]  # Get the most recently generated file
                    print(f"✅ {model_type_desc} generation completed")
                    print(f"💾 Output saved to: {output_path}")
                else:
                    print(f"⚠️  {model_type_desc} generation completed but no output path found in file_list")
            except Exception as e:
                print(f"⚠️  Could not retrieve output path from state: {e}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            raise

    # Convenience methods for specific generation types
    
    def generate_t2v(self, prompt: str, **kwargs) -> str:
        """Generate text-to-video content."""
        if not self._is_t2v():
            print(f"⚠️  Warning: Current model {self.current_model} may not be optimized for T2V")
        return self.generate(prompt=prompt, **kwargs)
    
    def generate_vace(self, 
                     prompt: str, 
                     video_guide: str,
                     video_mask: Optional[str] = None,
                     video_guide2: Optional[str] = None,  # NEW: Secondary guide
                     video_mask2: Optional[str] = None,   # NEW: Secondary mask
                     video_prompt_type: str = "VP",
                     control_net_weight: float = 1.0,
                     control_net_weight2: float = 1.0,
                     **kwargs) -> str:
        """Generate VACE controlled video content.
        
        Args:
            prompt: Text prompt for generation
            video_guide: Path to primary control video (required)
            video_mask: Path to primary mask video (optional)
            video_guide2: Path to secondary control video for dual encoding (NEW)
            video_mask2: Path to secondary mask video for dual encoding (NEW)
            video_prompt_type: VACE encoding type (e.g., "VP", "VPD", "VPDA")
            control_net_weight: Strength for first VACE encoding
            control_net_weight2: Strength for second VACE encoding
            **kwargs: Additional parameters
            
        Returns:
            Path to generated video file
        """
        if not self._is_vace():
            print(f"⚠️  Warning: Current model {self.current_model} may not be a VACE model")
        
        return self.generate(
            prompt=prompt,
            video_guide=video_guide,
            video_mask=video_mask,
            video_guide2=video_guide2,  # Now properly supported in WGP
            video_mask2=video_mask2,    # Now properly supported in WGP
            video_prompt_type=video_prompt_type,
            control_net_weight=control_net_weight,
            control_net_weight2=control_net_weight2,
            **kwargs
        )

    def generate_flux(self, prompt: str, images: int = 4, **kwargs) -> str:
        """Generate Flux images.
        
        Args:
            prompt: Text prompt for generation
            images: Number of images to generate (uses video_length parameter)
            **kwargs: Additional parameters
            
        Returns:
            Path to generated image(s)
        """
        if not self._is_flux():
            print(f"⚠️  Warning: Current model {self.current_model} may not be a Flux model")
        
        return self.generate(
            prompt=prompt,
            video_length=images,  # For Flux, video_length = number of images
            **kwargs
        )

# Backward compatibility
WanContentOrchestrator = WanOrchestrator