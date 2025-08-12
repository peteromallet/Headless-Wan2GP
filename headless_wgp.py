"""
WanGP Content Generation Orchestrator

A thin wrapper around wgp.py's generate_video() function for programmatic use.
Supports T2V, VACE, and Flux generation without running the Gradio UI.
"""

import os
import sys
from typing import Optional, List, Union

# Import structured logging
from source.logging_utils import orchestrator_logger, model_logger, generation_logger


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
            
            # Debug: Check if model definitions are loaded
            model_logger.debug(f"Available models after WGP import: {list(wgp.models_def.keys())}")
            if not wgp.models_def:
                model_logger.warning("No model definitions found - this may cause model loading issues")
                    
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
        
        orchestrator_logger.success(f"WanOrchestrator initialized with WGP at {wan_root}")
        
    def _load_missing_model_definition(self, model_key: str, json_path: str):
        """
        Dynamically load a missing model definition from JSON file.
        Replicates WGP's model loading logic for individual models.
        """
        import json
        import wgp
        
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                json_def = json.load(f)
            except Exception as e:
                raise Exception(f"Error while parsing Model Definition File '{json_path}': {str(e)}")
        
        model_def = json_def["model"]
        model_def["path"] = json_path
        del json_def["model"]      
        settings = json_def   
        
        existing_model_def = wgp.models_def.get(model_key, None) 
        if existing_model_def is not None:
            existing_settings = existing_model_def.get("settings", None)
            if existing_settings is not None:
                existing_settings.update(settings)
            existing_model_def.update(model_def)
        else:
            wgp.models_def[model_key] = model_def # partial def
            model_def = wgp.init_model_def(model_key, model_def)
            wgp.models_def[model_key] = model_def # replace with full def
            model_def["settings"] = settings
        
    def load_model(self, model_key: str):
        """Load and validate a model type using WGP's exact generation-time pattern.
        
        Args:
            model_key: Model identifier (e.g., "t2v", "vace_14B", "flux")
            
        This replicates the exact model loading logic from WGP's generate_video function
        (lines 4249-4258) rather than the UI preloading function.
        """
        if self._get_base_model_type(model_key) is None:
            raise ValueError(f"Unknown model: {model_key}")
        
        import wgp
        import gc
        
        # Debug: Check if model definition is missing and diagnose why
        model_def = wgp.get_model_def(model_key)
        if model_def is None:
            available_models = list(wgp.models_def.keys())
            current_dir = os.getcwd()
            model_logger.warning(f"Model definition for '{model_key}' not found!")
            model_logger.debug(f"Current working directory: {current_dir}")
            model_logger.debug(f"Available models: {available_models}")
            model_logger.debug(f"Looking for model file: {self.wan_root}/defaults/{model_key}.json")
            
            # Check if the JSON file exists and try to load it dynamically
            json_path = os.path.join(self.wan_root, "defaults", f"{model_key}.json")
            if os.path.exists(json_path):
                model_logger.warning(f"Model JSON file exists at {json_path} but wasn't loaded into models_def - attempting dynamic load")
                try:
                    self._load_missing_model_definition(model_key, json_path)
                    model_def = wgp.get_model_def(model_key)  # Try again after loading
                    if model_def:
                        model_logger.success(f"Successfully loaded missing model definition for {model_key}")
                    else:
                        model_logger.error(f"Failed to load model definition for {model_key} even after dynamic loading")
                except Exception as e:
                    model_logger.error(f"Failed to dynamically load model definition: {e}")
            else:
                model_logger.error(f"Model JSON file missing at {json_path}")
        
        architecture = model_def.get('architecture') if model_def else 'unknown'
        modules = wgp.get_model_recursive_prop(model_key, "modules", return_list=True)
        model_logger.debug(f"Model Info: {model_key} | Architecture: {architecture} | Modules: {modules}")
        
        # Use WGP's EXACT model loading pattern from generate_video (lines 4249-4258)
        current_model_info = f"(current: {wgp.transformer_type})" if wgp.transformer_type else "(no model loaded)"
        
        if model_key != wgp.transformer_type or wgp.reload_needed:
            model_logger.info(f"🔄 MODEL SWITCH: Using WGP's generate_video pattern - switching from {current_model_info} to {model_key}")
            
            # Replicate WGP's exact unloading pattern (lines 4250-4254)
            wgp.wan_model = None
            if wgp.offloadobj is not None:
                wgp.offloadobj.release()
                wgp.offloadobj = None
            gc.collect()
            
            # Replicate WGP's exact loading pattern (lines 4255-4258)
            model_logger.debug(f"Loading model {wgp.get_model_name(model_key)}...")
            wgp.wan_model, wgp.offloadobj = wgp.load_models(model_key)
            model_logger.debug("Model loaded")
            wgp.reload_needed = False
            
            # Note: transformer_type is set automatically by load_models() at line 2929
            
            model_logger.info(f"✅ MODEL: Loaded using WGP's exact generate_video pattern")
        else:
            model_logger.debug(f"📋 MODEL: Model {model_key} already loaded, no switch needed")
        
        # Update our tracking to match WGP's state
        self.current_model = model_key
        self.state["model_type"] = model_key
        self.offloadobj = wgp.offloadobj  # Keep reference to WGP's offload object
        
        family = self._get_model_family(model_key, for_ui=True)
        model_logger.success(f"✅ MODEL Loaded model: {model_key} ({family}) using WGP's exact generate_video pattern")
    
    def unload_model(self):
        """Unload the current model using WGP's native unload function."""
        import wgp
        
        if self.current_model and wgp.wan_model is not None:
            model_logger.info(f"🔄 MODEL UNLOAD: Unloading {self.current_model} using WGP's unload_model_if_needed")
            
            # Create a state object that WGP functions expect
            temp_state = {"model_type": self.current_model}
            
            # Use WGP's native unload function
            try:
                wgp.unload_model_if_needed(temp_state)
                model_logger.info(f"✅ MODEL: WGP unload_model_if_needed completed")
                
                # Clear our tracking
                self.current_model = None
                self.offloadobj = None
                self.state["model_type"] = None
                
            except Exception as e:
                model_logger.error(f"WGP unload_model_if_needed failed: {e}")
                raise
        else:
            model_logger.debug(f"📋 MODEL: No model to unload")
    
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
                model_logger.debug(f"Discovered {len(loras)} LoRAs for {model_type}: {[os.path.basename(l) for l in loras[:3]]}{'...' if len(loras) > 3 else ''}")
            else:
                model_logger.debug(f"No LoRAs found for {model_type}")
                
        except Exception as e:
            model_logger.warning(f"LoRA discovery failed for {model_type}: {e}")
            # Keep empty defaults to prevent crashes
            self.state["loras"] = []
            self.state["loras_names"] = []
            self.state["loras_presets"] = {}

    def _create_vace_fixed_generate_video(self, original_generate_video):
        """Create a wrapper around generate_video for VACE models.
        
        VACE models are properly loaded via load_models() now, so no special handling needed.
        Also handles parameter name mapping for compatibility.
        """
        def vace_fixed_generate_video(*args, **kwargs):
            # Map parameter names for compatibility
            if "denoise_strength" in kwargs:
                kwargs["denoising_strength"] = kwargs.pop("denoise_strength")
            
            # VACE modules are now properly loaded via load_models() - no patching needed
            return original_generate_video(*args, **kwargs)
        
        return vace_fixed_generate_video

    def _is_vace(self) -> bool:
        """Check if current model is a VACE model."""
        return self._test_vace_module(self.current_model)
    
    def is_model_vace(self, model_name: str) -> bool:
        """Check if a given model name is a VACE model (model-agnostic).
        
        This method doesn't require the model to be loaded, making it suitable
        for VACE detection during task processing when the orchestrator may not
        have the model loaded yet.
        
        Args:
            model_name: The model identifier to check (e.g., "vace_14B", "t2v")
            
        Returns:
            True if the model is a VACE model, False otherwise
        """
        return self._test_vace_module(model_name)

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

        # Determine model types for generation
        base_model_type = self._get_base_model_type(self.current_model)
        vace_test_result = self._test_vace_module(self.current_model)
        
        is_vace = self._is_vace()
        is_flux = self._is_flux()
        is_t2v = self._is_t2v()
        
        generation_logger.debug(f"Model detection - VACE: {is_vace}, Flux: {is_flux}, T2V: {is_t2v}")
        generation_logger.debug(f"Generation parameters - prompt: '{prompt[:50]}...', resolution: {resolution}, length: {video_length}")
        
        if is_vace:
            generation_logger.debug(f"VACE parameters - guide: {video_guide}, type: {video_prompt_type}, weights: {control_net_weight}/{control_net_weight2}")
            if video_guide2:
                generation_logger.debug(f"VACE secondary guide: {video_guide2}")

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

        # Prepare LoRA parameters first (needed for wgp_params)
        activated_loras = lora_names if lora_names else []
        # WGP expects loras_multipliers as string, not list
        if lora_multipliers:
            # Convert list of floats to space-separated string
            loras_multipliers_str = " ".join(str(m) for m in lora_multipliers)
        else:
            loras_multipliers_str = ""

        # Create minimal task and callback objects (needed for wgp_params)
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

        # Build parameter dictionary with defaults that can be overridden
        # This allows ANY parameter to be overridden via kwargs
        wgp_params = {
            # Core parameters with defaults
            'task': task,
            'send_cmd': send_cmd,
            'state': self.state,
            'model_type': self.current_model,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'resolution': resolution,
            'video_length': actual_video_length,
            'batch_size': actual_batch_size,
            'seed': seed,
            'force_fps': "auto",
            'num_inference_steps': num_inference_steps,
            'guidance_scale': actual_guidance,
            'guidance2_scale': actual_guidance,
            'switch_threshold': kwargs.get('switch_threshold', 500),  # Get from kwargs with default (0-1000 scale)
            'embedded_guidance_scale': embedded_guidance_scale if is_flux else 0.0,
            'image_mode': image_mode,
            
            # VACE control parameters
            'video_guide': video_guide,
            'video_mask': video_mask,
            'video_guide2': video_guide2,
            'video_mask2': video_mask2,
            'video_prompt_type': video_prompt_type,
            'control_net_weight': control_net_weight,
            'control_net_weight2': control_net_weight2,
            'denoising_strength': 1.0,
            
            # LoRA parameters
            'activated_loras': activated_loras,
            'loras_multipliers': loras_multipliers_str,
            
            # Standard defaults for other parameters
            'audio_guidance_scale': 1.0,
            'repeat_generation': 1,
            'multi_prompts_gen_type': 0,
            'multi_images_gen_type': 0,
            'skip_steps_cache_type': "",
            'skip_steps_multiplier': 1.0,
            'skip_steps_start_step_perc': 0.0,
            
            # Image parameters
            'image_prompt_type': "disabled",
            'image_start': None,
            'image_end': None,
            'image_refs': [],
            'frames_positions': "",
            'image_guide': None,
            'image_mask': None,
            
            # Video parameters
            'model_mode': "generate",
            'video_source': None,
            'keep_frames_video_source': "",
            'keep_frames_video_guide': "",
            'video_guide_outpainting': "0 0 0 0",
            'mask_expand': 0,
            
            # Audio parameters
            'audio_guide': None,
            'audio_guide2': None,
            'audio_source': None,
            'audio_prompt_type': "",
            'speakers_locations': "",
            
            # Sliding window
            'sliding_window_size': 129,
            'sliding_window_overlap': 0,
            'sliding_window_color_correction_strength': 0.0,
            'sliding_window_overlap_noise': 0.1,
            'sliding_window_discard_last_frames': 0,
            
            # Post-processing
            'remove_background_images_ref': 0,
            'temporal_upsampling': "",
            'spatial_upsampling': "",
            'film_grain_intensity': 0.0,
            'film_grain_saturation': 0.0,
            'MMAudio_setting': 0,
            'MMAudio_prompt': "",
            'MMAudio_neg_prompt': "",
            
            # Advanced parameters
            'RIFLEx_setting': 0,
            'NAG_scale': 0.0,
            'NAG_tau': 1.0,
            'NAG_alpha': 0.0,
            'slg_switch': 0,
            'slg_layers': "",
            'slg_start_perc': 0.0,
            'slg_end_perc': 100.0,
            'apg_switch': 0,
            'cfg_star_switch': 0,
            'cfg_zero_step': 0,
            'prompt_enhancer': 0,
            'min_frames_if_references': 9,
            
            # Mode and filename
            'mode': "generate",
            'model_filename': "",
            
            # Critical parameters from kwargs with defaults
            'flow_shift': kwargs.get('flow_shift', 7.0),
            'sample_solver': kwargs.get('sample_solver', "euler"),
        }
        
        # Override any parameters provided in kwargs
        # This allows ANY parameter to be customized
        for key, value in kwargs.items():
            if key not in wgp_params:
                # Add any additional parameters from kwargs that aren't in defaults
                wgp_params[key] = value
                generation_logger.debug(f"Adding extra parameter from kwargs: {key}={value}")

        # Generate content type description
        content_type = "images" if is_flux else "video"
        model_type_desc = "Flux" if is_flux else ("VACE" if is_vace else "T2V")
        count_desc = f"{video_length} {'images' if is_flux else 'frames'}"
        
        generation_logger.essential(f"Generating {model_type_desc} {content_type}: {resolution}, {count_desc}")
        if is_vace:
            encodings = [c for c in video_prompt_type if c in "PDSLCMUA"]
            generation_logger.debug(f"VACE encodings: {encodings}")
            if video_guide2:
                generation_logger.debug(f"Using secondary guide: {video_guide2}")
        if activated_loras:
            generation_logger.debug(f"LoRAs: {activated_loras}")

        try:
            generation_logger.debug("Calling WGP generate_video with VACE module support")
            
            # Log critical parameters being passed to WGP
            if is_vace:
                generation_logger.debug(f"VACE parameters to WGP - model: {self.current_model}, guide: {video_guide}, type: {video_prompt_type}")
            else:
                generation_logger.debug(f"Standard parameters to WGP - model: {self.current_model}")
            
            # [CausVidDebugTrace] Log parameters being passed to generate_video
            generation_logger.info(f"[CausVidDebugTrace] WanOrchestrator.generate calling _generate_video with:")
            generation_logger.info(f"[CausVidDebugTrace]   model_type: {self.current_model}")
            generation_logger.info(f"[CausVidDebugTrace]   num_inference_steps: {num_inference_steps}")
            generation_logger.info(f"[CausVidDebugTrace]   guidance_scale: {actual_guidance}")
            generation_logger.info(f"[CausVidDebugTrace]   guidance2_scale: {actual_guidance}")
            generation_logger.info(f"[CausVidDebugTrace]   activated_loras: {activated_loras}")
            generation_logger.info(f"[CausVidDebugTrace]   loras_multipliers_str: {loras_multipliers_str}")
            
            # ARCHITECTURAL FIX: Pre-populate WGP UI state for LoRA compatibility
            # WGP was designed for UI usage where state["loras"] gets populated through UI
            # In headless mode, we need to pre-populate this state to ensure LoRA loading works
            original_loras = self.state.get("loras", [])
            if activated_loras and len(activated_loras) > 0:
                generation_logger.info(f"[CausVidDebugTrace] WanOrchestrator: Pre-populating WGP state with {len(activated_loras)} LoRAs")
                self.state["loras"] = activated_loras.copy()  # Populate UI state for WGP compatibility
                generation_logger.debug(f"[CausVidDebugTrace] WanOrchestrator: state['loras'] = {self.state['loras']}")
            
            try:
                # Call the VACE-fixed generate_video with the unified parameter dictionary
                # This allows ANY parameter to be overridden via kwargs
                result = self._generate_video(**wgp_params)
            
            finally:
                # ARCHITECTURAL FIX: Restore original UI state after generation
                if activated_loras and len(activated_loras) > 0:
                    self.state["loras"] = original_loras  # Restore original state
                    generation_logger.debug(f"[CausVidDebugTrace] WanOrchestrator: Restored original state['loras'] = {self.state['loras']}")

            # WGP doesn't return the path, but stores it in state["gen"]["file_list"]
            output_path = None
            try:
                file_list = self.state["gen"]["file_list"]
                if file_list:
                    output_path = file_list[-1]  # Get the most recently generated file
                    generation_logger.success(f"{model_type_desc} generation completed")
                    generation_logger.essential(f"Output saved to: {output_path}")
                else:
                    generation_logger.warning(f"{model_type_desc} generation completed but no output path found in file_list")
            except Exception as e:
                generation_logger.warning(f"Could not retrieve output path from state: {e}")
            
            return output_path
            
        except Exception as e:
            generation_logger.error(f"Generation failed: {e}")
            raise

    # Convenience methods for specific generation types
    
    def generate_t2v(self, prompt: str, **kwargs) -> str:
        """Generate text-to-video content."""
        if not self._is_t2v():
            generation_logger.warning(f"Current model {self.current_model} may not be optimized for T2V")
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
            generation_logger.warning(f"Current model {self.current_model} may not be a VACE model")
        
        # [CausVidDebugTrace] Log LoRA parameters at VACE level
        generation_logger.info(f"[CausVidDebugTrace] generate_vace received kwargs: {list(kwargs.keys())}")
        if "lora_names" in kwargs:
            generation_logger.info(f"[CausVidDebugTrace] generate_vace lora_names: {kwargs['lora_names']}")
        if "lora_multipliers" in kwargs:
            generation_logger.info(f"[CausVidDebugTrace] generate_vace lora_multipliers: {kwargs['lora_multipliers']}")
        
        return self.generate(
            prompt=prompt,
            video_guide=video_guide,
            video_mask=video_mask,
            video_guide2=video_guide2,  # Now properly supported in WGP
            video_mask2=video_mask2,    # Now properly supported in WGP
            video_prompt_type=video_prompt_type,
            control_net_weight=control_net_weight,
            control_net_weight2=control_net_weight2,
            **kwargs  # Any additional parameters including switch_threshold
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
            generation_logger.warning(f"Current model {self.current_model} may not be a Flux model")
        
        return self.generate(
            prompt=prompt,
            video_length=images,  # For Flux, video_length = number of images
            **kwargs
        )

# Backward compatibility
WanContentOrchestrator = WanOrchestrator