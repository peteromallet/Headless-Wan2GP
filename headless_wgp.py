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

# Import debug mode flag
try:
    from worker import debug_mode
except ImportError:
    debug_mode = False


def _verify_wgp_directory(logger, context: str = ""):
    """
    Verify we're still in the Wan2GP directory after any operation.

    Wan2GP expects to run from its directory and uses relative paths.
    Call this after any wgp operation to catch directory changes early.

    Args:
        logger: Logger instance to use
        context: Description of what just happened (e.g., "after wgp.generate_video()")
    """
    current_dir = os.getcwd()
    expected_substr = "Wan2GP"

    if expected_substr not in current_dir:
        logger.warning(
            f"[PATH_CHECK] {context}: Current directory may be wrong!\n"
            f"  Current: {current_dir}\n"
            f"  Expected: Path containing 'Wan2GP'\n"
            f"  This could cause issues with wgp.py's relative paths!"
        )
    else:
        logger.debug(f"[PATH_CHECK] {context}: Still in Wan2GP directory ✓ ({current_dir})")

    # Also verify critical dirs still accessible
    if not os.path.exists("defaults"):
        logger.error(
            f"[PATH_CHECK] {context}: CRITICAL - defaults/ no longer accessible!\n"
            f"  Current directory: {current_dir}"
        )

    return current_dir


class WanOrchestrator:
    """Thin adapter around `wgp.generate_video` for easier programmatic use."""

    def __init__(self, wan_root: str):
        """Initialize orchestrator with WanGP directory.

        Args:
            wan_root: Path to WanGP repository root directory (MUST be absolute path to Wan2GP/)

        IMPORTANT: Caller MUST have already changed to wan_root directory before calling this.
        wgp.py uses relative paths and expects to run from Wan2GP/.
        """
        import logging
        _init_logger = logging.getLogger('HeadlessQueue')

        # Store the wan_root (should match current directory)
        self.wan_root = os.path.abspath(wan_root)
        current_dir = os.getcwd()

        if debug_mode:
            _init_logger.info(f"[INIT_DEBUG] WanOrchestrator.__init__ called with wan_root: {self.wan_root}")
            _init_logger.info(f"[INIT_DEBUG] Current working directory: {current_dir}")

        # CRITICAL CHECK: Verify caller already changed to the correct directory
        # wgp.py will import and execute module-level code that uses relative paths like "defaults/*.json"
        # If we're in the wrong directory, wgp will load 0 models and fail mysteriously
        if current_dir != self.wan_root:
            error_msg = (
                f"CRITICAL: WanOrchestrator must be initialized from Wan2GP directory!\n"
                f"  Current directory: {current_dir}\n"
                f"  Expected directory: {self.wan_root}\n"
                f"  Caller must chdir() before creating WanOrchestrator instance."
            )
            if debug_mode:
                _init_logger.error(f"[INIT_DEBUG] {error_msg}")
            raise RuntimeError(error_msg)

        # Verify Wan2GP structure
        if not os.path.isdir("defaults"):
            raise RuntimeError(
                f"defaults/ directory not found in {current_dir}. "
                f"This doesn't appear to be a valid Wan2GP directory!"
            )
        if not os.path.isdir("models"):
            if debug_mode:
                _init_logger.warning(f"[INIT_DEBUG] models/ directory not found in {current_dir}")

        # Ensure Wan2GP is first in sys.path so wgp.py imports correctly
        if self.wan_root in sys.path:
            sys.path.remove(self.wan_root)
        sys.path.insert(0, self.wan_root)
        if debug_mode:
            _init_logger.info(f"[INIT_DEBUG] Added {self.wan_root} to sys.path[0]")
        
        # Optional smoke/CPU-only modes
        self.smoke_mode = bool(os.environ.get("HEADLESS_WAN2GP_SMOKE", ""))
        force_cpu = os.environ.get("HEADLESS_WAN2GP_FORCE_CPU", "0") == "1"

        # Force CPU if requested and guard CUDA capability queries before importing WGP
        if force_cpu and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            import torch  # type: ignore
            # If CUDA isn't available, stub capability query used by upstream on import
            if force_cpu or not torch.cuda.is_available():
                try:
                    def _safe_get_device_capability(device=None):
                        return (8, 0)
                    torch.cuda.get_device_capability = _safe_get_device_capability  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass

        # Force a headless Matplotlib backend to avoid Tkinter requirements during upstream imports
        if not os.environ.get("MPLBACKEND"):
            os.environ["MPLBACKEND"] = "Agg"
        try:
            import matplotlib  # type: ignore
            _orig_use = matplotlib.use  # type: ignore
            def _force_agg(_backend=None, *args, **kwargs):
                try:
                    return _orig_use('Agg', force=True)
                except Exception:
                    return None
            matplotlib.use = _force_agg  # type: ignore
        except Exception:
            pass

        # Pre-import config hygiene for upstream wgp.py
        # Some older configs may store preload_model_policy as an int, but upstream now expects a list
        try:
            cfg_path = os.path.join(self.wan_root, "wgp_config.json")
            if os.path.isfile(cfg_path):
                import json as _json
                with open(cfg_path, "r", encoding="utf-8") as _r:
                    _cfg = _json.load(_r)
                changed = False
                pmp = _cfg.get("preload_model_policy", [])
                if isinstance(pmp, int):
                    _cfg["preload_model_policy"] = []  # disable preloading and fix type
                    changed = True
                # Ensure save paths exist and are strings
                if not isinstance(_cfg.get("save_path", ""), str):
                    _cfg["save_path"] = "outputs"
                    changed = True
                if not isinstance(_cfg.get("image_save_path", ""), str):
                    _cfg["image_save_path"] = "outputs"
                    changed = True
                if changed:
                    with open(cfg_path, "w", encoding="utf-8") as _w:
                        _json.dump(_cfg, _w, indent=4)
        except Exception:
            # Config hygiene should not block import
            pass

        # Import WGP components after path setup (skip entirely in smoke mode)
        if not self.smoke_mode:
            try:
                if debug_mode:
                    _init_logger.info(f"[INIT_DEBUG] About to import wgp module")
                current_dir = os.getcwd()
                if debug_mode:
                    _init_logger.info(f"[INIT_DEBUG] Current directory: {current_dir}")
                    _init_logger.info(f"[INIT_DEBUG] sys.path[0]: {sys.path[0] if sys.path else 'empty'}")
                    _init_logger.info(f"[INIT_DEBUG] wgp already in sys.modules: {'wgp' in sys.modules}")

                # Double-check we're in the right directory before importing
                if current_dir != self.wan_root:
                    if debug_mode:
                        _init_logger.error(f"[INIT_DEBUG] CRITICAL: Current directory {current_dir} != expected {self.wan_root}")
                    raise RuntimeError(f"Directory changed unexpectedly before wgp import: {current_dir} != {self.wan_root}")

                # If wgp was previously imported from wrong directory, remove it so it reimports
                if 'wgp' in sys.modules:
                    if debug_mode:
                        _init_logger.warning(f"[INIT_DEBUG] wgp already in sys.modules - removing to force reimport from correct directory")
                    del sys.modules['wgp']

                _saved_argv = list(sys.argv)
                sys.argv = ["headless_wgp.py"]
                from wgp import (
                    generate_video, get_base_model_type, get_model_family,
                    test_vace_module, apply_changes
                )

                # Verify directory didn't change during wgp import
                _verify_wgp_directory(_init_logger, "after importing wgp module")

                # Apply VACE fix wrapper to generate_video
                self._generate_video = self._create_vace_fixed_generate_video(generate_video)
                self._get_base_model_type = get_base_model_type
                self._get_model_family = get_model_family
                self._test_vace_module = test_vace_module
                self._apply_changes = apply_changes

                # Monkeypatch upstream to support Qwen family without modifying Wan2GP files
                import wgp as wgp

                # Patch load_wan_model to route Qwen to its dedicated handler
                try:
                    _orig_load_wan_model = wgp.load_wan_model

                    def _patched_load_wan_model(model_filename, model_type, base_model_type, model_def,
                                                quantizeTransformer=False, dtype=None, VAE_dtype=None,
                                                mixed_precision_transformer=False, save_quantized=False):
                        try:
                            base = wgp.get_base_model_type(base_model_type)
                        except Exception:
                            base = base_model_type
                        if isinstance(base, str) and "qwen" in base:
                            model_logger.debug("[QWEN_LOAD_DEBUG] Routing to Qwen family loader via monkeypatch")
                            from models.qwen.qwen_handler import family_handler as _qwen_handler  # type: ignore
                            pipe_processor, pipe = _qwen_handler.load_model(
                                model_filename=model_filename,
                                model_type=model_type,
                                base_model_type=base_model_type,
                                model_def=model_def,
                                quantizeTransformer=quantizeTransformer,
                                text_encoder_quantization=wgp.text_encoder_quantization,
                                dtype=dtype,
                                VAE_dtype=VAE_dtype,
                                mixed_precision_transformer=mixed_precision_transformer,
                                save_quantized=save_quantized,
                            )
                            return pipe_processor, pipe
                        # Fallback to original WAN loader
                        return _orig_load_wan_model(
                            model_filename, model_type, base_model_type, model_def,
                            quantizeTransformer=quantizeTransformer, dtype=dtype, VAE_dtype=VAE_dtype,
                            mixed_precision_transformer=mixed_precision_transformer, save_quantized=save_quantized
                        )

                    wgp.load_wan_model = _patched_load_wan_model  # type: ignore
                except Exception as _e:
                    model_logger.debug(f"[QWEN_LOAD_DEBUG] Failed to monkeypatch load_wan_model: {_e}")

                # Patch get_lora_dir to redirect Qwen models to loras_qwen if available
                try:
                    _orig_get_lora_dir = wgp.get_lora_dir

                    def _patched_get_lora_dir(model_type: str):
                        try:
                            mt = (model_type or "").lower()
                            if "qwen" in mt:
                                qwen_dir = os.path.join(self.wan_root, "loras_qwen")
                                if os.path.isdir(qwen_dir):
                                    return qwen_dir
                        except Exception:
                            pass
                        return _orig_get_lora_dir(model_type)

                    wgp.get_lora_dir = _patched_get_lora_dir  # type: ignore
                except Exception as _e:
                    model_logger.debug(f"[QWEN_LOAD_DEBUG] Failed to monkeypatch get_lora_dir: {_e}")

                # Harmonize LoRA multiplier parsing across pipelines:
                # Use the 3-phase capable parser so Qwen pipeline (which expects phase3/shared)
                # receives a compatible slists_dict. This is backward compatible for 2-phase models.
                try:
                    from shared.utils import loras_mutipliers as _shared_lora_utils  # type: ignore
                    wgp.parse_loras_multipliers = _shared_lora_utils.parse_loras_multipliers  # type: ignore
                    # preparse is identical, but patching it keeps the source consistent
                    wgp.preparse_loras_multipliers = _shared_lora_utils.preparse_loras_multipliers  # type: ignore
                except Exception as _e:
                    model_logger.debug(f"[QWEN_LOAD_DEBUG] Failed to monkeypatch lora parsers: {_e}")

                # Optionally disable Qwen's built-in inpainting LoRA (preload_URLs) in headless mode.
                # Default: disabled, unless HEADLESS_WAN2GP_ENABLE_QWEN_INPAINTING_LORA=1 is set.
                try:
                    from models.qwen import qwen_main as _qwen_main  # type: ignore
                    _orig_qwen_get_loras_transformer = _qwen_main.model_factory.get_loras_transformer  # type: ignore

                    def _patched_qwen_get_loras_transformer(self, get_model_recursive_prop, model_type, model_mode, **kwargs):  # type: ignore
                        try:
                            if os.environ.get("HEADLESS_WAN2GP_ENABLE_QWEN_INPAINTING_LORA", "0") != "1":
                                return [], []
                        except Exception:
                            # If env check fails, fall back to disabled behavior
                            return [], []
                        return _orig_qwen_get_loras_transformer(self, get_model_recursive_prop, model_type, model_mode, **kwargs)

                    _qwen_main.model_factory.get_loras_transformer = _patched_qwen_get_loras_transformer  # type: ignore
                except Exception as _e:
                    model_logger.debug(f"[QWEN_LOAD_DEBUG] Failed to monkeypatch Qwen get_loras_transformer: {_e}")
                
                # Initialize WGP global state (normally done by UI)
                import wgp
                # Set absolute output path to avoid issues when working directory changes
                absolute_outputs_path = os.path.abspath(os.path.join(os.path.dirname(self.wan_root), 'outputs'))
                for attr, default in {
                    'wan_model': None, 'offloadobj': None, 'reload_needed': True,
                    'transformer_type': None, 'server_config': {'save_path': absolute_outputs_path}
                }.items():
                    if not hasattr(wgp, attr):
                        setattr(wgp, attr, default)
                
                # Debug: Check if model definitions are loaded
                if debug_mode:
                    model_logger.info(f"[INIT_DEBUG] Available models after WGP import: {list(wgp.models_def.keys())}")
                _verify_wgp_directory(model_logger, "after wgp setup and monkeypatching")

                if not wgp.models_def:
                    error_msg = (
                        f"CRITICAL: No model definitions found after importing wgp! "
                        f"Current directory: {os.getcwd()}, "
                        f"Expected: {self.wan_root}, "
                        f"defaults/ exists: {os.path.exists('defaults')}, "
                        f"finetunes/ exists: {os.path.exists('finetunes')}"
                    )
                    model_logger.error(error_msg)
                    raise RuntimeError(error_msg)
            except ImportError as e:
                raise ImportError(f"Failed to import wgp module. Ensure {wan_root} contains wgp.py: {e}")
            finally:
                try:
                    sys.argv = _saved_argv
                except Exception:
                    pass

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
        if not self.smoke_mode:
            # Ensure model_type is set before applying config, as upstream generate_header/get_overridden_attention
            # will look up the current model's definition via state["model_type"].
            try:
                import wgp as _w
                default_model_type = getattr(_w, "transformer_type", None) or "t2v"
            except Exception:
                default_model_type = "t2v"
            self.state["model_type"] = default_model_type

            # Upstream apply_changes signature accepts a single save_path_choice
            outputs_dir = "outputs/"
            self._apply_changes(
                self.state,
                transformer_types_choices=["t2v"],
                transformer_dtype_policy_choice="auto",
                text_encoder_quantization_choice="bf16",
                VAE_precision_choice="fp32",
                mixed_precision_choice=0,
                save_path_choice=outputs_dir,
                image_save_path_choice=outputs_dir,
                attention_choice="auto",
                compile_choice=0,
                profile_choice=4,
                vae_config_choice="default",
                metadata_choice="none",
                quantization_choice="int8",
                preload_model_policy_choice=[]
            )

            # Verify directory after apply_changes (it may have done file operations)
            _verify_wgp_directory(orchestrator_logger, "after apply_changes()")

        else:
            # Provide stubbed helpers for smoke mode
            self._get_base_model_type = lambda model_key: ("t2v" if "flux" not in (model_key or "") else "flux")
            self._get_model_family = lambda model_key, for_ui=False: ("VACE" if "vace" in (model_key or "") else ("Flux" if "flux" in (model_key or "") else "T2V"))
            self._test_vace_module = lambda model_name: ("vace" in (model_name or ""))
        self.current_model = None
        self.offloadobj = None  # Store WGP's offload object
        self.passthrough_mode = False  # Flag for explicit passthrough mode

        orchestrator_logger.success(f"WanOrchestrator initialized with WGP at {wan_root}")

        # Final notice for smoke mode
        if self.smoke_mode:
            orchestrator_logger.warning("HEADLESS_WAN2GP_SMOKE enabled: generation will return sample outputs only")
        
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
        if self.smoke_mode:
            # In smoke mode, skip heavy WGP model loading
            self.current_model = model_key
            self.state["model_type"] = model_key
            model_logger.info(f"[SMOKE] Pretending to load model: {model_key}")
            return

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
            
            # CRITICAL: Clear CUDA cache after unloading to free reserved VRAM before loading new model
            # Without this, old model's reserved memory persists and new model OOMs during loading
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                model_logger.debug("Cleared CUDA cache after model unload")
            
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
        if self.smoke_mode:
            model_logger.info(f"[SMOKE] Unload model: {self.current_model}")
            self.current_model = None
            self.offloadobj = None
            self.state["model_type"] = None
            return
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

    def _is_qwen(self) -> bool:
        """Check if current model is a Qwen image model."""
        try:
            if self._get_model_family(self.current_model) == "qwen":
                return True
        except Exception:
            pass
        base_type = (self._get_base_model_type(self.current_model) or "").lower()
        return base_type.startswith("qwen")

    def _load_image(self, path: Optional[str], mask: bool = False):
        if not path:
            return None
        try:
            from PIL import Image  # type: ignore
            p = self._resolve_media_path(path)
            img = Image.open(p)
            if mask:
                try:
                    return img.convert("L")
                except Exception:
                    return img
            else:
                try:
                    return img.convert("RGB")
                except Exception:
                    return img
        except Exception as e:
            generation_logger.warning(f"Could not load image from {path}: {e}")
            return None

    def _resolve_media_path(self, path: Optional[str]) -> Optional[str]:
        """Resolve media paths relative to the local Headless-Wan2GP repo.

        - If the path exists as-is (absolute), return it.
        - If relative, prefer repo root (agent_tasks/Headless-Wan2GP), then Wan2GP.
        """
        if not path:
            return path
        try:
            from pathlib import Path
            p = Path(path)
            if p.exists():
                return str(p.resolve())
            wan_root = Path(self.wan_root)
            repo_root = wan_root.parent
            if not p.is_absolute():
                candidate = repo_root / p
                if candidate.exists():
                    return str(candidate.resolve())
                candidate = wan_root / p
                if candidate.exists():
                    return str(candidate.resolve())
        except Exception:
            pass
        return path

    def _resolve_parameters(self, model_type: str, task_params: dict) -> dict:
        """
        Resolve generation parameters with explicit precedence:
        1. Task explicit parameters (highest priority)
        2. Model JSON configuration (medium priority)
        3. System defaults (lowest priority)
        
        Args:
            model_type: Model identifier (e.g., "optimised-t2i")
            task_params: Parameters explicitly provided by the task/user
            
        Returns:
            Resolved parameter dictionary with proper precedence
        """
        # 1. Start with system defaults (lowest priority) - migrated from worker.py
        resolved_params = {
            "resolution": "1280x720",
            "video_length": 49,
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "guidance2_scale": 7.5,
            "flow_shift": 7.0,
            "sample_solver": "euler",
            "switch_threshold": 500,
            "seed": -1,  # Random seed (matches worker.py behavior)
            "negative_prompt": "",
            "activated_loras": [],
            "loras_multipliers": "",
        }
        
        # 2. Apply model JSON configuration (medium priority)
        try:
            import wgp
            model_defaults = wgp.get_default_settings(model_type)
            generation_logger.debug(f"get_default_settings('{model_type}') returned: {model_defaults}")
            generation_logger.debug(f"Type: {type(model_defaults)}")
            
            if model_defaults:
                generation_logger.debug(f"Before applying model config - resolved_params: {resolved_params}")

                for param, value in model_defaults.items():
                    # JSON passthrough mode: Allow activated_loras and loras_multipliers to pass directly
                    if param not in ["prompt"]:
                        old_value = resolved_params.get(param, "NOT_SET")
                        resolved_params[param] = value
                        generation_logger.debug(f"Applied {param}: {old_value} → {value}")

                generation_logger.debug(f"After applying model config - resolved_params: {resolved_params}")
                generation_logger.debug(f"Applied model config for '{model_type}': {len(model_defaults)} parameters")
            else:
                generation_logger.warning(f"No model configuration found for '{model_type}'")
                
        except Exception as e:
            generation_logger.warning(f"Could not load model configuration for '{model_type}': {e}")
            generation_logger.debug(f"Exception details: {str(e)}")
            import traceback
            generation_logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # 3. Apply task explicit parameters (highest priority)
        generation_logger.debug(f"Task explicit parameters: {task_params}")
        generation_logger.debug(f"Before applying task params - resolved_params: {resolved_params}")

        for param, value in task_params.items():
            if value is not None:  # Don't override with None values
                old_value = resolved_params.get(param, "NOT_SET")
                resolved_params[param] = value
                generation_logger.debug(f"Task override {param}: {old_value} → {value}")

        generation_logger.debug(f"FINAL resolved_params: {resolved_params}")
        generation_logger.debug(f"Parameter resolution for '{model_type}': {len(task_params)} task overrides applied")
        return resolved_params

    def generate(self, 
                prompt: str,
                model_type: str = None,
                # Common parameters - None means "use model/system defaults"
                resolution: Optional[str] = None,
                video_length: Optional[int] = None,
                num_inference_steps: Optional[int] = None,
                guidance_scale: Optional[float] = None,
                seed: Optional[int] = None,
                # VACE parameters
                video_guide: Optional[str] = None,
                video_mask: Optional[str] = None,
                video_prompt_type: Optional[str] = None,
                control_net_weight: Optional[float] = None,
                control_net_weight2: Optional[float] = None,
                # Flux parameters
                embedded_guidance_scale: Optional[float] = None,
                # LoRA parameters
                lora_names: Optional[List[str]] = None,
                lora_multipliers: Optional[List] = None,  # Can be List[float] or List[str] for phase-config
                # Other parameters
                negative_prompt: Optional[str] = None,
                batch_size: Optional[int] = None,
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

        # Smoke-mode short-circuit: create a sample output and return its path
        if self.smoke_mode:
            from pathlib import Path
            import shutil
            out_dir = Path(os.getcwd()) / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            sample_src = Path(os.path.abspath(os.path.join(Path(__file__).parent, "samples", "test.mp4")))
            if not sample_src.exists():
                # Fallback to project-level samples directory
                sample_src = Path(os.path.abspath(os.path.join(Path(__file__).parent, "samples", "video.mp4")))
            ts = __import__("time").strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"smoke_{self.current_model}_{ts}.mp4"
            try:
                shutil.copyfile(str(sample_src), str(out_path))
            except Exception:
                # If copy fails for any reason, create an empty file as placeholder
                out_path.write_bytes(b"")
            # Mimic WGP state behavior
            try:
                self.state["gen"]["file_list"].append(str(out_path))
            except Exception:
                pass
            generation_logger.info(f"[SMOKE] Generated placeholder output at: {out_path}")
            return str(out_path)

        # Use provided model_type or current loaded model
        effective_model_type = model_type or self.current_model
        
        # Build task explicit parameters: only non-None values are considered "explicit"
        # This ensures method signature defaults don't override model/system defaults
        task_explicit_params = {"prompt": prompt}  # prompt is always required
        
        # Add all non-None parameters from method signature
        param_values = {
            "resolution": resolution,
            "video_length": video_length,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "video_guide": self._resolve_media_path(video_guide),
            "video_mask": self._resolve_media_path(video_mask),
            "video_prompt_type": video_prompt_type,
            "control_net_weight": control_net_weight,
            "control_net_weight2": control_net_weight2,
            "embedded_guidance_scale": embedded_guidance_scale,
            "lora_names": lora_names,
            "lora_multipliers": lora_multipliers,
            "negative_prompt": negative_prompt,
            "batch_size": batch_size,
        }
        
        # Only include explicitly passed parameters (non-None values)
        for param, value in param_values.items():
            if value is not None:
                task_explicit_params[param] = value
        
        # Add any additional kwargs
        task_explicit_params.update(kwargs)

        # Initialize LoRA variables (needed for both modes)
        activated_loras = []
        loras_multipliers_str = ""

        # Resolve final parameters with proper precedence (skip in passthrough mode)
        if self.passthrough_mode:
            # In passthrough mode, use task parameters directly without any resolution
            resolved_params = task_explicit_params.copy()
            generation_logger.info(f"[PASSTHROUGH] Using task parameters directly without resolution: {len(resolved_params)} params")
        else:
            resolved_params = self._resolve_parameters(effective_model_type, task_explicit_params)
            generation_logger.info(f"[PARAM_RESOLUTION] Final parameters for '{effective_model_type}': num_inference_steps={resolved_params.get('num_inference_steps')}, guidance_scale={resolved_params.get('guidance_scale')}")

        # Determine model types for generation
        base_model_type = self._get_base_model_type(self.current_model)
        vace_test_result = self._test_vace_module(self.current_model)
        
        is_vace = self._is_vace()
        is_flux = self._is_flux()
        is_qwen = self._is_qwen()
        is_t2v = self._is_t2v()
        
        generation_logger.debug(f"Model detection - VACE: {is_vace}, Flux: {is_flux}, T2V: {is_t2v}")
        generation_logger.debug(f"Generation parameters - prompt: '{prompt[:50]}...', resolution: {resolution}, length: {video_length}")
        
        if is_vace:
            # Ensure sane defaults for required VACE controls
            if not video_prompt_type:
                video_prompt_type = "VP"
            if control_net_weight is None:
                control_net_weight = 1.0
            if control_net_weight2 is None:
                control_net_weight2 = 1.0
            generation_logger.debug(f"VACE parameters - guide: {video_guide}, type: {video_prompt_type}, weights: {control_net_weight}/{control_net_weight2}")

        # Validate model-specific requirements
        # Resolve media paths before validation to ensure correct location is used
        video_guide = self._resolve_media_path(video_guide)
        video_mask = self._resolve_media_path(video_mask)

        if is_vace and not video_guide:
            raise ValueError("VACE models require video_guide parameter")

        # Extract resolved parameters
        final_video_length = resolved_params.get("video_length", 49)
        final_batch_size = resolved_params.get("batch_size", 1)
        final_guidance_scale = resolved_params.get("guidance_scale", 7.5)
        final_embedded_guidance = resolved_params.get("embedded_guidance_scale", 3.0)

        # JSON parameters pass through directly to WGP - no processing needed
        
        # Configure model-specific parameters
        if is_flux:
            image_mode = 1
            # For Flux, video_length means number of images
            actual_video_length = 1
            actual_batch_size = final_video_length
            # Use embedded guidance for Flux
            actual_guidance = final_embedded_guidance
        elif is_qwen:
            # Qwen is an image model
            image_mode = 1
            actual_video_length = 1
            actual_batch_size = resolved_params.get("batch_size", 1)
            actual_guidance = final_guidance_scale
        else:
            image_mode = 0
            actual_video_length = final_video_length
            actual_batch_size = final_batch_size
            actual_guidance = final_guidance_scale

        # Set up VACE parameters
        # Only disable VACE if model doesn't support it AND no VACE params were provided
        if not is_vace and not video_guide and not video_mask:
            video_guide = None
            video_mask = None
            video_prompt_type = "disabled"
            control_net_weight = 0.0
            control_net_weight2 = 0.0

        # Check if we're in JSON passthrough mode - don't process anything, just pass the JSON
        # This happens when --passthrough flag is used or when we have a full model definition
        is_passthrough_mode = self.passthrough_mode

        if is_passthrough_mode:
            # Complete passthrough mode: bypass ALL parameter processing
            generation_logger.info("Using JSON passthrough mode - ALL parameters pass through directly from JSON")

            # Don't set LoRA parameters - let WGP extract them from model JSON via get_transformer_loras()
            # WGP will call get_transformer_loras(model_type) to get LoRAs from the temp model file
        else:
            # Normal mode: process LoRAs from loras/loras_multipliers_list
            # Filter out LoRAs that are not present on disk to avoid hard failures in offline setups.
            filtered_multipliers: List[float] = []
            if lora_names:
                try:
                    import wgp as _w
                    from pathlib import Path as _P
                    default_dir = _P(_w.get_lora_dir(self.current_model))
                    # Common alternate directory used by tests for Qwen LoRAs
                    alt_dir = default_dir.parent / "loras_qwen"
                    cand_dirs = [default_dir]
                    if alt_dir not in cand_dirs:
                        cand_dirs.append(alt_dir)
                    # Also include a plain "loras" at Wan root in case get_lora_dir changes working dir
                    wan_root_loras = _P(self.wan_root) / "loras"
                    if wan_root_loras not in cand_dirs:
                        cand_dirs.append(wan_root_loras)
                    # And include the dedicated Qwen LoRA directory at Wan root used by tests
                    wan_root_loras_qwen = _P(self.wan_root) / "loras_qwen"
                    if wan_root_loras_qwen not in cand_dirs:
                        cand_dirs.append(wan_root_loras_qwen)

                    def _exists_any(name: str) -> bool:
                        try:
                            npath = _P(name)
                            # Absolute path provided
                            if npath.is_absolute() and npath.exists():
                                return True
                            # Relative to Wan root
                            if (_P(self.wan_root) / npath).exists():
                                return True
                            # Join with candidate dirs (full name and basename)
                            for d in cand_dirs:
                                if (d / npath).exists() or (d / npath.name).exists():
                                    return True
                        except Exception:
                            pass
                        return False

                    def _resolve_first(name: str) -> str:
                        """Return absolute path to the first matching file among candidates."""
                        npath = _P(name)
                        try:
                            if npath.is_absolute() and npath.exists():
                                return str(npath.resolve())
                        except Exception:
                            pass
                        # Relative to Wan root
                        try:
                            p = (_P(self.wan_root) / npath)
                            if p.exists():
                                return str(p.resolve())
                        except Exception:
                            pass
                        # Search candidate dirs
                        for d in cand_dirs:
                            for candidate in (d / npath, d / npath.name):
                                try:
                                    if candidate.exists():
                                        return str(candidate.resolve())
                                except Exception:
                                    pass
                        # Fallback to original name
                        return name

                    for idx, name in enumerate(lora_names):
                        # Skip empty URLs from phase_config (phases without LoRAs)
                        if not name or not name.strip():
                            generation_logger.info(
                                f"Skipping empty LoRA URL at index {idx} (phase without LoRA)"
                            )
                            continue
                        if _exists_any(name):
                            # Use absolute path so WGP join does not override it
                            resolved = _resolve_first(name)
                            activated_loras.append(resolved)
                            if lora_multipliers and idx < len(lora_multipliers):
                                filtered_multipliers.append(lora_multipliers[idx])
                        else:
                            # Try to download from model config URL before skipping
                            try:
                                # Get the LoRA URLs from the model config
                                model_loras = task.parameters.get("model", {}).get("loras", [])
                                lora_url = None
                                if idx < len(model_loras):
                                    lora_url = model_loras[idx]

                                if lora_url and lora_url.startswith("http"):
                                    generation_logger.info(
                                        f"LoRA not found locally, attempting download from URL: '{name}' from {lora_url}"
                                    )

                                    # Use wget to download the file
                                    import subprocess
                                    import os

                                    # Ensure loras directory exists
                                    loras_dir = "loras"
                                    os.makedirs(loras_dir, exist_ok=True)

                                    # Download file
                                    local_path = os.path.join(loras_dir, name)
                                    result = subprocess.run(
                                        ["wget", "-O", local_path, lora_url],
                                        capture_output=True,
                                        text=True
                                    )

                                    if result.returncode == 0 and _exists_any(name):
                                        resolved = _resolve_first(name)
                                        activated_loras.append(resolved)
                                        if lora_multipliers and idx < len(lora_multipliers):
                                            filtered_multipliers.append(lora_multipliers[idx])
                                        generation_logger.info(
                                            f"Successfully downloaded and activated LoRA: '{name}'"
                                        )
                                    else:
                                        generation_logger.warning(
                                            f"Failed to download LoRA from {lora_url}: {result.stderr}"
                                        )
                                else:
                                    generation_logger.warning(
                                        f"LoRA not found and no valid URL provided for download: '{name}'"
                                    )

                            except Exception as e:
                                generation_logger.warning(
                                    f"LoRA not found and download failed with error: '{name}' - {e}"
                                )
                    if lora_names and not activated_loras:
                        generation_logger.info(
                            "No requested LoRAs were found in known directories; proceeding without LoRAs."
                        )
                except Exception as _e:
                    # On any error, fall back to original list to avoid accidental removal
                    generation_logger.warning(f"LoRA filtering failed; using provided list as-is: {_e}")
                    activated_loras = lora_names.copy()
                    filtered_multipliers = (lora_multipliers or []).copy()
            # WGP expects loras_multipliers as string, not list
            # Detect phase-config format (contains semicolons) vs regular format
            multipliers_to_convert = filtered_multipliers if filtered_multipliers else (lora_multipliers or [])
            if multipliers_to_convert:
                # Check if phase-config format (any multiplier contains semicolon)
                is_phase_config = any(";" in str(m) for m in multipliers_to_convert)
                if is_phase_config:
                    # Phase-config format: ["1.0;0", "0;1.0"] → space-separated "1.0;0 0;1.0"
                    loras_multipliers_str = " ".join(str(m) for m in multipliers_to_convert)
                else:
                    # Regular format: [1.0, 0.8] → comma-separated "1.0,0.8"
                    loras_multipliers_str = ",".join(str(m) for m in multipliers_to_convert)
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

        if is_passthrough_mode:
            # COMPLETE PASSTHROUGH MODE: Pass ALL parameters from JSON with required defaults
            wgp_params = {
                # Core parameters (fixed, not overridable)
                'task': task,
                'send_cmd': send_cmd,
                'state': self.state,
                'model_type': self.current_model,
                'image_mode': image_mode,

                # Required parameters with defaults
                'prompt': resolved_params.get('prompt', ''),
                'negative_prompt': resolved_params.get('negative_prompt', ''),
                'resolution': resolved_params.get('resolution', '1280x720'),
                'video_length': resolved_params.get('video_length', 81),
                'batch_size': resolved_params.get('batch_size', 1),
                'seed': resolved_params.get('seed', 42),
                'force_fps': 'auto',

                # VACE control parameters
                'video_guide': video_guide,
                'video_mask': video_mask,
                'video_guide2': None,
                'video_mask2': None,
                'video_prompt_type': video_prompt_type or 'VM',
                'control_net_weight': control_net_weight or 1.0,
                'control_net_weight2': control_net_weight2 or 1.0,
                'denoising_strength': 1.0,

                # LoRA parameters
                'activated_loras': [],
                'loras_multipliers': '',

                # Audio parameters
                'audio_guidance_scale': 1.0,
                'embedded_guidance_scale': resolved_params.get('embedded_guidance_scale', 0.0),
                'repeat_generation': 1,
                'multi_prompts_gen_type': 0,
                'multi_images_gen_type': 0,
                'skip_steps_cache_type': '',
                'skip_steps_multiplier': 1.0,
                'skip_steps_start_step_perc': 0.0,

                # Image parameters
                'image_prompt_type': 'disabled',
                'image_start': None,
                'image_end': None,
                'model_mode': 0,
                'video_source': None,
                'keep_frames_video_source': '',
                'image_refs': None,
                'frames_positions': '',
                'image_guide': None,
                'keep_frames_video_guide': '',
                'video_guide_outpainting': '0 0 0 0',
                'image_mask': None,
                'mask_expand': 0,

                # Audio parameters
                'audio_guide': None,
                'audio_guide2': None,
                'audio_source': None,
                'audio_prompt_type': '',
                'speakers_locations': '',

                # Sliding window parameters
                'sliding_window_size': 129,
                'sliding_window_overlap': 0,
                'sliding_window_color_correction_strength': 0.0,
                'sliding_window_overlap_noise': 0.1,
                'sliding_window_discard_last_frames': 0,
                'image_refs_relative_size': 50,

                # Post-processing parameters
                'remove_background_images_ref': 0,
                'temporal_upsampling': '',
                'spatial_upsampling': '',
                'film_grain_intensity': 0.0,
                'film_grain_saturation': 0.0,
                'MMAudio_setting': 0,
                'MMAudio_prompt': '',
                'MMAudio_neg_prompt': '',

                # Advanced parameters
                'RIFLEx_setting': 0,
                'NAG_scale': 0.0,
                'NAG_tau': 1.0,
                'NAG_alpha': 0.0,
                'slg_switch': 0,
                'slg_layers': '',
                'slg_start_perc': 0.0,
                'slg_end_perc': 100.0,
                'apg_switch': 0,
                'cfg_star_switch': 0,
                'cfg_zero_step': 0,
                'prompt_enhancer': 0,
                'min_frames_if_references': 9,
                'override_profile': -1,

                # Mode and filename
                'mode': 'generate',
                'model_filename': '',
            }

            # Override with ALL parameters from JSON (this preserves your exact JSON values)
            for param_key, param_value in resolved_params.items():
                if param_key not in ['task', 'send_cmd', 'state', 'model_type']:  # Don't override core system params
                    wgp_params[param_key] = param_value
                    if param_key == 'guidance2_scale':
                        generation_logger.info(f"[PASSTHROUGH_DEBUG] Setting {param_key} = {param_value} from JSON")

            # Debug: Check final guidance2_scale value before WGP call
            generation_logger.info(f"[PASSTHROUGH_DEBUG] Final wgp_params guidance2_scale = {wgp_params.get('guidance2_scale', 'NOT_SET')}")

        else:
            # Build parameter dictionary from resolved parameters
            # Supply defaults for required WGP args that may be unused depending on phases/model
            guidance3_scale_value = resolved_params.get(
                "guidance3_scale",
                resolved_params.get("guidance2_scale", actual_guidance),
            )
            switch_threshold2_value = resolved_params.get("switch_threshold2", 0)
            guidance_phases_value = resolved_params.get("guidance_phases", 1)
            model_switch_phase_value = resolved_params.get("model_switch_phase", 1)
            image_refs_relative_size_value = resolved_params.get("image_refs_relative_size", 50)
            override_profile_value = resolved_params.get("override_profile", -1)

            wgp_params = {
                # Core parameters (fixed, not overridable)
                'task': task,
                'send_cmd': send_cmd,
                'state': self.state,
                'model_type': self.current_model,
                'prompt': resolved_params.get("prompt", prompt),
                'negative_prompt': resolved_params.get("negative_prompt", ""),
                'resolution': resolved_params.get("resolution", "1280x720"),
                'video_length': actual_video_length,
                'batch_size': actual_batch_size,
                'seed': resolved_params.get("seed", 42),
                'force_fps': "auto",
                'image_mode': image_mode,

                # VACE control parameters (only pass supported fields upstream)
                'video_guide': video_guide,
                'video_mask': video_mask,
                'video_guide2': None,
                'video_mask2': None,
                'video_prompt_type': video_prompt_type,
                'control_net_weight': control_net_weight,
                'control_net_weight2': control_net_weight2,
                'denoising_strength': 1.0,

                # LoRA parameters in normal mode
                'activated_loras': activated_loras,
                'loras_multipliers': loras_multipliers_str,

                # Overridable parameters from resolved configuration
                'num_inference_steps': resolved_params.get("num_inference_steps", 25),
                'guidance_scale': actual_guidance,
                'guidance2_scale': resolved_params.get("guidance2_scale", actual_guidance),
                'guidance3_scale': guidance3_scale_value,
                'switch_threshold': resolved_params.get("switch_threshold", 500),
                'switch_threshold2': switch_threshold2_value,
                'guidance_phases': guidance_phases_value,
                'model_switch_phase': model_switch_phase_value,
                'embedded_guidance_scale': final_embedded_guidance if is_flux else 0.0,
                'flow_shift': resolved_params.get("flow_shift", 7.0),
                'sample_solver': resolved_params.get("sample_solver", "euler"),
            }

            # Standard defaults for other parameters - extend the dictionary
            wgp_params.update({
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
            'image_refs': None,
            'frames_positions': "",
            'image_guide': None,
            'image_mask': None,
            
            # Video parameters
            'model_mode': 0,
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
            'image_refs_relative_size': image_refs_relative_size_value,
            
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
            'override_profile': override_profile_value,
            
            # Mode and filename
            'mode': "generate",
            'model_filename': "",
        })
        


        # Override ANY parameter provided in kwargs
        # This allows complete customization of generation parameters
        for key, value in kwargs.items():
            if key in wgp_params:
                generation_logger.debug(f"Overriding parameter from kwargs: {key}={wgp_params[key]} -> {value}")
                wgp_params[key] = value
            else:
                generation_logger.debug(f"Adding new parameter from kwargs: {key}={value}")
                wgp_params[key] = value

        # Generate content type description
        content_type = "images" if (is_flux or is_qwen) else "video"
        model_type_desc = (
            "Flux" if is_flux else ("Qwen" if is_qwen else ("VACE" if is_vace else "T2V"))
        )
        count_desc = f"{video_length} {'images' if is_flux else 'frames'}"
        
        generation_logger.essential(f"Generating {model_type_desc} {content_type}: {resolution}, {count_desc}")
        if is_vace:
            encodings = [c for c in video_prompt_type if c in "VPDSLCMUA"]
            generation_logger.debug(f"VACE encodings: {encodings}")
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
            if is_passthrough_mode:
                generation_logger.info(f"[CausVidDebugTrace]   guidance2_scale: {wgp_params.get('guidance2_scale', 'NOT_SET')} (passthrough)")
            else:
                generation_logger.info(f"[CausVidDebugTrace]   guidance2_scale: {wgp_params.get('guidance2_scale', 'NOT_SET')} (normal)")
            generation_logger.info(f"[CausVidDebugTrace]   activated_loras: {activated_loras}")
            generation_logger.info(f"[CausVidDebugTrace]   loras_multipliers_str: {loras_multipliers_str}")
            
            # ARCHITECTURAL FIX: Pre-populate WGP UI state for LoRA compatibility
            # WGP was designed for UI usage where state["loras"] gets populated through UI
            # In headless mode, we need to pre-populate this state to ensure LoRA loading works
            original_loras = self.state.get("loras", [])
            if activated_loras and len(activated_loras) > 0:
                generation_logger.info(f"[CausVidDebugTrace] WanOrchestrator: Pre-populating WGP state with {len(activated_loras)} LoRAs")
                generation_logger.info(f"[LORA_APPLICATION_TRACE] ═══════════════════════════════════════════════════════")
                generation_logger.info(f"[LORA_APPLICATION_TRACE] COMPLETE LoRA APPLICATION BREAKDOWN:")
                generation_logger.info(f"[LORA_APPLICATION_TRACE] ═══════════════════════════════════════════════════════")

                # Parse multipliers string to show per-LoRA breakdown
                multiplier_list = loras_multipliers_str.split() if loras_multipliers_str else []

                for idx, lora_path in enumerate(activated_loras):
                    # Extract filename from path (handles both Unix and Windows paths)
                    lora_filename = str(lora_path).split('/')[-1].split('\\')[-1]
                    mult_str = multiplier_list[idx] if idx < len(multiplier_list) else "1.0"
                    phases = mult_str.split(";")

                    generation_logger.info(f"[LORA_APPLICATION_TRACE]")
                    generation_logger.info(f"[LORA_APPLICATION_TRACE] LoRA #{idx+1}: {lora_filename}")
                    generation_logger.info(f"[LORA_APPLICATION_TRACE]   Full path: {lora_path}")
                    generation_logger.info(f"[LORA_APPLICATION_TRACE]   Multiplier string: {mult_str}")
                    generation_logger.info(f"[LORA_APPLICATION_TRACE]   Phase breakdown:")

                    if len(phases) == 1:
                        generation_logger.info(f"[LORA_APPLICATION_TRACE]     - All phases: {phases[0]} (constant strength)")
                    else:
                        generation_logger.info(f"[LORA_APPLICATION_TRACE]     - Phase 1 (steps 0-1): {phases[0] if len(phases) > 0 else '1.0'}")
                        generation_logger.info(f"[LORA_APPLICATION_TRACE]     - Phase 2 (steps 2-3): {phases[1] if len(phases) > 1 else '1.0'}")
                        generation_logger.info(f"[LORA_APPLICATION_TRACE]     - Phase 3 (steps 4-5): {phases[2] if len(phases) > 2 else '1.0'}")

                generation_logger.info(f"[LORA_APPLICATION_TRACE]")
                generation_logger.info(f"[LORA_APPLICATION_TRACE] SUMMARY:")
                generation_logger.info(f"[LORA_APPLICATION_TRACE]   Total LoRAs: {len(activated_loras)}")
                generation_logger.info(f"[LORA_APPLICATION_TRACE]   Model config LoRAs: {len(activated_loras) - 1 if lora_names and len(lora_names) > 0 else len(activated_loras)}")
                generation_logger.info(f"[LORA_APPLICATION_TRACE]   amount_of_motion LoRA: {'Yes' if lora_names and len(lora_names) > 0 else 'No'}")
                generation_logger.info(f"[LORA_APPLICATION_TRACE] ═══════════════════════════════════════════════════════")

                self.state["loras"] = activated_loras.copy()  # Populate UI state for WGP compatibility
                generation_logger.debug(f"[CausVidDebugTrace] WanOrchestrator: state['loras'] = {self.state['loras']}")
            
            try:
                # Call the VACE-fixed generate_video with the unified parameter dictionary
                # IMPORTANT: Do not pass unsupported keys upstream
                # Pre-initialize WGP process status to avoid None in early callback
                try:
                    if isinstance(self.state.get("gen"), dict):
                        self.state["gen"]["process_status"] = "process:main"
                except Exception:
                    pass
                # For image-based models, load PIL images instead of passing paths
                if is_qwen or image_mode == 1:
                    if wgp_params.get('image_guide') and isinstance(wgp_params['image_guide'], str):
                        wgp_params['image_guide'] = self._load_image(wgp_params['image_guide'], mask=False)
                    if wgp_params.get('image_mask') and isinstance(wgp_params['image_mask'], str):
                        wgp_params['image_mask'] = self._load_image(wgp_params['image_mask'], mask=True)

                    # Ensure proper parameter coordination for Qwen models
                    if is_qwen:
                        # Ensure image_mask is truly None if not provided, not empty string or other falsy value
                        if not wgp_params.get('image_mask'):
                            wgp_params['image_mask'] = None
                            generation_logger.debug("[PREFLIGHT] Ensured image_mask=None for Qwen regular generation")
                        else:
                            # If there IS a mask, set model_mode=1 for inpainting
                            wgp_params['model_mode'] = 1
                            generation_logger.info("[PREFLIGHT] Set model_mode=1 for Qwen inpainting (image_mask present)")
                    # Preflight logs for image models
                    try:
                        ig = wgp_params.get('image_guide')
                        if ig is not None:
                            from PIL import Image as _PILImage  # type: ignore
                            if isinstance(ig, _PILImage.Image):
                                generation_logger.info("[PREFLIGHT] image_guide resolved to PIL.Image with size %sx%s and mode %s" % (ig.size[0], ig.size[1], ig.mode))
                            else:
                                generation_logger.warning(f"[PREFLIGHT] image_guide is not a PIL image (type={type(ig)})")
                        else:
                            generation_logger.warning("[PREFLIGHT] image_guide is None for image model")
                    except Exception as _e:
                        generation_logger.warning(f"[PREFLIGHT] Could not inspect image_guide: {_e}")
                # Sanitize image_refs: WGP expects None when there are no refs
                try:
                    if isinstance(wgp_params.get('image_refs'), list) and len(wgp_params['image_refs']) == 0:
                        wgp_params['image_refs'] = None
                except Exception:
                    pass

                # Filter out unsupported parameters to match upstream signature exactly
                try:
                    import inspect as _inspect
                    import wgp as _wgp  # use the upstream function, not our wrapper
                    _sig = _inspect.signature(_wgp.generate_video)
                    _allowed = set(_sig.parameters.keys())
                    _filtered_params = {k: v for k, v in wgp_params.items() if k in _allowed}
                    _dropped = sorted(set(wgp_params.keys()) - _allowed)
                    if _dropped:
                        generation_logger.debug(f"[PARAM_SANITIZE] Dropping unsupported params: {_dropped}")
                except Exception as _e:
                    # On any error, fall back to original params
                    _filtered_params = wgp_params

                # COMPREHENSIVE LOGGING: Show all final parameters being sent to WGP
                generation_logger.info("=" * 80)
                generation_logger.info("FINAL PARAMETERS BEING SENT TO WGP generate_video():")
                generation_logger.info("=" * 80)
                for key, value in sorted(_filtered_params.items()):
                    # Skip or truncate extremely verbose parameters
                    if key == 'state':
                        # State dict is huge - just show key info
                        state_summary = {
                            'model_type': value.get('model_type'),
                            'gen_file_count': len(value.get('gen', {}).get('file_list', [])),
                            'loras_count': len(value.get('loras', []))
                        }
                        generation_logger.info(f"[FINAL_PARAMS] {key}: {state_summary} (truncated)")
                    elif key in ['guidance_scale', 'guidance2_scale', 'guidance3_scale', 'num_inference_steps', 'switch_threshold', 'switch_threshold2']:
                        generation_logger.info(f"[FINAL_PARAMS] {key}: {value} ⭐")  # Star important guidance params
                    else:
                        generation_logger.info(f"[FINAL_PARAMS] {key}: {value}")
                generation_logger.info("=" * 80)

                result = self._generate_video(**_filtered_params)

                # Verify directory after generation (wgp may have changed it during file operations)
                _verify_wgp_directory(generation_logger, "after wgp.generate_video()")

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

            # Memory monitoring
            try:
                import torch
                import psutil

                # RAM usage
                ram = psutil.virtual_memory()
                ram_used_gb = ram.used / (1024**3)
                ram_total_gb = ram.total / (1024**3)
                ram_percent = ram.percent

                # VRAM usage
                if torch.cuda.is_available():
                    vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    vram_percent = (vram_reserved / vram_total) * 100

                    generation_logger.essential(f"💾 RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram_percent:.0f}%) | VRAM: {vram_reserved:.1f}GB / {vram_total:.1f}GB ({vram_percent:.0f}%) [Allocated: {vram_allocated:.1f}GB]")
                else:
                    generation_logger.essential(f"💾 RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram_percent:.0f}%)")
            except Exception as e:
                generation_logger.debug(f"Could not retrieve memory stats: {e}")

            return output_path
            
        except Exception as e:
            generation_logger.error(f"Generation failed: {e}")
            raise

    # Convenience methods for specific generation types
    
    def generate_t2v(self, prompt: str, model_type: str = None, **kwargs) -> str:
        """Generate text-to-video content."""
        if not self._is_t2v():
            generation_logger.warning(f"Current model {self.current_model} may not be optimized for T2V")
        return self.generate(prompt=prompt, model_type=model_type, **kwargs)
    
    def generate_vace(self, 
                     prompt: str, 
                     video_guide: str,
                     model_type: str = None,
                     video_mask: Optional[str] = None,
                     video_prompt_type: str = "VP",
                     control_net_weight: float = 1.0,
                     control_net_weight2: float = 1.0,
                     **kwargs) -> str:
        """Generate VACE controlled video content.
        
        Args:
            prompt: Text prompt for generation
            video_guide: Path to primary control video (required)
            video_mask: Path to primary mask video (optional)
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
            model_type=model_type,
            video_guide=video_guide,
            video_mask=video_mask,
            video_prompt_type=video_prompt_type,
            control_net_weight=control_net_weight,
            control_net_weight2=control_net_weight2,
            **kwargs  # Any additional parameters including switch_threshold
        )

    def generate_flux(self, prompt: str, images: int = 4, model_type: str = None, **kwargs) -> str:
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
            model_type=model_type,
            video_length=images,  # For Flux, video_length = number of images
            **kwargs
        )

# Backward compatibility
WanContentOrchestrator = WanOrchestrator
