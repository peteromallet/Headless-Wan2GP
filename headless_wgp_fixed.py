# Fixed version of headless_wgp.py with proper VACE handling

import os
import sys
import json
import threading
import time
from typing import Dict, Any, Optional, List, Tuple

# ... (all the imports and class definition would be here, but for now just the key fix)

class WanOrchestrator:
    def __init__(self):
        self.current_model = None
        self.offloadobj = None
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
            "validate_success": 0,
            "apply_success": 0,
            "refresh": 0,
            "all_settings": {},
            "image_mode_tab": 0,
            "prompt": ""
        }

    def _test_vace_module(self, model_key: str) -> bool:
        """Test if a model is a VACE model by checking its base model type."""
        import wgp
        base_model_type = wgp.get_base_model_type(model_key)
        return base_model_type in ['vace_14B', 'vace_1.3B', 'vace_multitalk_14B']

    def load_model(self, model_key: str):
        """Load a model using WGP's proper loading flow.
        
        This is the CORRECT approach - let load_models() handle everything:
        - Building the complete model file list (base + modules)
        - Module discovery and loading
        - LoRA setup
        """
        import wgp
        
        print(f"🔄 Loading model: {model_key}")
        
        # For VACE models, ensure config.json exists in ckpts/ directory
        if self._test_vace_module(model_key):
            print(f"🔧 [VACE_CONFIG] Setting up config.json for VACE model '{model_key}'")
            
            wgp_dir = os.path.dirname(os.path.abspath(wgp.__file__))
            ckpts_dir = os.path.join(wgp_dir, "ckpts")
            config_target = os.path.join(ckpts_dir, "config.json")
            
            if not os.path.exists(config_target):
                print(f"🔧 [VACE_CONFIG] Creating config.json at: {config_target}")
                
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
                
                os.makedirs(ckpts_dir, exist_ok=True)
                with open(config_target, 'w') as f:
                    json.dump(vace_config, f, indent=2)
                print(f"🔧 [VACE_CONFIG] ✅ Created config.json successfully")
            else:
                print(f"🔧 [VACE_CONFIG] ✅ Config.json already exists")

        # Debug: Check module discovery BEFORE calling load_models
        modules = wgp.get_model_recursive_prop(model_key, "modules", return_list=True)
        print(f"🔧 [MODULE_DISCOVERY] Modules found for '{model_key}': {modules}")
        
        model_def = wgp.get_model_def(model_key)
        print(f"🔧 [MODEL_DEF] Model definition: {model_def}")
        
        # Let WGP handle the complete loading process
        # This will build the proper file list with modules internally
        loras, loras_names, _, _, _, _, _ = wgp.setup_loras(model_key, None, wgp.get_lora_dir(model_key), '', None)
        print(f"🎨 Discovered {len(loras)} LoRAs: {loras_names}")
        
        # This is the KEY call - it builds the complete model+module file list internally
        wan_model, self.offloadobj = wgp.load_models(model_key)
        
        self.current_model = model_key
        self.state["model_type"] = model_key
        
        # Store globally for WGP
        wgp.wan_model = wan_model
        wgp.offloadobj = self.offloadobj
        
        print(f"✅ Model '{model_key}' loaded successfully")
        return True

# Key insight: The issue was that we were calling get_model_filename() BEFORE load_models()
# This returned only the base model file, not the complete list with modules.
# 
# The correct flow is:
# 1. Let load_models() handle building the complete file list internally
# 2. It calls get_model_recursive_prop() to find modules  
# 3. It builds model_file_list = [base_model] + [module1, module2, ...]
# 4. It loads everything together
#
# We should NOT be calling get_model_filename() ourselves!