import argparse
import sys
import os
import types
from pathlib import Path
from PIL import Image
import json
import time
# import shutil # No longer moving files, tasks are removed from tasks.json
import traceback
import requests # For downloading the LoRA
import inspect # Added import

# -----------------------------------------------------------------------------
# 1. Parse arguments for the server
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("WanGP Headless Server (Single tasks.json poller)")

    pgroup_server = parser.add_argument_group("Server Settings")
    pgroup_server.add_argument("--tasks-file", type=str, required=True,
                               help="Path to the tasks.json file to monitor.")
    pgroup_server.add_argument("--main-output-dir", type=str, default="./headless_outputs",
                               help="Base directory where outputs for each task will be saved (in subdirectories)")
    pgroup_server.add_argument("--poll-interval", type=int, default=10,
                               help="How often (in seconds) to check tasks.json for new tasks.")

    # Advanced wgp.py Global Config Overrides (Optional) - Applied once at server start
    pgroup_wgp_globals = parser.add_argument_group("WGP Global Config Overrides (Applied at Server Start)")
    pgroup_wgp_globals.add_argument("--wgp-attention-mode", type=str, default=None,
                                choices=["auto", "sdpa", "sage", "sage2", "flash", "xformers"])
    pgroup_wgp_globals.add_argument("--wgp-compile", type=str, default=None, choices=["", "transformer"])
    pgroup_wgp_globals.add_argument("--wgp-profile", type=int, default=None)
    pgroup_wgp_globals.add_argument("--wgp-vae-config", type=int, default=None)
    pgroup_wgp_globals.add_argument("--wgp-boost", type=int, default=None)
    pgroup_wgp_globals.add_argument("--wgp-transformer-quantization", type=str, default=None, choices=["int8", "bf16"])
    pgroup_wgp_globals.add_argument("--wgp-transformer-dtype-policy", type=str, default=None, choices=["", "fp16", "bf16"])
    pgroup_wgp_globals.add_argument("--wgp-text-encoder-quantization", type=str, default=None, choices=["int8", "bf16"])
    pgroup_wgp_globals.add_argument("--wgp-vae-precision", type=str, default=None, choices=["16", "32"])
    pgroup_wgp_globals.add_argument("--wgp-mixed-precision", type=str, default=None, choices=["0", "1"])
    pgroup_wgp_globals.add_argument("--wgp-preload-policy", type=str, default=None,
                                help="Set wgp.py's preload_model_policy (e.g., 'P,S' or 'P'. Avoid 'U' to keep models loaded longer).")

    return parser.parse_args()

# -----------------------------------------------------------------------------
# 2. Gradio Monkey-Patching (same as before)
# -----------------------------------------------------------------------------
def patch_gradio():
    import gradio as gr
    gr.Info = lambda msg: print(f"[INFO] {msg}")
    gr.Warning = lambda msg: print(f"[WARNING] {msg}")
    class _GrError(RuntimeError): pass
    def _raise(msg, *a, **k): raise _GrError(msg)
    gr.Error = _raise
    class _Dummy:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return None
        def update(self, **kwargs): return None
    dummy_attrs = ["HTML", "DataFrame", "Gallery", "Button", "Row", "Column", "Accordion", "Progress", "Dropdown", "Slider", "Textbox", "Checkbox", "Radio", "Image", "Video", "Audio", "DownloadButton", "UploadButton", "Markdown", "Tabs", "State", "Text", "Number"]
    for attr in dummy_attrs:
        setattr(gr, attr, _Dummy)
    gr.update = lambda *a, **k: None
    def dummy_event_handler(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    gr.on = dummy_event_handler
    gr.Request = type('Request', (object,), {'client': type('Client', (object,), {'host': 'localhost'})})
    gr.SelectData = type('SelectData', (), {'index': None, '_data': None})
    gr.EventData = type('EventData', (), {'target':None, '_data':None})

# -----------------------------------------------------------------------------
# 3. Minimal send_cmd implementation (task_id instead of task_name)
# -----------------------------------------------------------------------------
def make_send_cmd(task_id):
    def _send(cmd, data=None):
        prefix = f"[Task ID: {task_id}]"
        if cmd == "progress":
            if isinstance(data, list) and len(data) >= 2:
                prog, txt = data[0], data[1]
                if isinstance(prog, tuple) and len(prog) == 2: step, total = prog; print(f"{prefix}[Progress] {step}/{total} – {txt}")
                else: print(f"{prefix}[Progress] {txt}")
        elif cmd == "status": print(f"{prefix}[Status] {data}")
        elif cmd == "info": print(f"{prefix}[INFO] {data}")
        elif cmd == "error": print(f"{prefix}[ERROR] {data}"); raise RuntimeError(f"wgp.py error for {task_id}: {data}")
        elif cmd == "output": print(f"{prefix}[Output] video written.")
    
    print(f"DEBUG: Signature of _send for task {task_id}: {inspect.signature(_send)}") # Added diagnostic print
    return _send

# -----------------------------------------------------------------------------
# 4. State builder for a single task (same as before)
# -----------------------------------------------------------------------------
def build_task_state(wgp_mod, model_filename, task_params_dict, all_loras_for_model):
    state = {
        "model_filename": model_filename,
        "validate_success": 1,
        "advanced": True,
        "gen": {"queue": [], "file_list": [], "prompt_no": 1, "prompts_max": 1},
        "loras": all_loras_for_model,
    }
    model_type_key = wgp_mod.get_model_type(model_filename)
    ui_defaults = wgp_mod.get_default_settings(model_filename).copy()

    # Override with task_params from JSON, but preserve some crucial ones if CausVid is used
    causvid_active = task_params_dict.get("use_causvid_lora", False)

    for key, value in task_params_dict.items():
        if key not in ["output_sub_dir", "model", "task_id", "use_causvid_lora"]:
            if causvid_active and key in ["steps", "guidance_scale", "flow_shift", "activated_loras", "loras_multipliers"]:
                continue # These will be set by causvid logic if flag is true
            ui_defaults[key] = value
    
    ui_defaults["prompt"] = task_params_dict.get("prompt", "Default prompt")
    ui_defaults["resolution"] = task_params_dict.get("resolution", "832x480")
    # Allow task to specify frames/video_length, steps, guidance_scale, flow_shift unless overridden by CausVid
    if not causvid_active:
        ui_defaults["video_length"] = task_params_dict.get("frames", task_params_dict.get("video_length", 81))
        ui_defaults["num_inference_steps"] = task_params_dict.get("steps", task_params_dict.get("num_inference_steps", 30))
        ui_defaults["guidance_scale"] = task_params_dict.get("guidance_scale", ui_defaults.get("guidance_scale", 5.0))
        ui_defaults["flow_shift"] = task_params_dict.get("flow_shift", ui_defaults.get("flow_shift", 3.0))
    else: # CausVid specific defaults if not touched by its logic yet
        ui_defaults["video_length"] = task_params_dict.get("frames", task_params_dict.get("video_length", 81))
        # steps, guidance_scale, flow_shift will be set below by causvid logic

    ui_defaults["seed"] = task_params_dict.get("seed", -1)
    ui_defaults["lset_name"] = "" 

    def load_pil_images(paths_list_or_str, wgp_convert_func):
        if paths_list_or_str is None: return None
        paths_list = paths_list_or_str if isinstance(paths_list_or_str, list) else [paths_list_or_str]
        images = []
        for p_str in paths_list:
            p = Path(p_str.strip())
            if not p.is_file(): print(f"[WARNING] Image file not found: {p}"); continue
            try:
                img = Image.open(p)
                images.append(wgp_convert_func(img))
            except Exception as e:
                print(f"[WARNING] Failed to load image {p}: {e}")
        return images if images else None

    if task_params_dict.get("image_start_paths"):
        loaded = load_pil_images(task_params_dict["image_start_paths"], wgp_mod.convert_image)
        if loaded: ui_defaults["image_start"] = loaded
    if task_params_dict.get("image_end_paths"):
        loaded = load_pil_images(task_params_dict["image_end_paths"], wgp_mod.convert_image)
        if loaded: ui_defaults["image_end"] = loaded
    if task_params_dict.get("image_refs_paths"):
        loaded = load_pil_images(task_params_dict["image_refs_paths"], wgp_mod.convert_image)
        if loaded: ui_defaults["image_refs"] = loaded
    
    for key in ["video_source_path", "video_guide_path", "video_mask_path", "audio_guide_path"]:
        if task_params_dict.get(key):
            ui_defaults[key.replace("_path","")] = task_params_dict[key]

    if task_params_dict.get("prompt_enhancer_mode"):
        ui_defaults["prompt_enhancer"] = task_params_dict["prompt_enhancer_mode"]
        wgp_mod.server_config["enhancer_enabled"] = 1
    elif "prompt_enhancer" not in task_params_dict:
        ui_defaults["prompt_enhancer"] = ""
        wgp_mod.server_config["enhancer_enabled"] = 0

    # Apply CausVid LoRA specific settings if the flag is true
    if causvid_active:
        print(f"[Task ID: {task_params_dict.get('task_id')}] Applying CausVid LoRA settings.")
        
        # If steps are specified in the task JSON for a CausVid task, use them; otherwise, default to 9.
        if "steps" in task_params_dict:
            ui_defaults["num_inference_steps"] = task_params_dict["steps"]
            print(f"[Task ID: {task_params_dict.get('task_id')}] CausVid task using specified steps: {ui_defaults['num_inference_steps']}")
        elif "num_inference_steps" in task_params_dict:
            ui_defaults["num_inference_steps"] = task_params_dict["num_inference_steps"]
            print(f"[Task ID: {task_params_dict.get('task_id')}] CausVid task using specified num_inference_steps: {ui_defaults['num_inference_steps']}")
        else:
            ui_defaults["num_inference_steps"] = 9 # Default for CausVid if not specified in task
            print(f"[Task ID: {task_params_dict.get('task_id')}] CausVid task defaulting to steps: {ui_defaults['num_inference_steps']}")

        ui_defaults["guidance_scale"] = 1.0 # Still overridden
        ui_defaults["flow_shift"] = 1.0     # Still overridden
        
        causvid_lora_basename = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
        current_activated = ui_defaults.get("activated_loras", [])
        if not isinstance(current_activated, list):
             try: current_activated = [str(item).strip() for item in str(current_activated).split(',') if item.strip()] 
             except: current_activated = []

        if causvid_lora_basename not in current_activated:
            current_activated.append(causvid_lora_basename)
        ui_defaults["activated_loras"] = current_activated

        current_multipliers_str = ui_defaults.get("loras_multipliers", "")
        # Basic handling: if multipliers exist, prepend; otherwise, set directly.
        # More sophisticated merging might be needed if specific order or pairing is critical.
        # This assumes multipliers are space-separated.
        if current_multipliers_str:
            multipliers_list = current_multipliers_str.split()
            lora_names_list = [Path(lora_path).name for lora_path in all_loras_for_model 
                               if Path(lora_path).name in current_activated and Path(lora_path).name != causvid_lora_basename]
            
            final_multipliers = []
            final_loras = []

            # Add CausVid first
            final_loras.append(causvid_lora_basename)
            final_multipliers.append("0.7")

            # Add existing, ensuring no duplicate multiplier for already present CausVid (though it shouldn't be)
            processed_other_loras = set()
            for i, lora_name in enumerate(current_activated):
                if lora_name == causvid_lora_basename: continue # Already handled
                if lora_name not in processed_other_loras:
                    final_loras.append(lora_name)
                    if i < len(multipliers_list):
                         final_multipliers.append(multipliers_list[i])
                    else:
                         final_multipliers.append("1.0") # Default if not enough multipliers
                    processed_other_loras.add(lora_name)
            
            ui_defaults["activated_loras"] = final_loras # ensure order matches multipliers
            ui_defaults["loras_multipliers"] = " ".join(final_multipliers)
        else:
            ui_defaults["loras_multipliers"] = "0.7"
            ui_defaults["activated_loras"] = [causvid_lora_basename] # ensure only causvid if no others

    state[model_type_key] = ui_defaults
    return state, ui_defaults

# -----------------------------------------------------------------------------
# 5. Download utility
# -----------------------------------------------------------------------------
def download_file(url, dest_folder, filename):
    dest_path = Path(dest_folder) / filename
    if dest_path.exists():
        print(f"[INFO] File {filename} already exists in {dest_folder}.")
        return True
    try:
        print(f"Downloading {filename} from {url} to {dest_folder}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        if dest_path.exists(): # Attempt to clean up partial download
            try: os.remove(dest_path)
            except: pass
        return False

# -----------------------------------------------------------------------------
# 6. Process a single task dictionary from the tasks.json list
# -----------------------------------------------------------------------------

def process_single_task(wgp_mod, task_params_dict, main_output_dir):
    task_id = task_params_dict.get("task_id", "unknown_task_" + str(time.time()))
    print(f"--- Processing task ID: {task_id} ---")

    task_model_type_logical = task_params_dict.get("model", "t2v")
    # Determine the actual model filename before checking/downloading LoRA, as LoRA path depends on it
    model_filename_for_task = wgp_mod.get_model_filename(task_model_type_logical,
                                                         wgp_mod.transformer_quantization,
                                                         wgp_mod.transformer_dtype_policy)
    
    use_causvid = task_params_dict.get("use_causvid_lora", False)
    causvid_lora_basename = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
    causvid_lora_url = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

    if use_causvid:
        # Determine LoRA directory based on the *actual* model being used for the task
        # This assumes CausVid LoRA is for 14B T2V models primarily.
        # For simplicity, we use a fixed sub-path, but ideally, wgp_mod.get_lora_dir should be more granular
        # or we need to ensure the model chosen is compatible (e.g. a 14B t2v model).
        
        # Heuristic: if it's a 14B model, place it in a 14B subfolder. Otherwise, root lora dir.
        # This might need refinement based on wgp.py's exact lora dir logic for different model families/sizes.
        base_lora_dir_for_model = Path(wgp_mod.get_lora_dir(model_filename_for_task))
        target_causvid_lora_dir = base_lora_dir_for_model
        
        # Attempt to check if model is 14B for more specific LoRA path (best effort)
        if "14B" in model_filename_for_task and "t2v" in model_filename_for_task.lower():
             # A common pattern is loras/<model_type_or_size>/, e.g. loras/14B/
             # This is a heuristic, wgp.py's get_lora_dir is the source of truth for where to place it.
             # For simplicity, let's ensure the base_lora_dir_for_model exists, then place it.
             pass # get_lora_dir should give the correct final path
        elif "14B" in model_filename_for_task: # General 14B model
             pass 

        if not Path(target_causvid_lora_dir / causvid_lora_basename).exists():
            print(f"[Task ID: {task_id}] CausVid LoRA not found. Attempting download...")
            if not download_file(causvid_lora_url, target_causvid_lora_dir, causvid_lora_basename):
                print(f"[WARNING Task ID: {task_id}] Failed to download CausVid LoRA. Proceeding without it or with default settings.")
                # Potentially revert use_causvid flag or handle error more gracefully if LoRA is essential
                task_params_dict["use_causvid_lora"] = False # Disable if download fails
            else:
                 # Important: Refresh available LoRAs in wgp_mod if it caches them globally, 
                 # or ensure build_task_state re-scans. For now, we assume build_task_state will see it.
                 pass 
        if not "14B" in model_filename_for_task or not "t2v" in model_filename_for_task.lower():
            print(f"[WARNING Task ID: {task_id}] CausVid LoRA is intended for 14B T2V models. Current model is {model_filename_for_task}. Results may vary.")

    print(f"[Task ID: {task_id}] Using model file: {model_filename_for_task}")

    task_output_sub_dir = task_params_dict.get("output_sub_dir", task_id)
    task_specific_output_path = Path(main_output_dir) / task_output_sub_dir
    task_specific_output_path.mkdir(parents=True, exist_ok=True)
    original_wgp_save_path = wgp_mod.save_path
    wgp_mod.save_path = str(task_specific_output_path)

    # Get all LoRAs available for the *actual* model being used for the task
    # This list is used by build_task_state to resolve activated_loras names to paths if needed by wgp.py
    # and to correctly merge multipliers if CausVid LoRA is added.
    lora_dir_for_active_model = wgp_mod.get_lora_dir(model_filename_for_task)
    all_loras_for_active_model, _, _, _, _, _, _ = wgp_mod.setup_loras(
        model_filename_for_task, None, lora_dir_for_active_model, "", None
    )

    state, ui_params = build_task_state(wgp_mod, model_filename_for_task, task_params_dict, all_loras_for_active_model)
    
    gen_task_placeholder = {"id": 1, "prompt": ui_params.get("prompt"), "params": {}}
    send_cmd = make_send_cmd(task_id)

    tea_cache_value = ui_params.get("tea_cache_setting", ui_params.get("tea_cache", 0.0))

    print(f"[Task ID: {task_id}] Starting generation with effective params: {json.dumps(ui_params, default=lambda o: 'Unserializable' if isinstance(o, Image.Image) else o.__dict__ if hasattr(o, '__dict__') else str(o), indent=2)}")
    success = False
    try:
        wgp_mod.generate_video(
            task=gen_task_placeholder, send_cmd=send_cmd,
            prompt=ui_params["prompt"],
            negative_prompt=ui_params.get("negative_prompt", ""),
            resolution=ui_params["resolution"],
            video_length=ui_params.get("video_length", 81),
            seed=ui_params["seed"],
            num_inference_steps=ui_params.get("num_inference_steps", 30),
            guidance_scale=ui_params.get("guidance_scale", 5.0),
            audio_guidance_scale=ui_params.get("audio_guidance_scale", 5.0),
            flow_shift=ui_params.get("flow_shift", wgp_mod.get_default_flow(model_filename_for_task, wgp_mod.test_class_i2v(model_filename_for_task))),
            embedded_guidance_scale=ui_params.get("embedded_guidance_scale", 6.0),
            repeat_generation=ui_params.get("repeat_generation", 1),
            multi_images_gen_type=ui_params.get("multi_images_gen_type", 0),
            tea_cache_setting=tea_cache_value,
            tea_cache_start_step_perc=ui_params.get("tea_cache_start_step_perc", 0),
            activated_loras=ui_params.get("activated_loras", []),
            loras_multipliers=ui_params.get("loras_multipliers", ""),
            image_prompt_type=ui_params.get("image_prompt_type", "T"),
            image_start=ui_params.get("image_start", None),
            image_end=ui_params.get("image_end", None),
            model_mode=ui_params.get("model_mode", 0),
            video_source=ui_params.get("video_source", None),
            keep_frames_video_source=ui_params.get("keep_frames_video_source", ""),
            video_prompt_type=ui_params.get("video_prompt_type", ""),
            image_refs=ui_params.get("image_refs", None),
            video_guide=ui_params.get("video_guide", None),
            keep_frames_video_guide=ui_params.get("keep_frames_video_guide", ""),
            video_mask=ui_params.get("video_mask", None),
            audio_guide=ui_params.get("audio_guide", None),
            sliding_window_size=ui_params.get("sliding_window_size", 81),
            sliding_window_overlap=ui_params.get("sliding_window_overlap", 5),
            sliding_window_overlap_noise=ui_params.get("sliding_window_overlap_noise", 20),
            sliding_window_discard_last_frames=ui_params.get("sliding_window_discard_last_frames", 0),
            remove_background_image_ref=ui_params.get("remove_background_image_ref", 1),
            temporal_upsampling=ui_params.get("temporal_upsampling", ""),
            spatial_upsampling=ui_params.get("spatial_upsampling", ""),
            RIFLEx_setting=ui_params.get("RIFLEx_setting", 0),
            slg_switch=ui_params.get("slg_switch", 0),
            slg_layers=ui_params.get("slg_layers", [9]),
            slg_start_perc=ui_params.get("slg_start_perc", 10),
            slg_end_perc=ui_params.get("slg_end_perc", 90),
            cfg_star_switch=ui_params.get("cfg_star_switch", 0),
            cfg_zero_step=ui_params.get("cfg_zero_step", -1),
            prompt_enhancer=ui_params.get("prompt_enhancer", ""),
            state=state,
            model_filename=model_filename_for_task
        )
        print(f"[Task ID: {task_id}] Generation completed. Check {wgp_mod.save_path}")
        success = True
    except Exception as e:
        print(f"[ERROR] Task ID {task_id} failed during generation: {e}")
        traceback.print_exc()
        # success remains False
    finally:
        wgp_mod.save_path = original_wgp_save_path
    
    print(f"--- Finished task ID: {task_id} (Success: {success}) ---")
    return success

# -----------------------------------------------------------------------------
# 7. Main server loop
# -----------------------------------------------------------------------------

def main():
    cli_args = parse_args()
    tasks_file_path = Path(cli_args.tasks_file)
    main_output_dir = Path(cli_args.main_output_dir)
    main_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"WanGP Headless Server Started (Single tasks.json poller).")
    print(f"Monitoring tasks file: {tasks_file_path}")
    print(f"Outputs will be saved under: {main_output_dir}")
    print(f"Polling interval: {cli_args.poll_interval} seconds.")

    original_argv = sys.argv.copy()
    sys.argv = ["wgp.py"]
    patch_gradio()
    import wgp as wgp_mod
    sys.argv = original_argv

    # Apply wgp.py global config overrides
    if cli_args.wgp_attention_mode is not None: wgp_mod.attention_mode = cli_args.wgp_attention_mode
    if cli_args.wgp_compile is not None: wgp_mod.compile = cli_args.wgp_compile
    # ... (all other wgp global config settings as before) ...
    if cli_args.wgp_profile is not None: wgp_mod.profile = cli_args.wgp_profile
    if cli_args.wgp_vae_config is not None: wgp_mod.vae_config = cli_args.wgp_vae_config
    if cli_args.wgp_boost is not None: wgp_mod.boost = cli_args.wgp_boost
    if cli_args.wgp_transformer_quantization is not None: wgp_mod.transformer_quantization = cli_args.wgp_transformer_quantization
    if cli_args.wgp_transformer_dtype_policy is not None: wgp_mod.transformer_dtype_policy = cli_args.wgp_transformer_dtype_policy
    if cli_args.wgp_text_encoder_quantization is not None: wgp_mod.text_encoder_quantization = cli_args.wgp_text_encoder_quantization
    if cli_args.wgp_vae_precision is not None: wgp_mod.server_config["vae_precision"] = cli_args.wgp_vae_precision
    if cli_args.wgp_mixed_precision is not None: wgp_mod.server_config["mixed_precision"] = cli_args.wgp_mixed_precision
    if cli_args.wgp_preload_policy is not None:
        wgp_mod.server_config["preload_model_policy"] = [flag.strip() for flag in cli_args.wgp_preload_policy.split(',')]

    try:
        while True:
            tasks_to_process = []
            if tasks_file_path.is_file():
                try:
                    with open(tasks_file_path, 'r') as f:
                        tasks_to_process = json.load(f)
                    if not isinstance(tasks_to_process, list):
                        print(f"[ERROR] {tasks_file_path} does not contain a JSON list. Skipping cycle.")
                        tasks_to_process = []
                except json.JSONDecodeError:
                    print(f"[ERROR] Could not decode JSON from {tasks_file_path}. Is it valid JSON? Skipping cycle.")
                    tasks_to_process = []
                except Exception as e:
                    print(f"[ERROR] Could not read {tasks_file_path}: {e}. Skipping cycle.")
                    tasks_to_process = []
            
            if not tasks_to_process:
                time.sleep(cli_args.poll_interval)
                continue

            current_task_data = tasks_to_process[0]
            print(f"Found task: {current_task_data.get('task_id', 'N/A')}")

            task_succeeded = process_single_task(wgp_mod, current_task_data, main_output_dir)

            try:
                remaining_tasks = []
                if tasks_file_path.is_file():
                    with open(tasks_file_path, 'r') as f_read:
                        all_current_tasks_in_file = json.load(f_read)
                        if isinstance(all_current_tasks_in_file, list) and all_current_tasks_in_file:
                            remaining_tasks = all_current_tasks_in_file[1:] 
                else: 
                    print(f"[INFO] {tasks_file_path} was removed. No tasks to update.")

                with open(tasks_file_path, 'w') as f_write:
                    json.dump(remaining_tasks, f_write, indent=4)
                if task_succeeded:
                    print(f"Successfully processed and removed task {current_task_data.get('task_id', 'N/A')} from {tasks_file_path}")
                else:
                    print(f"Failed and removed task {current_task_data.get('task_id', 'N/A')} from {tasks_file_path}. Review logs for errors.")

            except Exception as e_write:
                print(f"[ERROR] Could not update {tasks_file_path} after processing task: {e_write}")
            
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nServer shutting down gracefully...")
    finally:
        if hasattr(wgp_mod, 'offloadobj') and wgp_mod.offloadobj is not None:
            try:
                print("Attempting to release wgp.py offload object...")
                wgp_mod.offloadobj.release()
            except Exception as e_release:
                print(f"Error during offloadobj release: {e_release}")
        print("Server stopped.")

if __name__ == "__main__":
    main() 