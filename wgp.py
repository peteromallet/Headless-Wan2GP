import os
import time
import sys
import threading
import argparse
from mmgp import offload, safetensors2, profile_type 
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from datetime import datetime
import gradio as gr
import random
import json
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS, SUPPORTED_SIZES, VACE_SIZE_CONFIGS
from wan.utils.utils import cache_video
from wan.modules.attention import get_attention_modes, get_supported_attention_modes
import torch
import gc
import traceback
import math
import typing
import asyncio
import inspect
from wan.utils import prompt_parser
import base64
import io
from PIL import Image
PROMPT_VARS_MAX = 10

target_mmgp_version = "3.3.4"
from importlib.metadata import version
mmgp_version = version("mmgp")
if mmgp_version != target_mmgp_version:
    print(f"Incorrect version of mmgp ({mmgp_version}), version {target_mmgp_version} is needed. Please upgrade with the command 'pip install -r requirements.txt'")
    exit()
lock = threading.Lock()
current_task_id = None
task_id = 0
# progress_tracker = {}
# tracker_lock = threading.Lock()

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def pil_to_base64_uri(pil_image, format="png", quality=75):
    if pil_image is None:
        return None

    if isinstance(pil_image, str):
        from wan.utils.utils import get_video_frame
        pil_image = get_video_frame(pil_image, 0)

    buffer = io.BytesIO()
    try:
        img_to_save = pil_image
        if format.lower() == 'jpeg' and pil_image.mode == 'RGBA':
            img_to_save = pil_image.convert('RGB')
        elif format.lower() == 'png' and pil_image.mode not in ['RGB', 'RGBA', 'L', 'P']:
             img_to_save = pil_image.convert('RGBA')
        elif pil_image.mode == 'P':
             img_to_save = pil_image.convert('RGBA' if 'transparency' in pil_image.info else 'RGB')
        if format.lower() == 'jpeg':
            img_to_save.save(buffer, format=format, quality=quality)
        else:
            img_to_save.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        encoded_string = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting PIL to base64: {e}")
        return None


def process_prompt_and_add_tasks(state, model_choice):
 
    if state.get("validate_success",0) != 1:
        return
    
    state["validate_success"] = 0

    model_filename = state["model_filename"]

    if model_choice != get_model_type(model_filename):
        raise gr.Error("Webform can not be used as the App has been restarted since the form was displayed. Please refresh the page")
 
    inputs = state.get(get_model_type(model_filename), None)
    inputs["state"] =  state
    if inputs == None:
        return
    prompt = inputs["prompt"]
    if len(prompt) ==0:
        return
    prompt, errors = prompt_parser.process_template(prompt)
    if len(errors) > 0:
        gr.Info("Error processing prompt template: " + errors)
        return
    
    inputs["model_filename"] = model_filename
    prompts = prompt.replace("\r", "").split("\n")
    prompts = [prompt.strip() for prompt in prompts if len(prompt.strip())>0 and not prompt.startswith("#")]
    if len(prompts) ==0:
        return

    resolution = inputs["resolution"]
    width, height = resolution.split("x")
    width, height = int(width), int(height)
    if test_class_i2v(model_filename):
        if "480p" in  model_filename and not "Fun" in model_filename and width * height > 848*480:
            gr.Info("You must use the 720P image to video model to generate videos with a resolution equivalent to 720P")
            return
        resolution = str(width) + "*" + str(height)  
        if  resolution not in ['720*1280', '1280*720', '480*832', '832*480']:
            gr.Info(f"Resolution {resolution} not supported by image 2 video")
            return

    if "1.3B" in  model_filename and width * height > 848*480:
        gr.Info("You must use the 14B model to generate videos with a resolution equivalent to 720P")
        return


    if "Vace" in model_filename:
        video_prompt_type = inputs["video_prompt_type"]
        image_refs = inputs["image_refs"]
        video_guide = inputs["video_guide"]
        video_mask = inputs["video_mask"]
        if "Vace" in model_filename and "1.3B" in model_filename :                
            resolution_reformated = str(height) + "*" + str(width) 
            if not resolution_reformated in VACE_SIZE_CONFIGS:
                res = VACE_SIZE_CONFIGS.keys().join(" and ")
                gr.Info(f"Video Resolution for Vace model is not supported. Only {res} resolutions are allowed.")
                return
        if not "I" in video_prompt_type:
            image_refs = None
        if not "V" in video_prompt_type:
            video_guide = None
        if not "M" in video_prompt_type:
            video_mask = None

        if isinstance(image_refs, list):
            image_refs = [ convert_image(tup[0]) for tup in image_refs ]

            from wan.utils.utils import resize_and_remove_background
            image_refs = resize_and_remove_background(image_refs, width, height, inputs["remove_background_image_ref"] ==1)
        

        for single_prompt  in prompts:
            extra_inputs = {
                "prompt" : single_prompt,
                "image_refs": image_refs,
                "video_guide" : video_guide,
                "video_mask" : video_mask ,
            }
            inputs.update(extra_inputs) 
            add_video_task(**inputs)
    elif "image2video" in model_filename or "Fun_InP" in model_filename :
        image_prompt_type = inputs["image_prompt_type"]

        image_start = inputs["image_start"]
        image_end = inputs["image_end"]
        if image_start == None or isinstance(image_start, list) and len(image_start) == 0:
            return
        if not "E" in image_prompt_type:
            image_end = None
        if isinstance(image_start, list):
            image_start = [ convert_image(tup[0]) for tup in image_start ]
        else:
            image_start = [convert_image(image_start)]
        if image_end != None:
            if isinstance(image_end , list):
                image_end  = [ convert_image(tup[0]) for tup in image_end  ]
            else:
                image_end  = [convert_image(image_end) ]
            if len(image_start) != len(image_end):
                gr.Info("The number of start and end images should be the same ")
                return 
                            
        if inputs["multi_images_gen_type"] == 0:
            new_prompts = []
            new_image_start = []
            new_image_end = []
            for i in range(len(prompts) * len(image_start) ):
                new_prompts.append(  prompts[ i % len(prompts)] )
                new_image_start.append(image_start[i // len(prompts)] )
                if image_end != None:
                    new_image_end.append(image_end[i // len(prompts)] )
            prompts = new_prompts
            image_start = new_image_start 
            if image_end != None:
                image_end = new_image_end 
        else:
            if len(prompts) >= len(image_start):
                if len(prompts) % len(image_start) != 0:
                    raise gr.Error("If there are more text prompts than input images the number of text prompts should be dividable by the number of images")
                rep = len(prompts) // len(image_start)
                new_image_start = []
                new_image_end = []
                for i, _ in enumerate(prompts):
                    new_image_start.append(image_start[i//rep] )
                    if image_end != None:
                        new_image_end.append(image_end[i//rep] )
                image_start = new_image_start 
                if image_end != None:
                    image_end = new_image_end 
            else: 
                if len(image_start) % len(prompts)  !=0:
                    raise gr.Error("If there are more input images than text prompts the number of images should be dividable by the number of text prompts")
                rep = len(image_start) // len(prompts)  
                new_prompts = []
                for i, _ in enumerate(image_start):
                    new_prompts.append(  prompts[ i//rep] )
                prompts = new_prompts

  
        if image_start == None:
            image_start = [None] * len(prompts)
        if image_end == None:
            image_end = [None] * len(prompts)

        for single_prompt, start, end in zip(prompts, image_start, image_end) :
            extra_inputs = {
                "prompt" : single_prompt,
                "image_start": start,
                "image_end" : end,
            }
            inputs.update(extra_inputs) 
            add_video_task(**inputs)
    else:
        for single_prompt in prompts :
            extra_inputs = {
                "prompt" : single_prompt,
            }
            inputs.update(extra_inputs) 
            add_video_task(**inputs)

    gen = get_gen_info(state)
    gen["prompts_max"] = len(prompts) + gen.get("prompts_max",0)
    state["validate_success"] = 1
    queue= gen.get("queue", [])
    return update_queue_data(queue)




def add_video_task(**inputs):
    global task_id
    state = inputs["state"]
    gen = get_gen_info(state)
    queue = gen["queue"]
    task_id += 1
    current_task_id = task_id
    inputs_to_query = ["image_start", "image_end", "image_refs", "video_guide", "video_mask"]
    start_image_data = None
    end_image_data = None
    for name in inputs_to_query:
        image= inputs.get(name, None)
        if image != None:
            image= [image] if not isinstance(image, list) else image
            if start_image_data == None:
                start_image_data = image
            else:
                end_image_data = image
                break

    queue.append({
        "id": current_task_id,
        "params": inputs.copy(),
        "repeats": inputs["repeat_generation"],
        "length": inputs["video_length"],
        "steps": inputs["num_inference_steps"],
        "prompt": inputs["prompt"],
        "start_image_data": start_image_data,
        "end_image_data": end_image_data,
        "start_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in start_image_data] if start_image_data != None else None,
        "end_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in end_image_data] if end_image_data != None else None
    })
    return update_queue_data(queue)

def move_up(queue, selected_indices):
    if not selected_indices or len(selected_indices) == 0:
        return update_queue_data(queue)
    idx = selected_indices[0]
    if isinstance(idx, list):
        idx = idx[0]
    idx = int(idx)
    with lock:
        if idx > 0:
            idx += 1
            queue[idx], queue[idx-1] = queue[idx-1], queue[idx]
    return update_queue_data(queue)

def move_down(queue, selected_indices):
    if not selected_indices or len(selected_indices) == 0:
        return update_queue_data(queue)
    idx = selected_indices[0]
    if isinstance(idx, list):
        idx = idx[0]
    idx = int(idx)
    with lock:
        idx += 1
        if idx < len(queue)-1:
            queue[idx], queue[idx+1] = queue[idx+1], queue[idx]
    return update_queue_data(queue)

def remove_task(queue, selected_indices):
    if not selected_indices or len(selected_indices) == 0:
        return update_queue_data(queue)
    idx = selected_indices[0]
    if isinstance(idx, list):
        idx = idx[0]
    idx = int(idx) + 1
    with lock:
        if idx < len(queue):
            if idx == 0:
                wan_model._interrupt = True
            del queue[idx]
    return update_queue_data(queue)



def get_queue_table(queue):
    data = []
    if len(queue) == 1:
        return data 

    # def td(l, content, width =None):
    #     if width !=None:
    #         l.append("<TD WIDTH="+ str(width) + "px>" + content + "</TD>")
    #     else:
    #         l.append("<TD>" + content + "</TD>")

    # data.append("<STYLE> .TB, .TB  th, .TB td {border: 1px solid #CCCCCC};></STYLE><TABLE CLASS=TB><TR BGCOLOR=#F2F2F2><TD Style='Bold'>Qty</TD><TD>Prompt</TD><TD>Steps</TD><TD></TD><TD><TD></TD><TD></TD><TD></TD></TR>")

    for i, item in enumerate(queue):
        if i==0:
            continue
        truncated_prompt = (item['prompt'][:97] + '...') if len(item['prompt']) > 100 else item['prompt']
        full_prompt = item['prompt'].replace('"', '&quot;')
        prompt_cell = f'<span title="{full_prompt}">{truncated_prompt}</span>'
        start_img_uri =item.get('start_image_data_base64')
        start_img_uri = start_img_uri[0] if start_img_uri !=None else None
        end_img_uri = item.get('end_image_data_base64')
        end_img_uri = end_img_uri[0] if end_img_uri !=None else None
        thumbnail_size = "50px"
        num_steps = item.get('steps')
        length = item.get('length')
        start_img_md = ""
        end_img_md = ""
        if start_img_uri:
            start_img_md = f'<img src="{start_img_uri}" alt="Start" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display: block; margin: auto; object-fit: contain;" />'
        if end_img_uri:
            end_img_md = f'<img src="{end_img_uri}" alt="End" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display: block; margin: auto; object-fit: contain;" />'
    #     if i % 2 == 1:
    #         data.append("<TR>")
    #     else:
    #         data.append("<TR BGCOLOR=#F2F2F2>")

    #     td(data,str(item.get('repeats', "1")) )
    #     td(data, prompt_cell, "100%")
    #     td(data, num_steps, "100%")
    #     td(data, start_img_md)
    #     td(data, end_img_md)
    #     td(data, "↑")
    #     td(data, "↓")
    #     td(data, "✖")
    #     data.append("</TR>")
    # data.append("</TABLE>")
    # return ''.join(data)

        data.append([item.get('repeats', "1"),
                    prompt_cell,
                    length,
                    num_steps,
                    start_img_md,
                    end_img_md,
                    "↑",
                    "↓",
                    "✖"
                    ])    
    return data
def update_queue_data(queue):

    data = get_queue_table(queue)

    # if len(data) == 0:
    #     return gr.HTML(visible=False)
    # else:
    #     return gr.HTML(value=data, visible= True)
    if len(data) == 0:
        return gr.DataFrame(visible=False)
    else:
        return gr.DataFrame(value=data, visible= True)

def create_html_progress_bar(percentage=0.0, text="Idle", is_idle=True):
    bar_class = "progress-bar-custom idle" if is_idle else "progress-bar-custom"
    bar_text_html = f'<div class="progress-bar-text">{text}</div>'

    html = f"""
    <div class="progress-container-custom">
        <div class="{bar_class}" style="width: {percentage:.1f}%;" role="progressbar" aria-valuenow="{percentage:.1f}" aria-valuemin="0" aria-valuemax="100">
           {bar_text_html}
        </div>
    </div>
    """
    return html


def update_generation_status(html_content):
    if(html_content):
        return gr.update(value=html_content)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt or image using Gradio")

    parser.add_argument(
        "--quantize-transformer",
        action="store_true",
        help="On the fly 'transformer' quantization"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shared URL to access webserver remotely"
    )

    parser.add_argument(
        "--lock-config",
        action="store_true",
        help="Prevent modifying the configuration from the web interface"
    )

    parser.add_argument(
        "--lock-model",
        action="store_true",
        help="Prevent switch models"
    )

    parser.add_argument(
        "--preload",
        type=str,
        default="0",
        help="Megabytes of the diffusion model to preload in VRAM"
    )

    parser.add_argument(
        "--multiple-images",
        action="store_true",
        help="Allow inputting multiple images with image to video"
    )


    parser.add_argument(
        "--lora-dir-i2v",
        type=str,
        default="",
        help="Path to a directory that contains Loras for i2v"
    )


    parser.add_argument(
        "--lora-dir",
        type=str,
        default="", 
        help="Path to a directory that contains Loras"
    )

    parser.add_argument(
        "--check-loras",
        action="store_true",
        help="Filter Loras that are not valid"
    )


    parser.add_argument(
        "--lora-preset",
        type=str,
        default="",
        help="Lora preset to preload"
    )

    # parser.add_argument(
    #     "--i2v-settings",
    #     type=str,
    #     default="i2v_settings.json",
    #     help="Path to settings file for i2v"
    # )

    # parser.add_argument(
    #     "--t2v-settings",
    #     type=str,
    #     default="t2v_settings.json",
    #     help="Path to settings file for t2v"
    # )

    # parser.add_argument(
    #     "--lora-preset-i2v",
    #     type=str,
    #     default="",
    #     help="Lora preset to preload for i2v"
    # )

    parser.add_argument(
        "--profile",
        type=str,
        default=-1,
        help="Profile No"
    )

    parser.add_argument(
        "--verbose",
        type=str,
        default=1,
        help="Verbose level"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="default denoising steps"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="default number of frames"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="default generation seed"
    )

    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Access advanced options by default"
    )

    parser.add_argument(
        "--server-port",
        type=str,
        default=0,
        help="Server port"
    )

    parser.add_argument(
        "--server-name",
        type=str,
        default="",
        help="Server name"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="",
        help="Default GPU Device"
    )

    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open browser"
    )

    parser.add_argument(
        "--t2v",
        action="store_true",
        help="text to video mode"
    )

    parser.add_argument(
        "--i2v",
        action="store_true",
        help="image to video mode"
    )

    parser.add_argument(
        "--t2v-14B",
        action="store_true",
        help="text to video mode 14B model"
    )

    parser.add_argument(
        "--t2v-1-3B",
        action="store_true",
        help="text to video mode 1.3B model"
    )

    parser.add_argument(
        "--vace-1-3B",
        action="store_true",
        help="Vace ControlNet 1.3B model"
    )    
    parser.add_argument(
        "--i2v-1-3B",
        action="store_true",
        help="Fun InP image to video mode 1.3B model"
    )

    parser.add_argument(
        "--i2v-14B",
        action="store_true",
        help="image to video mode 14B model"
    )


    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable pytorch compilation"
    )

    parser.add_argument(
        "--listen",
        action="store_true",
        help="Server accessible on local network"
    )

    # parser.add_argument(
    #     "--fast",
    #     action="store_true",
    #     help="use Fast model"
    # )

    # parser.add_argument(
    #     "--fastest",
    #     action="store_true",
    #     help="activate the best config"
    # )

    parser.add_argument(
    "--attention",
    type=str,
    default="",
    help="attention mode"
    )

    parser.add_argument(
    "--vae-config",
    type=str,
    default="",
    help="vae config mode"
    )    

    args = parser.parse_args()

    return args

def get_lora_dir(model_filename):
    lora_dir =args.lora_dir
    i2v = test_class_i2v(model_filename)
    if i2v and len(lora_dir)==0:
        lora_dir =args.lora_dir_i2v
    if len(lora_dir) > 0:
        return lora_dir

    root_lora_dir = "loras_i2v" if i2v else "loras"

    if  "1.3B" in model_filename :
        lora_dir_1_3B = os.path.join(root_lora_dir, "1.3B")
        if os.path.isdir(lora_dir_1_3B ):
            return lora_dir_1_3B
    else:
        lora_dir_14B = os.path.join(root_lora_dir, "14B")
        if os.path.isdir(lora_dir_14B ):
            return lora_dir_14B
    return root_lora_dir    


attention_modes_installed = get_attention_modes()
attention_modes_supported = get_supported_attention_modes()
args = _parse_args()
args.flow_reverse = True
processing_device = args.gpu
if len(processing_device) == 0:
    processing_device ="cuda"
# torch.backends.cuda.matmul.allow_fp16_accumulation = True
lock_ui_attention = False
lock_ui_transformer = False
lock_ui_compile = False

preload =int(args.preload)
force_profile_no = int(args.profile)
verbose_level = int(args.verbose)
quantizeTransformer = args.quantize_transformer
check_loras = args.check_loras ==1
advanced = args.advanced

transformer_choices_t2v=["ckpts/wan2.1_text2video_1.3B_bf16.safetensors", "ckpts/wan2.1_text2video_14B_bf16.safetensors", "ckpts/wan2.1_text2video_14B_quanto_int8.safetensors", "ckpts/wan2.1_Vace_1.3B_preview_bf16.safetensors"]   
transformer_choices_i2v=["ckpts/wan2.1_image2video_480p_14B_bf16.safetensors", "ckpts/wan2.1_image2video_480p_14B_quanto_int8.safetensors", "ckpts/wan2.1_image2video_720p_14B_bf16.safetensors", "ckpts/wan2.1_image2video_720p_14B_quanto_int8.safetensors", "ckpts/wan2.1_Fun_InP_1.3B_bf16.safetensors", "ckpts/wan2.1_Fun_InP_14B_bf16.safetensors", "ckpts/wan2.1_Fun_InP_14B_quanto_int8.safetensors", ]
transformer_choices = transformer_choices_t2v + transformer_choices_i2v
text_encoder_choices = ["ckpts/models_t5_umt5-xxl-enc-bf16.safetensors", "ckpts/models_t5_umt5-xxl-enc-quanto_int8.safetensors"]
server_config_filename = "wgp_config.json"

if not os.path.isfile(server_config_filename) and os.path.isfile("gradio_config.json"):
    import shutil 
    shutil.move("gradio_config.json", server_config_filename) 

if not Path(server_config_filename).is_file():
    server_config = {"attention_mode" : "auto",  
                     "transformer_types": [], 
                     "transformer_quantization": "int8",
                     "text_encoder_filename" : text_encoder_choices[1],
                     "save_path": os.path.join(os.getcwd(), "gradio_outputs"),
                     "compile" : "",
                     "metadata_type": "metadata",
                     "default_ui": "t2v",
                     "boost" : 1,
                     "clear_file_list" : 0,
                     "vae_config": 0,
                     "profile" : profile_type.LowRAM_LowVRAM,
                     "reload_model": 2 }

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))
else:
    with open(server_config_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    server_config = json.loads(text)


model_types = [ "t2v_1.3B", "vace_1.3B", "fun_inp_1.3B", "t2v", "i2v", "i2v_720p", "fun_inp"]
model_signatures = {"t2v": "text2video_14B", "t2v_1.3B" : "text2video_1.3B",   "fun_inp_1.3B" : "Fun_InP_1.3B",  "fun_inp" :  "Fun_InP_14B", 
                    "i2v" : "image2video_480p", "i2v_720p" : "image2video_720p" , "vace_1.3B" : "Vace_1.3B" }


def get_model_type(model_filename):
    if "text2video" in model_filename and "14B" in model_filename:
        return "t2v"
    elif "text2video" in model_filename and "1.3B" in model_filename:
        return "t2v_1.3B"
    elif "Fun_InP" in model_filename and "1.3B" in model_filename:
        return "fun_inp_1.3B"
    elif "Fun_InP" in model_filename and "14B" in model_filename:
        return "fun_inp"
    elif "image2video_480p" in model_filename :
        return "i2v" 
    elif "image2video_720p" in model_filename :
        return "i2v_720p" 
    elif "Vace" in model_filename and "1.3B" in model_filename:
        return "vace_1.3B"
    elif "Vace" in model_filename and "14B" in model_filename:
        return "vace"
    else:
        raise Exception("Unknown model:" + model_filename)

def test_class_i2v(model_filename):
    return "image2video" in model_filename or "Fun_InP" in model_filename 


def get_model_filename(model_type, quantization):
    signature = model_signatures[model_type]

    choices = [ name for name in transformer_choices if signature in name]
    if len(quantization) == 0:
        quantization = "bf16"

    if len(choices) <= 1:
        return choices[0]
    
    sub_choices = [ name for name in choices if quantization in name]
    if len(sub_choices) > 0:
        return sub_choices[0]
    else:
        return choices[0]
    
def get_settings_file_name(model_filename):
    return  get_model_type(model_filename) + "_settings.json"

def get_default_settings(filename):
    def get_default_prompt(i2v):
        if i2v:
            return "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field."
        else:
            return "A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect."
    i2v = "image2video" in file_name
    defaults_filename = get_settings_file_name(filename)
    if not Path(defaults_filename).is_file():
        ui_defaults = {
            "prompts": get_default_prompt(i2v),
            "resolution": "832x480",
            "video_length": 81,
            "num_inference_steps": 30,
            "seed": -1,
            "repeat_generation": 1,
            "multi_images_gen_type": 0,        
            "guidance_scale": 5.0,
            "flow_shift": get_default_flow(filename, i2v),
            "negative_prompt": "",
            "activated_loras": [],
            "loras_multipliers": "",
            "tea_cache": 0.0,
            "tea_cache_start_step_perc": 0,
            "RIFLEx_setting": 0,
            "slg_switch": 0,
            "slg_layers": [9],
            "slg_start_perc": 10,
            "slg_end_perc": 90
        }
        with open(defaults_filename, "w", encoding="utf-8") as f:
            json.dump(ui_defaults, f, indent=4)
    else:
        with open(defaults_filename, "r", encoding="utf-8") as f:
            ui_defaults = json.load(f)
        prompts = ui_defaults.get("prompts", "")
        if len(prompts) > 0:
            ui_defaults["prompt"] = prompts
        image_prompt_type = ui_defaults.get("image_prompt_type", None)
        if image_prompt_type !=None and not isinstance(image_prompt_type, str):
            ui_defaults["image_prompt_type"] = "S" if image_prompt_type  == 0 else "SE"

    default_seed = args.seed
    if default_seed > -1:
        ui_defaults["seed"] = default_seed
    default_number_frames = args.frames
    if default_number_frames > 0:
        ui_defaults["video_length"] = default_number_frames
    default_number_steps = args.steps
    if default_number_steps > 0:
        ui_defaults["num_inference_steps"] = default_number_steps
    return ui_defaults

transformer_types = server_config.get("transformer_types", [])
transformer_type = transformer_types[0] if len(transformer_types) > 0 else  model_types[0]
transformer_quantization =server_config.get("transformer_quantization", "int8")
transformer_filename = get_model_filename(transformer_type, transformer_quantization)
text_encoder_filename = server_config["text_encoder_filename"]
attention_mode = server_config["attention_mode"]
if len(args.attention)> 0:
    if args.attention in ["auto", "sdpa", "sage", "sage2", "flash", "xformers"]:
        attention_mode = args.attention
        lock_ui_attention = True
    else:
        raise Exception(f"Unknown attention mode '{args.attention}'")

profile =  force_profile_no if force_profile_no >=0 else server_config["profile"]
compile = server_config.get("compile", "")
boost = server_config.get("boost", 1)
vae_config = server_config.get("vae_config", 0)
if len(args.vae_config) > 0:
    vae_config = int(args.vae_config)

reload_needed = False
default_ui = server_config.get("default_ui", "t2v") 
save_path = server_config.get("save_path", os.path.join(os.getcwd(), "gradio_outputs"))
reload_model = server_config.get("reload_model", 2) 


if args.t2v_14B or args.t2v: 
    transformer_filename = get_model_filename("t2v", transformer_quantization)

if args.i2v_14B or args.i2v: 
    transformer_filename = get_model_filename("i2v", transformer_quantization)

if args.t2v_1_3B:
    transformer_filename = get_model_filename("t2v_1.3B", transformer_quantization)

if args.i2v_1_3B:
    transformer_filename = get_model_filename("fun_inp_1.3B", transformer_quantization)

if args.vace_1_3B: 
    transformer_filename = get_model_filename("vace_1.3B", transformer_quantization)

only_allow_edit_in_advanced = False
lora_preselected_preset = args.lora_preset
lora_preset_model = transformer_filename

if  args.compile: #args.fastest or
    compile="transformer"
    lock_ui_compile = True

model_filename = ""
#attention_mode="sage"
#attention_mode="sage2"
#attention_mode="flash"
#attention_mode="sdpa"
#attention_mode="xformers"
# compile = "transformer"

def preprocess_loras(sd):
    if wan_model == None:
        return sd
    model_filename = wan_model._model_file_name

    first = next(iter(sd), None)
    if first == None:
        return sd
    
    if first.startswith("lora_unet_"):
        new_sd = {}
        print("Converting Lora Safetensors format to Lora Diffusers format")
        alphas = {}
        repl_list = ["cross_attn", "self_attn", "ffn"]
        src_list = ["_" + k + "_" for k in repl_list]
        tgt_list = ["." + k + "." for k in repl_list]

        for k,v in sd.items():
            k = k.replace("lora_unet_blocks_","diffusion_model.blocks.")

            for s,t in zip(src_list, tgt_list):
                k = k.replace(s,t)

            k = k.replace("lora_up","lora_B")
            k = k.replace("lora_down","lora_A")

            if "alpha" in k:
                alphas[k] = v
            else:
                new_sd[k] = v

        new_alphas = {}
        for k,v in new_sd.items():
            if "lora_B" in k:
                dim = v.shape[1]
            elif "lora_A" in k:
                dim = v.shape[0]
            else:
                continue
            alpha_key = k[:-len("lora_X.weight")] +"alpha"
            if alpha_key in alphas:
                scale = alphas[alpha_key] / dim
                new_alphas[alpha_key] = scale
            else:
                print(f"Lora alpha'{alpha_key}' is missing")
        new_sd.update(new_alphas)
        sd = new_sd

    if "text2video" in model_filename:
        new_sd = {}
        # convert loras for i2v to t2v
        for k,v in sd.items():
            if  any(layer in k for layer in ["cross_attn.k_img", "cross_attn.v_img"]):
                continue
            new_sd[k] = v
        sd = new_sd

    return sd


def download_models(transformer_filename, text_encoder_filename):
    def computeList(filename):
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        return [filename]        
    
    from huggingface_hub import hf_hub_download, snapshot_download    
    repoId = "DeepBeepMeep/Wan2.1" 
    sourceFolderList = ["xlm-roberta-large", "",  ]
    fileList = [ [], ["Wan2.1_VAE_bf16.safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors", "flownet.pkl" ] + computeList(text_encoder_filename) + computeList(transformer_filename) ]   
    targetRoot = "ckpts/" 
    for sourceFolder, files in zip(sourceFolderList,fileList ):
        if len(files)==0:
            if not Path(targetRoot + sourceFolder).exists():
                snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= targetRoot)
        else:
             for onefile in files:     
                if len(sourceFolder) > 0: 
                    if not os.path.isfile(targetRoot + sourceFolder + "/" + onefile ):          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot, subfolder=sourceFolder)
                else:
                    if not os.path.isfile(targetRoot + onefile ):          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot)


offload.default_verboseLevel = verbose_level
to_remove = ["models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", "Wan2.1_VAE.pth"]
for file_name in to_remove:
    file_name = os.path.join("ckpts",file_name)
    if os.path.isfile(file_name):
        try:
            os.remove(file_name)
        except:
            pass

download_models(transformer_filename, text_encoder_filename) 

def sanitize_file_name(file_name, rep =""):
    return file_name.replace("/",rep).replace("\\",rep).replace(":",rep).replace("|",rep).replace("?",rep).replace("<",rep).replace(">",rep).replace("\"",rep) 

def extract_preset(model_filename, lset_name, loras):
    loras_choices = []
    loras_choices_files = []
    loras_mult_choices = ""
    prompt =""
    full_prompt =""
    lset_name = sanitize_file_name(lset_name)
    lora_dir = get_lora_dir(model_filename)
    if not lset_name.endswith(".lset"):
        lset_name_filename = os.path.join(lora_dir, lset_name + ".lset" ) 
    else:
        lset_name_filename = os.path.join(lora_dir, lset_name ) 
    error = ""
    if not os.path.isfile(lset_name_filename):
        error = f"Preset '{lset_name}' not found "
    else:
        missing_loras = []

        with open(lset_name_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        lset = json.loads(text)

        loras_choices_files = lset["loras"]
        for lora_file in loras_choices_files:
            choice = os.path.join(lora_dir, lora_file)
            if choice not in loras:
                missing_loras.append(lora_file)
            else:
                loras_choice_no = loras.index(choice)
                loras_choices.append(str(loras_choice_no))

        if len(missing_loras) > 0:
            error = f"Unable to apply Lora preset '{lset_name} because the following Loras files are missing or invalid: {missing_loras}"
        
        loras_mult_choices = lset["loras_mult"]
        prompt = lset.get("prompt", "")
        full_prompt = lset.get("full_prompt", False)
    return loras_choices, loras_mult_choices, prompt, full_prompt, error


    
def setup_loras(model_filename, transformer,  lora_dir, lora_preselected_preset, split_linear_modules_map = None):
    loras =[]
    loras_names = []
    default_loras_choices = []
    default_loras_multis_str = ""
    loras_presets = []
    default_lora_preset = ""
    default_lora_preset_prompt = ""

    from pathlib import Path

    lora_dir = get_lora_dir(model_filename)
    if lora_dir != None :
        if not os.path.isdir(lora_dir):
            raise Exception("--lora-dir should be a path to a directory that contains Loras")


    if lora_dir != None:
        import glob
        dir_loras =  glob.glob( os.path.join(lora_dir , "*.sft") ) + glob.glob( os.path.join(lora_dir , "*.safetensors") ) 
        dir_loras.sort()
        loras += [element for element in dir_loras if element not in loras ]

        dir_presets =  glob.glob( os.path.join(lora_dir , "*.lset") ) 
        dir_presets.sort()
        loras_presets = [ Path(Path(file_path).parts[-1]).stem for file_path in dir_presets]

    if transformer !=None:
        loras = offload.load_loras_into_model(transformer, loras,  activate_all_loras=False, check_only= True, preprocess_sd=preprocess_loras, split_linear_modules_map = split_linear_modules_map) #lora_multiplier,

    if len(loras) > 0:
        loras_names = [ Path(lora).stem for lora in loras  ]

    if len(lora_preselected_preset) > 0:
        if not os.path.isfile(os.path.join(lora_dir, lora_preselected_preset + ".lset")):
            raise Exception(f"Unknown preset '{lora_preselected_preset}'")
        default_lora_preset = lora_preselected_preset
        default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, _ , error = extract_preset(model_filename, default_lora_preset, loras)
        if len(error) > 0:
            print(error[:200])
    return loras, loras_names, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset


def load_t2v_model(model_filename, value):

    cfg = WAN_CONFIGS['t2v-14B']
    # cfg = WAN_CONFIGS['t2v-1.3B']    
    print(f"Loading '{model_filename}' model...")

    wan_model = wan.WanT2V(
        config=cfg,
        checkpoint_dir="ckpts",
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        model_filename=model_filename,
        text_encoder_filename= text_encoder_filename
    )

    pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model,  "vae": wan_model.vae.model } 

    return wan_model, pipe

def load_i2v_model(model_filename, value):

    print(f"Loading '{model_filename}' model...")

    if value == '720P':
        cfg = WAN_CONFIGS['i2v-14B']
        wan_model = wan.WanI2V(
            config=cfg,
            checkpoint_dir="ckpts",
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            i2v720p= True,
            model_filename=model_filename,
            text_encoder_filename=text_encoder_filename
        )            
        pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model,  "text_encoder_2": wan_model.clip.model, "vae": wan_model.vae.model } #

    elif value == '480P':
        cfg = WAN_CONFIGS['i2v-14B']
        wan_model = wan.WanI2V(
            config=cfg,
            checkpoint_dir="ckpts",
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            i2v720p= False,
            model_filename=model_filename,
            text_encoder_filename=text_encoder_filename

        )
        pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model,  "text_encoder_2": wan_model.clip.model, "vae": wan_model.vae.model } #
    else:
        raise Exception("Model i2v {value} not supported")
    return wan_model, pipe



def load_models(model_filename):
    global transformer_filename

    transformer_filename = model_filename
    download_models(model_filename, text_encoder_filename)
    if test_class_i2v(model_filename):
        res720P = "720p" in model_filename
        wan_model, pipe = load_i2v_model(model_filename, "720P" if res720P else "480P")
    else:
        wan_model, pipe = load_t2v_model(model_filename, "")
    wan_model._model_file_name = model_filename
    kwargs = { "extraModelsToQuantize": None}
    if profile == 2 or profile == 4:
        kwargs["budgets"] = { "transformer" : 100 if preload  == 0 else preload, "text_encoder" : 100, "*" : 1000 }
        # if profile == 4:
        #     kwargs["partialPinning"] = True
    elif profile == 3:
        kwargs["budgets"] = { "*" : "70%" }
    offloadobj = offload.profile(pipe, profile_no= profile, compile = compile, quantizeTransformer = quantizeTransformer, loras = "transformer", coTenantsMap= {}, **kwargs)  
    if len(args.gpu) > 0:
        torch.set_default_device(args.gpu)

    return wan_model, offloadobj, pipe["transformer"] 

if reload_model ==3:
    wan_model, offloadobj, transformer = None, None, None
    reload_needed = True
else:
    wan_model, offloadobj, transformer = load_models(transformer_filename)
    if check_loras:
        setup_loras(model_filename, transformer,  get_lora_dir(transformer_filename), "", None)
        exit()
    del transformer

gen_in_progress = False

def get_auto_attention():
    for attn in ["sage2","sage","sdpa"]:
        if attn in attention_modes_supported:
            return attn
    return "sdpa"

def get_default_flow(filename, i2v):
    return 7.0 if "480p" in filename and i2v else 5.0 


def get_model_name(model_filename):
    if "Fun" in model_filename:
        model_name = "Fun InP image2video"
        model_name += " 14B" if "14B" in model_filename else " 1.3B"
    elif "Vace" in model_filename:
        model_name = "Vace ControlNet"
        model_name += " 14B" if "14B" in model_filename else " 1.3B"
    elif "image" in model_filename:
        model_name = "Wan2.1 image2video"
        model_name += " 720p" if "720p" in model_filename else " 480p"
    else:
        model_name = "Wan2.1 text2video"
        model_name += " 14B" if "14B" in model_filename else " 1.3B"

    return model_name

# def generate_header(model_filename, compile, attention_mode):
    
#     header = "<div class='title-with-lines'><div class=line></div><h2>"
    
#     model_name = get_model_name(model_filename)

#     header += model_name 
#     header += " (attention mode: " + (attention_mode if attention_mode!="auto" else "auto/" + get_auto_attention() )
#     if attention_mode not in attention_modes_installed:
#         header += " -NOT INSTALLED-"
#     elif attention_mode not in attention_modes_supported:
#         header += " -NOT SUPPORTED-"

#     if compile:
#         header += ", pytorch compilation ON"
#     header += ") </h2><div class=line></div>    "


#     return header


def generate_header(model_filename, compile, attention_mode):
    
    header = "<DIV style='align:right;width:100%'><FONT SIZE=3>Attention mode <B>" + (attention_mode if attention_mode!="auto" else "auto/" + get_auto_attention() )
    if attention_mode not in attention_modes_installed:
        header += " -NOT INSTALLED-"
    elif attention_mode not in attention_modes_supported:
        header += " -NOT SUPPORTED-"
    header += "</B>"

    if compile:
        header += ", Pytorch compilation <B>ON</B>"
    if "int8" in model_filename:
        header += ", Quantization <B>Int8</B>"
    header += "<FONT></DIV>"

    return header

def apply_changes(  state,
                    transformer_types_choices,
                    text_encoder_choice,
                    save_path_choice,
                    attention_choice,
                    compile_choice,
                    profile_choice,
                    vae_config_choice,
                    metadata_choice,
                    quantization_choice,
                    boost_choice = 1,
                    clear_file_list = 0,
                    reload_choice = 1,
):
    if args.lock_config:
        return
    if gen_in_progress:
        return "<DIV ALIGN=CENTER>Unable to change config when a generation is in progress</DIV>"
    global offloadobj, wan_model, server_config, loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset, loras_presets
    server_config = {"attention_mode" : attention_choice,  
                     "transformer_types": transformer_types_choices, 
                     "text_encoder_filename" : text_encoder_choices[text_encoder_choice],
                     "save_path" : save_path_choice,
                     "compile" : compile_choice,
                     "profile" : profile_choice,
                     "vae_config" : vae_config_choice,
                     "metadata_choice": metadata_choice,
                     "transformer_quantization" : quantization_choice,
                     "boost" : boost_choice,
                     "clear_file_list" : clear_file_list,
                     "reload_model" : reload_choice,
                       }

    if Path(server_config_filename).is_file():
        with open(server_config_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        old_server_config = json.loads(text)
        if lock_ui_transformer:
            server_config["transformer_filename"] = old_server_config["transformer_filename"]
        if lock_ui_attention:
            server_config["attention_mode"] = old_server_config["attention_mode"]
        if lock_ui_compile:
            server_config["compile"] = old_server_config["compile"]

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))

    changes = []
    for k, v in server_config.items():
        v_old = old_server_config.get(k, None)
        if v != v_old:
            changes.append(k)

    global attention_mode, profile, compile, transformer_filename, text_encoder_filename, vae_config, boost, lora_dir, reload_needed, reload_model, transformer_quantization, transformer_types
    attention_mode = server_config["attention_mode"]
    profile = server_config["profile"]
    compile = server_config["compile"]
    text_encoder_filename = server_config["text_encoder_filename"]
    vae_config = server_config["vae_config"]
    boost = server_config["boost"]
    reload_model = server_config["reload_model"]
    transformer_quantization = server_config["transformer_quantization"]
    transformer_types = server_config["transformer_types"]
    transformer_type = get_model_type(transformer_filename)
    if not transformer_type in transformer_types:
        transformer_type = transformer_types[0] if len(transformer_types) > 0 else  model_types[0]
        transformer_filename = get_model_filename(transformer_type, transformer_quantization)
    if  all(change in ["attention_mode", "vae_config", "boost", "save_path", "metadata_choice", "clear_file_list"] for change in changes ):
        model_choice = gr.Dropdown()
    else:
        reload_needed = True
        model_choice = generate_dropdown_model_list()

    header = generate_header(transformer_filename, compile=compile, attention_mode= attention_mode)
    return "<DIV ALIGN=CENTER>The new configuration has been succesfully applied</DIV>", header, model_choice



from moviepy.editor import ImageSequenceClip
import numpy as np

def save_video(final_frames, output_path, fps=24):
    assert final_frames.ndim == 4 and final_frames.shape[3] == 3, f"invalid shape: {final_frames} (need t h w c)"
    if final_frames.dtype != np.uint8:
        final_frames = (final_frames * 255).astype(np.uint8)
    ImageSequenceClip(list(final_frames), fps=fps).write_videofile(output_path, verbose= False, logger = None)


def get_gen_info(state):
    cache = state.get("gen", None)
    if cache == None:
        cache = dict()
        state["gen"] = cache
    return cache

def build_callback(state, pipe, progress, status, num_inference_steps):
    def callback(step_idx, force_refresh, read_state = False):
        gen = get_gen_info(state)
        refresh_id =  gen.get("refresh", -1)
        if force_refresh or step_idx >= 0:
            pass
        else:
            refresh_id =  gen.get("refresh", -1)
            if refresh_id < 0:
                return
            UI_refresh = state.get("refresh", 0)
            if UI_refresh >= refresh_id:
                return  

        status = gen["progress_status"]
        state["refresh"] = refresh_id
        if read_state:
            phase, step_idx  = gen["progress_phase"] 
        else:
            step_idx += 1         
            if gen.get("abort", False):
                # pipe._interrupt = True
                phase = " - Aborting"    
            elif step_idx  == num_inference_steps:
                phase = " - VAE Decoding"    
            else:
                phase = " - Denoising"   
            gen["progress_phase"] = (phase, step_idx)
        status_msg = status + phase      
        if step_idx >= 0:
            progress_args = [(step_idx , num_inference_steps) , status_msg  ,  num_inference_steps]
        else:
            progress_args = [0, status_msg]
        
        progress(*progress_args)
        gen["progress_args"] = progress_args
            
    return callback
def abort_generation(state):
    gen = get_gen_info(state)
    if "in_progress" in gen:

        gen["abort"] = True
        gen["extra_orders"] = 0
        wan_model._interrupt= True
        msg = "Processing Request to abort Current Generation"
        gr.Info(msg)
        return msg, gr.Button(interactive=  False)
    else:
        return "", gr.Button(interactive=  True)



def refresh_gallery(state, msg):
    gen = get_gen_info(state)

    gen["last_msg"] = msg
    file_list = gen.get("file_list", None)      
    choice = gen.get("selected",0)
    in_progress = "in_progress" in gen
    if in_progress:
        if gen.get("last_selected", True):
            choice = max(len(file_list) - 1,0)  

    queue = gen.get("queue", [])
    abort_interactive = not gen.get("abort", False)
    if not in_progress or len(queue) == 0:
        return gr.Gallery(selected_index=choice, value = file_list), gr.HTML("", visible= False),  gr.Button(visible=True), gr.Button(visible=False), gr.Row(visible=False), update_queue_data(queue), gr.Button(interactive=  abort_interactive)
    else:
        task = queue[0]
        start_img_md = ""
        end_img_md = ""
        prompt =  task["prompt"]

        start_img_uri = task.get('start_image_data_base64')
        start_img_uri = start_img_uri[0] if start_img_uri !=None else None
        end_img_uri = task.get('end_image_data_base64')
        end_img_uri = end_img_uri[0] if end_img_uri !=None else None
        thumbnail_size = "100px"
        if start_img_uri:
            start_img_md = f'<img src="{start_img_uri}" alt="Start" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display: block; margin: auto; object-fit: contain;" />'
        if end_img_uri:
            end_img_md = f'<img src="{end_img_uri}" alt="End" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display: block; margin: auto; object-fit: contain;" />'
        
        label = f"Prompt of Video being Generated"            
 
        html = "<STYLE> #PINFO, #PINFO  th, #PINFO td {border: 1px solid #CCCCCC;background-color:#FFFFFF;}</STYLE><TABLE WIDTH=100% ID=PINFO ><TR><TD width=100%>" + prompt + "</TD>"
        if start_img_md != "":
            html += "<TD>" + start_img_md +  "</TD>"
        if end_img_md != "":
            html += "<TD>" + end_img_md +  "</TD>" 

        html += "</TR></TABLE>" 
        html_output = gr.HTML(html, visible= True)
        return gr.Gallery(selected_index=choice, value = file_list), html_output, gr.Button(visible=False), gr.Button(visible=True), gr.Row(visible=True), update_queue_data(queue), gr.Button(interactive=  abort_interactive)



def finalize_generation(state):
    gen = get_gen_info(state)
    choice = gen.get("selected",0)
    if "in_progress" in gen:
        del gen["in_progress"]
    if gen.get("last_selected", True):
        file_list = gen.get("file_list", [])
        choice = len(file_list) - 1


    gen["extra_orders"] = 0
    time.sleep(0.2)
    global gen_in_progress
    gen_in_progress = False
    return gr.Gallery(selected_index=choice), gr.Button(interactive=  True), gr.Button(visible= True), gr.Button(visible= False), gr.Column(visible= False), gr.HTML(visible= False, value="")


def refresh_gallery_on_trigger(state):
    gen = get_gen_info(state)

    if(gen.get("update_gallery", False)):
        gen['update_gallery'] = False
        return gr.update(value=gen.get("file_list", []))

def select_video(state , event_data: gr.EventData):
    data=  event_data._data
    gen = get_gen_info(state)

    if data!=None:
        choice = data.get("index",0)
        file_list = gen.get("file_list", [])
        gen["last_selected"] = (choice + 1) >= len(file_list)
        gen["selected"] = choice
    return 

def expand_slist(slist, num_inference_steps ):
    new_slist= []
    inc =  len(slist) / num_inference_steps 
    pos = 0
    for i in range(num_inference_steps):
        new_slist.append(slist[ int(pos)])
        pos += inc
    return new_slist
def convert_image(image):

    from PIL import ExifTags, ImageOps
    from typing import cast

    return cast(Image, ImageOps.exif_transpose(image))
    # image = image.convert('RGB')
    # for orientation in ExifTags.TAGS.keys():
    #     if ExifTags.TAGS[orientation]=='Orientation':
    #         break            
    # exif = image.getexif()
    #     return image
    # if not orientation in exif:
    # if exif[orientation] == 3:
    #     image=image.rotate(180, expand=True)
    # elif exif[orientation] == 6:
    #     image=image.rotate(270, expand=True)
    # elif exif[orientation] == 8:
    #     image=image.rotate(90, expand=True)
    # return image

def generate_video(
    task_id,
    progress,
    prompt,
    negative_prompt,    
    resolution,
    video_length,
    seed,
    num_inference_steps,
    guidance_scale,
    flow_shift,
    embedded_guidance_scale,
    repeat_generation,
    multi_images_gen_type,
    tea_cache_setting,
    tea_cache_start_step_perc,    
    activated_loras,
    loras_multipliers,
    image_prompt_type,
    image_start,
    image_end,
    video_prompt_type,
    image_refs,
    video_guide,
    video_mask,
    max_frames,
    remove_background_image_ref,
    temporal_upsampling,
    spatial_upsampling,
    RIFLEx_setting,
    slg_switch,
    slg_layers,    
    slg_start_perc,
    slg_end_perc,
    cfg_star_switch,
    cfg_zero_step,
    state,
    model_filename

):

    global wan_model, offloadobj, reload_needed
    gen = get_gen_info(state)

    file_list = gen["file_list"]
    prompt_no = gen["prompt_no"]


    # if wan_model == None:
    #     gr.Info("Unable to generate a Video while a new configuration is being applied.")
    #     return

    if reload_model !=3 :
        while wan_model == None:
            time.sleep(1)
        
    if model_filename !=  transformer_filename or reload_needed:
        wan_model = None
        if offloadobj is not None:
            offloadobj.release()
            offloadobj = None
        gc.collect()
        yield f"Loading model {get_model_name(model_filename)}..."
        wan_model, offloadobj, trans = load_models(model_filename)
        yield f"Model loaded"
        reload_needed=  False

    if attention_mode == "auto":
        attn = get_auto_attention()
    elif attention_mode in attention_modes_supported:
        attn = attention_mode
    else:
        gr.Info(f"You have selected attention mode '{attention_mode}'. However it is not installed or supported on your system. You should either install it or switch to the default 'sdpa' attention.")
        return
    
    width, height = resolution.split("x")
    width, height = int(width), int(height)
    resolution_reformated = str(height) + "*" + str(width) 

    if slg_switch == 0:
        slg_layers = None

    offload.shared_state["_attention"] =  attn
 
     # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
    if vae_config == 0:
        if device_mem_capacity >= 24000:
            use_vae_config = 1            
        elif device_mem_capacity >= 8000:
            use_vae_config = 2
        else:          
            use_vae_config = 3
    else:
        use_vae_config = vae_config

    if use_vae_config == 1:
        VAE_tile_size = 0  
    elif use_vae_config == 2:
        VAE_tile_size = 256  
    else: 
        VAE_tile_size = 128  

    trans = wan_model.model

    temp_filename = None

    loras = state["loras"]
    if len(loras) > 0:
        def is_float(element: any) -> bool:
            if element is None: 
                return False
            try:
                float(element)
                return True
            except ValueError:
                return False
        list_mult_choices_nums = []
        if len(loras_multipliers) > 0:
            loras_mult_choices_list = loras_multipliers.replace("\r", "").split("\n")
            loras_mult_choices_list = [multi for multi in loras_mult_choices_list if len(multi)>0 and not multi.startswith("#")]
            loras_multipliers = " ".join(loras_mult_choices_list)
            list_mult_choices_str = loras_multipliers.split(" ")
            for i, mult in enumerate(list_mult_choices_str):
                mult = mult.strip()
                if "," in mult:
                    multlist = mult.split(",")
                    slist = []
                    for smult in multlist:
                        if not is_float(smult):                
                            raise gr.Error(f"Lora sub value no {i+1} ({smult}) in Multiplier definition '{multlist}' is invalid")
                        slist.append(float(smult))
                    slist = expand_slist(slist, num_inference_steps )
                    list_mult_choices_nums.append(slist)
                else:
                    if not is_float(mult):                
                        raise gr.Error(f"Lora Multiplier no {i+1} ({mult}) is invalid")
                    list_mult_choices_nums.append(float(mult))
        if len(list_mult_choices_nums ) < len(activated_loras):
            list_mult_choices_nums  += [1.0] * ( len(activated_loras) - len(list_mult_choices_nums ) )        
        loras_selected = [ lora for lora in loras if os.path.basename(lora) in activated_loras]
        pinnedLora = profile !=5 #False # # # 
        offload.load_loras_into_model(trans, loras_selected, list_mult_choices_nums, activate_all_loras=True, preprocess_sd=preprocess_loras, pinnedLora=pinnedLora, split_linear_modules_map = None) 
        errors = trans._loras_errors
        if len(errors) > 0:
            error_files = [msg for _ ,  msg  in errors]
            raise gr.Error("Error while loading Loras: " + ", ".join(error_files))
    seed = None if seed == -1 else seed
    # negative_prompt = "" # not applicable in the inference
    image2video = test_class_i2v(model_filename)
    enable_RIFLEx = RIFLEx_setting == 0 and video_length > (6* 16) or RIFLEx_setting == 1
    # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(None).total_memory / 1048576

    joint_pass = boost ==1 #and profile != 1 and profile != 3  
   # TeaCache   
    trans.enable_teacache = tea_cache_setting > 0
    if trans.enable_teacache:
        trans.teacache_multiplier = tea_cache_setting
        trans.rel_l1_thresh = 0
        trans.teacache_start_step =  int(tea_cache_start_step_perc*num_inference_steps/100)

        if image2video:
            if '480p' in model_filename: 
                # teacache_thresholds = [0.13, .19, 0.26]
                trans.coefficients = [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01,  5.87348440e+00, -2.01973289e-01]
            elif '720p' in model_filename:
                teacache_thresholds = [0.18, 0.2 , 0.3]
                trans.coefficients = [-114.36346466,   65.26524496,  -18.82220707,    4.91518089,   -0.23412683]
            else:
                raise gr.Error("Teacache not supported for this model")
        else:
            if '1.3B' in model_filename:
                # teacache_thresholds= [0.05, 0.07, 0.08]
                trans.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
            elif '14B' in model_filename:
                # teacache_thresholds = [0.14, 0.15, 0.2]
                trans.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
            else:
                    raise gr.Error("Teacache not supported for this model")

    if "Vace" in model_filename:
        src_video, src_mask, src_ref_images = wan_model.prepare_source([video_guide],
                                                                [video_mask],
                                                                [image_refs],
                                                                video_length, VACE_SIZE_CONFIGS[resolution_reformated], "cpu",
                                                                trim_video=max_frames)
    else:
        src_video, src_mask, src_ref_images = None, None, None


    import random
    if seed == None or seed <0:
        seed = random.randint(0, 999999999)

    global save_path
    os.makedirs(save_path, exist_ok=True)
    video_no = 0
    abort = False
    gc.collect()
    torch.cuda.empty_cache()
    wan_model._interrupt = False
    gen["abort"] = False
    gen["prompt"] = prompt    
    repeat_no = 0
    extra_generation = 0
    while True: 
        extra_generation += gen.get("extra_orders",0)
        gen["extra_orders"] = 0
        total_generation = repeat_generation + extra_generation
        gen["total_generation"] = total_generation         
        if abort or repeat_no >= total_generation:
            break
        repeat_no +=1
        gen["repeat_no"] = repeat_no
        prompts_max = gen["prompts_max"]
        status = get_generation_status(prompt_no, prompts_max, repeat_no, total_generation)

        yield status

        gen["progress_status"] = status 
        gen["progress_phase"] = (" - Encoding Prompt", -1 )
        callback = build_callback(state, trans, progress, status, num_inference_steps)
        progress_args = [0, status + " - Encoding Prompt"]
        progress(*progress_args )   
        gen["progress_args"] = progress_args

        try:
            start_time = time.time()
            # with tracker_lock:
            #     progress_tracker[task_id] = {
            #         'current_step': 0,
            #         'total_steps': num_inference_steps,
            #         'start_time': start_time,
            #         'last_update': start_time,
            #         'repeats': repeat_generation, # f"{video_no}/{repeat_generation}",
            #         'status': "Encoding Prompt"
            #     }
            if trans.enable_teacache:
                trans.teacache_counter = 0
                trans.num_steps = num_inference_steps                
                trans.teacache_skipped_steps = 0    
                trans.previous_residual_uncond = None
                trans.previous_residual_cond = None

            video_no += 1
            if image2video:
                samples = wan_model.generate(
                    prompt,
                    image_start,  
                    image_end if image_end != None else None,  
                    frame_num=(video_length // 4)* 4 + 1,
                    max_area=MAX_AREA_CONFIGS[resolution_reformated], 
                    shift=flow_shift,
                    sampling_steps=num_inference_steps,
                    guide_scale=guidance_scale,
                    n_prompt=negative_prompt,
                    seed=seed,
                    offload_model=False,
                    callback=callback,
                    enable_RIFLEx = enable_RIFLEx,
                    VAE_tile_size = VAE_tile_size,
                    joint_pass = joint_pass,
                    slg_layers = slg_layers,
                    slg_start = slg_start_perc/100,
                    slg_end = slg_end_perc/100,
                    cfg_star_switch = cfg_star_switch,
                    cfg_zero_step = cfg_zero_step,
                    add_frames_for_end_image = not "Fun_InP" in model_filename,
                )
            else:
                samples = wan_model.generate(
                    prompt,
                    input_frames = src_video,
                    input_ref_images=  src_ref_images,
                    input_masks = src_mask,
                    frame_num=(video_length // 4)* 4 + 1,
                    size=(width, height),
                    shift=flow_shift,
                    sampling_steps=num_inference_steps,
                    guide_scale=guidance_scale,
                    n_prompt=negative_prompt,
                    seed=seed,
                    offload_model=False,
                    callback=callback,
                    enable_RIFLEx = enable_RIFLEx,
                    VAE_tile_size = VAE_tile_size,
                    joint_pass = joint_pass,
                    slg_layers = slg_layers,
                    slg_start = slg_start_perc/100,
                    slg_end = slg_end_perc/100,
                    cfg_star_switch = cfg_star_switch,
                    cfg_zero_step = cfg_zero_step,
                )
            # samples = torch.empty( (1,2)) #for testing
        except Exception as e:
            if temp_filename!= None and  os.path.isfile(temp_filename):
                os.remove(temp_filename)
            offload.last_offload_obj.unload_all()
            offload.unload_loras_from_model(trans)
            # if compile:
            #     cache_size = torch._dynamo.config.cache_size_limit                                      
            #     torch.compiler.reset()
            #     torch._dynamo.config.cache_size_limit = cache_size

            gc.collect()
            torch.cuda.empty_cache()
            s = str(e)
            keyword_list = ["vram", "VRAM", "memory","allocat"]
            VRAM_crash= False
            if any( keyword in s for keyword in keyword_list):
                VRAM_crash = True
            else:
                stack = traceback.extract_stack(f=None, limit=5)
                for frame in stack:
                    if any( keyword in frame.name for keyword in keyword_list):
                        VRAM_crash = True
                        break

            state["prompt"] = ""
            if VRAM_crash:
                new_error = "The generation of the video has encountered an error: it is likely that you have unsufficient VRAM and you should therefore reduce the video resolution or its number of frames."
            else:
                new_error =  gr.Error(f"The generation of the video has encountered an error, please check your terminal for more information. '{s}'")
            tb = traceback.format_exc().split('\n')[:-1] 
            print('\n'.join(tb))
            raise gr.Error(new_error, print_exception= False)

        finally:
            pass
            # with tracker_lock:
            #     if task_id in progress_tracker:
            #         del progress_tracker[task_id]

        if trans.enable_teacache:
            print(f"Teacache Skipped Steps:{trans.teacache_skipped_steps}/{num_inference_steps}" )
            trans.previous_residual_uncond = None
            trans.previous_residual_cond = None

        if samples != None:
            samples = samples.to("cpu")
        offload.last_offload_obj.unload_all()
        gc.collect()
        torch.cuda.empty_cache()

        if samples == None:
            end_time = time.time()
            abort = True
            state["prompt"] = ""
            # yield f"Video generation was aborted. Total Generation Time: {end_time-start_time:.1f}s"
        else:
            sample = samples.cpu()

            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
            if os.name == 'nt':
                file_name = f"{time_flag}_seed{seed}_{sanitize_file_name(prompt[:50]).strip()}.mp4"
            else:
                file_name = f"{time_flag}_seed{seed}_{sanitize_file_name(prompt[:100]).strip()}.mp4"
            video_path = os.path.join(save_path, file_name)
            # if False: # for testing
            #     torch.save(sample, "output.pt")
            # else:
            #     sample =torch.load("output.pt")
            exp = 0
            fps = 16

            if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0:                
                progress_args = [(num_inference_steps , num_inference_steps) , status + " - Upsampling"  ,  num_inference_steps]
                progress(*progress_args )   
                gen["progress_args"] = progress_args

            if temporal_upsampling == "rife2":
                exp = 1
            elif temporal_upsampling == "rife4":
                exp = 2
            
            if exp > 0: 
                from rife.inference import temporal_interpolation
                sample = temporal_interpolation( os.path.join("ckpts", "flownet.pkl"), sample, exp, device=processing_device)
                fps = fps * 2**exp

            if len(spatial_upsampling) > 0:
                from wan.utils.utils import resize_lanczos # need multithreading or to do lanczos with cuda
                if spatial_upsampling == "lanczos1.5":
                    scale = 1.5
                else:
                    scale = 2
                sample = (sample + 1) / 2
                h, w = sample.shape[-2:]
                h *= scale
                w *= scale
                h = int(h)
                w = int(w)
                new_frames =[]
                for i in range( sample.shape[1] ):
                    frame = sample[:, i]
                    frame = resize_lanczos(frame, h, w)
                    frame = frame.unsqueeze(1)
                    new_frames.append(frame)
                sample = torch.cat(new_frames, dim=1)
                new_frames = None
                sample = sample * 2 - 1


            cache_video(
                tensor=sample[None],
                save_file=video_path,
                fps=fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))

            inputs = get_function_arguments(generate_video, locals())
            inputs.pop("progress")
            configs = prepare_inputs_dict("metadata", inputs)

            metadata_choice = server_config.get("metadata_choice","metadata")
            if metadata_choice == "json":
                with open(video_path.replace('.mp4', '.json'), 'w') as f:
                    json.dump(configs, f, indent=4)
            elif metadata_choice == "metadata":
                from mutagen.mp4 import MP4
                file = MP4(video_path)
                file.tags['©cmt'] = [json.dumps(configs)]
                file.save()

            print(f"New video saved to Path: "+video_path)
            file_list.append(video_path)
            state['update_gallery'] = True
        seed += 1
  
    if temp_filename!= None and  os.path.isfile(temp_filename):
        os.remove(temp_filename)
    offload.unload_loras_from_model(trans)

def prepare_generate_video(state):    
    if state.get("validate_success",0) != 1:
        return gr.Button(visible= True), gr.Button(visible= False), gr.Column(visible= False)
    else:
        return gr.Button(visible= False), gr.Button(visible= True), gr.Column(visible= True)


def process_tasks(state, progress=gr.Progress()):
    gen = get_gen_info(state)
    queue = gen.get("queue", [])

    if len(queue) == 0:
        return
    gen = get_gen_info(state)
    clear_file_list = server_config.get("clear_file_list", 0)    
    file_list = gen.get("file_list", [])
    if clear_file_list > 0:
        file_list_current_size = len(file_list)
        keep_file_from = max(file_list_current_size - clear_file_list, 0)
        files_removed = keep_file_from
        choice = gen.get("selected",0)
        choice = max(choice- files_removed, 0)
        file_list = file_list[ keep_file_from: ]
    else:
        file_list = []
        choice = 0
    gen["selected"] = choice         
    gen["file_list"] = file_list    

    start_time = time.time()

    global gen_in_progress
    gen_in_progress = True
    gen["in_progress"] = True

    prompt_no = 0
    while len(queue) > 0:
        prompt_no += 1
        gen["prompt_no"] = prompt_no
        task = queue[0]
        task_id = task["id"] 
        params = task['params']
        iterator = iter(generate_video(task_id,   progress, **params))
        while True:
            try:
                ok = False
                status = next(iterator, "#")
                ok = True
                if status == "#":
                    break
            except Exception as e:
                _ , exc_value, exc_traceback = sys.exc_info()
                raise exc_value.with_traceback(exc_traceback)
            finally:
                if not ok:
                    queue.clear()
            yield status

        queue[:] = [item for item in queue if item['id'] != task['id']]

    gen["prompts_max"] = 0
    gen["prompt"] = ""
    end_time = time.time()
    if gen.get("abort"):
        yield f"Video generation was aborted. Total Generation Time: {end_time-start_time:.1f}s"
    else:
        yield f"Total Generation Time: {end_time-start_time:.1f}s"


def get_generation_status(prompt_no, prompts_max, repeat_no, repeat_max):
    if prompts_max == 1:
        if repeat_max == 1:
            return "Video"
        else:
            return f"Sample {repeat_no}/{repeat_max}"
    else:
        if repeat_max == 1:
            return f"Prompt {prompt_no}/{prompts_max}"
        else:
            return f"Prompt {prompt_no}/{prompts_max}, Sample {repeat_no}/{repeat_max}"


refresh_id = 0

def get_new_refresh_id():
    global refresh_id
    refresh_id += 1
    return refresh_id

def update_status(state):
    gen = get_gen_info(state)
    prompt_no = gen["prompt_no"] 
    prompts_max = gen.get("prompts_max",0)
    total_generation = gen["total_generation"] 
    repeat_no = gen["repeat_no"]
    status = get_generation_status(prompt_no, prompts_max, repeat_no, total_generation)
    gen["progress_status"] = status
    gen["refresh"] = get_new_refresh_id()


def one_more_sample(state):
    gen = get_gen_info(state)
    extra_orders = gen.get("extra_orders", 0)
    extra_orders += 1
    gen["extra_orders"]  = extra_orders
    in_progress = gen.get("in_progress", False)
    if not in_progress :
        return state
    prompt_no = gen["prompt_no"] 
    prompts_max = gen.get("prompts_max",0)
    total_generation = gen["total_generation"] + extra_orders
    repeat_no = gen["repeat_no"]
    status = get_generation_status(prompt_no, prompts_max, repeat_no, total_generation)


    gen["progress_status"] = status
    gen["refresh"] = get_new_refresh_id()
    gr.Info(f"An extra sample generation is planned for a total of {total_generation} videos for this prompt")

    return state 

def get_new_preset_msg(advanced = True):
    if advanced:
        return "Enter here a Name for a Lora Preset or Choose one in the List"
    else:
        return "Choose a Lora Preset in this List to Apply a Special Effect"


def validate_delete_lset(lset_name):
    if len(lset_name) == 0 or lset_name == get_new_preset_msg(True) or lset_name == get_new_preset_msg(False):
        gr.Info(f"Choose a Preset to delete")
        return  gr.Button(visible= True), gr.Checkbox(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False) 
    else:
        return  gr.Button(visible= False), gr.Checkbox(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= True), gr.Button(visible= True) 
    
def validate_save_lset(lset_name):
    if len(lset_name) == 0 or lset_name == get_new_preset_msg(True) or lset_name == get_new_preset_msg(False):
        gr.Info("Please enter a name for the preset")
        return  gr.Button(visible= True), gr.Checkbox(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False),gr.Checkbox(visible= False) 
    else:
        return  gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= True), gr.Button(visible= True),gr.Checkbox(visible= True)

def cancel_lset():
    return gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Checkbox(visible= False)



def save_lset(state, lset_name, loras_choices, loras_mult_choices, prompt, save_lset_prompt_cbox):    
    loras_presets = state["loras_presets"] 
    loras = state["loras"]
    if state.get("validate_success",0) == 0:
        pass
    if len(lset_name) == 0 or lset_name == get_new_preset_msg(True) or lset_name == get_new_preset_msg(False):
        gr.Info("Please enter a name for the preset")
        lset_choices =[("Please enter a name for a Lora Preset","")]
    else:
        lset_name = sanitize_file_name(lset_name)

        loras_choices_files = [ Path(loras[int(choice_no)]).parts[-1] for choice_no in loras_choices  ]
        lset  = {"loras" : loras_choices_files, "loras_mult" : loras_mult_choices}
        if save_lset_prompt_cbox!=1:
            prompts = prompt.replace("\r", "").split("\n")
            prompts = [prompt for prompt in prompts if len(prompt)> 0 and prompt.startswith("#")]
            prompt = "\n".join(prompts)

        if len(prompt) > 0:
            lset["prompt"] = prompt
        lset["full_prompt"] = save_lset_prompt_cbox ==1
        

        lset_name_filename = lset_name + ".lset" 
        full_lset_name_filename = os.path.join(get_lora_dir(state["model_filename"]), lset_name_filename) 

        with open(full_lset_name_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(lset, indent=4))

        if lset_name in loras_presets:
            gr.Info(f"Lora Preset '{lset_name}' has been updated")
        else:
            gr.Info(f"Lora Preset '{lset_name}' has been created")
            loras_presets.append(Path(Path(lset_name_filename).parts[-1]).stem )
        lset_choices = [ ( preset, preset) for preset in loras_presets ]
        lset_choices.append( (get_new_preset_msg(), ""))
        state["loras_presets"] = loras_presets
    return gr.Dropdown(choices=lset_choices, value= lset_name), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False), gr.Checkbox(visible= False)

def delete_lset(state, lset_name):
    loras_presets = state["loras_presets"]
    lset_name_filename = os.path.join( get_lora_dir(state["model_filename"]),  sanitize_file_name(lset_name) + ".lset" )
    if len(lset_name) > 0 and lset_name != get_new_preset_msg(True) and  lset_name != get_new_preset_msg(False):
        if not os.path.isfile(lset_name_filename):
            raise gr.Error(f"Preset '{lset_name}' not found ")
        os.remove(lset_name_filename)
        pos = loras_presets.index(lset_name) 
        gr.Info(f"Lora Preset '{lset_name}' has been deleted")
        loras_presets.remove(lset_name)
    else:
        pos = len(loras_presets) 
        gr.Info(f"Choose a Preset to delete")

    state["loras_presets"] = loras_presets

    lset_choices = [ (preset, preset) for preset in loras_presets]
    lset_choices.append((get_new_preset_msg(), ""))
    return  gr.Dropdown(choices=lset_choices, value= lset_choices[pos][1]), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Checkbox(visible= False)

def refresh_lora_list(state, lset_name, loras_choices):
    loras_names = state["loras_names"]
    prev_lora_names_selected = [ loras_names[int(i)] for i in loras_choices]
    model_filename= state["model_filename"]
    loras, loras_names, loras_presets, _, _, _, _  = setup_loras(model_filename, None,  get_lora_dir(model_filename), lora_preselected_preset, None)
    state["loras"] = loras
    state["loras_names"] = loras_names
    state["loras_presets"] = loras_presets

    gc.collect()
    new_loras_choices = [ (loras_name, str(i)) for i,loras_name in enumerate(loras_names)]
    new_loras_dict = { loras_name: str(i) for i,loras_name in enumerate(loras_names) }
    lora_names_selected = []
    for lora in prev_lora_names_selected:
        lora_id = new_loras_dict.get(lora, None)
        if lora_id!= None:
            lora_names_selected.append(lora_id)

    lset_choices = [ (preset, preset) for preset in loras_presets]
    lset_choices.append((get_new_preset_msg( state["advanced"]), "")) 
    if lset_name in loras_presets:
        pos = loras_presets.index(lset_name) 
    else:
        pos = len(loras_presets)
        lset_name =""
    
    errors = getattr(wan_model.model, "_loras_errors", "")
    if errors !=None and len(errors) > 0:
        error_files = [path for path, _ in errors]
        gr.Info("Error while refreshing Lora List, invalid Lora files: " + ", ".join(error_files))
    else:
        gr.Info("Lora List has been refreshed")


    return gr.Dropdown(choices=lset_choices, value= lset_choices[pos][1]), gr.Dropdown(choices=new_loras_choices, value= lora_names_selected) 

def apply_lset(state, wizard_prompt_activated, lset_name, loras_choices, loras_mult_choices, prompt):

    state["apply_success"] = 0

    if len(lset_name) == 0 or lset_name== get_new_preset_msg(True) or lset_name== get_new_preset_msg(False):
        gr.Info("Please choose a preset in the list or create one")
    else:
        loras = state["loras"]
        loras_choices, loras_mult_choices, preset_prompt, full_prompt, error = extract_preset(state["model_filename"],  lset_name, loras)
        if len(error) > 0:
            gr.Info(error)
        else:
            if full_prompt:
                prompt = preset_prompt
            elif len(preset_prompt) > 0:
                prompts = prompt.replace("\r", "").split("\n")
                prompts = [prompt for prompt in prompts if len(prompt)>0 and not prompt.startswith("#")]
                prompt = "\n".join(prompts) 
                prompt = preset_prompt + '\n' + prompt
            gr.Info(f"Lora Preset '{lset_name}' has been applied")
            state["apply_success"] = 1
            wizard_prompt_activated = "on"

    return wizard_prompt_activated, loras_choices, loras_mult_choices, prompt


def extract_prompt_from_wizard(state, variables_names, prompt, wizard_prompt, allow_null_values, *args):

    prompts = wizard_prompt.replace("\r" ,"").split("\n")

    new_prompts = [] 
    macro_already_written = False
    for prompt in prompts:
        if not macro_already_written and not prompt.startswith("#") and "{"  in prompt and "}"  in prompt:
            variables =  variables_names.split("\n")   
            values = args[:len(variables)]
            macro = "! "
            for i, (variable, value) in enumerate(zip(variables, values)):
                if len(value) == 0 and not allow_null_values:
                    return prompt, "You need to provide a value for '" + variable + "'" 
                sub_values= [ "\"" + sub_value + "\"" for sub_value in value.split("\n") ]
                value = ",".join(sub_values)
                if i>0:
                    macro += " : "    
                macro += "{" + variable + "}"+ f"={value}"
            if len(variables) > 0:
                macro_already_written = True
                new_prompts.append(macro)
            new_prompts.append(prompt)
        else:
            new_prompts.append(prompt)

    prompt = "\n".join(new_prompts)
    return prompt, ""

def validate_wizard_prompt(state, wizard_prompt_activated, wizard_variables_names, prompt, wizard_prompt, *args):
    state["validate_success"] = 0

    if wizard_prompt_activated != "on":
        state["validate_success"] = 1
        return prompt

    prompt, errors = extract_prompt_from_wizard(state, wizard_variables_names, prompt, wizard_prompt, False, *args)
    if len(errors) > 0:
        gr.Info(errors)
        return prompt

    state["validate_success"] = 1

    return prompt

def fill_prompt_from_wizard(state, wizard_prompt_activated, wizard_variables_names, prompt, wizard_prompt, *args):

    if wizard_prompt_activated == "on":
        prompt, errors = extract_prompt_from_wizard(state, wizard_variables_names, prompt,  wizard_prompt, True, *args)
        if len(errors) > 0:
            gr.Info(errors)

        wizard_prompt_activated = "off"

    return wizard_prompt_activated, "", gr.Textbox(visible= True, value =prompt) , gr.Textbox(visible= False), gr.Column(visible = True), *[gr.Column(visible = False)] * 2,  *[gr.Textbox(visible= False)] * PROMPT_VARS_MAX

def extract_wizard_prompt(prompt):
    variables = []
    values = {}
    prompts = prompt.replace("\r" ,"").split("\n")
    if sum(prompt.startswith("!") for prompt in prompts) > 1:
        return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"

    new_prompts = [] 
    errors = ""
    for prompt in prompts:
        if prompt.startswith("!"):
            variables, errors = prompt_parser.extract_variable_names(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
            if len(variables) > PROMPT_VARS_MAX:
                return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"
            values, errors = prompt_parser.extract_variable_values(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
        else:
            variables_extra, errors = prompt_parser.extract_variable_names(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
            variables += variables_extra
            variables = [var for pos, var in enumerate(variables) if var not in variables[:pos]]
            if len(variables) > PROMPT_VARS_MAX:
                return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"

            new_prompts.append(prompt)
    wizard_prompt = "\n".join(new_prompts)
    return  wizard_prompt, variables, values, errors

def fill_wizard_prompt(state, wizard_prompt_activated, prompt, wizard_prompt):
    def get_hidden_textboxes(num = PROMPT_VARS_MAX ):
        return [gr.Textbox(value="", visible=False)] * num

    hidden_column =  gr.Column(visible = False)
    visible_column =  gr.Column(visible = True)

    wizard_prompt_activated  = "off"  
    if state["advanced"] or state.get("apply_success") != 1:
        return wizard_prompt_activated, gr.Text(), prompt, wizard_prompt, gr.Column(), gr.Column(), hidden_column,  *get_hidden_textboxes() 
    prompt_parts= []

    wizard_prompt, variables, values, errors =  extract_wizard_prompt(prompt)
    if len(errors) > 0:
        gr.Info( errors )
        return wizard_prompt_activated, "", gr.Textbox(prompt, visible=True), gr.Textbox(wizard_prompt, visible=False), visible_column, *[hidden_column] * 2, *get_hidden_textboxes()

    for variable in variables:
        value = values.get(variable, "")
        prompt_parts.append(gr.Textbox( placeholder=variable, info= variable, visible= True, value= "\n".join(value) ))
    any_macro = len(variables) > 0

    prompt_parts += get_hidden_textboxes(PROMPT_VARS_MAX-len(prompt_parts))

    variables_names= "\n".join(variables)
    wizard_prompt_activated  = "on"

    return wizard_prompt_activated, variables_names,  gr.Textbox(prompt, visible = False),  gr.Textbox(wizard_prompt, visible = True),   hidden_column, visible_column, visible_column if any_macro else hidden_column, *prompt_parts

def switch_prompt_type(state, wizard_prompt_activated_var, wizard_variables_names, prompt, wizard_prompt, *prompt_vars):
    if state["advanced"]:
        return fill_prompt_from_wizard(state, wizard_prompt_activated_var, wizard_variables_names, prompt, wizard_prompt, *prompt_vars)
    else:
        state["apply_success"] = 1
        return fill_wizard_prompt(state, wizard_prompt_activated_var, prompt, wizard_prompt)

visible= False
def switch_advanced(state, new_advanced, lset_name):
    state["advanced"] = new_advanced
    loras_presets = state["loras_presets"]
    lset_choices = [ (preset, preset) for preset in loras_presets]
    lset_choices.append((get_new_preset_msg(new_advanced), ""))
    if lset_name== get_new_preset_msg(True) or lset_name== get_new_preset_msg(False) or lset_name=="":
        lset_name =  get_new_preset_msg(new_advanced)

    if only_allow_edit_in_advanced:
        return  gr.Row(visible=new_advanced), gr.Row(visible=new_advanced), gr.Button(visible=new_advanced), gr.Row(visible= not new_advanced), gr.Dropdown(choices=lset_choices, value= lset_name)
    else:
        return  gr.Row(visible=new_advanced), gr.Row(visible=True), gr.Button(visible=True), gr.Row(visible= False), gr.Dropdown(choices=lset_choices, value= lset_name)


def prepare_inputs_dict(target, inputs ):
    
    state = inputs.pop("state")
    loras = state["loras"]
    if "loras_choices" in inputs:
        loras_choices = inputs.pop("loras_choices")
        inputs.pop("model_filename", None)
        activated_loras = [Path( loras[int(no)]).parts[-1]  for no in loras_choices ]
        inputs["activated_loras"] = activated_loras

    if target == "state":
        return inputs

    unsaved_params = ["image_start", "image_end", "image_refs", "video_guide", "video_mask"]
    for k in unsaved_params:
        inputs.pop(k)

    model_filename = state["model_filename"]
    inputs["type"] = "Wan2.1GP by DeepBeepMeep - " +  get_model_name(model_filename)

    if target == "settings":
        return inputs
    
    if not any(k in model_filename for k in ["image2video", "Fun_InP"]):
        inputs.pop("image_prompt_type")


    if not "Vace" in model_filename:
        unsaved_params = ["video_prompt_type", "max_frames", "remove_background_image_ref"]
        for k in unsaved_params:
            inputs.pop(k)

    if target == "metadata":
        inputs = {k: v for k,v in inputs.items() if v != None  }

    return inputs

def get_function_arguments(func, locals):
    args_names = list(inspect.signature(func).parameters)
    kwargs = typing.OrderedDict()
    for k in args_names:
        kwargs[k] = locals[k]
    return kwargs
            

def save_inputs(
            target,
            prompt,
            negative_prompt,
            resolution,
            video_length,
            seed,
            num_inference_steps,
            guidance_scale,
            flow_shift,
            embedded_guidance_scale,
            repeat_generation,
            multi_images_gen_type,
            tea_cache_setting,
            tea_cache_start_step_perc,
            loras_choices,
            loras_multipliers,
            image_prompt_type,
            image_start,
            image_end,
            video_prompt_type,            
            image_refs,
            video_guide,
            video_mask,
            max_frames,
            remove_background_image_ref,
            temporal_upsampling,
            spatial_upsampling,
            RIFLEx_setting,
            slg_switch, 
            slg_layers,
            slg_start_perc,
            slg_end_perc,
            cfg_star_switch,
            cfg_zero_step,
            state,
):

  
    # if state.get("validate_success",0) != 1:
    #     return
    model_filename = state["model_filename"]
    inputs = get_function_arguments(save_inputs, locals())
    inputs.pop("target")
    cleaned_inputs = prepare_inputs_dict(target, inputs)
    if target == "settings":
        defaults_filename = get_settings_file_name(model_filename)

        with open(defaults_filename, "w", encoding="utf-8") as f:
            json.dump(cleaned_inputs, f, indent=4)

        gr.Info("New Default Settings saved")
    elif target == "state":
        state[get_model_type(model_filename)] = cleaned_inputs

def download_loras():
    from huggingface_hub import  snapshot_download    
    yield gr.Row(visible=True), "<B><FONT SIZE=3>Please wait while the Loras are being downloaded</B></FONT>", *[gr.Column(visible=False)] * 2
    lora_dir = get_lora_dir(get_model_filename("i2v"), quantizeTransformer)
    log_path = os.path.join(lora_dir, "log.txt")
    if not os.path.isfile(log_path):
        import shutil 
        tmp_path = os.path.join(lora_dir, "tmp_lora_dowload")
        import glob
        snapshot_download(repo_id="DeepBeepMeep/Wan2.1",  allow_patterns="loras_i2v/*", local_dir= tmp_path)
        for f in glob.glob(os.path.join(tmp_path, "loras_i2v", "*.*")):
            target_file = os.path.join(lora_dir,  Path(f).parts[-1] )
            if os.path.isfile(target_file):
                os.remove(f)
            else:
                shutil.move(f, lora_dir) 
    try:
        os.remove(tmp_path)
    except:
        pass
    yield gr.Row(visible=True), "<B><FONT SIZE=3>Loras have been completely downloaded</B></FONT>", *[gr.Column(visible=True)] * 2

    from datetime import datetime
    dt = datetime.today().strftime('%Y-%m-%d')
    with open( log_path, "w", encoding="utf-8") as writer:
        writer.write(f"Loras downloaded on the {dt} at {time.time()} on the {time.time()}")
    return

def refresh_image_prompt_type(state, image_prompt_type):
    if args.multiple_images:
        return gr.Gallery(visible = "S" in image_prompt_type ), gr.Gallery(visible = "E" in image_prompt_type )
    else:
        return gr.Image(visible = "S" in image_prompt_type ), gr.Image(visible = "E" in image_prompt_type )

def refresh_video_prompt_type(state, video_prompt_type):
    return gr.Gallery(visible = "I" in video_prompt_type), gr.Video(visible= "V" in video_prompt_type),gr.Video(visible= "M" in video_prompt_type ), gr.Text(visible= "V" in video_prompt_type) , gr.Checkbox(visible= "I" in video_prompt_type)


def handle_celll_selection(state, evt: gr.SelectData):
    gen = get_gen_info(state)
    queue = gen.get("queue", [])

    if evt.index is None:
        return gr.update(), gr.update(), gr.update(visible=False)
    row_index, col_index = evt.index
    cell_value = None
    if col_index in [6, 7, 8]:
        if col_index == 6: cell_value = "↑"
        elif col_index == 7: cell_value = "↓"
        elif col_index == 8: cell_value = "✖"
    if col_index == 6:
        new_df_data = move_up(queue, [row_index])
        return new_df_data, gr.update(), gr.update(visible=False)
    elif col_index == 7:
        new_df_data = move_down(queue, [row_index])
        return new_df_data, gr.update(), gr.update(visible=False)
    elif col_index == 8:
        new_df_data = remove_task(queue, [row_index])
        gen["prompts_max"] = gen.get("prompts_max",0) - 1
        update_status(state)
        return new_df_data, gr.update(), gr.update(visible=False)
    start_img_col_idx = 4
    end_img_col_idx = 5
    image_data_to_show = None
    if col_index == start_img_col_idx:
        with lock:
            row_index += 1
            if row_index < len(queue):
                image_data_to_show = queue[row_index].get('start_image_data')
    elif col_index == end_img_col_idx:
        with lock:
            row_index += 1
            if row_index < len(queue):
                image_data_to_show = queue[row_index].get('end_image_data')

    if image_data_to_show:
        return gr.update(), gr.update(value=image_data_to_show[0]), gr.update(visible=True)
    else:
        return gr.update(), gr.update(), gr.update(visible=False)


def change_model(state, model_choice):
    if model_choice == None:
        return
    model_filename = get_model_filename(model_choice, transformer_quantization)
    state["model_filename"] = model_filename
    header = generate_header(model_filename, compile=compile, attention_mode=attention_mode)
    return header

def fill_inputs(state):
    model_filename = state["model_filename"]
    prefix = get_model_type(model_filename)
    ui_defaults = state.get(prefix, None)
    if ui_defaults == None:
        ui_defaults = get_default_settings(model_filename)
 
    return generate_video_tab(update_form = True, state_dict = state, ui_defaults = ui_defaults)

def preload_model(state):
    global reload_needed, wan_model, offloadobj
    if reload_model == 1:
        model_filename = state["model_filename"] 
        if  state["model_filename"] !=  transformer_filename:
            wan_model = None
            if offloadobj is not None:
                offloadobj.release()
                offloadobj = None
            gc.collect()
            yield f"Loading model {get_model_name(model_filename)}..."
            wan_model, offloadobj, _ = load_models(model_filename)
            yield f"Model loaded"
            reload_needed=  False 
        return   
    return gr.Text()

def unload_model_if_needed(state):
    global reload_needed, wan_model, offloadobj
    if reload_model == 3:
        if wan_model != None:
            wan_model = None
            if offloadobj is not None:
                offloadobj.release()
                offloadobj = None
            gc.collect()
            reload_needed=  True


def generate_video_tab(update_form = False, state_dict = None, ui_defaults = None, model_choice = None, header = None):
    global inputs_names #, advanced

    if update_form:
        model_filename = state_dict["model_filename"]
        advanced_ui = state_dict["advanced"]  
    else:
        model_filename = transformer_filename
        advanced_ui = advanced
        ui_defaults=  get_default_settings(model_filename)
        state_dict = {}
        state_dict["model_filename"] = model_filename
        state_dict["advanced"] = advanced_ui
        gen = dict()
        gen["queue"] = []
        state_dict["gen"] = gen

    preset_to_load = lora_preselected_preset if lora_preset_model == model_filename else "" 

    loras, loras_names, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset = setup_loras(model_filename,  None,  get_lora_dir(model_filename), preset_to_load, None)

    state_dict["loras"] = loras
    state_dict["loras_presets"] = loras_presets
    state_dict["loras_names"] = loras_names

    launch_prompt = ""
    launch_preset = ""
    launch_loras = []
    launch_multis_str = ""

    if update_form:
        pass
    if len(default_lora_preset) > 0 and lora_preset_model == model_filename:
        launch_preset = default_lora_preset
        launch_prompt = default_lora_preset_prompt 
        launch_loras = default_loras_choices
        launch_multis_str = default_loras_multis_str

    if len(launch_prompt) == 0:
        launch_prompt = ui_defaults.get("prompt","")
    if len(launch_loras) == 0:
        launch_multis_str = ui_defaults.get("loras_multipliers","")
        activated_loras = ui_defaults.get("activated_loras",[])
        if len(activated_loras) > 0:
            lora_filenames = [os.path.basename(lora_path) for lora_path in loras]
            activated_indices = []
            for lora_file in ui_defaults["activated_loras"]:
                try:
                    idx = lora_filenames.index(lora_file)
                    activated_indices.append(str(idx))
                except ValueError:
                    print(f"Warning: Lora file {lora_file} from config not found in loras directory")
            launch_loras = activated_indices

    with gr.Row():
        with gr.Column():
            with gr.Column(visible=False, elem_id="image-modal-container") as modal_container:
                with gr.Row(elem_id="image-modal-close-button-row"):
                     close_modal_button = gr.Button("❌", size="sm")
                modal_image_display = gr.Image(label="Full Resolution Image", interactive=False, show_label=False)
            with gr.Row(visible= True): #len(loras)>0) as presets_column:
                lset_choices = [ (preset, preset) for preset in loras_presets ] + [(get_new_preset_msg(advanced_ui), "")]
                with gr.Column(scale=6):
                    lset_name = gr.Dropdown(show_label=False, allow_custom_value= True, scale=5, filterable=True, choices= lset_choices, value=launch_preset)
                with gr.Column(scale=1):
                    with gr.Row(height=17):
                        apply_lset_btn = gr.Button("Apply Lora Preset", size="sm", min_width= 1)
                        refresh_lora_btn = gr.Button("Refresh", size="sm", min_width= 1, visible=advanced_ui or not only_allow_edit_in_advanced)
                        save_lset_prompt_drop= gr.Dropdown(
                            choices=[
                                ("Save Prompt Comments Only", 0),
                                ("Save Full Prompt", 1)
                            ],  show_label= False, container=False, value =1, visible= False
                        ) 
                    with gr.Row(height=17, visible=False) as refresh2_row:
                        refresh_lora_btn2 = gr.Button("Refresh", size="sm", min_width= 1)

                    with gr.Row(height=17, visible=advanced_ui or not only_allow_edit_in_advanced) as preset_buttons_rows:
                        confirm_save_lset_btn = gr.Button("Go Ahead Save it !", size="sm", min_width= 1, visible=False) 
                        confirm_delete_lset_btn = gr.Button("Go Ahead Delete it !", size="sm", min_width= 1, visible=False) 
                        save_lset_btn = gr.Button("Save", size="sm", min_width= 1)
                        delete_lset_btn = gr.Button("Delete", size="sm", min_width= 1)
                        cancel_lset_btn = gr.Button("Don't do it !", size="sm", min_width= 1 , visible=False)  

            if not update_form:
                state = gr.State(state_dict)     
            trigger_refresh_input_type = gr.Text(interactive= False, visible= False)
            with gr.Column(visible= "image2video" in model_filename or "Fun_InP" in model_filename ) as image_prompt_column: 
                image_prompt_type_value= ui_defaults.get("image_prompt_type","S")
                image_prompt_type = gr.Radio( [("Use only a Start Image", "S"),("Use both a Start and an End Image", "SE")], value =image_prompt_type_value, label="Location", show_label= False, scale= 3)

                if args.multiple_images:  
                    image_start = gr.Gallery(
                            label="Images as starting points for new videos", type ="pil", #file_types= "image", 
                            columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= True, value= ui_defaults.get("image_start", None), visible= "S" in image_prompt_type_value)
                else:
                    image_start = gr.Image(label= "Image as a starting point for a new video", type ="pil",value= ui_defaults.get("image_start", None), visible= "S" in image_prompt_type_value )

                if args.multiple_images:  
                    image_end  = gr.Gallery(
                            label="Images as ending points for new videos", type ="pil", #file_types= "image", 
                            columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= True, visible="E" in image_prompt_type_value, value= ui_defaults.get("image_end", None))
                else:
                    image_end = gr.Image(label= "Last Image for a new video", type ="pil", visible="E" in image_prompt_type_value, value= ui_defaults.get("image_end", None))

            with gr.Column(visible= "Vace" in model_filename ) as video_prompt_column: 
                video_prompt_type_value= ui_defaults.get("video_prompt_type","I")
                video_prompt_type = gr.Radio( [("Use Images Ref", "I"),("a Video", "V"), ("Images + a Video", "IV"), ("Video + Video Mask", "VM"), ("Images + Video + Mask", "IVM")], value =video_prompt_type_value, label="Location", show_label= False, scale= 3)
                image_refs = gr.Gallery(
                        label="Reference Images of Faces and / or Object to be found in the Video", type ="pil",  
                        columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= True, visible= "I" in video_prompt_type_value, value= ui_defaults.get("image_refs", None) )

                video_guide = gr.Video(label= "Reference Video", visible= "V" in video_prompt_type_value, value= ui_defaults.get("video_guide", None) )
                with gr.Row():
                    max_frames = gr.Slider(0, 100, value=ui_defaults.get("max_frames",0), step=1, label="Nb of frames in Ref. Video (0 = as many as possible)", visible= "V" in video_prompt_type_value, scale = 2 ) 
                    remove_background_image_ref = gr.Checkbox(value=ui_defaults.get("remove_background_image_ref",1), label= "Remove Images Ref. Background", visible= "I" in video_prompt_type_value, scale =1 ) 

                video_mask = gr.Video(label= "Video Mask (white pixels = Mask)", visible= "M" in video_prompt_type_value, value= ui_defaults.get("video_mask", None) ) 


            advanced_prompt = advanced_ui
            prompt_vars=[]

            if advanced_prompt:
                default_wizard_prompt, variables, values= None, None, None
            else:                 
                default_wizard_prompt, variables, values, errors =  extract_wizard_prompt(launch_prompt)
                advanced_prompt  = len(errors) > 0
            with gr.Column(visible= advanced_prompt) as prompt_column_advanced:
                prompt = gr.Textbox( visible= advanced_prompt, label="Prompts (each new line of prompt will generate a new video, # lines = comments, ! lines = macros)", value=launch_prompt, lines=3)

            with gr.Column(visible=not advanced_prompt and len(variables) > 0) as prompt_column_wizard_vars:
                gr.Markdown("<B>Please fill the following input fields to adapt automatically the Prompt:</B>")
                wizard_prompt_activated = "off"
                wizard_variables = ""
                with gr.Row():
                    if not advanced_prompt:
                        for variable in variables:
                            value = values.get(variable, "")
                            prompt_vars.append(gr.Textbox( placeholder=variable, min_width=80, show_label= False, info= variable, visible= True, value= "\n".join(value) ))
                        wizard_prompt_activated = "on"
                        if len(variables) > 0:
                            wizard_variables = "\n".join(variables)
                    for _ in range( PROMPT_VARS_MAX - len(prompt_vars)):
                        prompt_vars.append(gr.Textbox(visible= False, min_width=80, show_label= False))
 			
            with gr.Column(not advanced_prompt) as prompt_column_wizard:
                wizard_prompt = gr.Textbox(visible = not advanced_prompt, label="Prompts (each new line of prompt will generate a new video, # lines = comments)", value=default_wizard_prompt, lines=3)
                wizard_prompt_activated_var = gr.Text(wizard_prompt_activated, visible= False)
                wizard_variables_var = gr.Text(wizard_variables, visible = False)
            with gr.Row():
                if "image2video" in model_filename or "Fun_InP" in model_filename:
                    resolution = gr.Dropdown(
                        choices=[
                            # 720p
                            ("720p", "1280x720"),
                            ("480p", "832x480"),
                        ],
                        value=ui_defaults.get("resolution","480p"),
                        label="Resolution (video will have the same height / width ratio than the original image)"
                    )
                else:
                    resolution = gr.Dropdown(
                        choices=[
                            # 720p
                            ("1280x720 (16:9, 720p)", "1280x720"),
                            ("720x1280 (9:16, 720p)", "720x1280"), 
                            ("1024x1024 (4:3, 720p)", "1024x024"),
                            # ("832x1104 (3:4, 720p)", "832x1104"),
                            # ("960x960 (1:1, 720p)", "960x960"),
                            # 480p
                            # ("960x544 (16:9, 480p)", "960x544"),
                            ("832x480 (16:9, 480p)", "832x480"),
                            ("480x832 (9:16, 480p)", "480x832"),
                            # ("832x624 (4:3, 540p)", "832x624"), 
                            # ("624x832 (3:4, 540p)", "624x832"),
                            # ("720x720 (1:1, 540p)", "720x720"),
                        ],
                        value=ui_defaults.get("resolution","832x480"),
                        label="Resolution"
                    )
            with gr.Row():
                with gr.Column():
                    video_length = gr.Slider(5, 193, value=ui_defaults.get("video_length", 81), step=4, label="Number of frames (16 = 1s)")
                with gr.Column():
                    num_inference_steps = gr.Slider(1, 100, value=ui_defaults.get("num_inference_steps",30), step=1, label="Number of Inference Steps")
            show_advanced = gr.Checkbox(label="Advanced Mode", value=advanced_ui)
            with gr.Row(visible=advanced_ui) as advanced_row:
                with gr.Column():
                    seed = gr.Slider(-1, 999999999, value=ui_defaults["seed"], step=1, label="Seed (-1 for random)") 
                    with gr.Row():
                        repeat_generation = gr.Slider(1, 25.0, value=ui_defaults.get("repeat_generation",1), step=1, label="Default Number of Generated Videos per Prompt") 
                        multi_images_gen_type = gr.Dropdown( value=ui_defaults.get("multi_images_gen_type",0), 
                            choices=[
                                ("Generate every combination of images and texts", 0),
                                ("Match images and text prompts", 1),
                            ], visible= args.multiple_images, label= "Multiple Images as Texts Prompts"
                        )
                    with gr.Row():
                        guidance_scale = gr.Slider(1.0, 20.0, value=ui_defaults.get("guidance_scale",5), step=0.5, label="Guidance Scale", visible=True)
                        embedded_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.5, label="Embedded Guidance Scale", visible=False)
                        flow_shift = gr.Slider(0.0, 25.0, value=ui_defaults.get("flow_shift",3), step=0.1, label="Shift Scale") 
                    with gr.Row():
                        negative_prompt = gr.Textbox(label="Negative Prompt", value=ui_defaults.get("negative_prompt", "") )
                    with gr.Column(visible = True): #as loras_column:
                        gr.Markdown("<B>Loras can be used to create special effects on the video by mentioning a trigger word in the Prompt. You can save Loras combinations in presets.</B>")
                        loras_choices = gr.Dropdown(
                            choices=[
                                (lora_name, str(i) ) for i, lora_name in enumerate(loras_names)
                            ],
                            value= launch_loras,
                            multiselect= True,
                            label="Activated Loras"
                        )
                        loras_multipliers = gr.Textbox(label="Loras Multipliers (1.0 by default) separated by space characters or carriage returns, line that starts with # are ignored", value=launch_multis_str)
                    with gr.Row():
                        gr.Markdown("<B>Tea Cache accelerates by skipping intelligently some steps, the more steps are skipped the lower the quality of the video (Tea Cache consumes also VRAM)</B>")
                    with gr.Row():
                        tea_cache_setting = gr.Dropdown(
                            choices=[
                                ("Tea Cache Disabled", 0),
                                ("around x1.5 speed up", 1.5), 
                                ("around x1.75 speed up", 1.75), 
                                ("around x2 speed up", 2.0), 
                                ("around x2.25 speed up", 2.25), 
                                ("around x2.5 speed up", 2.5), 
                            ],
                            value=float(ui_defaults.get("tea_cache_setting",0)),
                            visible=True,
                            label="Tea Cache Global Acceleration"
                        )
                        tea_cache_start_step_perc = gr.Slider(0, 100, value=ui_defaults.get("tea_cache_start_step_perc",0), step=1, label="Tea Cache starting moment in % of generation") 

                    with gr.Row():
                        gr.Markdown("<B>Upsampling - postprocessing that may improve fluidity and the size of the video</B>")
                    with gr.Row():
                        temporal_upsampling = gr.Dropdown(
                            choices=[
                                ("Disabled", ""),
                                ("Rife x2 (32 frames/s)", "rife2"), 
                                ("Rife x4 (64 frames/s)", "rife4"), 
                            ],
                            value=ui_defaults.get("temporal_upsampling", ""),
                            visible=True,
                            scale = 1,
                            label="Temporal Upsampling"
                        )
                        spatial_upsampling = gr.Dropdown(
                            choices=[
                                ("Disabled", ""),
                                ("Lanczos x1.5", "lanczos1.5"), 
                                ("Lanczos x2.0", "lanczos2"), 
                            ],
                            value=ui_defaults.get("spatial_upsampling", ""),
                            visible=True,
                            scale = 1,
                            label="Spatial Upsampling"
                        )

                    gr.Markdown("<B>With Riflex you can generate videos longer than 5s which is the default duration of videos used to train the model</B>")
                    RIFLEx_setting = gr.Dropdown(
                        choices=[
                            ("Auto (ON if Video longer than 5s)", 0),
                            ("Always ON", 1), 
                            ("Always OFF", 2), 
                        ],
                        value=ui_defaults.get("RIFLEx_setting",0),
                        label="RIFLEx positional embedding to generate long video"
                    )
                    with gr.Row():
                        gr.Markdown("<B>Experimental: Skip Layer Guidance, should improve video quality</B>")
                    with gr.Row():
                        slg_switch = gr.Dropdown(
                            choices=[
                                ("OFF", 0),
                                ("ON", 1), 
                            ],
                            value=ui_defaults.get("slg_switch",0),
                            visible=True,
                            scale = 1,
                            label="Skip Layer guidance"
                        )
                        slg_layers = gr.Dropdown(
                            choices=[
                                (str(i), i ) for i in range(40)
                            ],
                            value=ui_defaults.get("slg_layers", ["9"]),
                            multiselect= True,
                            label="Skip Layers",
                            scale= 3
                        )
                    with gr.Row():
                        slg_start_perc = gr.Slider(0, 100, value=ui_defaults.get("slg_start_perc",10), step=1, label="Denoising Steps % start") 
                        slg_end_perc = gr.Slider(0, 100, value=ui_defaults.get("slg_end_perc",90), step=1, label="Denoising Steps % end") 

                    with gr.Row():
                        gr.Markdown("<B>Experimental: Classifier-Free Guidance Zero Star, better adherence to Text Prompt")
                    with gr.Row():
                        cfg_star_switch = gr.Dropdown(
                            choices=[
                                ("OFF", 0),
                                ("ON", 1), 
                            ],
                            value=ui_defaults.get("cfg_star_switch",0),
                            visible=True,
                            scale = 1,
                            label="CFG Star"
                        )
                        with gr.Row():
                            cfg_zero_step = gr.Slider(-1, 39, value=ui_defaults.get("cfg_zero_step",-1), step=1, label="CFG Zero below this Layer (Extra Process)") 

                    with gr.Row():
                        save_settings_btn = gr.Button("Set Settings as Default", visible = not args.lock_config)

        if not update_form:
            with gr.Column():
                gen_status = gr.Text(interactive= False, label = "Status")
                output = gr.Gallery( label="Generated videos", show_label=False, elem_id="gallery" , columns=[3], rows=[1], object_fit="contain", height=450, selected_index=0, interactive= False)
                generate_btn = gr.Button("Generate")
                add_to_queue_btn = gr.Button("Add New Prompt To Queue", visible = False)

                with gr.Column(visible= False) as current_gen_column:
                    with gr.Row():
                        gen_info = gr.HTML(visible=False, min_height=1)
                    with gr.Row():
                        onemore_btn = gr.Button("One More Sample Please !")
                        abort_btn = gr.Button("Abort")

                    queue_df = gr.DataFrame(
                        headers=["Qty","Prompt", "Length","Steps","", "", "", "", ""],
                        datatype=[ "str","markdown","str", "markdown", "markdown", "markdown", "str", "str", "str"],
                        column_widths= ["50","", "65","55", "60", "60", "30", "30", "35"],
                        interactive=False,
                        col_count=(9, "fixed"),
                        wrap=True,
                        value=[],
                        line_breaks= True,
                        visible= False,
                        elem_id="queue_df"
                    )

        extra_inputs = prompt_vars + [wizard_prompt, wizard_variables_var, wizard_prompt_activated_var, video_prompt_column, image_prompt_column,
                                      prompt_column_advanced, prompt_column_wizard_vars, prompt_column_wizard, lset_name, advanced_row] # show_advanced presets_column,
        if update_form:
            locals_dict = locals()
            gen_inputs = [state_dict if k=="state" else locals_dict[k]  for k in inputs_names] + [state_dict] + extra_inputs
            return gen_inputs
        else:
            target_state = gr.Text(value = "state", interactive= False, visible= False)
            target_settings = gr.Text(value = "settings", interactive= False, visible= False)

            image_prompt_type.change(fn=refresh_image_prompt_type, inputs=[state, image_prompt_type], outputs=[image_start, image_end]) 
            video_prompt_type.change(fn=refresh_video_prompt_type, inputs=[state, video_prompt_type], outputs=[image_refs, video_guide, video_mask, max_frames, remove_background_image_ref])
            show_advanced.change(fn=switch_advanced, inputs=[state, show_advanced, lset_name], outputs=[advanced_row, preset_buttons_rows, refresh_lora_btn, refresh2_row ,lset_name ]).then(
                fn=switch_prompt_type, inputs = [state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars], outputs = [wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, prompt_column_advanced, prompt_column_wizard, prompt_column_wizard_vars, *prompt_vars])
            queue_df.select( fn=handle_celll_selection, inputs=state, outputs=[queue_df, modal_image_display, modal_container])
            save_lset_btn.click(validate_save_lset, inputs=[lset_name], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_save_lset_btn, cancel_lset_btn, save_lset_prompt_drop])
            confirm_save_lset_btn.click(fn=validate_wizard_prompt, inputs =[state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars] , outputs= [prompt]).then(
            save_lset, inputs=[state, lset_name, loras_choices, loras_multipliers, prompt, save_lset_prompt_drop], outputs=[lset_name, apply_lset_btn,refresh_lora_btn, delete_lset_btn, save_lset_btn, confirm_save_lset_btn, cancel_lset_btn, save_lset_prompt_drop])
            delete_lset_btn.click(validate_delete_lset, inputs=[lset_name], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_delete_lset_btn, cancel_lset_btn ])
            confirm_delete_lset_btn.click(delete_lset, inputs=[state, lset_name], outputs=[lset_name, apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_delete_lset_btn, cancel_lset_btn ])
            cancel_lset_btn.click(cancel_lset, inputs=[], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn, confirm_delete_lset_btn,confirm_save_lset_btn, cancel_lset_btn,save_lset_prompt_drop ])
            apply_lset_btn.click(apply_lset, inputs=[state, wizard_prompt_activated_var, lset_name,loras_choices, loras_multipliers, prompt], outputs=[wizard_prompt_activated_var, loras_choices, loras_multipliers, prompt]).then(
                fn = fill_wizard_prompt, inputs = [state, wizard_prompt_activated_var, prompt, wizard_prompt], outputs = [ wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, prompt_column_advanced, prompt_column_wizard, prompt_column_wizard_vars, *prompt_vars]
            )
            refresh_lora_btn.click(refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])
            refresh_lora_btn2.click(refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])
            output.select(select_video, state, None )
                
            gen_status.change(refresh_gallery,
                inputs = [state, gen_status], 
                outputs = [output, gen_info, generate_btn, add_to_queue_btn, current_gen_column,  queue_df, abort_btn])
            

            abort_btn.click(abort_generation, [state], [gen_status, abort_btn] ) #.then(refresh_gallery, inputs = [state, gen_info], outputs = [output, gen_info, queue_df] )
            onemore_btn.click(fn=one_more_sample,inputs=[state], outputs= [state])

            inputs_names= list(inspect.signature(save_inputs).parameters)[1:-1]
            locals_dict = locals()
            gen_inputs = [locals_dict[k] for k in inputs_names] + [state]
            save_settings_btn.click( fn=validate_wizard_prompt, inputs =[state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] , outputs= [prompt]).then(
            save_inputs, inputs =[target_settings] + gen_inputs, outputs = [])


            model_choice.change(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt]
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
                ).then(fn= change_model,
                inputs=[state, model_choice],
                outputs= [header]
                ).then(fn= fill_inputs, 
                inputs=[state],
                outputs=gen_inputs + extra_inputs
            ).then(fn= preload_model, 
                inputs=[state],
                outputs=[gen_status])

            generate_btn.click(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt]
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=process_prompt_and_add_tasks,
                inputs = [state, model_choice],
                outputs= queue_df
            ).then(fn=prepare_generate_video,
                inputs= [state],
                outputs= [generate_btn, add_to_queue_btn, current_gen_column],
            ).then(fn=process_tasks,
                inputs= [state],
                outputs= [gen_status],             
            ).then(finalize_generation,
                inputs= [state], 
                outputs= [output, abort_btn, generate_btn, add_to_queue_btn, current_gen_column, gen_info]
            ).then(unload_model_if_needed,
                inputs= [state], 
                outputs= []
            )

            add_to_queue_btn.click(fn=validate_wizard_prompt, 
                inputs =[state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt]
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=process_prompt_and_add_tasks,
                inputs = [state, model_choice],
                outputs=queue_df
            ).then(
                fn=update_status,
                inputs = [state],
            )

            close_modal_button.click(
                lambda: gr.update(visible=False),
                inputs=[],
                outputs=[modal_container]
            )

    return loras_choices, lset_name, state 

def generate_download_tab(lset_name,loras_choices, state):
    with gr.Row():
        with gr.Row(scale =2):
            gr.Markdown("<I>WanGP's Lora Festival ! Press the following button to download i2v <B>Remade_AI</B> Loras collection (and bonuses Loras).")
        with gr.Row(scale =1):
            download_loras_btn = gr.Button("---> Let the Lora's Festival Start !", scale =1)
        with gr.Row(scale =1):
            gr.Markdown("")
    with gr.Row() as download_status_row: 
        download_status = gr.Markdown()

    download_loras_btn.click(fn=download_loras, inputs=[], outputs=[download_status_row, download_status]).then(fn=refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])

    
def generate_configuration_tab(header, model_choice):
    state_dict = {}
    state = gr.State(state_dict)
    gr.Markdown("Please click Apply Changes at the bottom so that the changes are effective. Some choices below may be locked if the app has been launched by specifying a config preset.")
    with gr.Column():
        model_list = []

        for model_type in model_types:
            choice = get_model_filename(model_type, transformer_quantization)
            model_list.append(choice)
        dropdown_choices = [ ( get_model_name(choice),  get_model_type(choice) ) for choice in model_list]
        transformer_types_choices = gr.Dropdown(
            choices= dropdown_choices,
            value= transformer_types,
            label= "Selectable Wan Transformer Models (keep empty to get All of them)",
            scale= 2,
            multiselect= True
            )

        quantization_choice = gr.Dropdown(
            choices=[
                ("Int8 Quantization (recommended)", "int8"),
                ("BF16 (no quantization)", "bf16"),
            ],
            value= transformer_quantization,
            label="Wan Transformer Model Quantization Type (if available)",
         )                

        index = text_encoder_choices.index(text_encoder_filename)
        index = 0 if index ==0 else index
        text_encoder_choice = gr.Dropdown(
            choices=[
                ("UMT5 XXL 16 bits - unquantized text encoder, better quality uses more RAM", 0),
                ("UMT5 XXL quantized to 8 bits - quantized text encoder, slightly worse quality but uses less RAM", 1),
            ],
            value= index,
            label="Text Encoder model"
         )
        save_path_choice = gr.Textbox(
            label="Output Folder for Generated Videos",
            value=server_config.get("save_path", save_path)
        )
        def check(mode): 
            if not mode in attention_modes_installed:
                return " (NOT INSTALLED)"
            elif not mode in attention_modes_supported:
                return " (NOT SUPPORTED)"
            else:
                return ""
        attention_choice = gr.Dropdown(
            choices=[
                ("Auto : pick sage2 > sage > sdpa depending on what is installed", "auto"),
                ("Scale Dot Product Attention: default, always available", "sdpa"),
                ("Flash" + check("flash")+ ": good quality - requires additional install (usually complex to set up on Windows without WSL)", "flash"),
                # ("Xformers" + check("xformers")+ ": good quality - requires additional install (usually complex, may consume less VRAM to set up on Windows without WSL)", "xformers"),
                ("Sage" + check("sage")+ ": 30% faster but slightly worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage"),
                ("Sage2" + check("sage2")+ ": 40% faster but slightly worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage2"),
            ],
            value= attention_mode,
            label="Attention Type",
            interactive= not lock_ui_attention
         )
        gr.Markdown("Beware: when restarting the server or changing a resolution or video duration, the first step of generation for a duration / resolution may last a few minutes due to recompilation")
        compile_choice = gr.Dropdown(
            choices=[
                ("ON: works only on Linux / WSL", "transformer"),
                ("OFF: no other choice if you have Windows without using WSL", "" ),
            ],
            value= compile,
            label="Compile Transformer (up to 50% faster and 30% more frames but requires Linux / WSL and Flash or Sage attention)",
            interactive= not lock_ui_compile
         )              
        vae_config_choice = gr.Dropdown(
            choices=[
        ("Auto", 0),
        ("Disabled (faster but may require up to 22 GB of VRAM)", 1),
        ("256 x 256 : If at least 8 GB of VRAM", 2),
        ("128 x 128 : If at least 6 GB of VRAM", 3),
            ],
            value= vae_config,
            label="VAE Tiling - reduce the high VRAM requirements for VAE decoding and VAE encoding (if enabled it will be slower)"
         )
        boost_choice = gr.Dropdown(
            choices=[
                # ("Auto (ON if Video longer than 5s)", 0),
                ("ON", 1), 
                ("OFF", 2), 
            ],
            value=boost,
            label="Boost: Give a 10% speed speedup without losing quality at the cost of a litle VRAM (up to 1GB for max frames and resolution)"
        )
        profile_choice = gr.Dropdown(
            choices=[
        ("HighRAM_HighVRAM, profile 1: at least 48 GB of RAM and 24 GB of VRAM, the fastest for short videos a RTX 3090 / RTX 4090", 1),
        ("HighRAM_LowVRAM, profile 2 (Recommended): at least 48 GB of RAM and 12 GB of VRAM, the most versatile profile with high RAM, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos", 2),
        ("LowRAM_HighVRAM, profile 3: at least 32 GB of RAM and 24 GB of VRAM, adapted for RTX 3090 / RTX 4090 with limited RAM for good speed short video",3),
        ("LowRAM_LowVRAM, profile 4 (Default): at least 32 GB of RAM and 12 GB of VRAM, if you have little VRAM or want to generate longer videos",4),
        ("VerylowRAM_LowVRAM, profile 5: (Fail safe): at least 16 GB of RAM and 10 GB of VRAM, if you don't have much it won't be fast but maybe it will work",5)
            ],
            value= profile,
            label="Profile (for power users only, not needed to change it)"
         )
        # default_ui_choice = gr.Dropdown(
        #     choices=[
        #         ("Text to Video", "t2v"),
        #         ("Image to Video", "i2v"),
        #     ],
        #     value= default_ui,
        #     label="Default mode when launching the App if not '--t2v' ot '--i2v' switch is specified when launching the server ",
        #  )                
        metadata_choice = gr.Dropdown(
            choices=[
                ("Export JSON files", "json"),
                ("Add metadata to video", "metadata"),
                ("Neither", "none")
            ],
            value=server_config.get("metadata_type", "metadata"),
            label="Metadata Handling"
        )
        reload_choice = gr.Dropdown(
            choices=[
                ("Load Model When Changing Model", 1), 
                ("Load Model When Pressing Generate", 2), 
                ("Load Model When Pressing Generate and Unload Model when Finished", 3), 
            ],
            value=server_config.get("reload_model",2),
            label="RAM Loading / Unloading Model Policy (in any case VRAM will be freed once the queue has been processed)"
        )

        clear_file_list_choice = gr.Dropdown(
            choices=[
                ("None", 0),
                ("Keep the last video", 1),
                ("Keep the last 5 videos", 5),
                ("Keep the last 10 videos", 10),
                ("Keep the last 20 videos", 20),
                ("Keep the last 30 videos", 30),
            ],
            value=server_config.get("clear_file_list", 0),
            label="Keep Previously Generated Videos when starting a Generation Batch"
        )

        
        msg = gr.Markdown()            
        apply_btn  = gr.Button("Apply Changes")
        apply_btn.click(
                fn=apply_changes,
                inputs=[
                    state,
                    transformer_types_choices,
                    text_encoder_choice,
                    save_path_choice,
                    attention_choice,
                    compile_choice,                            
                    profile_choice,
                    vae_config_choice,
                    metadata_choice,
                    quantization_choice,
                    boost_choice,
                    clear_file_list_choice,
                    reload_choice,
                ],
                outputs= [msg , header, model_choice]
        )

def generate_about_tab():
    gr.Markdown("<H2>WanGP - Wan 2.1 model for the GPU Poor by <B>DeepBeepMeep</B> (<A HREF='https://github.com/deepbeepmeep/Wan2GP'>GitHub</A>)</H2>")
    gr.Markdown("Original Wan 2.1 Model by <B>Alibaba</B> (<A HREF='https://github.com/Wan-Video/Wan2.1'>GitHub</A>)")
    gr.Markdown("Many thanks to:")
    gr.Markdown("- <B>Alibaba Wan team for the best open source video generator")
    gr.Markdown("- <B>Cocktail Peanuts</B> : QA and simple installation via Pinokio.computer")
    gr.Markdown("- <B>Tophness</B> : created multi tabs and queuing frameworks")
    gr.Markdown("- <B>AmericanPresidentJimmyCarter</B> : added original support for Skip Layer Guidance")
    gr.Markdown("- <B>Remade_AI</B> : for creating their awesome Loras collection")

def generate_info_tab():
    gr.Markdown("<FONT SIZE=3>Welcome to WanGP a super fast and low VRAM AI Video Generator !</FONT>")
    
    gr.Markdown("The VRAM requirements will depend greatly of the resolution and the duration of the video, for instance :")
    gr.Markdown("- 848 x 480 with a 14B model: 80 frames (5s) : 8 GB of VRAM")
    gr.Markdown("- 848 x 480 with the 1.3B model: 80 frames (5s) : 5 GB of VRAM")
    gr.Markdown("- 1280 x 720 with a 14B model: 80 frames (5s): 11 GB of VRAM")
    gr.Markdown("It is not recommmended to generate a video longer than 8s (128 frames) even if there is still some VRAM left as some artifacts may appear")
    gr.Markdown("Please note that if your turn on compilation, the first denoising step of the first video generation will be slow due to the compilation. Therefore all your tests should be done with compilation turned off.")


def generate_dropdown_model_list():
    dropdown_types= transformer_types if len(transformer_types) > 0 else model_types 
    current_model_type = get_model_type(transformer_filename)
    if current_model_type not in dropdown_types:
        dropdown_types.append(current_model_type)
    model_list = []
    for model_type in dropdown_types:
        choice = get_model_filename(model_type, transformer_quantization)
        model_list.append(choice)
    dropdown_choices = [ ( get_model_name(choice),  get_model_type(choice) ) for choice in model_list]
    return gr.Dropdown(
        choices= dropdown_choices,
        value= current_model_type,
        show_label= False,
        scale= 2
        )



def create_demo():
    css = """
        .title-with-lines {
            display: flex;
            align-items: center;
            margin: 30px 0;
        }
        .line {
            flex-grow: 1;
            height: 1px;
            background-color: #333;
        }
        h2 {
            margin: 0 20px;
            white-space: nowrap;
        }
        .queue-item {
            border: 1px solid #ccc;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        .current {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
        }
        .task-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .progress-container {
            height: 10px;
            background: #e9ecef;
            border-radius: 5px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: #007bff;
            transition: width 0.3s ease;
        }
        .task-details {
            display: flex;
            justify-content: space-between;
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }
        .task-prompt {
            font-size: 0.8em;
            color: #868e96;
            margin-top: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        #queue_df th {
            pointer-events: none;
            text-align: center;
            vertical-align: middle;
            font-size:11px;
        }
        #xqueue_df table {
            width: 100%;
            overflow: hidden !important;
        }
        #xqueue_df::-webkit-scrollbar {
            display: none !important;
        }
        #xqueue_df {
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }
        .selection-button {
            display: none;
        }
        .cell-selected {
            --ring-color: none;
        }
        #queue_df th:nth-child(1),
        #queue_df td:nth-child(1) {
            width: 60px;
            text-align: center;
            vertical-align: middle;
            cursor: default !important;
            pointer-events: none;
        }
        #xqueue_df th:nth-child(2),
        #queue_df td:nth-child(2) {
            text-align: center;
            vertical-align: middle;
            white-space: normal;
        }
        #queue_df td:nth-child(2) {
            cursor: default !important;
        }
        #queue_df th:nth-child(3),
        #queue_df td:nth-child(3) {
            width: 60px;
            text-align: center;
            vertical-align: middle;
            cursor: default !important;
            pointer-events: none;
        }
        #queue_df th:nth-child(4),
        #queue_df td:nth-child(4) {
            width: 60px;
            text-align: center;
            white-space: nowrap;
            cursor: default !important;
            pointer-events: none;
        }
        #queue_df th:nth-child(5), #queue_df td:nth-child(7),
        #queue_df th:nth-child(6), #queue_df td:nth-child(8) {
            width: 60px;
            text-align: center;
            vertical-align: middle;
        }
        #queue_df td:nth-child(5) img,
        #queue_df td:nth-child(6) img {
            max-width: 50px;
            max-height: 50px;
            object-fit: contain;
            display: block;
            margin: auto;
            cursor: pointer;
        }
        #queue_df th:nth-child(7), #queue_df td:nth-child(9),
        #queue_df th:nth-child(8), #queue_df td:nth-child(10),
        #queue_df th:nth-child(9), #queue_df td:nth-child(11) {
            width: 20px;
            padding: 2px !important;
            cursor: pointer;
            text-align: center;
            font-weight: bold;
            vertical-align: middle;
        }
        #queue_df td:nth-child(5):hover,
        #queue_df td:nth-child(6):hover,
        #queue_df td:nth-child(7):hover,
        #queue_df td:nth-child(8):hover,
        #queue_df td:nth-child(9):hover {
            background-color: #e0e0e0;
        }
        #image-modal-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            justify-content: center;
            align-items: center;
            z-index: 1000;
            padding: 20px;
            box-sizing: border-box;
        }
        #image-modal-container > div {
             background-color: white;
             padding: 15px;
             border-radius: 8px;
             max-width: 90%;
             max-height: 90%;
             overflow: auto;
             position: relative;
             display: flex;
             flex-direction: column;
        }
         #image-modal-container img {
             max-width: 100%;
             max-height: 80vh;
             object-fit: contain;
             margin-top: 10px;
         }
         #image-modal-close-button-row {
             display: flex;
             justify-content: flex-end;
         }
         #image-modal-close-button-row button {
            cursor: pointer;
         }
        .progress-container-custom {
            width: 100%;
            background-color: #e9ecef;
            border-radius: 0.375rem;
            overflow: hidden;
            height: 25px;
            position: relative;
            margin-top: 5px;
            margin-bottom: 5px;
        }
        .progress-bar-custom {
            height: 100%;
            background-color: #0d6efd;
            transition: width 0.3s ease-in-out;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.9em;
            font-weight: bold;
            white-space: nowrap;
            overflow: hidden;
        }
        .progress-bar-custom.idle {
            background-color: #6c757d;
        }
        .progress-bar-text {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            mix-blend-mode: difference;
            font-size: 0.9em;
            font-weight: bold;
            white-space: nowrap;
            z-index: 2;
            pointer-events: none;
        }
    """
    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="sky", neutral_hue="slate", text_size="md"), title= "Wan2GP") as demo:
        gr.Markdown("<div align=center><H1>Wan<SUP>GP</SUP> v4.0 <FONT SIZE=4>by <I>DeepBeepMeep</I></FONT> <FONT SIZE=3>") # (<A HREF='https://github.com/deepbeepmeep/Wan2GP'>Updates</A>)</FONT SIZE=3></H1></div>")
        global model_list

        with gr.Tabs(selected="video_gen", ) as main_tabs:
            with gr.Tab("Video Generator", id="video_gen") as t2v_tab:
                with gr.Row():
                    if args.lock_model:    
                        gr.Markdown("<div class='title-with-lines'><div class=line></div><h2>" + get_model_name(transformer_filename) + "</h2><div class=line></div>")
                        model_choice = gr.Dropdown(visible=False, value= get_model_type(transformer_filename))
                    else:
                        gr.Markdown("<div class='title-with-lines'><div class=line width=100%></div></div>")
                        model_choice = generate_dropdown_model_list()
                        gr.Markdown("<div class='title-with-lines'><div class=line width=100%></div></div>")
                with gr.Row():
                    header = gr.Markdown(generate_header(transformer_filename, compile, attention_mode), visible= True)
                with gr.Row():

                    loras_choices, lset_name, state = generate_video_tab(model_choice = model_choice, header = header)
            with gr.Tab("Informations"):
                generate_info_tab()
            if not args.lock_config:
                with gr.Tab("Downloads", id="downloads") as downloads_tab:
                    generate_download_tab(lset_name, loras_choices, state)
                with gr.Tab("Configuration"):
                    generate_configuration_tab(header, model_choice)
            with gr.Tab("About"):
                generate_about_tab()

        return demo

if __name__ == "__main__":
    # threading.Thread(target=runner, daemon=True).start()
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_port = int(args.server_port)
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if server_port == 0:
        server_port = int(os.getenv("SERVER_PORT", "7860"))
    server_name = args.server_name
    if args.listen:
        server_name = "0.0.0.0"
    if len(server_name) == 0:
        server_name = os.getenv("SERVER_NAME", "localhost")      
    demo = create_demo()
    if args.open_browser:
        import webbrowser 
        if server_name.startswith("http"):
            url = server_name 
        else:
            url = "http://" + server_name 
        webbrowser.open(url + ":" + str(server_port), new = 0, autoraise = True)
    demo.launch(server_name=server_name, server_port=server_port, share=args.share, allowed_paths=[save_path])