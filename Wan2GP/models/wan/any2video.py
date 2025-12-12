# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from mmgp import offload
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from .distributed.fsdp import shard_model
from .modules.model import WanModel
from mmgp.offload import get_cache, clear_caches
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .modules.vae2_2 import Wan2_2_VAE

from .modules.clip import CLIPModel
from shared.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from shared.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .modules.posemb_layers import get_rotary_pos_embed, get_nd_rotary_pos_embed
from shared.utils.vace_preprocessor import VaceVideoProcessor
from shared.utils.basic_flowmatch import FlowMatchScheduler
from shared.utils.utils import get_outpainting_frame_location, resize_lanczos, calculate_new_dimensions, convert_image_to_tensor
from .multitalk.multitalk_utils import MomentumBuffer, adaptive_projected_guidance, match_and_blend_colors, match_and_blend_colors_with_mask
from mmgp import safetensors2

def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm
    
    return st_star

def timestep_transform(t, shift=5.0, num_timesteps=1000 ):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t
    
    
class WanAny2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        model_filename = None,
        submodel_no_list = None,
        model_type = None, 
        model_def = None,
        base_model_type = None,
        text_encoder_filename = None,
        quantizeTransformer = False,
        save_quantized = False,
        dtype = torch.bfloat16,
        VAE_dtype = torch.float32,
        mixed_precision_transformer = False
    ):
        self.device = torch.device(f"cuda")
        self.config = config
        self.VAE_dtype = VAE_dtype
        self.dtype = dtype
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.model_def = model_def
        self.model2 = None
        self.transformer_switch = model_def.get("URLs2", None) is not None
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=text_encoder_filename,
            tokenizer_path=os.path.join(checkpoint_dir, "umt5-xxl"),
            shard_fn= None)

        # base_model_type = "i2v2_2"
        if hasattr(config, "clip_checkpoint") and not base_model_type in ["i2v_2_2", "i2v_2_2_multitalk"]:
            self.clip = CLIPModel(
                dtype=config.clip_dtype,
                device=self.device,
                checkpoint_path=os.path.join(checkpoint_dir , 
                                            config.clip_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir , "xlm-roberta-large"))


        if base_model_type in ["ti2v_2_2"]:
            self.vae_stride = (4, 16, 16)
            vae_checkpoint = "Wan2.2_VAE.safetensors"
            vae = Wan2_2_VAE
        else:
            self.vae_stride = config.vae_stride
            vae_checkpoint = "Wan2.1_VAE.safetensors"
            vae = WanVAE
        self.patch_size = config.patch_size 
        
        self.vae = vae(
            vae_pth=os.path.join(checkpoint_dir, vae_checkpoint), dtype= VAE_dtype,
            device="cpu")
        self.vae.device = self.device
        
        # config_filename= "configs/t2v_1.3B.json"
        # import json
        # with open(config_filename, 'r', encoding='utf-8') as f:
        #     config = json.load(f)
        # sd = safetensors2.torch_load_file(xmodel_filename)
        # model_filename = "c:/temp/wan2.2i2v/low/diffusion_pytorch_model-00001-of-00006.safetensors"
        base_config_file = f"configs/{base_model_type}.json"
        forcedConfigPath = base_config_file if len(model_filename) > 1 else None
        # forcedConfigPath = base_config_file = f"configs/flf2v_720p.json"
        # model_filename[1] = xmodel_filename
        self.model = self.model2 = None
        source =  model_def.get("source", None)
        source2 = model_def.get("source2", None)
        module_source =  model_def.get("module_source", None)
        module_source2 =  model_def.get("module_source2", None)
        if module_source is not None:
            self.model = offload.fast_load_transformers_model(model_filename[:1] + [module_source], modelClass=WanModel,do_quantize= quantizeTransformer and not save_quantized, writable_tensors= False, defaultConfigPath=base_config_file , forcedConfigPath= forcedConfigPath)
        if module_source2 is not None:
            self.model2 = offload.fast_load_transformers_model(model_filename[1:2] + [module_source2], modelClass=WanModel,do_quantize= quantizeTransformer and not save_quantized, writable_tensors= False, defaultConfigPath=base_config_file , forcedConfigPath= forcedConfigPath)
        if source is not None:
            self.model = offload.fast_load_transformers_model(source, modelClass=WanModel, writable_tensors= False, forcedConfigPath= base_config_file)
        if source2 is not None:
            self.model2 = offload.fast_load_transformers_model(source2, modelClass=WanModel, writable_tensors= False, forcedConfigPath= base_config_file)

        if self.model is not None or self.model2 is not None:
            from wgp import save_model
            from mmgp.safetensors2 import torch_load_file
        else:
            if self.transformer_switch:
                if 0 in submodel_no_list[2:] and 1 in submodel_no_list[2:]:
                    raise Exception("Shared and non shared modules at the same time across multipe models is not supported")
                
                if 0 in submodel_no_list[2:]:
                    shared_modules= {}
                    self.model = offload.fast_load_transformers_model(model_filename[:1], modules = model_filename[2:], modelClass=WanModel,do_quantize= quantizeTransformer and not save_quantized, writable_tensors= False, defaultConfigPath=base_config_file , forcedConfigPath= forcedConfigPath,  return_shared_modules= shared_modules)
                    self.model2 = offload.fast_load_transformers_model(model_filename[1:2], modules = shared_modules, modelClass=WanModel,do_quantize= quantizeTransformer and not save_quantized, writable_tensors= False, defaultConfigPath=base_config_file , forcedConfigPath= forcedConfigPath)
                    shared_modules = None
                else:
                    modules_for_1 =[ file_name for file_name, submodel_no in zip(model_filename[2:],submodel_no_list[2:] ) if submodel_no ==1 ]
                    modules_for_2 =[ file_name for file_name, submodel_no in zip(model_filename[2:],submodel_no_list[2:] ) if submodel_no ==2 ]
                    self.model = offload.fast_load_transformers_model(model_filename[:1], modules = modules_for_1, modelClass=WanModel,do_quantize= quantizeTransformer and not save_quantized, writable_tensors= False, defaultConfigPath=base_config_file , forcedConfigPath= forcedConfigPath)
                    self.model2 = offload.fast_load_transformers_model(model_filename[1:2], modules = modules_for_2, modelClass=WanModel,do_quantize= quantizeTransformer and not save_quantized, writable_tensors= False, defaultConfigPath=base_config_file , forcedConfigPath= forcedConfigPath)

            else:
                self.model = offload.fast_load_transformers_model(model_filename, modelClass=WanModel,do_quantize= quantizeTransformer and not save_quantized, writable_tensors= False, defaultConfigPath=base_config_file , forcedConfigPath= forcedConfigPath)
        

        if self.model is not None:
            self.model.lock_layers_dtypes(torch.float32 if mixed_precision_transformer else dtype)
            offload.change_dtype(self.model, dtype, True)
            self.model.eval().requires_grad_(False)
        if self.model2 is not None:
            self.model2.lock_layers_dtypes(torch.float32 if mixed_precision_transformer else dtype)
            offload.change_dtype(self.model2, dtype, True)
            self.model2.eval().requires_grad_(False)

        if module_source is not None:
            save_model(self.model, model_type, dtype, None, is_module=True, filter=list(torch_load_file(module_source)), module_source_no=1)
        if module_source2 is not None:
            save_model(self.model2, model_type, dtype, None, is_module=True, filter=list(torch_load_file(module_source2)), module_source_no=2)
        if not source is None:
            save_model(self.model, model_type, dtype, None, submodel_no= 1)
        if not source2 is None:
            save_model(self.model2, model_type, dtype, None, submodel_no= 2)

        if save_quantized:
            from wgp import save_quantized_model
            if self.model is not None:
                save_quantized_model(self.model, model_type, model_filename[0], dtype, base_config_file)
            if self.model2 is not None:
                save_quantized_model(self.model2, model_type, model_filename[1], dtype, base_config_file, submodel_no=2)
        self.sample_neg_prompt = config.sample_neg_prompt

        if self.model.config.get("vace_in_dim", None) != None:
            self.vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(config.vae_stride, self.patch_size)]),
                                            min_area=480*832,
                                            max_area=480*832,
                                            min_fps=config.sample_fps,
                                            max_fps=config.sample_fps,
                                            zero_start=True,
                                            seq_len=32760,
                                            keep_last=True)

            self.adapt_vace_model(self.model)
            if self.model2 is not None: self.adapt_vace_model(self.model2)

        self.num_timesteps = 1000 
        self.use_timestep_transform = True 

    def vace_encode_frames(self, frames, ref_images, masks=None, tile_size = 0, overlapped_latents = None):
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = self.vae.encode(frames, tile_size = tile_size)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = self.vae.encode(inactive, tile_size = tile_size)

            if overlapped_latents  != None and False : # disabled as quality seems worse
                # inactive[0][:, 0:1] = self.vae.encode([frames[0][:, 0:1]], tile_size = tile_size)[0] # redundant
                for t in inactive:
                    t[:, 1:overlapped_latents.shape[1] + 1] = overlapped_latents
                overlapped_latents[: 0:1] = inactive[0][: 0:1]

            reactive = self.vae.encode(reactive, tile_size = tile_size)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = self.vae.encode(refs, tile_size = tile_size)
                else:
                    ref_latent = self.vae.encode(refs, tile_size = tile_size)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None):
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // self.vae_stride[0]) # nb latents token without (ref tokens not included)
            height = 2 * (int(height) // (self.vae_stride[1] * 2))
            width = 2 * (int(width) // (self.vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, self.vae_stride[1], width, self.vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                self.vae_stride[1] * self.vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros(mask.shape[0], length, *mask.shape[-2:], dtype=mask.dtype, device=mask.device)
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def fit_image_into_canvas(self, ref_img, image_size, canvas_tf_bg, device, full_frame = False, outpainting_dims = None, return_mask = False):
        from shared.utils.utils import save_image
        ref_width, ref_height = ref_img.size
        if (ref_height, ref_width) == image_size and outpainting_dims  == None:
            ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
            canvas = torch.zeros_like(ref_img) if return_mask else None
        else:
            if outpainting_dims != None:
                final_height, final_width = image_size
                canvas_height, canvas_width, margin_top, margin_left =   get_outpainting_frame_location(final_height, final_width,  outpainting_dims, 1)        
            else:
                canvas_height, canvas_width = image_size
            if full_frame:
                new_height = canvas_height
                new_width = canvas_width
                top = left = 0 
            else:
                # if fill_max  and (canvas_height - new_height) < 16:
                #     new_height = canvas_height
                # if fill_max  and (canvas_width - new_width) < 16:
                #     new_width = canvas_width
                scale = min(canvas_height / ref_height, canvas_width / ref_width)
                new_height = int(ref_height * scale)
                new_width = int(ref_width * scale)
                top = (canvas_height - new_height) // 2
                left = (canvas_width - new_width) // 2
            ref_img = ref_img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS) 
            ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
            if outpainting_dims != None:
                canvas = torch.full((3, 1, final_height, final_width), canvas_tf_bg, dtype= torch.float, device=device) # [-1, 1]
                canvas[:, :, margin_top + top:margin_top + top + new_height, margin_left + left:margin_left + left + new_width] = ref_img 
            else:
                canvas = torch.full((3, 1, canvas_height, canvas_width), canvas_tf_bg, dtype= torch.float, device=device) # [-1, 1]
                canvas[:, :, top:top + new_height, left:left + new_width] = ref_img 
            ref_img = canvas
            canvas = None
            if return_mask:
                if outpainting_dims != None:
                    canvas = torch.ones((3, 1, final_height, final_width), dtype= torch.float, device=device) # [-1, 1]
                    canvas[:, :, margin_top + top:margin_top + top + new_height, margin_left + left:margin_left + left + new_width] = 0
                else:
                    canvas = torch.ones((3, 1, canvas_height, canvas_width), dtype= torch.float, device=device) # [-1, 1]
                    canvas[:, :, top:top + new_height, left:left + new_width] = 0
                canvas = canvas.to(device)
        return ref_img.to(device), canvas

    def prepare_source(self, src_video, src_mask, src_ref_images, total_frames, image_size,  device, keep_video_guide_frames= [], pre_src_video = None, inject_frames = [], outpainting_dims = None, any_background_ref = False):
        image_sizes = []
        trim_video_guide = len(keep_video_guide_frames)
        def conv_tensor(t, device):
            return t.float().div_(127.5).add_(-1).permute(3, 0, 1, 2).to(device)

        for i, (sub_src_video, sub_src_mask, sub_pre_src_video) in enumerate(zip(src_video, src_mask,pre_src_video)):
            prepend_count = 0 if sub_pre_src_video == None else sub_pre_src_video.shape[1]
            num_frames = total_frames - prepend_count            
            num_frames = min(num_frames, trim_video_guide) if trim_video_guide > 0 and sub_src_video != None else num_frames
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i] = conv_tensor(sub_src_video[:num_frames], device)
                src_mask[i] = conv_tensor(sub_src_mask[:num_frames], device)
                # src_video is [-1, 1] (at this function output), 0 = inpainting area (in fact 127  in [0, 255])
                # src_mask is [-1, 1] (at this function output), 0 = preserve original video (in fact 127  in [0, 255]) and 1 = Inpainting (in fact 255  in [0, 255])
                if prepend_count > 0:
                    src_video[i] =  torch.cat( [sub_pre_src_video, src_video[i]], dim=1)
                    src_mask[i] =  torch.cat( [torch.full_like(sub_pre_src_video, -1.0), src_mask[i]] ,1)
                src_video_shape = src_video[i].shape
                if src_video_shape[1] != total_frames:
                    src_video[i] =  torch.cat( [src_video[i], src_video[i].new_zeros(src_video_shape[0], total_frames -src_video_shape[1], *src_video_shape[-2:])], dim=1)
                    src_mask[i] =  torch.cat( [src_mask[i], src_mask[i].new_ones(src_video_shape[0], total_frames -src_video_shape[1], *src_video_shape[-2:])], dim=1)
                src_mask[i] = torch.clamp((src_mask[i][:, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                if prepend_count > 0:
                    src_video[i] =  torch.cat( [sub_pre_src_video, torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)], dim=1)
                    src_mask[i] =  torch.cat( [torch.zeros_like(sub_pre_src_video), torch.ones((3, num_frames, image_size[0], image_size[1]), device=device)] ,1)
                else:
                    src_video[i] = torch.zeros((3, total_frames, image_size[0], image_size[1]), device=device)
                    src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i] = conv_tensor(sub_src_video[:num_frames], device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                if prepend_count > 0:
                    src_video[i] =  torch.cat( [sub_pre_src_video, src_video[i]], dim=1)
                    src_mask[i] =  torch.cat( [torch.zeros_like(sub_pre_src_video), src_mask[i]] ,1)
                src_video_shape = src_video[i].shape
                if src_video_shape[1] != total_frames:
                    src_video[i] =  torch.cat( [src_video[i], src_video[i].new_zeros(src_video_shape[0], total_frames -src_video_shape[1], *src_video_shape[-2:])], dim=1)
                    src_mask[i] =  torch.cat( [src_mask[i], src_mask[i].new_ones(src_video_shape[0], total_frames -src_video_shape[1], *src_video_shape[-2:])], dim=1)
                image_sizes.append(src_video[i].shape[2:])
            for k, keep in enumerate(keep_video_guide_frames):
                if not keep:
                    pos = prepend_count + k
                    src_video[i][:, pos:pos+1] = 0
                    src_mask[i][:, pos:pos+1] = 1

            for k, frame in enumerate(inject_frames):
                if frame != None:
                    pos = prepend_count + k
                    src_video[i][:, pos:pos+1], src_mask[i][:, pos:pos+1] = self.fit_image_into_canvas(frame, image_size, 0, device, True, outpainting_dims, return_mask= True)
        

        self.background_mask = None
        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None and not torch.is_tensor(ref_img):
                        if j==0 and any_background_ref:
                            if self.background_mask == None: self.background_mask = [None] * len(src_ref_images) 
                            src_ref_images[i][j], self.background_mask[i] = self.fit_image_into_canvas(ref_img, image_size, 0, device, True, outpainting_dims, return_mask= True)
                        else:
                            src_ref_images[i][j], _ = self.fit_image_into_canvas(ref_img, image_size, 1, device)
        if self.background_mask != None:
            self.background_mask = [ item if item != None else self.background_mask[0] for item in self.background_mask ] # deplicate background mask with double control net since first controlnet image ref modifed by ref
        return src_video, src_mask, src_ref_images

    def get_vae_latents(self, ref_images, device, tile_size= 0):
        ref_vae_latents = []
        for ref_image in ref_images:
            ref_image = TF.to_tensor(ref_image).sub_(0.5).div_(0.5).to(self.device)
            img_vae_latent = self.vae.encode([ref_image.unsqueeze(1)], tile_size= tile_size)
            ref_vae_latents.append(img_vae_latent[0])
                    
        return torch.cat(ref_vae_latents, dim=1)


    def generate(self,
        input_prompt,
        input_frames= None,
        input_masks = None,
        input_ref_images = None,
        input_video = None,
        image_start = None,
        image_end = None,
        denoising_strength = 1.0,
        target_camera=None,
        context_scale=None,
        width = 1280,
        height = 720,
        fit_into_canvas = True,
        frame_num=81,
        batch_size = 1,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=50,
        guide_scale=5.0,
        guide2_scale = 5.0,
        guide3_scale = 5.0,
        switch_threshold = 0,
        switch2_threshold = 0,
        guide_phases= 1 ,
        model_switch_phase = 1,
        n_prompt="",
        seed=-1,
        callback = None,
        enable_RIFLEx = None,
        VAE_tile_size = 0,
        joint_pass = False,
        slg_layers = None,
        slg_start = 0.0,
        slg_end = 1.0,
        cfg_star_switch = True,
        cfg_zero_step = 5,
        audio_scale=None,
        audio_cfg_scale=None,
        audio_proj=None,
        audio_context_lens=None,
        overlapped_latents  = None,
        return_latent_slice = None,
        overlap_noise = 0,
        conditioning_latents_size = 0,
        keep_frames_parsed = [],
        model_type = None,
        model_mode = None,
        loras_slists = None,
        NAG_scale = 0,
        NAG_tau = 3.5,
        NAG_alpha = 0.5,
        offloadobj = None,
        apg_switch = False,
        speakers_bboxes = None,
        color_correction_strength = 1,
        prefix_frames_count = 0,
        image_mode = 0,
        window_no = 0,
        set_header_text = None,
        pre_video_frame = None,
        video_prompt_type= "",
        original_input_ref_images = [],
        latent_noise_mask_strength = 0.0,  # 0.0 = disabled, 1.0 = full latent noise masking
        vid2vid_init_video = None,  # Path to video for vid2vid initialization (gap frames)
        vid2vid_init_strength = 0.7,  # 0.0 = pure vid2vid (keep original), 1.0 = pure txt2vid (random noise)
        **bbargs
                ):

        import time
        import torch
        _generate_start_time = time.time()

        # Log memory state at entry to detect fragmentation/leaks
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated() / 1024**3
            vram_reserved = torch.cuda.memory_reserved() / 1024**3
            vram_free = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**3
            print(f"[GENERATE_ENTRY] VRAM: allocated={vram_allocated:.2f}GB, reserved={vram_reserved:.2f}GB, free_in_reserved={vram_free:.2f}GB")

        # Check if model components are on expected devices
        print(f"[GENERATE_ENTRY] Model device: {self.device}")
        print(f"[GENERATE_ENTRY] Model dtype: {self.dtype}")
        if hasattr(self, 'text_encoder'):
            try:
                te_device = next(self.text_encoder.parameters()).device if hasattr(self.text_encoder, 'parameters') else 'unknown'
                print(f"[GENERATE_ENTRY] Text encoder device: {te_device}")
            except:
                print(f"[GENERATE_ENTRY] Text encoder device: could not detect")

        print(f"[GENERATE_TIMING] generate() called at {_generate_start_time:.3f}")

        if sample_solver =="euler":
            # prepare timesteps
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps][:-1]
            timesteps = torch.tensor(timesteps)
            sample_scheduler = None                  
        elif sample_solver == 'causvid':
            sample_scheduler = FlowMatchScheduler(num_inference_steps=sampling_steps, shift=shift, sigma_min=0, extra_one_step=True)
            timesteps = torch.tensor([1000, 934, 862, 756, 603, 410, 250, 140, 74])[:sampling_steps].to(self.device)
            sample_scheduler.timesteps =timesteps
            sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.], device=self.device)])
        elif sample_solver == 'unipc' or sample_solver == "":
            sample_scheduler = FlowUniPCMultistepScheduler( num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps( sampling_steps, device=self.device, shift=shift)
            
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        elif sample_solver == 'dpm++_sde':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                algorithm_type="sde-dpmsolver++",
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError(f"Unsupported Scheduler {sample_solver}")
        original_timesteps = timesteps

        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        image_outputs = image_mode == 1
        kwargs = {'pipeline': self, 'callback': callback}
        color_reference_frame = None
        if self._interrupt:
            return None

        _before_text_encode = time.time()
        print(f"[GENERATE_TIMING] Before text encoding at {_before_text_encode - _generate_start_time:.3f}s")

        # Check if text encoder needs to be moved to GPU (async loading issue)
        if hasattr(self, 'text_encoder'):
            try:
                te_device_before = next(self.text_encoder.parameters()).device if hasattr(self.text_encoder, 'parameters') else 'unknown'
                print(f"[TEXT_ENCODE_DEBUG] Text encoder on {te_device_before} before encoding")
            except:
                print(f"[TEXT_ENCODE_DEBUG] Could not detect text encoder device")

        # Text Encoder
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        context = self.text_encoder([input_prompt], self.device)[0]
        context_null = self.text_encoder([n_prompt], self.device)[0]
        context = context.to(self.dtype)
        context_null = context_null.to(self.dtype)

        _after_text_encode = time.time()
        text_encode_time = _after_text_encode - _before_text_encode
        print(f"[GENERATE_TIMING] Text encoding took {text_encode_time:.3f}s")

        # Flag if text encoding is abnormally slow (may indicate model loading from RAM)
        if text_encode_time > 5.0:
            print(f"[TEXT_ENCODE_WARNING] ⚠️  Text encoding took {text_encode_time:.1f}s (unusually slow - model may have been offloaded to RAM)")
        text_len = self.model.text_len
        context = torch.cat([context, context.new_zeros(text_len -context.size(0), context.size(1)) ]).unsqueeze(0) 
        context_null = torch.cat([context_null, context_null.new_zeros(text_len -context_null.size(0), context_null.size(1)) ]).unsqueeze(0) 
        if input_video is not None: height, width = input_video.shape[-2:]

        # NAG_prompt =  "static, low resolution, blurry"
        # context_NAG = self.text_encoder([NAG_prompt], self.device)[0]
        # context_NAG = context_NAG.to(self.dtype)
        # context_NAG = torch.cat([context_NAG, context_NAG.new_zeros(text_len -context_NAG.size(0), context_NAG.size(1)) ]).unsqueeze(0) 
        
        # from mmgp import offload
        # offloadobj.unload_all()

        offload.shared_state.update({"_nag_scale" : NAG_scale, "_nag_tau" : NAG_tau, "_nag_alpha":  NAG_alpha })
        if NAG_scale > 1: context = torch.cat([context, context_null], dim=0)
        # if NAG_scale > 1: context = torch.cat([context, context_NAG], dim=0)
        if self._interrupt: return None

        vace = model_type in ["vace_1.3B","vace_14B", "vace_multitalk_14B", "vace_standin_14B"]
        phantom = model_type in ["phantom_1.3B", "phantom_14B"]
        fantasy = model_type in ["fantasy"]
        multitalk = model_type in ["multitalk", "infinitetalk", "vace_multitalk_14B", "i2v_2_2_multitalk"]
        infinitetalk = model_type in ["infinitetalk"]
        standin = model_type in ["standin", "vace_standin_14B"]
        recam = model_type in ["recam_1.3B"]
        ti2v = model_type in ["ti2v_2_2"]
        start_step_no = 0
        ref_images_count = 0
        trim_frames = 0
        extended_overlapped_latents = None
        no_noise_latents_injection = infinitetalk
        timestep_injection = False
        lat_frames = int((frame_num - 1) // self.vae_stride[0]) + 1
        # image2video 
        if model_type in ["i2v", "i2v_2_2", "fun_inp_1.3B", "fun_inp", "fantasy", "multitalk", "infinitetalk", "i2v_2_2_multitalk", "flf2v_720p"]:
            any_end_frame = False
            if image_start is None:
                if infinitetalk:
                    new_shot = "Q" in video_prompt_type
                    if input_frames is not None:
                        image_ref = input_frames[:, 0]
                    else:
                        if input_ref_images is None:                        
                            if pre_video_frame is None: raise Exception("Missing Reference Image")
                            input_ref_images, new_shot = [pre_video_frame], False
                        new_shot = new_shot and window_no <= len(input_ref_images)
                        image_ref = convert_image_to_tensor(input_ref_images[ min(window_no, len(input_ref_images))-1 ])
                    if new_shot or input_video is None:  
                        input_video = image_ref.unsqueeze(1)
                    else:
                        color_correction_strength = 0 #disable color correction as transition frames between shots may have a complete different color level than the colors of the new shot
                _ , preframes_count, height, width = input_video.shape
                input_video = input_video.to(device=self.device).to(dtype= self.VAE_dtype)
                if infinitetalk:
                    image_start = image_ref.to(input_video)
                    control_pre_frames_count = 1 
                    control_video = image_start.unsqueeze(1)
                else:
                    image_start = input_video[:, -1]
                    control_pre_frames_count = preframes_count
                    control_video = input_video

                color_reference_frame = image_start.unsqueeze(1).clone()
            else:
                preframes_count = control_pre_frames_count = 1                
                height, width = image_start.shape[1:]
                control_video = image_start.unsqueeze(1).to(self.device)
                color_reference_frame = control_video.clone()

            any_end_frame = image_end is not None 
            add_frames_for_end_image = any_end_frame and model_type == "i2v"
            if any_end_frame:
                color_correction_strength = 0 #disable color correction as transition frames between shots may have a complete different color level than the colors of the new shot
                if add_frames_for_end_image:
                    frame_num +=1
                    lat_frames = int((frame_num - 2) // self.vae_stride[0] + 2)
                    trim_frames = 1

            lat_h, lat_w = height // self.vae_stride[1], width // self.vae_stride[2]

            if image_end is not None:
                img_end_frame = image_end.unsqueeze(1).to(self.device)

            if hasattr(self, "clip"):                                   
                clip_image_size = self.clip.model.image_size
                image_start = resize_lanczos(image_start, clip_image_size, clip_image_size)
                image_end = resize_lanczos(image_end, clip_image_size, clip_image_size) if image_end is not None else image_start
                if model_type == "flf2v_720p":                    
                    clip_context = self.clip.visual([image_start[:, None, :, :], image_end[:, None, :, :] if image_end is not None else image_start[:, None, :, :]])
                else:
                    clip_context = self.clip.visual([image_start[:, None, :, :]])
            else:
                clip_context = None

            if any_end_frame:
                enc= torch.concat([
                        control_video,
                        torch.zeros( (3, frame_num-control_pre_frames_count-1,  height, width), device=self.device, dtype= self.VAE_dtype),
                        img_end_frame,
                ], dim=1).to(self.device)
            else:
                enc= torch.concat([
                        control_video,
                        torch.zeros( (3, frame_num-control_pre_frames_count, height, width), device=self.device, dtype= self.VAE_dtype)
                ], dim=1).to(self.device)

            image_start = image_end = img_end_frame = image_ref = control_video = None

            msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
            if any_end_frame:
                msk[:, control_pre_frames_count: -1] = 0
                if add_frames_for_end_image:
                    msk = torch.concat([ torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:-1], torch.repeat_interleave(msk[:, -1:], repeats=4, dim=1) ], dim=1)
                else:
                    msk = torch.concat([ torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:] ], dim=1)
            else:
                msk[:, control_pre_frames_count:] = 0
                msk = torch.concat([ torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:] ], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]

            _before_vae_encode = time.time()
            print(f"[VAE_ENCODE_DEBUG] Starting VAE encode at {_before_vae_encode - _generate_start_time:.3f}s, frames={enc.shape if hasattr(enc, 'shape') else 'unknown'}")

            # Check VAE device location
            try:
                vae_device = next(self.vae.parameters()).device if hasattr(self.vae, 'parameters') else 'unknown'
                print(f"[VAE_ENCODE_DEBUG] VAE on device: {vae_device}")
            except:
                print(f"[VAE_ENCODE_DEBUG] Could not detect VAE device")

            # Check if input is on correct device
            if hasattr(enc, 'device'):
                print(f"[VAE_ENCODE_DEBUG] Input tensor on device: {enc.device}")

            lat_y = self.vae.encode([enc], VAE_tile_size, any_end_frame= any_end_frame and add_frames_for_end_image)[0]

            _after_vae_encode = time.time()
            vae_encode_time = _after_vae_encode - _before_vae_encode
            print(f"[GENERATE_TIMING] VAE encoding took {vae_encode_time:.3f}s")

            # Flag if VAE encoding is abnormally slow
            if vae_encode_time > 10.0:
                print(f"[VAE_ENCODE_WARNING] ⚠️  VAE encoding took {vae_encode_time:.1f}s (unusually slow - model may have been offloaded or data transfer is slow)")
            y = torch.concat([msk, lat_y])
            overlapped_latents_frames_num = int(1 + (preframes_count-1) // 4)
            # if overlapped_latents != None:
            if overlapped_latents_frames_num > 0:
                # disabled because looks worse
                if False and overlapped_latents_frames_num > 1: lat_y[:, :, 1:overlapped_latents_frames_num]  = overlapped_latents[:, 1:]
                if infinitetalk:
                    lat_y = self.vae.encode([input_video], VAE_tile_size)[0]
                extended_overlapped_latents = lat_y[:, :overlapped_latents_frames_num].clone().unsqueeze(0)
            # if control_pre_frames_count != pre_frames_count:

            lat_y = input_video = None
            kwargs.update({ 'y': y})
            if not clip_context is None:
                kwargs.update({'clip_fea': clip_context})

        # Recam Master
        if recam:
            target_camera = model_mode
            height,width = input_frames.shape[-2:]
            input_frames = input_frames.to(dtype=self.dtype , device=self.device)
            source_latents = self.vae.encode([input_frames])[0].unsqueeze(0) #.to(dtype=self.dtype, device=self.device)
            del input_frames
            # Process target camera (recammaster)
            from shared.utils.cammmaster_tools import get_camera_embedding
            cam_emb = get_camera_embedding(target_camera)       
            cam_emb = cam_emb.to(dtype=self.dtype, device=self.device)
            kwargs['cam_emb'] = cam_emb

        # Video 2 Video
        if denoising_strength < 1. and input_frames != None:
            height, width = input_frames.shape[-2:]
            source_latents = self.vae.encode([input_frames])[0].unsqueeze(0)
            injection_denoising_step = 0
            inject_from_start = False
            if input_frames != None and denoising_strength < 1 :
                color_reference_frame = input_frames[:, -1:].clone()
                if prefix_frames_count > 0:
                    overlapped_frames_num = prefix_frames_count
                    overlapped_latents_frames_num = (overlapped_latents_frames_num -1 // 4) + 1 
                    # overlapped_latents_frames_num = overlapped_latents.shape[2]
                    # overlapped_frames_num = (overlapped_latents_frames_num-1) * 4 + 1
                else: 
                    overlapped_latents_frames_num = overlapped_frames_num  = 0
                if len(keep_frames_parsed) == 0  or image_outputs or  (overlapped_frames_num + len(keep_frames_parsed)) == input_frames.shape[1] and all(keep_frames_parsed) : keep_frames_parsed = [] 
                injection_denoising_step = int( round(sampling_steps * (1. - denoising_strength),4) )
                latent_keep_frames = []
                if source_latents.shape[2] < lat_frames or len(keep_frames_parsed) > 0:
                    inject_from_start = True
                    if len(keep_frames_parsed) >0 :
                        if overlapped_frames_num > 0: keep_frames_parsed = [True] * overlapped_frames_num + keep_frames_parsed
                        latent_keep_frames =[keep_frames_parsed[0]]
                        for i in range(1, len(keep_frames_parsed), 4):
                            latent_keep_frames.append(all(keep_frames_parsed[i:i+4]))
                else:
                    timesteps = timesteps[injection_denoising_step:]
                    start_step_no = injection_denoising_step
                    if hasattr(sample_scheduler, "timesteps"): sample_scheduler.timesteps = timesteps
                    if hasattr(sample_scheduler, "sigmas"): sample_scheduler.sigmas= sample_scheduler.sigmas[injection_denoising_step:]
                    injection_denoising_step = 0

        # Phantom
        if phantom:
            input_ref_images_neg = None
            if input_ref_images != None: # Phantom Ref images
                input_ref_images = self.get_vae_latents(input_ref_images, self.device)
                input_ref_images_neg = torch.zeros_like(input_ref_images)
                ref_images_count = input_ref_images.shape[1] if input_ref_images != None else 0
                trim_frames = input_ref_images.shape[1]

        if ti2v:
            if input_video is None:
                height, width = (height // 32) * 32, (width // 32) * 32 
            else:
                height, width = input_video.shape[-2:]
                source_latents = self.vae.encode([input_video], tile_size = VAE_tile_size)[0].unsqueeze(0)
                timestep_injection = True

        # Initialize latent noise mask variables (will be populated if VACE and latent_noise_mask_strength > 0)
        latent_noise_mask_original = None
        latent_noise_mask_blend = None
        latent_noise_mask_noise = None  # Stored once for consistent blending across steps

        # Vace
        if vace :
            # vace context encode
            input_frames = [u.to(self.device) for u in input_frames]
            input_ref_images = [ None if u == None else [v.to(self.device) for v in u]  for u in input_ref_images]
            input_masks = [u.to(self.device) for u in input_masks]
            if self.background_mask != None: self.background_mask = [m.to(self.device) for m in self.background_mask]
            z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks, tile_size = VAE_tile_size, overlapped_latents = overlapped_latents )
            m0 = self.vace_encode_masks(input_masks, input_ref_images)
            if self.background_mask != None:
                color_reference_frame = input_ref_images[0][0].clone()
                zbg = self.vace_encode_frames([ref_img[0] for ref_img in input_ref_images], None, masks=self.background_mask, tile_size = VAE_tile_size )
                mbg = self.vace_encode_masks(self.background_mask, None)
                for zz0, mm0, zzbg, mmbg in zip(z0, m0, zbg, mbg):
                    zz0[:, 0:1] = zzbg
                    mm0[:, 0:1] = mmbg

                self.background_mask = zz0 = mm0 = zzbg = mmbg = None
            z = self.vace_latent(z0, m0)

            # Latent noise mask: Store original latents and mask for blending during denoising
            # z0[0] shape: [32, frames, h, w] where first 16 channels are inactive, last 16 are reactive
            # m0[0] shape: [64, frames, h, w] - the mask in latent space (0=preserve, 1=generate)
            latent_noise_mask_original = None
            latent_noise_mask_blend = None
            latent_noise_mask_noise = None
            if latent_noise_mask_strength > 0:
                # Get the inactive latents (first 16 channels of z0) - these are the preserved regions
                latent_noise_mask_original = z0[0][:16].clone().unsqueeze(0)  # [1, 16, frames, h, w]
                # Get the mask from m0 - average across the 64 channels to get a single mask
                # m0 values: 0 = preserve (black in mask video), 1 = generate (white in mask video)
                latent_noise_mask_blend = m0[0].mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, frames, h, w]
                # Store noise once for consistent blending across all denoising steps
                latent_noise_mask_noise = torch.randn_like(latent_noise_mask_original)
                print(f"[LATENT_NOISE_MASK] Enabled with strength={latent_noise_mask_strength}")
                print(f"[LATENT_NOISE_MASK] Original latents shape: {latent_noise_mask_original.shape}")
                print(f"[LATENT_NOISE_MASK] Mask blend shape: {latent_noise_mask_blend.shape}")

            ref_images_count = len(input_ref_images[0]) if input_ref_images != None and input_ref_images[0] != None else 0
            context_scale = context_scale if context_scale != None else [1.0] * len(z)
            kwargs.update({'vace_context' : z, 'vace_context_scale' : context_scale, "ref_images_count": ref_images_count })
            if overlapped_latents != None :
                overlapped_latents_size = overlapped_latents.shape[2]
                extended_overlapped_latents = z[0][:16, :overlapped_latents_size + ref_images_count].clone().unsqueeze(0)
            if prefix_frames_count > 0:
                color_reference_frame = input_frames[0][:, prefix_frames_count -1:prefix_frames_count].clone()

            target_shape = list(z0[0].shape)
            target_shape[0] = int(target_shape[0] / 2)
            lat_h, lat_w = target_shape[-2:] 
            height = self.vae_stride[1] * lat_h
            width = self.vae_stride[2] * lat_w

        else:
            target_shape = (self.vae.model.z_dim, lat_frames + ref_images_count, height // self.vae_stride[1], width // self.vae_stride[2])

        if multitalk:
            if audio_proj is None:
                audio_proj = [ torch.zeros( (1, 1, 5, 12, 768 ), dtype=self.dtype, device=self.device), torch.zeros( (1, (frame_num - 1) // 4, 8, 12, 768 ), dtype=self.dtype, device=self.device) ] 
            from .multitalk.multitalk import get_target_masks
            audio_proj = [audio.to(self.dtype) for audio in audio_proj]
            human_no = len(audio_proj[0])
            token_ref_target_masks = get_target_masks(human_no, lat_h, lat_w, height, width, face_scale = 0.05, bbox = speakers_bboxes).to(self.dtype) if human_no > 1 else None

        if fantasy and audio_proj != None:
            kwargs.update({ "audio_proj": audio_proj.to(self.dtype), "audio_context_lens": audio_context_lens, }) 


        if self._interrupt:
            return None

        expand_shape = [batch_size] + [-1] * len(target_shape)
        # Ropes
        if target_camera != None:
            shape = list(target_shape[1:])
            shape[0] *= 2
            freqs = get_rotary_pos_embed(shape, enable_RIFLEx= False) 
        else:
            freqs = get_rotary_pos_embed(target_shape[1:], enable_RIFLEx= enable_RIFLEx) 

        kwargs["freqs"] = freqs

        #Standin
        if standin:
            from preprocessing.face_preprocessor  import FaceProcessor 
            standin_ref_pos = 1 if "K" in video_prompt_type else 0
            if len(original_input_ref_images) < standin_ref_pos + 1: 
                if "I" in video_prompt_type and model_type in ["vace_standin_14B"]:
                    print("Warning: Missing Standin ref image, make sure 'Inject only People / Objets' is selected or if there is 'Landscape and then People or Objects' there are at least two ref images.")
            else: 
                standin_ref_pos = -1
                image_ref = original_input_ref_images[standin_ref_pos]
                face_processor = FaceProcessor()
                standin_ref = face_processor.process(image_ref, remove_bg = model_type in ["vace_standin_14B"])
                face_processor = None
                gc.collect()
                torch.cuda.empty_cache()
                standin_freqs = get_nd_rotary_pos_embed((-1, int(target_shape[-2]/2), int(target_shape[-1]/2) ), (-1, int(target_shape[-2]/2 + standin_ref.height/16), int(target_shape[-1]/2 + standin_ref.width/16) )) 
                standin_ref = self.vae.encode([ convert_image_to_tensor(standin_ref).unsqueeze(1) ], VAE_tile_size)[0].unsqueeze(0)
                kwargs.update({ "standin_freqs": standin_freqs, "standin_ref": standin_ref, }) 


        # Steps Skipping
        skip_steps_cache = self.model.cache
        if skip_steps_cache != None:
            cache_type = skip_steps_cache.cache_type
            x_count = 3 if phantom or fantasy or multitalk else 2
            skip_steps_cache.previous_residual = [None] * x_count
            if cache_type == "tea":
                self.model.compute_teacache_threshold(max(skip_steps_cache.start_step, start_step_no), original_timesteps, skip_steps_cache.multiplier)
            else: 
                self.model.compute_magcache_threshold(max(skip_steps_cache.start_step, start_step_no), original_timesteps, skip_steps_cache.multiplier)
                skip_steps_cache.accumulated_err, skip_steps_cache.accumulated_steps, skip_steps_cache.accumulated_ratio  = [0.0] * x_count, [0] * x_count, [1.0] * x_count
                skip_steps_cache.one_for_all = x_count > 2

        if callback != None:
            callback(-1, None, True)

        offload.shared_state["_chipmunk"] =  False
        chipmunk = offload.shared_state.get("_chipmunk", False)        
        if chipmunk:
            self.model.setup_chipmunk()

        # init denoising
        updated_num_steps= len(timesteps)

        _before_denoising_prep = time.time()
        print(f"[GENERATE_TIMING] Starting denoising prep at {_before_denoising_prep - _generate_start_time:.3f}s")

        # Check main transformer device (this is the big model that's often offloaded)
        try:
            model_device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'unknown'
            print(f"[MODEL_DEVICE_DEBUG] Main transformer on device: {model_device}")
            if self.model2 is not None:
                model2_device = next(self.model2.parameters()).device if hasattr(self.model2, 'parameters') else 'unknown'
                print(f"[MODEL_DEVICE_DEBUG] Secondary transformer on device: {model2_device}")
        except:
            print(f"[MODEL_DEVICE_DEBUG] Could not detect transformer device")

        denoising_extra = ""
        from shared.utils.loras_mutipliers import update_loras_slists, get_model_switch_steps

        # DEBUG: Log actual timesteps from scheduler
        print(f"[TIMESTEPS_DEBUG] sample_solver={sample_solver}, num_steps={updated_num_steps}, flow_shift={shift}")
        print(f"[TIMESTEPS_DEBUG] Actual timesteps from scheduler: {[f'{float(t):.1f}' for t in timesteps]}")
        print(f"[TIMESTEPS_DEBUG] switch_threshold={switch_threshold}, switch_threshold2={switch2_threshold}")

        phase_switch_step, phase_switch_step2, phases_description = get_model_switch_steps(timesteps, updated_num_steps, guide_phases, 0 if self.model2 is None else model_switch_phase, switch_threshold, switch2_threshold )
        if len(phases_description) > 0:  set_header_text(phases_description)
        guidance_switch_done =  guidance_switch2_done = False
        if guide_phases > 1: denoising_extra = f"Phase 1/{guide_phases} High Noise" if self.model2 is not None else f"Phase 1/{guide_phases}"
        def update_guidance(step_no, t, guide_scale, new_guide_scale, guidance_switch_done, switch_threshold, trans, phase_no, denoising_extra):
            if guide_phases >= phase_no and not guidance_switch_done and t <= switch_threshold:
                print(f"[PHASE_SWITCH_DEBUG] Step {step_no}: t={t:.1f} <= {switch_threshold} → Switching to phase {phase_no}/{guide_phases}")
                print(f"[PHASE_SWITCH_DEBUG] model_switch_phase={model_switch_phase}, phase_no-1={phase_no-1}, will switch model: {model_switch_phase == phase_no-1}")
                if model_switch_phase == phase_no-1 and self.model2 is not None: trans = self.model2
                guide_scale, guidance_switch_done = new_guide_scale, True
                denoising_extra = f"Phase {phase_no}/{guide_phases} {'Low Noise' if trans == self.model2 else 'High Noise'}" if self.model2 is not None else f"Phase {phase_no}/{guide_phases}"
                print(f"[PHASE_SWITCH_DEBUG] Calling callback({step_no}, denoising_extra='{denoising_extra}')")
                callback(step_no, denoising_extra = denoising_extra)
            return guide_scale, guidance_switch_done, trans, denoising_extra
        update_loras_slists(self.model, loras_slists, updated_num_steps, phase_switch_step= phase_switch_step, phase_switch_step2= phase_switch_step2)
        if self.model2 is not None: update_loras_slists(self.model2, loras_slists, updated_num_steps, phase_switch_step= phase_switch_step, phase_switch_step2= phase_switch_step2)
        callback(-1, None, True, override_num_inference_steps = updated_num_steps, denoising_extra = denoising_extra)

        def clear():
            clear_caches()
            gc.collect()
            torch.cuda.empty_cache()
            return None

        if sample_scheduler != None:
            scheduler_kwargs = {} if isinstance(sample_scheduler, FlowMatchScheduler) else {"generator": seed_g}
        # b, c, lat_f, lat_h, lat_w
        latents = torch.randn(batch_size, *target_shape, dtype=torch.float32, device=self.device, generator=seed_g)
        
        # Vid2vid initialization: Use provided video as starting point instead of pure noise
        # This is useful for VACE replace mode where we want to refine existing frames
        if vid2vid_init_video is not None and vid2vid_init_strength < 1.0:
            try:
                import cv2
                import numpy as np
                
                print(f"[VID2VID_INIT] Loading video for initialization: {vid2vid_init_video}")
                print(f"[VID2VID_INIT] Strength: {vid2vid_init_strength} (0=keep original, 1=random noise)")
                
                # Load video frames
                cap = cv2.VideoCapture(str(vid2vid_init_video))
                vid2vid_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB and normalize to [-1, 1]
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb).float().div_(127.5).sub_(1)
                    vid2vid_frames.append(frame_tensor)
                cap.release()
                
                if len(vid2vid_frames) > 0:
                    # Stack frames: [F, H, W, C] -> [C, F, H, W]
                    vid2vid_tensor = torch.stack(vid2vid_frames, dim=0).permute(3, 0, 1, 2).to(self.device)
                    print(f"[VID2VID_INIT] Loaded {len(vid2vid_frames)} frames, shape: {vid2vid_tensor.shape}")
                    
                    # Encode with VAE
                    with torch.no_grad():
                        vid2vid_latents = self.vae.encode([vid2vid_tensor], tile_size=VAE_tile_size)[0]
                    print(f"[VID2VID_INIT] Encoded to latent shape: {vid2vid_latents.shape}")
                    
                    # Handle frame count mismatch between vid2vid video and target
                    target_lat_frames = target_shape[1]  # Latent frame count
                    vid2vid_lat_frames = vid2vid_latents.shape[1]
                    
                    if vid2vid_lat_frames != target_lat_frames:
                        print(f"[VID2VID_INIT] Frame count mismatch: vid2vid has {vid2vid_lat_frames} latent frames, target has {target_lat_frames}")
                        if vid2vid_lat_frames > target_lat_frames:
                            # Trim excess frames
                            vid2vid_latents = vid2vid_latents[:, :target_lat_frames]
                        else:
                            # Pad with random noise for remaining frames
                            pad_size = target_lat_frames - vid2vid_lat_frames
                            padding = torch.randn(vid2vid_latents.shape[0], pad_size, *vid2vid_latents.shape[2:], 
                                                  device=self.device, dtype=vid2vid_latents.dtype)
                            vid2vid_latents = torch.cat([vid2vid_latents, padding], dim=1)
                    
                    # Add batch dimension if needed
                    if vid2vid_latents.dim() == 4:
                        vid2vid_latents = vid2vid_latents.unsqueeze(0)
                    
                    # Blend: latents = strength * noise + (1 - strength) * encoded
                    # Higher strength = more noise = more regeneration
                    # Lower strength = more original = less change
                    print(f"[VID2VID_INIT] Blending latents with strength {vid2vid_init_strength}")
                    latents = vid2vid_init_strength * latents + (1.0 - vid2vid_init_strength) * vid2vid_latents.to(latents.dtype)
                    
                    print(f"[VID2VID_INIT] Vid2vid initialization complete, final latent shape: {latents.shape}")
                else:
                    print(f"[VID2VID_INIT] Warning: Could not load any frames from {vid2vid_init_video}")
                    
            except Exception as e:
                print(f"[VID2VID_INIT] Error during vid2vid initialization: {e}")
                import traceback
                traceback.print_exc()
                # Continue with random latents on error
        
        if apg_switch != 0:
            apg_momentum = -0.75
            apg_norm_threshold = 55
            text_momentumbuffer  = MomentumBuffer(apg_momentum)
            audio_momentumbuffer = MomentumBuffer(apg_momentum)


        # denoising
        _before_denoising_loop = time.time()
        denoising_prep_time = _before_denoising_loop - _before_denoising_prep
        print(f"[GENERATE_TIMING] Denoising prep took {denoising_prep_time:.3f}s")
        print(f"[GENERATE_TIMING] Starting denoising loop at {_before_denoising_loop - _generate_start_time:.3f}s")

        # Final VRAM check before denoising
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated() / 1024**3
            vram_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[PRE_DENOISE_VRAM] VRAM before loop: allocated={vram_allocated:.2f}GB, reserved={vram_reserved:.2f}GB")

        trans = self.model
        _first_step_time = None
        for i, t in enumerate(tqdm(timesteps)):
            # Log timing of first denoising step to detect if there's a delay entering the loop
            if i == 0:
                _first_step_start = time.time()
                delay_before_first_step = _first_step_start - _before_denoising_loop
                if delay_before_first_step > 1.0:
                    print(f"[FIRST_STEP_DELAY] ⚠️  {delay_before_first_step:.1f}s delay before first denoising step (may indicate model loading)")
                _first_step_time = _first_step_start
            guide_scale, guidance_switch_done, trans, denoising_extra = update_guidance(i, t, guide_scale, guide2_scale, guidance_switch_done, switch_threshold, trans, 2, denoising_extra)
            guide_scale, guidance_switch2_done, trans, denoising_extra = update_guidance(i, t, guide_scale, guide3_scale, guidance_switch2_done, switch2_threshold, trans, 3, denoising_extra)
            offload.set_step_no_for_lora(trans, start_step_no + i)
            timestep = torch.stack([t])

            if timestep_injection:
                latents[:, :, :source_latents.shape[2]] = source_latents
                timestep = torch.full((target_shape[-3],), t, dtype=torch.int64, device=latents.device)
                timestep[:source_latents.shape[2]] = 0
                        
            kwargs.update({"t": timestep, "current_step": start_step_no + i})  
            kwargs["slg_layers"] = slg_layers if int(slg_start * sampling_steps) <= i < int(slg_end * sampling_steps) else None

            if denoising_strength < 1 and input_frames != None and i <= injection_denoising_step:
                sigma = t / 1000
                noise = torch.randn(batch_size, *target_shape, dtype=torch.float32, device=self.device, generator=seed_g)
                if inject_from_start:
                    new_latents = latents.clone()
                    new_latents[:,:, :source_latents.shape[2] ] = noise[:, :, :source_latents.shape[2] ] * sigma + (1 - sigma) * source_latents
                    for latent_no, keep_latent in enumerate(latent_keep_frames):
                        if not keep_latent:
                            new_latents[:, :, latent_no:latent_no+1 ] = latents[:, :, latent_no:latent_no+1]
                    latents = new_latents
                    new_latents = None
                else:
                    latents = noise * sigma + (1 - sigma) * source_latents
                noise = None

            if extended_overlapped_latents != None:
                if no_noise_latents_injection:
                    latents[:, :, :extended_overlapped_latents.shape[2]]   = extended_overlapped_latents 
                else:
                    latent_noise_factor = t / 1000
                    latents[:, :, :extended_overlapped_latents.shape[2]]   = extended_overlapped_latents  * (1.0 - latent_noise_factor) + torch.randn_like(extended_overlapped_latents ) * latent_noise_factor 
                if vace:
                    overlap_noise_factor = overlap_noise / 1000 
                    for zz in z:
                        zz[0:16, ref_images_count:extended_overlapped_latents.shape[2] ]   = extended_overlapped_latents[0, :, ref_images_count:]  * (1.0 - overlap_noise_factor) + torch.randn_like(extended_overlapped_latents[0, :, ref_images_count:] ) * overlap_noise_factor 

            if target_camera != None:
                latent_model_input = torch.cat([latents, source_latents.expand(*expand_shape)], dim=2)
            else:
                latent_model_input = latents

            any_guidance = guide_scale != 1
            if phantom:
                gen_args = {
                    "x" : ([ torch.cat([latent_model_input[:,:, :-ref_images_count], input_ref_images.unsqueeze(0).expand(*expand_shape)], dim=2) ] * 2 + 
                        [ torch.cat([latent_model_input[:,:, :-ref_images_count], input_ref_images_neg.unsqueeze(0).expand(*expand_shape)], dim=2)]),
                    "context": [context, context_null, context_null] ,
                }
            elif fantasy:
                gen_args = {
                    "x" : [latent_model_input, latent_model_input, latent_model_input],
                    "context" : [context, context_null, context_null],
                    "audio_scale": [audio_scale, None, None ]
                }
            elif multitalk and audio_proj != None:
                if guide_scale == 1:
                    gen_args = {
                        "x" : [latent_model_input, latent_model_input],
                        "context" : [context, context],
                        "multitalk_audio": [audio_proj, [torch.zeros_like(audio_proj[0][-1:]), torch.zeros_like(audio_proj[1][-1:])]],
                        "multitalk_masks": [token_ref_target_masks, None]
                    }
                    any_guidance = audio_cfg_scale != 1
                else:
                    gen_args = {
                        "x" : [latent_model_input, latent_model_input, latent_model_input],
                        "context" : [context, context_null, context_null],
                        "multitalk_audio": [audio_proj, audio_proj, [torch.zeros_like(audio_proj[0][-1:]), torch.zeros_like(audio_proj[1][-1:])]],
                        "multitalk_masks": [token_ref_target_masks, token_ref_target_masks, None]
                    }
            else:
                gen_args = {
                    "x" : [latent_model_input, latent_model_input],
                    "context": [context, context_null]
                }

            if joint_pass and any_guidance:
                # Log timing of first transformer forward pass (this is where MMGP async loading happens)
                if i == 0:
                    _before_first_forward = time.time()
                    print(f"[FIRST_FORWARD_DEBUG] Starting first transformer forward pass at {_before_first_forward - _generate_start_time:.3f}s")

                ret_values = trans( **gen_args , **kwargs)

                if i == 0:
                    _after_first_forward = time.time()
                    first_forward_time = _after_first_forward - _before_first_forward
                    print(f"[FIRST_FORWARD_TIMING] First forward pass took {first_forward_time:.3f}s")
                    if first_forward_time > 30.0:
                        print(f"[FIRST_FORWARD_WARNING] ⚠️  First forward took {first_forward_time:.1f}s (very slow - likely async loading from RAM to VRAM)")
                    elif first_forward_time > 10.0:
                        print(f"[FIRST_FORWARD_WARNING] ⚠️  First forward took {first_forward_time:.1f}s (slow - may indicate model loading)")

                if self._interrupt:
                    return clear()               
            else:
                size = len(gen_args["x"]) if any_guidance else 1 
                ret_values = [None] * size
                for x_id in range(size):
                    sub_gen_args = {k : [v[x_id]] for k, v in gen_args.items() }
                    ret_values[x_id] = trans( **sub_gen_args, x_id= x_id , **kwargs)[0]
                    if self._interrupt:
                        return clear()         
                sub_gen_args = None
            if not any_guidance:
                noise_pred = ret_values[0]       
            elif phantom:
                guide_scale_img= 5.0
                guide_scale_text= guide_scale #7.5
                pos_it, pos_i, neg = ret_values
                noise_pred = neg + guide_scale_img * (pos_i - neg) + guide_scale_text * (pos_it - pos_i)
                pos_it = pos_i = neg = None
            elif fantasy:
                noise_pred_cond, noise_pred_noaudio, noise_pred_uncond = ret_values
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_noaudio - noise_pred_uncond) + audio_cfg_scale * (noise_pred_cond  - noise_pred_noaudio) 
                noise_pred_noaudio = None
            elif multitalk and audio_proj != None:
                if apg_switch != 0:
                    if guide_scale == 1:
                        noise_pred_cond, noise_pred_drop_audio  = ret_values
                        noise_pred = noise_pred_cond + (audio_cfg_scale - 1)* adaptive_projected_guidance(noise_pred_cond - noise_pred_drop_audio, 
                                                                                        noise_pred_cond, 
                                                                                        momentum_buffer=audio_momentumbuffer, 
                                                                                        norm_threshold=apg_norm_threshold)

                    else:
                        noise_pred_cond, noise_pred_drop_text, noise_pred_uncond = ret_values
                        noise_pred = noise_pred_cond + (guide_scale - 1) * adaptive_projected_guidance(noise_pred_cond - noise_pred_drop_text, 
                                                                                                            noise_pred_cond, 
                                                                                                            momentum_buffer=text_momentumbuffer, 
                                                                                                            norm_threshold=apg_norm_threshold) \
                                + (audio_cfg_scale - 1) * adaptive_projected_guidance(noise_pred_drop_text - noise_pred_uncond, 
                                                                                        noise_pred_cond, 
                                                                                        momentum_buffer=audio_momentumbuffer, 
                                                                                        norm_threshold=apg_norm_threshold)
                else:
                    if guide_scale == 1:
                        noise_pred_cond, noise_pred_drop_audio  = ret_values
                        noise_pred = noise_pred_drop_audio + audio_cfg_scale* (noise_pred_cond - noise_pred_drop_audio)  
                    else:
                        noise_pred_cond, noise_pred_drop_text, noise_pred_uncond = ret_values
                        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_drop_text) + audio_cfg_scale * (noise_pred_drop_text - noise_pred_uncond)  
                    noise_pred_uncond = noise_pred_cond = noise_pred_drop_text = noise_pred_drop_audio = None
            else:
                noise_pred_cond, noise_pred_uncond = ret_values
                if apg_switch != 0:
                    noise_pred = noise_pred_cond + (guide_scale - 1) * adaptive_projected_guidance(noise_pred_cond - noise_pred_uncond, 
                                                                                                        noise_pred_cond, 
                                                                                                        momentum_buffer=text_momentumbuffer, 
                                                                                                        norm_threshold=apg_norm_threshold)
                else:
                    noise_pred_text = noise_pred_cond
                    if cfg_star_switch:
                        # CFG Zero *. Thanks to https://github.com/WeichenFan/CFG-Zero-star/
                        positive_flat = noise_pred_text.view(batch_size, -1)  
                        negative_flat = noise_pred_uncond.view(batch_size, -1)  

                        alpha = optimized_scale(positive_flat,negative_flat)
                        alpha = alpha.view(batch_size, 1, 1, 1)

                        if (i <= cfg_zero_step):
                            noise_pred = noise_pred_text*0. # it would be faster not to compute noise_pred...
                        else:
                            noise_pred_uncond *= alpha
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_text - noise_pred_uncond)            
            ret_values = noise_pred_uncond = noise_pred_cond = noise_pred_text = neg  = None
            
            if sample_solver == "euler":
                dt = timesteps[i] if i == len(timesteps)-1 else (timesteps[i] - timesteps[i + 1])
                dt = dt.item() / self.num_timesteps
                latents = latents - noise_pred * dt
            else:
                latents = sample_scheduler.step(
                    noise_pred[:, :, :target_shape[1]],
                    t,
                    latents,
                    **scheduler_kwargs)[0]

            # Latent Noise Mask: Blend denoised latents with noised original for preserved regions
            # This ensures preserved regions stay closer to the original throughout denoising
            # Similar to ComfyUI's SetLatentNoiseMask / WanVaceAdvanced approach
            if latent_noise_mask_strength > 0 and latent_noise_mask_original is not None and latent_noise_mask_blend is not None and latent_noise_mask_noise is not None:
                # Get next timestep (for calculating noise level at next step)
                next_t = timesteps[i+1] if i < len(timesteps) - 1 else torch.tensor([0.0], device=self.device)
                latent_noise_factor = next_t.item() / 1000.0 if hasattr(next_t, 'item') else float(next_t) / 1000.0
                
                # Get the original latents and stored noise for the frame range we're generating
                # Account for ref_images if present
                orig_start = ref_images_count if vace and ref_images_count > 0 else 0
                orig_latents = latent_noise_mask_original[:, :, orig_start:orig_start + latents.shape[2]]
                stored_noise = latent_noise_mask_noise[:, :, orig_start:orig_start + latents.shape[2]]
                mask_blend = latent_noise_mask_blend[:, :, orig_start:orig_start + latents.shape[2]]
                
                # Ensure shapes match
                if orig_latents.shape[2:] == latents.shape[2:]:
                    # Create noised version of original using stored noise (consistent across all steps)
                    noised_original = orig_latents * (1.0 - latent_noise_factor) + stored_noise * latent_noise_factor
                    
                    # Blend based on mask:
                    # mask=1 (white in mask video) = generate new content (use denoised latents)
                    # mask=0 (black in mask video) = preserve original (use noised original)
                    preserve_weight = (1.0 - mask_blend) * latent_noise_mask_strength
                    latents = latents * (1.0 - preserve_weight) + noised_original * preserve_weight
                    
                    noised_original = None

            if callback is not None:
                latents_preview = latents
                if vace and ref_images_count > 0: latents_preview = latents_preview[:, :, ref_images_count: ] 
                if trim_frames > 0:  latents_preview=  latents_preview[:, :,:-trim_frames]
                if image_outputs: latents_preview=  latents_preview[:, :,:1]
                if len(latents_preview) > 1: latents_preview = latents_preview.transpose(0,2)
                callback(i, latents_preview[0], False, denoising_extra =denoising_extra )
                latents_preview = None

        clear()
        if timestep_injection:
            latents[:, :, :source_latents.shape[2]] = source_latents

        if vace and ref_images_count > 0: latents = latents[:, :, ref_images_count:]
        if trim_frames > 0:  latents=  latents[:, :,:-trim_frames]
        if return_latent_slice != None:
            latent_slice = latents[:, :, return_latent_slice].clone()

        x0 =latents.unbind(dim=0)

        if chipmunk:
            self.model.release_chipmunk() # need to add it at every exit when in prod

        videos = self.vae.decode(x0, VAE_tile_size)

        if image_outputs:
            videos = torch.cat([video[:,:1] for video in videos], dim=1) if len(videos) > 1 else videos[0][:,:1]
        else:
            videos = videos[0] # return only first video
        if color_correction_strength > 0 and (prefix_frames_count > 0 and window_no > 1 or prefix_frames_count > 1 and window_no == 1):
            if vace and False:
                # videos = match_and_blend_colors_with_mask(videos.unsqueeze(0), input_frames[0].unsqueeze(0), input_masks[0][:1].unsqueeze(0), color_correction_strength,copy_mode= "progressive_blend").squeeze(0)
                videos = match_and_blend_colors_with_mask(videos.unsqueeze(0), input_frames[0].unsqueeze(0), input_masks[0][:1].unsqueeze(0), color_correction_strength,copy_mode= "reference").squeeze(0)
                # videos = match_and_blend_colors_with_mask(videos.unsqueeze(0), videos.unsqueeze(0), input_masks[0][:1].unsqueeze(0), color_correction_strength,copy_mode= "reference").squeeze(0)
            elif color_reference_frame is not None:
                videos = match_and_blend_colors(videos.unsqueeze(0), color_reference_frame.unsqueeze(0), color_correction_strength).squeeze(0)
            
        if return_latent_slice != None:
            return { "x" : videos, "latent_slice" : latent_slice }
        return videos

    def adapt_vace_model(self, model):
        modules_dict= { k: m for k, m in model.named_modules()}
        for model_layer, vace_layer in model.vace_layers_mapping.items():
            module = modules_dict[f"vace_blocks.{vace_layer}"]
            target = modules_dict[f"blocks.{model_layer}"]
            setattr(target, "vace", module )
        delattr(model, "vace_blocks")



