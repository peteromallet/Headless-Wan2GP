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

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.modules.posemb_layers import get_rotary_pos_embed

from PIL import Image

def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)

def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''

        c = b1.shape[-1]

        #norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        #normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        #zero when norms are zero
        b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0

        #slerp
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        #technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)

        #edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res


def common_upscale(samples, width, height, upscale_method, crop):
        orig_shape = tuple(samples.shape)
        if len(orig_shape) > 4:
            samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
            samples = samples.movedim(2, 1)
            samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
        if crop == "center":
            old_width = samples.shape[-1]
            old_height = samples.shape[-2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        else:
            s = samples

        if upscale_method == "bislerp":
            out = bislerp(s, width, height)
        elif upscale_method == "lanczos":
            out = lanczos(s, width, height)
        else:
            out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

        if len(orig_shape) == 4:
            return out

        out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
        return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))

class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        i2v720p= True,
        model_filename ="",
        text_encoder_filename="",
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            init_on_cpu (`bool`, *optional*, defaults to True):
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=text_encoder_filename,
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {model_filename}")
        from mmgp import offload

        self.model = offload.fast_load_transformers_model(model_filename, modelClass=WanModel)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        # if dist.is_initialized():
        #     dist.barrier()
        # if dit_fsdp:
        #     self.model = shard_fn(self.model)
        # else:
        #     if not init_on_cpu:
        #         self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
        input_prompt,
        img,
        img2 = None,
        max_area=720 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        callback = None,
        enable_RIFLEx = False,
        VAE_tile_size= 0,
        joint_pass = False,
        slg_layers = None,
        slg_start = 0.0,
        slg_end = 1.0,
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        lat_frames = int((frame_num - 1) // self.vae_stride[0] + 1)
        any_end_frame = img2 !=None 
        if any_end_frame:
            any_end_frame = True
            img2 = TF.to_tensor(img2).sub_(0.5).div_(0.5).to(self.device)
            frame_num +=1
            lat_frames = int((frame_num - 2) // self.vae_stride[0] + 2)

        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = lat_frames * lat_h * lat_w // ( self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(16, lat_frames, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=self.device)        

        msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
        if any_end_frame:
            msk[:, 1: -1] = 0
            msk = torch.concat([ torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:-1], torch.repeat_interleave(msk[:, -1:], repeats=4, dim=1) ], dim=1)
        else:
            msk[:, 1:] = 0
            msk = torch.concat([ torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:] ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            # self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        from mmgp import offload

        offload.last_offload_obj.unload_all()
        if any_end_frame:
            img_interpolated = torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1).to(torch.bfloat16)
            img2_interpolated = torch.nn.functional.interpolate(img2[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1).to(torch.bfloat16) 
            mean2 = 0
            enc= torch.concat([
                    img_interpolated,
                    torch.full( (3, frame_num-2,  h, w), mean2, device="cpu", dtype= torch.bfloat16),
                    img2_interpolated,
            ], dim=1).to(self.device)
        else:
            enc= torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(),   size=(h, w), mode='bicubic').transpose(0, 1).to(torch.bfloat16),
                    torch.zeros(3, frame_num-1, h, w, device="cpu", dtype= torch.bfloat16)
            ], dim=1).to(self.device)

        lat_y = self.vae.encode([enc], VAE_tile_size, any_end_frame= any_end_frame)[0]
        y = torch.concat([msk, lat_y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
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
        else:
            raise NotImplementedError("Unsupported solver.")

        # sample videos
        latent = noise

        freqs = get_rotary_pos_embed(latent.shape[1:],  enable_RIFLEx= enable_RIFLEx) 

        arg_c = {
            'context': [context[0]],
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],
            'freqs' : freqs,
            'pipeline' : self
        }

        arg_null = {
            'context': context_null,
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],
            'freqs' : freqs,
            'pipeline' : self
        }

        arg_both= {
            'context': [context[0]],
            'context2': context_null,
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],
            'freqs' : freqs,
            'pipeline' : self
        }

        if offload_model:
            torch.cuda.empty_cache()

        if self.model.enable_teacache:
            self.model.compute_teacache_threshold(self.model.teacache_start_step, timesteps, self.model.teacache_multiplier)

        # self.model.to(self.device)
        if callback != None:
            callback(-1, None)

        for i, t in enumerate(tqdm(timesteps)):
            offload.set_step_no_for_lora(self.model, i)
            slg_layers_local = None
            if int(slg_start * sampling_steps) <= i < int(slg_end * sampling_steps):
                slg_layers_local = slg_layers

            latent_model_input = [latent.to(self.device)]
            timestep = [t]

            timestep = torch.stack(timestep).to(self.device)
            if joint_pass:
                noise_pred_cond, noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, current_step=i, slg_layers=slg_layers_local, **arg_both)
                if self._interrupt:
                    return None                
            else:
                noise_pred_cond = self.model(
                    latent_model_input,
                    t=timestep,
                    current_step=i,
                    is_uncond=False,
                    **arg_c,
                )[0]
                if self._interrupt:
                    return None                
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input,
                    t=timestep,
                    current_step=i,
                    is_uncond=True,
                    slg_layers=slg_layers_local,
                    **arg_null,
                )[0]
                if self._interrupt:
                    return None                
            del latent_model_input
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)
            del noise_pred_uncond

            latent = latent.to(
                torch.device('cpu') if offload_model else self.device)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latent = temp_x0.squeeze(0)
            del temp_x0
            del timestep

            if callback is not None:
                callback(i, latent) 


        x0 = [latent.to(self.device, dtype=torch.bfloat16)]

        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()

        if self.rank == 0:
            # x0 = [lat_y]
            video = self.vae.decode(x0, VAE_tile_size, any_end_frame= any_end_frame)[0]

            if any_end_frame:
                # video[:,  -1:] = img2_interpolated
                video = video[:,  :-1]  

        else:
            video = None

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return video
