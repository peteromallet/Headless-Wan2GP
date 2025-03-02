# Wan2.1


<p align="center">
    💜 <a href=""><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.1">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="">Paper (Coming soon)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wanxai.com">Blog</a> &nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat Group</a>&nbsp&nbsp | &nbsp&nbsp 📖 <a href="https://discord.gg/p5XbdQV7">Discord</a>&nbsp&nbsp
<br>

-----

[**Wan2.1 GP by DeepBeepMeep based on Wan2.1's Alibaba: Open and Advanced Large-Scale Video Generative Models**]("") <be>

In this repository, we present **Wan2.1**, a comprehensive and open suite of video foundation models that pushes the boundaries of video generation. **Wan2.1** offers these key features:
- 👍 **SOTA Performance**: **Wan2.1** consistently outperforms existing open-source models and state-of-the-art commercial solutions across multiple benchmarks.
- 👍 **Supports Consumer-grade GPUs**: The T2V-1.3B model requires only 8.19 GB VRAM, making it compatible with almost all consumer-grade GPUs. It can generate a 5-second 480P video on an RTX 4090 in about 4 minutes (without optimization techniques like quantization). Its performance is even comparable to some closed-source models.
- 👍 **Multiple Tasks**: **Wan2.1** excels in Text-to-Video, Image-to-Video, Video Editing, Text-to-Image, and Video-to-Audio, advancing the field of video generation.
- 👍 **Visual Text Generation**: **Wan2.1** is the first video model capable of generating both Chinese and English text, featuring robust text generation that enhances its practical applications.
- 👍 **Powerful Video VAE**: **Wan-VAE** delivers exceptional efficiency and performance, encoding and decoding 1080P videos of any length while preserving temporal information, making it an ideal foundation for video and image generation.


## 🔥 Latest News!!

* Mar 03, 2025: 👋 Wan2.1GP DeepBeepMeep out of this World version ! Reduced memory consumption by 2, with possiblity to generate more than 10s of video at 720p
* Feb 25, 2025: 👋 We've released the inference code and weights of Wan2.1.
* Feb 27, 2025: 👋 Wan2.1 has been integrated into [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/wan/). Enjoy!


## Features
*GPU Poor version by **DeepBeepMeep**. This great video generator can now run smoothly on any GPU.*

This version has the following improvements over the original Alibaba model:
- Reduce greatly the RAM requirements and VRAM requirements 
- Much faster thanks to compilation and fast loading / unloading
- 5 profiles in order to able to run the model at a decent speed on a low end consumer config (32 GB of RAM and 12 VRAM) and to run it at a very good speed on a high end consumer config (48 GB of RAM and 24 GB of VRAM)
- Autodownloading of the needed model files
- Improved gradio interface with progression bar and more options
- Multiples prompts / multiple generations per prompt
- Support multiple pretrained Loras with 32 GB of RAM or less
- Switch easily between Hunyuan and Fast Hunyuan models and quantized / non quantized models
- Much simpler installation


This fork by DeepBeepMeep is an integration of the mmpg module on the original model

It is an illustration on how one can set up on an existing model some fast and properly working CPU offloading with changing only a few lines of code in the core model.

For more information on how to use the mmpg module, please go to: https://github.com/deepbeepmeep/mmgp

You will find the original Hunyuan Video repository here: https://github.com/deepbeepmeep/Wan2GP
 


## Installation Guide for Linux and Windows

We provide an `environment.yml` file for setting up a Conda environment.
Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

This app has been tested on Python 3.10 / 2.6.0  / Cuda 12.4.\

```shell
# 1 - conda. Prepare and activate a conda environment
conda env create -f environment.yml
conda activate Wan2

# OR

# 1 - venv. Alternatively create a python 3.10 venv and then do the following
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124  


# 2. Install pip dependencies
python -m pip install -r requirements.txt

# 3.1 optional Sage attention support (30% faster, easy to install on Linux but much harder on Windows)
python -m pip install sageattention==1.0.6 

# or for Sage Attention 2 (40% faster, sorry only manual compilation for the moment)
git pull https://github.com/thu-ml/SageAttention
cd sageattention 
pip install -e .

# 3.2 optional Flash attention support (easy to install on Linux but much harder on Windows)
python -m pip install flash-attn==2.7.2.post1




```

Note that *Flash attention* and *Sage attention* are quite complex to install on Windows but offers a better memory management (and consequently longer videos) than the default *sdpa attention*.
Likewise *Pytorch Compilation* will work on Windows only if you manage to install Triton. It is quite a complex process (see below for links).

### Ready to use python wheels for Windows users
I provide here links to simplify the installation for Windows users with Python 3.10 / Pytorch 2.51 / Cuda 12.4. As I am not hosting these files I won't be able to provide support neither guarantee they do what they should do.
- Triton attention (needed for *pytorch compilation* and *Sage attention*)
```
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post9/triton-3.2.0-cp310-cp310-win_amd64.whl # triton for pytorch 2.6.0
```

- Sage attention
```
pip install https://github.com/deepbeepmeep/SageAttention/raw/refs/heads/main/releases/sageattention-2.1.0-cp310-cp310-win_amd64.whl # for pytorch 2.6.0 (experimental, if it works, otherwise you you will need to install and compile manually, see above) 
 
```

## Run the application

### Run a Gradio Server on port 7860 (recommended)
```bash
python gradio_server.py
```


### Loras support

-- Ready to be used but theorical as no lora for Wan have been released as today. 

Every lora stored in the subfoler 'loras' will be automatically loaded. You will be then able to activate / desactive any of them when running the application.

For each activated Lora, you may specify a *multiplier* that is one float number that corresponds to its weight (default is 1.0), alternatively you may specify a list of floats multipliers separated by a "," that gives the evolution of this Lora's multiplier over the steps. For instance let's assume there are 30 denoising steps and the multiplier is *0.9,0.8,0.7* then for the steps ranges 0-9, 10-19 and 20-29 the Lora multiplier will be respectively 0.9, 0.8 and 0.7.

You can edit, save or delete Loras presets (combinations of loras with their corresponding multipliers) directly from the gradio interface. Each preset, is a file with ".lset" extension stored in the loras directory and can be shared with other users

Then you can pre activate loras corresponding to a preset when launching the gradio server:
```bash
python gradio_server.py --lora-preset  mylorapreset.lset # where 'mylorapreset.lset' is a preset stored in the 'loras' folder
```

Please note that command line parameters *--lora-weight* and *--lora-multiplier* have been deprecated since they are redundant with presets.

You will find prebuilt Loras on https://civitai.com/ or you will be able to build them with tools such as kohya or onetrainer.


### Command line parameters for Gradio Server
--profile no : default (4) : no of profile between 1 and 5\
--quantize-transformer bool: (default True) : enable / disable on the fly transformer quantization\
--lora-dir path : Path of directory that contains Loras in diffusers / safetensor format\
--lora-preset preset : name of preset gile (without the extension) to preload
--verbose level : default (1) : level of information between 0 and 2\
--server-port portno : default (7860) : Gradio port no\
--server-name name : default (0.0.0.0) : Gradio server name\
--open-browser : open automatically Browser when launching Gradio Server\
--compile : turn on pytorch compilation\
--attention mode: force attention mode among, sdpa, flash, sage, sage2\

### Profiles (for power users only)
You can choose between 5 profiles, these will try to leverage the most your hardware, but have little impact for HunyuanVideo GP:
- HighRAM_HighVRAM  (1):  the fastest well suited for a RTX 3090 / RTX 4090 but consumes much more VRAM, adapted for fast shorter video
- HighRAM_LowVRAM  (2): a bit slower, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos
- LowRAM_HighVRAM  (3): adapted for RTX 3090 / RTX 4090 with limited RAM  but at the cost of VRAM (shorter videos)
- LowRAM_LowVRAM  (4): if you have little VRAM or want to generate longer videos 
- VerylowRAM_LowVRAM  (5): at least 24 GB of RAM and 10 GB of VRAM : if you don't have much it won't be fast but maybe it will work

Profile 2 (High RAM) and 4 (Low RAM)are the most recommended profiles since they are versatile (support for long videos for a slight performance cost).\
However, a safe approach is to start from profile 5 (default profile) and then go down progressively to profile 4 and then to profile 2 as long as the app remains responsive or doesn't trigger any out of memory error.

### Other Models for the GPU Poor

- HuanyuanVideoGP: https://github.com/deepbeepmeep/HunyuanVideoGP :\
One of the best open source Text to Video generator

- Hunyuan3D-2GP: https://github.com/deepbeepmeep/Hunyuan3D-2GP :\
A great image to 3D and text to 3D tool by the Tencent team. Thanks to mmgp it can run with less than 6 GB of VRAM

- FluxFillGP: https://github.com/deepbeepmeep/FluxFillGP :\
One of the best inpainting / outpainting tools based on Flux that can run with less than 12 GB of VRAM.

- Cosmos1GP: https://github.com/deepbeepmeep/Cosmos1GP :\
This application include two models: a text to world generator and a image / video to world (probably the best open source image to video generator).

- OminiControlGP: https://github.com/deepbeepmeep/OminiControlGP :\
A Flux derived application very powerful that can be used to transfer an object of your choice in a prompted scene. With mmgp you can run it with only 6 GB of VRAM.

- YuE GP: https://github.com/deepbeepmeep/YuEGP :\
A great song generator (instruments + singer's voice) based on prompted Lyrics and a genre description. Thanks to mmgp you can run it with less than 10 GB of VRAM without waiting forever.


