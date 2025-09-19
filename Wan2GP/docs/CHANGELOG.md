# Changelog

## 🔥 Latest News
### August 29 2025: WanGP v8.21 -  Here Goes Your Weekend

- **InfiniteTalk Video to Video**: this feature can be used for Video Dubbing. Keep in mind that it is a *Sparse Video to Video*, that is internally only image is used by Sliding Window. However thanks to the new *Smooth Transition* mode, each new clip is connected to the previous and all the camera work is done by InfiniteTalk. If you dont get any transition, increase the number of frames of a Sliding Window (81 frames recommended)

- **StandIn**: very light model specialized in Identity Transfer. I have provided two versions of Standin: a basic one derived from the text 2 video model and another based on Vace. If used with Vace, the last reference frame given to Vace will be also used for StandIn

- **Flux ESO**: a new Flux dervied *Image Editing tool*, but this one is specialized both in *Identity Transfer* and *Style Transfer*. Style has to be understood in its wide meaning: give a reference picture of a person and another one of Sushis and you will turn this person into Sushis

### August 24 2025: WanGP v8.1 -  the RAM Liberator

- **Reserved RAM entirely freed when switching models**, you should get much less out of memory related to RAM. I have also added a button in *Configuration / Performance* that will release most of the RAM used by WanGP if you want to use another application without quitting WanGP 
- **InfiniteTalk** support: improved version of Multitalk that supposedly supports very long video generations based on an audio track. Exists in two flavors (*Single Speaker* and *Multi Speakers*) but doesnt seem to be compatible with Vace. One key new feature compared to Multitalk is that you can have different visual shots associated to the same audio: each Reference frame you provide you will be associated to a new Sliding Window. If only Reference frame is provided, it will be used for all windows. When Continuing a video, you can either continue the current shot (no Reference Frame) or add new shots (one or more Reference Frames).\
If you are not into audio, you can use still this model to generate infinite long image2video, just select "no speaker". Last but not least, Infinitetalk works works with all the Loras accelerators.
- **Flux Chroma 1 HD** support: uncensored flux based model and lighter than Flux (8.9B versus 12B) and can fit entirely in VRAM with only 16 GB of VRAM. Unfortunalely it is not distilled and you will need CFG at minimum 20 steps

### August 21 2025: WanGP v8.01 - the killer of seven

- **Qwen Image Edit** : Flux Kontext challenger (prompt driven image edition). Best results (including Identity preservation) will be obtained at 720p. Beyond you may get image outpainting and / or lose identity preservation. Below 720p prompt adherence will be worse. Qwen Image Edit works with Qwen Lora Lightning 4 steps. I have also unlocked all the resolutions for Qwen models. Bonus Zone: support for multiple image compositions but identity preservation won't be as good.
- **On demand Prompt Enhancer** (needs to be enabled in Configuration Tab) that you can use to Enhance a Text Prompt before starting a Generation. You can refine the Enhanced Prompt or change the original Prompt.
- Choice of a **Non censored Prompt Enhancer**. Beware this is one is VRAM hungry and will require 12 GB of VRAM to work
- **Memory Profile customizable per model** : useful to set for instance Profile 3 (preload the model entirely in VRAM) with only Image Generation models, if you have 24 GB of VRAM. In that case Generation will be much faster because with Image generators (contrary to Video generators) as a lot of time is wasted in offloading 
- **Expert Guidance Mode**: change the Guidance during the generation up to 2 times. Very useful with Wan 2.2 Ligthning to reduce the slow motion effect. The idea is to insert a CFG phase before the 2 accelerated phases that follow and have no Guidance. I have added the finetune *Wan2.2 Vace Lightning 3 Phases 14B* with a prebuilt configuration. Please note that it is a 8 steps process although the lora lightning is 4 steps. This expert guidance mode is also available with Wan 2.1.

*WanGP 8.01 update, improved Qwen Image Edit Identity Preservation*
### August 12 2025: WanGP v7.7777 - Lucky Day(s)

This is your lucky day ! thanks to new configuration options that will let you store generated Videos and Images in lossless compressed formats, you will find they in fact they look two times better without doing anything !

Just kidding, they will be only marginally better, but at least this opens the way to professionnal editing.

Support:
- Video: x264, x264 lossless, x265
- Images: jpeg, png, webp, wbp lossless
Generation Settings are stored in each of the above regardless of the format (that was the hard part).

Also you can now choose different output directories for images and videos.

unexpected luck: fixed lightning 8 steps for Qwen, and lightning 4 steps for Wan 2.2, now you just need 1x multiplier no weird numbers. 
*update 7.777 : oops got a crash a with FastWan ? Luck comes and goes, try a new update, maybe you will have a better chance this time*
*update 7.7777 : Sometime good luck seems to last forever. For instance what if Qwen Lightning 4 steps could also work with WanGP ?*
- https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors (Qwen Lightning 4 steps)
- https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors (new improved version of Qwen Lightning 8 steps)


### August 10 2025: WanGP v7.76 - Faster than the VAE ...
We have a funny one here today: FastWan 2.2 5B, the Fastest Video Generator, only 20s to generate 121 frames at 720p. The snag is that VAE is twice as slow... 
Thanks to Kijai for extracting the Lora that is used to build the corresponding finetune.

*WanGP 7.76: fixed the messed up I did to i2v models (loras path was wrong for Wan2.2 and Clip broken)*

### August 9 2025: WanGP v7.74 - Qwen Rebirth part 2
Added support for Qwen Lightning lora for a 8 steps generation (https://huggingface.co/lightx2v/Qwen-Image-Lightning/blob/main/Qwen-Image-Lightning-8steps-V1.0.safetensors). Lora is not normalized and you can use a multiplier around 0.1.

Mag Cache support for all the Wan2.2 models Don't forget to set guidance to 1 and 8 denoising steps , your gen will be 7x faster !

### August 8 2025: WanGP v7.73 - Qwen Rebirth
Ever wondered what impact not using Guidance has on a model that expects it ? Just look at Qween Image in WanGP 7.71 whose outputs were erratic. Somehow I had convinced myself that Qwen was a distilled model. In fact Qwen was dying for a negative prompt. And in WanGP 7.72 there is at last one for him.

As Qwen is not so picky after all I have added also quantized text encoder which reduces the RAM requirements of Qwen by 10 GB (the text encoder quantized version produced garbage before) 

Unfortunately still the Sage bug for older GPU architectures. Added Sdpa fallback for these architectures.

*7.73 update: still Sage / Sage2 bug for GPUs before RTX40xx. I have added a detection mechanism that forces Sdpa attention if that's the case*


### August 6 2025: WanGP v7.71 - Picky, picky

This release comes with two new models :
- Qwen Image: a Commercial grade Image generator capable to inject full sentences in the generated Image while still offering incredible visuals
- Wan 2.2 TextImage to Video 5B: the last Wan 2.2 needed if you want to complete your Wan 2.2 collection (loras for this folder can be stored in "\loras\5B"     )

There is catch though, they are very picky if you want to get good generations: first they both need lots of steps (50 ?) to show what they have to offer. Then for Qwen Image I had to hardcode the supported resolutions, because if you try anything else, you will get garbage. Likewise Wan 2.2 5B will remind you of Wan 1.0 if you don't ask for at least  720p. 

*7.71 update: Added VAE Tiling for both Qwen Image and Wan 2.2 TextImage to Video 5B, for low VRAM during a whole gen.*


### August 4 2025: WanGP v7.6 - Remuxed

With this new version you won't have any excuse if there is no sound in your video.

*Continue Video* now works with any video that has already some sound (hint: Multitalk ).

Also, on top of MMaudio and the various sound driven models I have added the ability to use your own soundtrack.

As a result you can apply a different sound source on each new video segment when doing a *Continue Video*. 

For instance:
- first video part: use Multitalk with two people speaking
- second video part: you apply your own soundtrack which will gently follow the multitalk conversation
- third video part: you use Vace effect and its corresponding control audio will be concatenated to the rest of the audio

To multiply the combinations I have also implemented *Continue Video* with the various image2video models.

Also:
- End Frame support added for LTX Video models
- Loras can now be targetted specifically at the High noise or Low noise models with Wan 2.2, check the Loras and Finetune guides
- Flux Krea Dev support

### July 30 2025: WanGP v7.5:  Just another release ... Wan 2.2 part 2
Here is now Wan 2.2 image2video a very good model if you want to set Start and End frames. Two Wan 2.2 models delivered, only one to go ...

Please note that although it is an image2video model it is structurally very close to Wan 2.2 text2video (same layers with only a different initial projection). Given that Wan 2.1 image2video loras don't work too well (half of their tensors are not supported), I have decided that this model will look for its loras in the text2video loras folder instead of the image2video folder.

I have also optimized RAM management with Wan 2.2 so that loras and modules will be loaded only once in RAM and Reserved RAM, this saves up to 5 GB of RAM which can make a difference...

And this time I really removed Vace Cocktail Light which gave a blurry vision.

### July 29 2025: WanGP v7.4:  Just another release ... Wan 2.2 Preview
Wan 2.2 is here.  The good news is that WanGP wont require a single byte of extra VRAM to run it and it will be as fast as Wan 2.1. The bad news is that you will need much more RAM if you want to leverage entirely this new model since it has twice has many parameters.

So here is a preview version of Wan 2.2 that is without the 5B model and Wan 2.2 image to video for the moment.

However as I felt bad to deliver only half of the wares, I gave you instead .....** Wan 2.2 Vace Experimental Cocktail** !

Very good surprise indeed, the loras and Vace partially work with Wan 2.2. We will need to wait for the official Vace 2.2 release since some Vace features are broken like identity preservation

Bonus zone: Flux multi images conditions has been added, or maybe not if I broke everything as I have been distracted by Wan...

7.4 update: I forgot to update the version number. I also removed Vace Cocktail light which didnt work well.

### July 27 2025: WanGP v7.3 : Interlude
While waiting for Wan 2.2, you will appreciate the model selection hierarchy which is very useful to collect even more models. You will also appreciate that WanGP remembers which model you used last in each model family.

### July 26 2025: WanGP v7.2 : Ode to Vace
I am really convinced that Vace can do everything the other models can do and in a better way especially as Vace can be combined with Multitalk.

Here are some new Vace improvements:
- I have provided a default finetune named *Vace Cocktail*  which is a model created on the fly using the Wan text 2 video model and the Loras used to build FusioniX. The weight of the *Detail Enhancer* Lora has been reduced to improve identity preservation. Copy the model definition in *defaults/vace_14B_cocktail.json* in the *finetunes/* folder to change the Cocktail composition. Cocktail contains already some Loras acccelerators so no need to add on top a Lora Accvid, Causvid or Fusionix, ... . The whole point of Cocktail is to be able  to build you own FusioniX (which originally is a combination of 4 loras) but without the inconvenient of FusioniX.
- Talking about identity preservation, it tends to go away when one generates a single Frame instead of a Video which is shame for our Vace photoshop. But there is a solution : I have added an Advanced Quality option, that tells WanGP to generate a little more than a frame (it will still keep only the first frame). It will be a little slower but you will be amazed how Vace Cocktail combined with this option will preserve identities (bye bye *Phantom*). 
- As in practise I have observed one switches frequently between *Vace text2video* and *Vace text2image* I have put them in the same place they are now just one tab away, no need to reload the model. Likewise *Wan text2video* and *Wan tex2image* have been merged.
- Color fixing when using Sliding Windows. A new postprocessing *Color Correction* applied automatically by default (you can disable it in the *Advanced tab Sliding Window*) will try to match the colors of the new window with that of the previous window. It doesnt fix all the unwanted artifacts of the new window but at least this makes the transition smoother. Thanks to the multitalk team for the original code.

Also you will enjoy our new real time statistics (CPU / GPU usage, RAM / VRAM used, ... ). Many thanks to **Redtash1** for providing the framework for this new feature ! You need to go in the Config tab to enable real time stats.


### July 21 2025: WanGP v7.12
- Flux Family Reunion : *Flux Dev* and *Flux Schnell* have been invited aboard WanGP. To celebrate that, Loras support for the Flux *diffusers* format has also been added.

- LTX Video upgraded to version 0.9.8: you can now generate 1800 frames (1 min of video !) in one go without a sliding window. With the distilled model it will take only 5 minutes with a RTX 4090 (you will need 22 GB of VRAM though). I have added options to select higher humber frames if you want to experiment (go to Configuration Tab / General / Increase the Max Number of Frames, change the value and restart the App)

- LTX Video ControlNet : it is a Control Net that allows you for instance to transfer a Human motion or Depth from a control video. It is not as powerful as Vace but can produce interesting things especially as now you can generate quickly a 1 min video. Under the scene IC-Loras (see below) for Pose, Depth and Canny are automatically loaded for you, no need to add them. 

- LTX IC-Lora support: these are special Loras that consumes a conditional image or video
Beside the pose, depth and canny IC-Loras transparently loaded there is the *detailer* (https://huggingface.co/Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8) which is basically an upsampler. Add the *detailer* as a Lora and use LTX Raw Format as control net choice to use it.

- Matanyone is now also for the GPU Poor as its VRAM requirements have been divided by 2! (7.12 shadow update)

- Easier way to select video resolution 

### July 15 2025: WanGP v7.0 is an AI Powered Photoshop
This release turns the Wan models into Image Generators. This goes way more than allowing to generate a video made of single frame :
- Multiple Images generated at the same time so that you can choose the one you like best.It is Highly VRAM optimized so that you can generate for instance 4 720p Images at the same time with less than 10 GB
- With the *image2image* the original text2video WanGP becomes an image upsampler / restorer
- *Vace image2image* comes out of the box with image outpainting, person / object replacement, ...
- You can use in one click a newly Image generated as Start Image or Reference Image for a Video generation

And to complete the full suite of AI Image Generators, Ladies and Gentlemen please welcome for the first time in WanGP : **Flux Kontext**.\
As a reminder Flux Kontext is an image editor : give it an image and a prompt and it will do the change for you.\
This highly optimized version of Flux Kontext will make you feel that you have been cheated all this time as WanGP Flux Kontext requires only 8 GB of VRAM to generate 4 images at the same time with no need for quantization.

WanGP v7 comes with *Image2image* vanilla and *Vace FusinoniX*. However you can build your own finetune where you will combine a text2video or Vace model with any combination of Loras.

Also in the news:
- You can now enter the *Bbox* for each speaker in *Multitalk* to precisely locate who is speaking. And to save some headaches the *Image Mask generator* will give you the *Bbox* coordinates of an area you have selected.
- *Film Grain* post processing to add a vintage look at your video
- *First Last Frame to Video* model should work much better now as I have discovered rencently its implementation was not complete
- More power for the finetuners, you can now embed Loras directly in the finetune definition. You can also override the default models (titles, visibility, ...) with your own finetunes. Check the doc that has been updated.


### July 10 2025: WanGP v6.7, is NAG a game changer ? you tell me
Maybe you knew that already but most *Loras accelerators* we use today (Causvid, FusioniX) don't use *Guidance* at all (that it is *CFG* is set to 1). This helps to get much faster generations but the downside is that *Negative Prompts* are completely ignored (including the default ones set by the models). **NAG** (https://github.com/ChenDarYen/Normalized-Attention-Guidance) aims to solve that by injecting the *Negative Prompt* during the *attention* processing phase.

So WanGP 6.7 gives you NAG, but not any NAG, a *Low VRAM* implementation, the default one ends being VRAM greedy. You will find NAG in the *General* advanced tab for most Wan models. 

Use NAG especially when Guidance is set to 1. To turn it on set the **NAG scale** to something around 10. There are other NAG parameters **NAG tau** and **NAG alpha** which I recommend to change only if you don't get good results by just playing with the NAG scale. Don't hesitate to share on this discord server the best combinations for these 3 parameters.

The authors of NAG claim that NAG can also be used when using a Guidance (CFG > 1) and to improve the prompt adherence.

### July 8 2025: WanGP v6.6, WanGP offers you **Vace Multitalk Dual Voices Fusionix Infinite** :
**Vace** our beloved super Control Net has been combined with **Multitalk** the new king in town that can animate up to two people speaking (**Dual Voices**). It is accelerated by the **Fusionix** model and thanks to *Sliding Windows* support and *Adaptive Projected Guidance* (much slower but should reduce the reddish effect with long videos) your two people will be able to talk for very a long time (which is an **Infinite** amount of time in the field of video generation).

Of course you will get as well *Multitalk* vanilla and also *Multitalk 720p* as a bonus.

And since I am mister nice guy I have enclosed as an exclusivity an *Audio Separator* that will save you time to isolate each voice when using Multitalk with two people.

As I feel like resting a bit I haven't produced yet a nice sample Video to illustrate all these new capabilities. But here is the thing, I ams sure you will publish in the *Share Your Best Video* channel your *Master Pieces*. The best ones will be added to the *Announcements Channel* and will bring eternal fame to its authors.

But wait, there is more:
- Sliding Windows support has been added anywhere with Wan models, so imagine with text2video recently upgraded in 6.5 into a video2video, you can now upsample very long videos regardless of your VRAM. The good old image2video model can now reuse the last image to produce new videos (as requested by many of you)
- I have added also the capability to transfer the audio of the original control video (Misc. advanced tab) and an option to preserve the fps into the generated video, so from now on you will be to upsample / restore your old families video and keep the audio at their original pace. Be aware that the duration will be limited to 1000 frames as I still need to add streaming support for unlimited video sizes.

Also, of interest too:
- Extract video info from Videos that have not been generated by WanGP, even better you can also apply post processing (Upsampling / MMAudio) on non WanGP videos
- Force the generated video fps to your liking, works wery well with Vace when using a Control Video
- Ability to chain URLs of Finetune models (for instance put the URLs of a model in your main finetune and reference this finetune in other finetune models to save time)

### July 2 2025: WanGP v6.5.1, WanGP takes care of you: lots of quality of life features:
- View directly inside WanGP the properties (seed, resolutions, length, most settings...) of the past generations
- In one click use the newly generated video as a Control Video or Source Video to be continued 
- Manage multiple settings for the same model and switch between them using a dropdown box 
- WanGP will keep the last generated videos in the Gallery and will remember the last model you used if you restart the app but kept the Web page open
- Custom resolutions : add a file in the WanGP folder with the list of resolutions you want to see in WanGP (look at the instruction readme in this folder)

Taking care of your life is not enough, you want new stuff to play with ?
- MMAudio directly inside WanGP : add an audio soundtrack that matches the content of your video. By the way it is a low VRAM MMAudio and 6 GB of VRAM should be sufficient. You will need to go in the *Extensions* tab of the WanGP *Configuration* to enable MMAudio
- Forgot to upsample your video during the generation ? want to try another MMAudio variation ? Fear not you can also apply upsampling or add an MMAudio track once the video generation is done. Even better you can ask WangGP for multiple variations of MMAudio to pick the one you like best
- MagCache support: a new step skipping approach, supposed to be better than TeaCache. Makes a difference if you usually generate with a high number of steps
- SageAttention2++ support : not just the compatibility but also a slightly reduced VRAM usage
- Video2Video in Wan Text2Video : this is the paradox, a text2video can become a video2video if you start the denoising process later on an existing video
- FusioniX upsampler: this is an illustration of Video2Video in Text2Video. Use the FusioniX text2video model with an output resolution of 1080p and a denoising strength of 0.25 and you will get one of the best upsamplers (in only 2/3 steps, you will need lots of VRAM though). Increase the denoising strength and you will get one of the best Video Restorer
- Choice of Wan Samplers / Schedulers
- More Lora formats support

**If you had upgraded to v6.5 please upgrade again to 6.5.1 as this will fix a bug that ignored Loras beyond the first one**

### June 23 2025: WanGP v6.3, Vace Unleashed. Thought we couldnt squeeze Vace even more ?
- Multithreaded preprocessing when possible for faster generations
- Multithreaded frames Lanczos Upsampling as a bonus
- A new Vace preprocessor : *Flow* to extract fluid motion
- Multi Vace Controlnets: you can now transfer several properties at the same time. This opens new possibilities to explore, for instance if you transfer *Human Movement* and *Shapes* at the same time for some reasons the lighting of your character will take into account much more the environment of your character.
- Injected Frames Outpainting, in case you missed it in WanGP 6.21

Don't know how to use all of the Vace features ? Check the Vace Guide embedded in WanGP as it has also been updated.


### June 19 2025: WanGP v6.2, Vace even more Powercharged
👋 Have I told you that I am a big fan of Vace ? Here are more goodies to unleash its power: 
- If you ever wanted to watch Star Wars in 4:3, just use the new *Outpainting* feature and it will add the missing bits of image at the top and the bottom of the screen. The best thing is *Outpainting* can be combined with all the other Vace modifications, for instance you can change the main character of your favorite movie at the same time  
- More processing can combined at the same time  (for instance the depth process can be applied outside the mask)
- Upgraded the depth extractor to Depth Anything 2 which is much more detailed

As a bonus, I have added two finetunes based on the Safe-Forcing technology (which requires only 4 steps to generate a video): Wan 2.1 text2video Self-Forcing and Vace Self-Forcing. I know there is Lora around but the quality of the Lora is worse (at least with Vace) compared to the full model. Don't hesitate to share your opinion about this on the discord server. 
### June 17 2025: WanGP v6.1, Vace Powercharged
👋 Lots of improvements for Vace the Mother of all Models:
- masks can now be combined with on the fly processing of a control video, for instance you can extract the motion of a specific person defined by a mask
- on the fly modification of masks : reversed masks (with the same mask you can modify the background instead of the people covered by the masks), enlarged masks (you can cover more area if for instance the person you are trying to inject is larger than the one in the mask), ...
- view these modified masks directly inside WanGP during the video generation to check they are really as expected
- multiple frames injections: multiples frames can be injected at any location of the video
- expand past videos in on click: just select one generated video to expand it

Of course all these new stuff work on all Vace finetunes (including Vace Fusionix).

Thanks also to Reevoy24 for adding a Notfication sound at the end of a generation and for fixing the background color of the current generation summary.

### June 12 2025: WanGP v6.0
👋 *Finetune models*: You find the 20 models supported by WanGP not sufficient ? Too impatient to wait for the next release to get the support for a newly released model ? Your prayers have been answered: if a new model is compatible with a model architecture supported by WanGP, you can add by yourself the support for this model in WanGP by just creating a finetune model definition. You can then store this model in the cloud (for instance in Huggingface) and the very light finetune definition file can be easily shared with other users. WanGP will download automatically the finetuned model for them.

To celebrate the new finetunes support, here are a few finetune gifts (directly accessible from the model selection menu):
- *Fast Hunyuan Video* : generate model t2v in only 6 steps
- *Hunyuan Vido AccVideo* : generate model t2v in only 5 steps
- *Wan FusioniX*: it is a combo of AccVideo / CausVid ans other models and can generate high quality Wan videos in only 8 steps

One more thing...

The new finetune system can be used to combine complementaty models : what happens when you combine  Fusionix Text2Video and Vace Control Net ?

You get **Vace FusioniX**: the Ultimate Vace Model, Fast (10 steps, no need for guidance) and with a much better quality Video than the original slower model (despite being the best Control Net out there). Here goes one more finetune...

Check the *Finetune Guide* to create finetune models definitions and share them on the WanGP discord server.

### June 11 2025: WanGP v5.5
👋 *Hunyuan Video Custom Audio*: it is similar to Hunyuan Video Avatar excpet there isn't any lower limit on the number of frames and you can use your reference images in a different context than the image itself\
*Hunyuan Video Custom Edit*: Hunyuan Video Controlnet, use it to do inpainting and replace a person in a video while still keeping his poses. Similar to Vace but less restricted than the Wan models in terms of content...

### June 6 2025: WanGP v5.41
👋 Bonus release: Support for **AccVideo** Lora to speed up x2 Video generations in Wan models. Check the Loras documentation to get the usage instructions of AccVideo.

### June 6 2025: WanGP v5.4
👋 World Exclusive : Hunyuan Video Avatar Support ! You won't need 80 GB of VRAM nor 32 GB oF VRAM, just 10 GB of VRAM will be sufficient to generate up to 15s of high quality speech / song driven Video at a high speed with no quality degradation. Support for TeaCache included.

### May 26, 2025: WanGP v5.3
👋 Happy with a Video generation and want to do more generations using the same settings but you can't remember what you did or you find it too hard to copy/paste one per one each setting from the file metadata? Rejoice! There are now multiple ways to turn this tedious process into a one click task:
- Select one Video recently generated in the Video Gallery and click *Use Selected Video Settings*
- Click *Drop File Here* and select a Video you saved somewhere, if the settings metadata have been saved with the Video you will be able to extract them automatically
- Click *Export Settings to File* to save on your harddrive the current settings. You will be able to use them later again by clicking *Drop File Here* and select this time a Settings json file

### May 23, 2025: WanGP v5.21
👋 Improvements for Vace: better transitions between Sliding Windows, Support for Image masks in Matanyone, new Extend Video for Vace, different types of automated background removal

### May 20, 2025: WanGP v5.2
👋 Added support for Wan CausVid which is a distilled Wan model that can generate nice looking videos in only 4 to 12 steps. The great thing is that Kijai (Kudos to him!) has created a CausVid Lora that can be combined with any existing Wan t2v model 14B like Wan Vace 14B. See [LORAS.md](LORAS.md) for instructions on how to use CausVid.

Also as an experiment I have added support for the MoviiGen, the first model that claims to be capable of generating 1080p videos (if you have enough VRAM (20GB...) and be ready to wait for a long time...). Don't hesitate to share your impressions on the Discord server.

### May 18, 2025: WanGP v5.1
👋 Bonus Day, added LTX Video 13B Distilled: generate in less than one minute, very high quality Videos!

### May 17, 2025: WanGP v5.0
👋 One App to Rule Them All! Added support for the other great open source architectures:
- **Hunyuan Video**: text 2 video (one of the best, if not the best t2v), image 2 video and the recently released Hunyuan Custom (very good identity preservation when injecting a person into a video)
- **LTX Video 13B** (released last week): very long video support and fast 720p generation. Wan GP version has been greatly optimized and reduced LTX Video VRAM requirements by 4!

Also:
- Added support for the best Control Video Model, released 2 days ago: Vace 14B
- New Integrated prompt enhancer to increase the quality of the generated videos

*You will need one more `pip install -r requirements.txt`*

### May 5, 2025: WanGP v4.5
👋 FantasySpeaking model, you can animate a talking head using a voice track. This works not only on people but also on objects. Also better seamless transitions between Vace sliding windows for very long videos. New high quality processing features (mixed 16/32 bits calculation and 32 bits VAE)

### April 27, 2025: WanGP v4.4
👋 Phantom model support, very good model to transfer people or objects into video, works quite well at 720p and with the number of steps > 30

### April 25, 2025: WanGP v4.3
👋 Added preview mode and support for Sky Reels v2 Diffusion Forcing for high quality "infinite length videos". Note that Skyreel uses causal attention that is only supported by Sdpa attention so even if you choose another type of attention, some of the processes will use Sdpa attention.

### April 18, 2025: WanGP v4.2
👋 FLF2V model support, official support from Wan for image2video start and end frames specialized for 720p.

### April 17, 2025: WanGP v4.1
👋 Recam Master model support, view a video from a different angle. The video to process must be at least 81 frames long and you should set at least 15 steps denoising to get good results.

### April 13, 2025: WanGP v4.0
👋 Lots of goodies for you!
- A new UI, tabs were replaced by a Dropdown box to easily switch models
- A new queuing system that lets you stack in a queue as many text2video, image2video tasks, ... as you want. Each task can rely on complete different generation parameters (different number of frames, steps, loras, ...). Many thanks to **Tophness** for being a big contributor on this new feature
- Temporal upsampling (Rife) and spatial upsampling (Lanczos) for a smoother video (32 fps or 64 fps) and to enlarge your video by x2 or x4. Check these new advanced options.
- Wan Vace Control Net support: with Vace you can inject in the scene people or objects, animate a person, perform inpainting or outpainting, continue a video, ... See [VACE.md](VACE.md) for introduction guide.
- Integrated *Matanyone* tool directly inside WanGP so that you can create easily inpainting masks used in Vace
- Sliding Window generation for Vace, create windows that can last dozens of seconds
- New optimizations for old generation GPUs: Generate 5s (81 frames, 15 steps) of Vace 1.3B with only 5GB and in only 6 minutes on a RTX 2080Ti and 5s of t2v 14B in less than 10 minutes.

### March 27, 2025
👋 Added support for the new Wan Fun InP models (image2video). The 14B Fun InP has probably better end image support but unfortunately existing loras do not work so well with it. The great novelty is the Fun InP image2 1.3B model: Image 2 Video is now accessible to even lower hardware configuration. It is not as good as the 14B models but very impressive for its size. Many thanks to the VideoX-Fun team (https://github.com/aigc-apps/VideoX-Fun)

### March 26, 2025
👋 Good news! Official support for RTX 50xx please check the [installation instructions](INSTALLATION.md).

### March 24, 2025: Wan2.1GP v3.2
👋 
- Added Classifier-Free Guidance Zero Star. The video should match better the text prompt (especially with text2video) at no performance cost: many thanks to the **CFG Zero * Team**. Don't hesitate to give them a star if you appreciate the results: https://github.com/WeichenFan/CFG-Zero-star
- Added back support for PyTorch compilation with Loras. It seems it had been broken for some time
- Added possibility to keep a number of pregenerated videos in the Video Gallery (useful to compare outputs of different settings)

*You will need one more `pip install -r requirements.txt`*

### March 19, 2025: Wan2.1GP v3.1
👋 Faster launch and RAM optimizations (should require less RAM to run)

*You will need one more `pip install -r requirements.txt`*

### March 18, 2025: Wan2.1GP v3.0
👋 
- New Tab based interface, you can switch from i2v to t2v conversely without restarting the app
- Experimental Dual Frames mode for i2v, you can also specify an End frame. It doesn't always work, so you will need a few attempts.
- You can save default settings in the files *i2v_settings.json* and *t2v_settings.json* that will be used when launching the app (you can also specify the path to different settings files)
- Slight acceleration with loras

*You will need one more `pip install -r requirements.txt`*

Many thanks to *Tophness* who created the framework (and did a big part of the work) of the multitabs and saved settings features

### March 18, 2025: Wan2.1GP v2.11
👋 Added more command line parameters to prefill the generation settings + customizable output directory and choice of type of metadata for generated videos. Many thanks to *Tophness* for his contributions.

*You will need one more `pip install -r requirements.txt` to reflect new dependencies*

### March 18, 2025: Wan2.1GP v2.1
👋 More Loras!: added support for 'Safetensors' and 'Replicate' Lora formats.

*You will need to refresh the requirements with a `pip install -r requirements.txt`*

### March 17, 2025: Wan2.1GP v2.0
👋 The Lora festival continues:
- Clearer user interface
- Download 30 Loras in one click to try them all (expand the info section)
- Very easy to use Loras as now Lora presets can input the subject (or other needed terms) of the Lora so that you don't have to modify manually a prompt
- Added basic macro prompt language to prefill prompts with different values. With one prompt template, you can generate multiple prompts.
- New Multiple images prompts: you can now combine any number of images with any number of text prompts (need to launch the app with --multiple-images)
- New command lines options to launch directly the 1.3B t2v model or the 14B t2v model

### March 14, 2025: Wan2.1GP v1.7
👋 
- Lora Fest special edition: very fast loading/unload of loras for those Loras collectors around. You can also now add/remove loras in the Lora folder without restarting the app.
- Added experimental Skip Layer Guidance (advanced settings), that should improve the image quality at no extra cost. Many thanks to the *AmericanPresidentJimmyCarter* for the original implementation

*You will need to refresh the requirements `pip install -r requirements.txt`*

### March 13, 2025: Wan2.1GP v1.6
👋 Better Loras support, accelerated loading Loras.

*You will need to refresh the requirements `pip install -r requirements.txt`*

### March 10, 2025: Wan2.1GP v1.5
👋 Official Teacache support + Smart Teacache (find automatically best parameters for a requested speed multiplier), 10% speed boost with no quality loss, improved lora presets (they can now include prompts and comments to guide the user)

### March 7, 2025: Wan2.1GP v1.4
👋 Fix PyTorch compilation, now it is really 20% faster when activated

### March 4, 2025: Wan2.1GP v1.3
👋 Support for Image to Video with multiples images for different images/prompts combinations (requires *--multiple-images* switch), and added command line *--preload x* to preload in VRAM x MB of the main diffusion model if you find there is too much unused VRAM and you want to (slightly) accelerate the generation process.

*If you upgrade you will need to do a `pip install -r requirements.txt` again.*

### March 4, 2025: Wan2.1GP v1.2
👋 Implemented tiling on VAE encoding and decoding. No more VRAM peaks at the beginning and at the end

### March 3, 2025: Wan2.1GP v1.1
👋 Added Tea Cache support for faster generations: optimization of kijai's implementation (https://github.com/kijai/ComfyUI-WanVideoWrapper/) of teacache (https://github.com/ali-vilab/TeaCache)

### March 2, 2025: Wan2.1GP by DeepBeepMeep v1
👋 Brings:
- Support for all Wan including the Image to Video model
- Reduced memory consumption by 2, with possibility to generate more than 10s of video at 720p with a RTX 4090 and 10s of video at 480p with less than 12GB of VRAM. Many thanks to REFLEx (https://github.com/thu-ml/RIFLEx) for their algorithm that allows generating nice looking video longer than 5s.
- The usual perks: web interface, multiple generations, loras support, sage attention, auto download of models, ...

## Original Wan Releases

### February 25, 2025
👋 We've released the inference code and weights of Wan2.1.

### February 27, 2025
👋 Wan2.1 has been integrated into [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/wan/). Enjoy! 