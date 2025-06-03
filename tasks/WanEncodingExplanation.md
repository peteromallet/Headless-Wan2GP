# VACE Encoding in Wan2GP Explained

This document explains how the VACE (Video AutoEncoder ControlNet-like) encoding is applied within the Wan2GP framework. VACE enables ControlNet-like functionalities for video generation, such as video-to-video and reference-image-to-video tasks.

## Overview

The VACE encoding process involves several key components and steps:

1.  **Initialization**: When a VACE model is selected, specific VACE components are initialized and integrated into the main `WanModel`.
2.  **Input Preprocessing**: Video frames, reference images, and masks are preprocessed.
3.  **VAE Encoding**: The preprocessed inputs are encoded into a latent representation using the `WanVAE`.
    *   Frames are encoded by `WanVAE`. If masks are present, "inactive" and "reactive" parts of the frames are encoded separately and then combined.
    *   Reference images are also encoded by `WanVAE`.
    *   Masks are processed (reshaped and interpolated) to match the VAE's latent dimensions.
4.  **VACE Context Creation**: The encoded frames, reference images, and masks are combined to form the `vace_context`.
5.  **Integration into `WanModel`**: The `vace_context` is injected into the `WanModel`'s attention blocks at specified layers during the diffusion process.

## Detailed Steps and File References

### 1. Initialization and Model Adaptation

*   **File**: `Wan2GP/wan/text2video.py`
*   **Class**: `WanT2V`
    *   In the `__init__` method (around lines 42-98):
        *   If the `model_filename` contains "Vace", a `VaceVideoProcessor` is initialized (line 90).
        *   The crucial step is `self.adapt_vace_model()` (line 97).
    *   **`adapt_vace_model()` method (lines 569-576)**:
        *   This method is responsible for integrating VACE functionality into the main `WanModel`.
        *   It iterates through `model.vace_layers_mapping` (which is defined in `Wan2GP/wan/modules/model.py` during `WanModel` initialization if `vace_layers` are specified).
        *   For each VACE layer, it takes a pre-initialized VACE-specific attention block from `model.vace_blocks` (these are `VaceWanAttentionBlock` instances) and assigns it as an attribute (named `vace`) to the corresponding standard `WanAttentionBlock` in `model.blocks`.
        *   Example: `setattr(target, "vace", module)`, where `target` is a `WanAttentionBlock` and `module` is a `VaceWanAttentionBlock`.
        *   After this, the original `model.vace_blocks` attribute is deleted from the model.

### 2. VACE Input Encoding

*   **File**: `Wan2GP/wan/text2video.py`
*   **Class**: `WanT2V`
    *   **`vace_encode_frames(frames, ref_images, masks, ...)` method (lines 100-130)**:
        *   This method encodes the input video frames and reference images.
        *   It uses `self.vae.encode(...)` for the actual encoding. `self.vae` is an instance of `WanVAE` (from `Wan2GP/wan/modules/vae.py`).
        *   **If masks are provided**:
            *   Frames are split into `inactive` parts (`i * (1 - m)`) and `reactive` parts (`i * m`).
            *   Both parts are encoded separately by `self.vae.encode()`.
            *   The resulting latents are concatenated: `latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]`. This doubles the channel dimension of the latents for masked regions.
        *   **If reference images (`ref_images`) are provided**:
            *   Reference images are encoded using `self.vae.encode()`.
            *   If masks were also provided, the reference latents are padded with zeros on one half of the channel dimension (`ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]`).
            *   The reference latents are concatenated with the frame latents along the temporal dimension (`dim=1`).
    *   **`vace_encode_masks(masks, ref_images=None)` method (lines 132-160)**:
        *   This method processes the input masks.
        *   Masks are reshaped and then interpolated using `F.interpolate(..., mode='nearest-exact')` to match the spatial and temporal dimensions of the VAE's latent space (considering `self.vae_stride`).
        *   If reference images are provided, the mask is padded with zeros at the beginning of the temporal dimension to account for the concatenated reference latents.
    *   **`vace_latent(z, m)` method (line 162)**:
        *   A simple utility that concatenates the latents from `vace_encode_frames` (`z`) and the processed masks from `vace_encode_masks` (`m`) along the channel dimension (`dim=0`).
        *   `z` itself (from `vace_encode_frames`) might already have a doubled channel dimension if input masks were used. So, if an input mask was provided, `z` would be `[2*C, T, H, W]` (inactive_latent, reactive_latent) and `m` would be `[C_mask, T, H, W]`. This function seems to concatenate them further, but `z` in the `generate` function (where `vace_latent` is called) is `z0` which is the output of `vace_encode_frames`. If masks were used in `vace_encode_frames`, `z0` already contains both inactive and reactive parts. The `m0` (encoded masks) are concatenated to this `z0`.
        *   More accurately, looking at the `generate` function (line 376-377), `z0 = self.vace_encode_frames(...)` and `m0 = self.vace_encode_masks(...)`, then `z = self.vace_latent(z0, m0)`. If `vace_encode_frames` already concatenated inactive/reactive parts, `z0` has shape `[B, 2*channels, T, H, W]`. Then `m0` (the VAE-encoded mask itself) is concatenated, leading to `[B, 2*channels + mask_channels, T, H, W]`. This resulting `z` is the `vace_context`.

### 3. `WanVAE` Encoding Details

*   **File**: `Wan2GP/wan/modules/vae.py`
*   **Class**: `WanVAE`
    *   The `encode(videos, tile_size, ...)` method (lines 813-823) is a wrapper.
    *   It calls `self.model.encode(...)` or `self.model.spatial_tiled_encode(...)`. `self.model` is an instance of `WanVAE_` (note the underscore).
*   **Class**: `WanVAE_`
    *   **`encode(x, scale=None, ...)` method (lines 534-569)**:
        *   The input video tensor `x` is processed in temporal chunks (first frame, then 4-frame chunks, then last frame).
        *   Each chunk goes through `self.encoder` (an `Encoder3d` instance).
        *   The outputs are concatenated temporally.
        *   A final convolution `self.conv1(out).chunk(2, dim=1)` produces `mu` and `log_var`.
        *   Only `mu` is returned (potentially scaled) and used as the latent representation.

### 4. Injection into `WanModel`

*   **File**: `Wan2GP/wan/text2video.py`
*   **Class**: `WanT2V`
    *   In the `generate()` method (around lines 275-473):
        *   If VACE is active (`vace = "Vace" in model_filename`):
            *   `z0 = self.vace_encode_frames(...)` (line 376)
            *   `m0 = self.vace_encode_masks(...)` (line 377)
            *   `z = self.vace_latent(z0, m0)` (line 378). This `z` is the `vace_context`.
            *   This `vace_context` is added to the keyword arguments passed to the diffusion model's sampling loop: `kwargs.update({'vace_context' : z, 'vace_context_scale' : context_scale})` (line 442).

*   **File**: `Wan2GP/wan/modules/model.py`
*   **Class**: `WanModel`
    *   **`__init__(..., vace_layers=None, vace_in_dim=None, ...)` constructor (lines 656-808)**:
        *   If `vace_layers` is provided (which happens if a VACE model is being loaded):
            *   `self.vace_layers_mapping` is created.
            *   `self.blocks` (list of `WanAttentionBlock`) are recreated, and each `WanAttentionBlock` that corresponds to a VACE layer gets a `block_id` (line 773).
            *   `self.vace_blocks` is initialized as a `nn.ModuleList` of `VaceWanAttentionBlock` instances (lines 779-783). These are the blocks that were later assigned using `setattr` in `WanT2V.adapt_vace_model`.
            *   `self.vace_patch_embedding = nn.Conv3d(...)` is created (lines 786-788). This layer processes the raw `vace_context` before it's fed into the attention blocks.
    *   **`forward(..., vace_context=None, vace_context_scale=1.0, ...)` method (lines 902-1081)**:
        *   If `vace_context` is provided (lines 1017-1024):
            *   The `vace_context` is first processed by `self.vace_patch_embedding`.
            *   This processed context `c` is then packaged into `hints_list = [ [c] for _ in range(len(x_list)) ]`.
        *   Inside the loop iterating through `self.blocks` (line 1049, though the loop itself is not shown in this snippet but implied by how `block(x, e, hints=hints_list[i], ...)` is called):
            *   Each `block` (which is a `WanAttentionBlock`) receives the `hints` (the processed `vace_context`).

*   **File**: `Wan2GP/wan/modules/model.py`
*   **Class**: `WanAttentionBlock`
    *   **`forward(..., hints=None, context_scale=1.0, ...)` method (lines 397-499)**:
        *   If `self.block_id is not None` (meaning it's a VACE-designated layer) and `hints` are provided (line 417):
            *   `hint = self.vace(hints, x, **kwargs)` (line 423 or 425). Here, `self.vace` is the `VaceWanAttentionBlock` instance that was attached during `adapt_vace_model`.
            *   The returned `hint` from the `VaceWanAttentionBlock` is added to the main feature map `x`: `x.add_(hint, alpha=context_scale)` (line 497).

*   **File**: `Wan2GP/wan/modules/model.py`
*   **Class**: `VaceWanAttentionBlock` (inherits from `WanAttentionBlock`)
    *   **`__init__(...)` (lines 504-522)**:
        *   Initializes `before_proj` and `after_proj` linear layers with zero weights and biases.
    *   **`forward(self, hints, x, **kwargs)` method (lines 526-535)**:
        *   This is where the VACE conditioning is directly applied.
        *   `c = hints[0]` gets the VACE context features (output of `vace_patch_embedding` from `WanModel.forward`).
        *   If it's the first VACE block (`self.block_id == 0`), it applies `self.before_proj` to `c` and then adds the current timestep's feature map `x`: `c += x`.
        *   It then calls `super().forward(c, **kwargs)`. This means the (potentially modified) VACE context `c` is processed through a standard `WanAttentionBlock`'s attention and FFN layers.
        *   The output of this is `c`.
        *   `c_skip = self.after_proj(c)`: The result is passed through another projection.
        *   `hints[0] = c`: The result *before* the `after_proj` is stored back into `hints[0]`. This is interesting, as it means subsequent VACE blocks in the *same timestep* would receive the output of the previous VACE block's main processing path, not the skip connection. (Correction: `hints` is a list passed down. `hints[0]` is modified in place. This `c` will be the input `hints[0]` for the *next* `VaceWanAttentionBlock` if multiple are chained directly without an intermediate normal `WanAttentionBlock`. However, the structure `adapt_vace_model` sets up `target.vace = module`, so each `WanAttentionBlock.forward` calls its own `self.vace` instance with the *original* hint from `WanModel.forward`.)
        *   The actual returned value is `c_skip`, which is then added to the main path in `WanAttentionBlock.forward`.

## Summary of VACE Context Flow

1.  **`WanT2V.generate`**:
    *   `input_frames`, `input_masks`, `input_ref_images` -> `vace_encode_frames` & `vace_encode_masks` (using `WanVAE`) -> `z0`, `m0`.
    *   `vace_latent(z0, m0)` -> `vace_context`.
2.  **`WanModel.forward`**:
    *   `vace_context` -> `self.vace_patch_embedding` -> `c` (processed VACE context).
    *   `c` is put into `hints_list`.
3.  **`WanModel`'s loop over `blocks`**:
    *   For each `block` (a `WanAttentionBlock`): `block(x, ..., hints=hints_list[...])`.
4.  **`WanAttentionBlock.forward`**:
    *   If it's a VACE layer (`self.block_id is not None`):
        *   `hint_output = self.vace(hints, x, ...)` (where `self.vace` is a `VaceWanAttentionBlock`).
        *   `x = x + hint_output * context_scale`.
5.  **`VaceWanAttentionBlock.forward`**:
    *   Receives `hints` (containing processed `vace_context` `c`) and `x` (current features).
    *   `c_modified = c` (or `before_proj(c) + x` if first VACE block).
    *   `c_processed = super().forward(c_modified, ...)` (pass through standard attention block).
    *   `c_skip = self.after_proj(c_processed)`.
    *   Returns `c_skip`.

This detailed flow shows how VACE conditions the generation process by injecting its encoded representation of frames, masks, and reference images into specific layers of the main `WanModel` transformer via specialized attention blocks.

## Interaction with `wgp.py` (Gradio UI)

The main application script, `wgp.py`, handles user inputs from the Gradio web interface and orchestrates the video generation process, including the VACE-specific parts.

### 1. UI Input Collection

*   **File**: `Wan2GP/wgp.py`
*   **Function**: `generate_video_tab` (around lines 4288-4842, VACE specific UI around 4458-4507)
    *   This function defines the Gradio UI elements.
    *   For VACE, it creates:
        *   `video_prompt_type_video_guide`: A dropdown to select the type of video guidance (None, Pose, Depth, Color, VACE general, VACE with Mask). This determines letters like 'P', 'D', 'C', 'V', 'M' in the `video_prompt_type` string.
        *   `video_prompt_type_image_refs`: A dropdown to enable/disable "Inject custom Faces / Objects", adding 'I' to `video_prompt_type`.
        *   `video_guide`: A `gr.Video` component for the control video.
        *   `keep_frames_video_guide`: A `gr.Text` input to specify frames to keep/mask from the `video_guide`.
        *   `image_refs`: A `gr.Gallery` for reference images.
        *   `remove_background_image_ref`: A `gr.Checkbox` for background removal on reference images.
        *   `video_mask`: A `gr.Video` component for an explicit video mask (for inpainting/outpainting).
    *   The choices made in these UI elements are collated into variables that are then passed to the backend processing.

### 2. Input Validation and Task Creation

*   **File**: `Wan2GP/wgp.py`
*   **Function**: `process_prompt_and_add_tasks` (around lines 129-458, VACE specific logic around 301-368)
    *   This function is triggered when a user adds a generation task.
    *   If the selected model is a VACE model (checks if "Vace" is in `model_filename`):
        *   It retrieves `video_prompt_type`, `image_refs`, `video_guide`, `video_mask`, etc., from the current UI state.
        *   It validates that necessary inputs are provided based on the selected `video_prompt_type` (e.g., if 'I' is selected, `image_refs` must exist).
        *   Reference images from the gallery are converted using `convert_image`.
        *   These validated and potentially pre-processed inputs are then used to populate the arguments for `add_video_task`.

### 3. Video Generation Orchestration

*   **File**: `Wan2GP/wgp.py`
*   **Function**: `generate_video` (lines 2648-3276)
    *   This is the main worker function that performs video generation. It receives all parameters, including those for VACE, which were set up by the UI and `process_prompt_and_add_tasks`.
    *   It sets a boolean `vace = "Vace" in model_filename` (line 2805).
    *   **Background Removal for Image References (lines 2818-2824)**: If `image_refs` are present and VACE is active, it calls `wan.utils.utils.resize_and_remove_background` if the corresponding UI checkbox is ticked.
    *   **VACE Input Preparation (lines 2986-3028, within a loop for sliding window processing)**:
        *   If `vace` is true:
            *   It makes copies of `image_refs`, `video_guide`, `video_mask` (as the `prepare_source` method might modify them).
            *   **Control Video Preprocessing**: If `video_prompt_type` indicates 'P' (Pose), 'D' (Depth), or 'G' (Grayscale), it calls `preprocess_video` (defined in `wgp.py` around line 2540) to apply these effects to the `video_guide_copy`.
            *   `parse_keep_frames_video_guide` processes the string specifying which frames to keep from the control video.
            *   **`wan_model.prepare_source` Call**:
                ```python
                src_video, src_mask, src_ref_images = wan_model.prepare_source(
                    [video_guide_copy],
                    [video_mask_copy],
                    [image_refs_copy], 
                    video_length, # Current window's length
                    image_size=image_size, 
                    device="cpu",
                    original_video="O" in video_prompt_type, # "O" for original video / alternate ending
                    keep_frames=keep_frames_parsed,
                    start_frame=guide_start_frame,
                    pre_src_video=[pre_video_guide], # For sliding window continuity
                    fit_into_canvas=(fit_canvas == 1)
                )
                ```
                - `wan_model` here is an instance of `WanT2V` (from `Wan2GP/wan/text2video.py`).
                - The `prepare_source` method (defined in `Wan2GP/wan/text2video.py`, lines 164-250) is crucial. It takes the raw video paths/data and converts them into the actual PyTorch tensors. It handles:
                    - Loading video frames.
                    - Resizing them to the target `image_size`.
                    - Applying the `keep_frames` logic: frames not in `keep_frames_parsed` will have their corresponding `src_mask` set to 1 (indicating inpainting/regeneration), and `src_video` pixels might be zeroed out.
                    - Padding videos/masks to the required `video_length`.
                    - Preparing reference images.
                - The outputs `src_video`, `src_mask`, and `src_ref_images` are the tensors that will be fed into the VAE encoding part of the `WanT2V.generate` method.
    *   **Calling `WanT2V.generate` (lines 3053-3091)**:
        *   The `src_video`, `src_mask`, and `src_ref_images` tensors (prepared above) are passed to `wan_model.generate()` as `input_frames`, `input_masks`, and `input_ref_images` respectively.
        *   This triggers the VACE encoding pipeline within `WanT2V` (using `vace_encode_frames`, `vace_encode_masks`, `vace_latent`) and the subsequent diffusion process detailed in the sections above.

This flow shows how `wgp.py` translates user interactions and media uploads from the Gradio interface into the structured tensor inputs required by the `WanT2V` class for VACE-conditioned video generation.

## Supporting Multiple Wan Encodings (Proposed Extension)

The existing VACE system processes a single set of control inputs (frames, masks, reference images) to create one `vace_context`. To support multiple, separate Wan Encodings for a single generation, the following modifications could be considered:

### 1. Input Handling and Encoding

*   **UI Enhancements (`wgp.py`)**:
    *   The user interface would need to allow users to define multiple "control groups". Each group could consist of its own video guide, reference images, masks, and associated parameters (e.g., type of control like Pose, Depth, general VACE).
*   **Task Processing (`wgp.py`)**:
    *   `process_prompt_and_add_tasks` would need to gather these multiple input groups, validating each one.
*   **Source Preparation (`WanT2V.prepare_source`)**:
    *   This method would need to be called for each control group, or be adapted to process a list of input groups. It would output lists of tensors, e.g., `list_of_src_videos`, `list_of_src_masks`, `list_of_src_ref_images`.
*   **VACE Encoding (`WanT2V.generate`)**:
    *   The method would receive these lists of source tensors.
    *   It would iterate through each set of `src_video_i`, `src_mask_i`, `src_ref_images_i`.
    *   For each set, it would call `self.vace_encode_frames(...)` and `self.vace_encode_masks(...)` to produce an individual `vace_context_i`.
    *   The result would be a list of VACE contexts: `list_of_vace_contexts = [context1, context2, ..., contextN]`.
    *   This list would be passed to the `WanModel`'s forward pass, for example, via `kwargs.update({'vace_contexts': list_of_vace_contexts, ...})`.

### 2. `WanModel` Adaptations

*   **Initialization (`WanModel.__init__`)**:
    *   The model would need to be aware of the number of VACE streams (e.g., via a `num_vace_streams` parameter).
    *   `self.vace_patch_embedding`: This would likely become an `nn.ModuleList` of `nn.Conv3d` layers. Each convolutional layer in this list would correspond to one VACE stream, processing its respective context. These layers would be designed to output features that can be combined (e.g., summed).
*   **Forward Pass (`WanModel.forward`)**:
    *   The method would accept `vace_contexts` (a list of tensors) instead of a single `vace_context`.
    *   It would iterate through the `vace_contexts` list and the `self.vace_patch_embeddings` ModuleList, applying the corresponding patch embedding to each context: `processed_c_i = self.vace_patch_embeddings[i](vace_context_i)`.
    *   This results in a list of processed context tensors: `processed_contexts = [pc1, pc2, ..., pcN]`.
    *   The `hints_list` provided to the model's blocks would be structured so that each VACE-enabled block receives this full list `[pc1, pc2, ..., pcN]`. For example, `hints_list = [processed_contexts for _ in range(len(x_list))]`.

### 3. `WanAttentionBlock` and `VaceWanAttentionBlock` Adaptations

*   **`WanAttentionBlock.forward`**:
    *   When `self.block_id is not None` (it's a VACE layer), the `hints` argument it receives will be the list `[pc1, pc2, ..., pcN]`.
    *   It passes this entire list to its `self.vace` module (the `VaceWanAttentionBlock` instance): `hint_output = self.vace(hints, x, **kwargs)`.
    *   `hint_output` is expected to be a single tensor representing the combined influence from all VACE streams for that block.
    *   This combined `hint_output` is added to the main feature map `x`: `x.add_(hint_output, alpha=context_scale)`. The `context_scale` would apply to the combined hint.
*   **`VaceWanAttentionBlock.forward`**: This block is responsible for combining the multiple VACE streams.
    *   It receives the `hints` list (`[pc1, pc2, ..., pcN]`) and the current feature map `x` from the main U-Net path.
    *   **Context Combination**: The list of processed context tensors `pc_i` are combined into a single tensor. Summation is a straightforward approach: `combined_pc = torch.sum(torch.stack(hints), dim=0)`. This assumes the patch embeddings have produced features in a compatible space.
    *   **Feature Integration**:
        *   Let `c_attention_input = combined_pc`.
        *   If `self.block_id == 0` (typically the first VACE-enabled block in the model hierarchy for a given resolution): The combined context is first projected by `self.before_proj` and then added to the main path's features `x`. `c_attention_input = self.before_proj(combined_pc) + x`.
        *   Else (for subsequent VACE blocks): `c_attention_input` can be just `combined_pc`, which is then passed to `super().forward()`. (Alternatively, `self.before_proj(combined_pc)` could be used consistently if `before_proj` is seen as a general adapter for the combined context before attention, regardless of `block_id`.)
    *   **Core Processing**: The `c_attention_input` (which is the combined VACE information, potentially mixed with `x`) is processed through the standard attention and FFN layers of the parent `WanAttentionBlock` by calling `c_processed = super().forward(c_attention_input, **kwargs)`.
    *   **Final Projection**: The output `c_processed` is passed through `self.after_proj` to get the final skip connection value for this block: `c_skip = self.after_proj(c_processed)`.
    *   **Return Value**: The `c_skip` tensor is returned. This tensor represents the aggregated influence of all active Wan Encodings for the current block.
    *   The original in-place update `hints[0] = c` (where `c` was `c_processed`) would likely be removed or re-evaluated, as its role in a multi-context scenario with independent streams is less clear and the documentation suggests hints are passed fresh from `WanModel.forward` in each step.

By implementing these changes, the Wan2GP framework could leverage multiple, diverse control signals simultaneously, offering more nuanced and complex control over the video generation process. 