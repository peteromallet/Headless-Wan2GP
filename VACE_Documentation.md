# VACE (Video Auto-Completion Enhancement) Technical Documentation

## Overview
VACE is a video generation system that allows controlled video synthesis using various types of guidance videos. It enables precise control over video generation through different preprocessing methods applied to input control videos.

## Core Components

### 1. Video Input Processing
VACE accepts video input through the `video_guide` parameter, which serves as the primary control mechanism for video generation.

**Key Functions:**
- `get_resampled_video()`: Resamples input videos to target frame rate and length
- `preprocess_video_with_mask()`: Main preprocessing function that applies control transformations

### 2. Video Prompt Types
VACE uses a letter-based encoding system (`video_prompt_type`) to specify how the control video should be processed:

#### Primary Control Types (Single Letter Codes):
- **`P`** - **Pose**: Extract human pose/motion information using DWPose
- **`D`** - **Depth**: Extract depth information using Depth Anything V2
- **`S`** - **Scribble/Shapes**: Extract edge/shape information
- **`L`** - **Flow**: Extract optical flow information using RAFT
- **`C`** - **Gray**: Convert to grayscale for color guidance
- **`M`** - **Inpaint**: Use for inpainting/masking operations
- **`U`** - **Identity**: Keep original video unchanged
- **`V`** - **Raw**: Use video in raw format without preprocessing

#### Advanced Combinations:
- **Dual Processing**: Can combine two types (e.g., `"PDV"` for Pose + Depth)
- **Masking Modifiers**: 
  - `"A"` - Apply masking operations
  - `"N"` - Negate mask
  - `"G"` - Apply denoising strength control

#### Outside Mask Processing:
- **`Y`** - Apply depth processing outside mask areas
- **`W`** - Apply scribble processing outside mask areas  
- **`X`** - Apply inpainting outside mask areas
- **`Z`** - Apply flow processing outside mask areas

### 3. Preprocessing Pipeline

#### Step 1: Video Analysis
```python
# Input video is resampled to target specifications
video = get_resampled_video(video_guide, start_frame, max_frames, target_fps)
```

#### Step 2: Control Type Selection
The system parses the `video_prompt_type` to determine preprocessing:
```python
process_map_video_guide = {
    "P": "pose",      # Human pose extraction
    "D": "depth",     # Depth map generation  
    "S": "scribble",  # Edge/shape detection
    "L": "flow",      # Optical flow computation
    "C": "gray",      # Grayscale conversion
    "M": "inpaint",   # Mask generation
    "U": "identity"   # No processing
}
```

#### Step 3: Preprocessing Execution
Each control type uses specialized neural networks:

- **Pose (`P`)**: DWPose with YOLO detection + pose estimation
- **Depth (`D`)**: Depth Anything V2 (ViTL or ViTB variants)
- **Scribble (`S`)**: Edge detection with pretrained models
- **Flow (`L`)**: RAFT optical flow estimation
- **Gray (`C`)**: Simple color space conversion
- **Inpaint (`M`)**: Mask generation for inpainting

### 4. Strength and Control Parameters

#### Control Strengths:
- **`control_net_weight`**: Primary control strength (0.0-2.0, default 1.0)
- **`control_net_weight2`**: Secondary control strength for dual processing
- **`denoising_strength`**: Controls how much of the original structure to preserve (0.0-1.0)

#### Dual Processing:
When using dual control types (e.g., `"PDV"`):
- Primary strength = `control_net_weight / 2`
- Secondary strength = `control_net_weight2 / 2`

### 5. Advanced Features

#### Frame Alignment:
- **`keep_frames_video_guide`**: Specify which frames to preserve from control video
- Supports absolute frame numbers and relative positioning

#### Outpainting:
- **`video_guide_outpainting`**: Extend video canvas in specified directions
- Format: `[top%, bottom%, left%, right%]`

#### Mask Integration:
- **`video_mask`**: Optional mask video for selective control
- **`mask_expand`**: Expand mask boundaries by specified pixels

### 6. Generation Process

#### Model Input Preparation:
1. **Preprocessing**: Apply selected control extraction methods
2. **Tensor Conversion**: Convert processed frames to PyTorch tensors
3. **Batching**: Organize frames for model input
4. **Context Scaling**: Apply strength modifiers

#### Model Generation Call:
```python
samples = wan_model.generate(
    input_prompt=prompt,
    input_frames=src_video,           # Processed control frames
    input_masks=src_mask,             # Optional mask frames
    denoising_strength=denoising_strength,
    context_scale=context_scale,      # Control strengths
    # ... other parameters
)
```

### 7. Quality Control Features

#### Sliding Window Support:
- Process long videos in overlapping segments
- **`sliding_window_size`**: Segment length
- **`sliding_window_overlap`**: Frame overlap between segments
- **`sliding_window_discard_last_frames`**: Remove final frames for seamless stitching

#### Color Correction:
- **`sliding_window_color_correction_strength`**: Maintain color consistency across segments

### 8. Output Processing

#### Frame Assembly:
1. **Decode**: Convert latent representations to video frames
2. **Upsampling**: Optional temporal/spatial upsampling
3. **Post-processing**: Film grain, color correction
4. **Audio**: Optional MMAudio integration

## Usage Examples

### Basic Pose Control:
```python
generate_vace(
    prompt="A person dancing in a garden",
    video_guide="control_dance.mp4",
    video_prompt_type="PV",  # Pose control
    control_net_weight=1.0
)
```

### Dual Control (Pose + Depth):
```python
generate_vace(
    prompt="Character walking through scene",
    video_guide="reference.mp4", 
    video_prompt_type="PDV",  # Pose + Depth
    control_net_weight=0.8,
    control_net_weight2=0.6
)
```

### Inpainting with Mask:
```python
generate_vace(
    prompt="New background scene",
    video_guide="original.mp4",
    video_mask="mask.mp4",
    video_prompt_type="VM",  # Video + Mask
    control_net_weight=1.2
)
```

### Travel Between Images (Optimized):
```python
# Travel segments use intelligent masking for seamless transitions
generate_vace(
    prompt="Journey through magical forest",
    video_guide="interpolated_guide.mp4",  # Created from start/end images
    video_mask="frame_mask.mp4",           # Masks overlapping frames
    video_prompt_type="VM",                # Video guide + frame masking
    image_refs=["start.jpg", "end.jpg"],   # Key frame references
    control_net_weight=1.0
)
```

## Travel Between Images Integration

### Intelligent Frame Masking
Travel segments use sophisticated frame masking to create seamless video transitions:

- **Overlap Control**: Masks frames reused from previous segments
- **Key Frame Preservation**: Preserves start/end frames for smooth transitions
- **Active Frame Generation**: Only generates new content for transition frames

### Automatic Video Guide Creation
- Creates interpolated motion guides between keyframe images
- Supports pose-based interpolation for human movement
- Handles depth-based guides for camera movements

### Processing Flow:
1. **Guide Generation**: Create control video from start/end images
2. **Mask Creation**: Generate frame mask for active/inactive regions
3. **VACE Processing**: Apply `VM` (Video + Mask) control
4. **Segment Stitching**: Blend overlapping frames for seamless result

### Configuration Options:
- `vace_preprocessing`: Control type ("M" for mask-only, "P" for pose, etc.)
- `mask_active_frames`: Enable/disable intelligent frame masking
- `frame_overlap_*`: Control overlap between segments

## Technical Notes

- **Memory Management**: VACE automatically handles model loading/unloading
- **Device Optimization**: Utilizes GPU acceleration when available
- **Format Support**: Accepts standard video formats (MP4, AVI, MOV, etc.)
- **Resolution Handling**: Automatically resizes to model-compatible dimensions
- **Batch Processing**: Supports multiple video segments for long sequences
- **Travel Optimization**: Specialized handling for multi-segment journeys

## Performance Considerations

- **Preprocessing Time**: Varies by control type (Pose > Depth > Flow > Gray)
- **Memory Usage**: Dual processing requires additional VRAM
- **Quality vs Speed**: Higher control weights increase processing time but improve adherence
- **Optimal Settings**: `control_net_weight` 0.8-1.2 for most use cases
- **Travel Segments**: Frame masking reduces processing time by 30-50%