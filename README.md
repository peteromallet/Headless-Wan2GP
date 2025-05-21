# WanGP Headless Processing

This document describes the headless processing feature for WanGP, enabling automated video generation by monitoring a `tasks.json` file.

## Overview

The `headless.py` script allows users to run WanGP without the Gradio web interface. It continuously polls a specified JSON file (e.g., `tasks.json`) for video generation tasks. When a new task is found, it processes it using the `wgp.py` engine and saves the output to a designated directory. After processing, the task is removed from the JSON file.

This is useful for batch processing, automated workflows, or running WanGP on servers without a graphical interface.

## How it Works

1.  **Task Definition (`tasks.json`)**:
    Users define a list of video generation tasks in a JSON file. Each task is an object with parameters that would normally be set in the Gradio UI.
    *   `task_id`: A unique identifier for the task.
    *   `prompt`: The text prompt for the video generation.
    *   `model`: The model to use (e.g., "t2v", "vace_14B").
    *   `resolution`: The output video resolution (e.g., "960x544").
    *   `frames`: The number of frames to generate.
    *   `seed`: The random seed for generation.
    *   `output_sub_dir`: The subdirectory name within the main output directory where the generated video will be saved.
    *   `use_causvid_lora`: (Optional) Boolean, if `true`, applies specific settings for the CausVid LoRA, including attempting to download it if not found.
    *   Other parameters corresponding to `wgp.py` settings can also be included (e.g., `video_guide_path`, `video_mask_path`, `steps`, `guidance_scale`, `activated_loras`, `loras_multipliers`, `image_start_paths`, etc.).

    **Example `tasks.json`:**
    ```json
    [
        {
          "task_id": "causvid_lora_example_001",
          "prompt": "A cyberpunk cityscape at night, rain, neon lights, cinematic, highly detailed",
          "model": "t2v",
          "resolution": "960x544",
          "frames": 81,
          "seed": 2024,
          "use_causvid_lora": true,
          "output_sub_dir": "causvid_cyberpunk_city"
        },
        {
          "task_id": "vace_14b_example_001",
          "prompt": "A surreal landscape with floating islands and giant flowers, dreamlike, vibrant colors",
          "model": "vace_14B",
          "resolution": "960x544",
          "frames": 81,
          "seed": 12345,
          "video_prompt_type": "VM",
          "video_guide_path": "path/to/your/video.mp4",
          "video_mask_path": "path/to/your/mask.mp4",
          "output_sub_dir": "vace_surreal_landscape",
          "use_causvid_lora": true
        }
    ]
    ```

2.  **Polling Mechanism**:
    The `headless.py` script periodically checks the `tasks.json` file. The interval is configurable via the `--poll-interval` command-line argument.

3.  **Task Processing**:
    *   When tasks are found, the script takes the first task from the list.
    *   It prepares the state for `wgp.py` based on the task parameters.
    *   If `use_causvid_lora` is true, it will attempt to download the "Wan21_CausVid_14B_T2V_lora_rank32.safetensors" LoRA if it's not already present in the appropriate LoRA directory (determined by `wgp.py`'s logic, often within a `loras/14B/` or similar subdirectory). It also applies specific parameters for CausVid (guidance scale 1.0, flow shift 7.0, and a LoRA multiplier of 0.3 for the CausVid LoRA itself). If the specified model is not a 14B T2V model, a warning will be issued.
    *   It invokes the `wgp_mod.generate_video()` function.
    *   Output videos are saved in a subdirectory (named after `output_sub_dir` or `task_id`) within the directory specified by `--main-output-dir`.

4.  **Task Completion**:
    *   After a task is processed (whether successfully or with an error), it is removed from the `tasks.json` file. The script then picks up the next task in the list during the subsequent poll.
    *   Progress and status messages are printed to the console.

## Usage

To run the headless server:

```bash
python headless.py --tasks-file /path/to/your/tasks.json --main-output-dir /path/to/your/outputs
```

### Command-Line Arguments for `headless.py`:

*   **Server Settings**:
    *   `--tasks-file` (required): Path to the `tasks.json` file to monitor.
    *   `--main-output-dir`: Base directory where outputs for each task will be saved (in subdirectories named by `output_sub_dir` or `task_id`). Defaults to `./headless_outputs`.
    *   `--poll-interval`: How often (in seconds) to check `tasks.json` for new tasks. Defaults to 10 seconds.

*   **WGP Global Config Overrides (Optional - Applied once at server start)**:
    These arguments allow you to override global configurations in `wgp.py` for the duration of the headless server's run.
    *   `--wgp-attention-mode`: Choices: "auto", "sdpa", "sage", "sage2", "flash", "xformers".
    *   `--wgp-compile`: Choices: "", "transformer".
    *   `--wgp-profile`: Integer value.
    *   `--wgp-vae-config`: Integer value.
    *   `--wgp-boost`: Integer value.
    *   `--wgp-transformer-quantization`: Choices: "int8", "bf16".
    *   `--wgp-transformer-dtype-policy`: Choices: "", "fp16", "bf16".
    *   `--wgp-text-encoder-quantization`: Choices: "int8", "bf16".
    *   `--wgp-vae-precision`: Choices: "16", "32".
    *   `--wgp-mixed-precision`: Choices: "0", "1".
    *   `--wgp-preload-policy`: Set `wgp.py`'s preload_model_policy (e.g., 'P,S' or 'P'. Avoid 'U' to keep models loaded longer).

## Key Features in `headless.py`

*   **Gradio Monkey-Patching**: Disables actual Gradio UI elements to allow `wgp.py` to run in a non-UI environment.
*   **Task State Management**: Dynamically builds the necessary state for `wgp.py` from the JSON task parameters.
*   **Output Redirection**: Manages output paths for each task.
*   **LoRA Handling**: Includes logic to download the CausVid LoRA if specified in a task and not present.
*   **Error Handling**: Logs errors and continues to the next task if one fails.
*   **Graceful Shutdown**: Can be stopped with Ctrl+C, attempting to release resources.

This headless mode provides a powerful way to integrate WanGP into automated video generation pipelines.