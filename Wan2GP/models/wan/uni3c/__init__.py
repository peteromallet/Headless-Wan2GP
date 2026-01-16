"""
Uni3C ControlNet integration for Wan2GP.

Provides video-to-video control using structure guidance from a reference video.
Ported from Kijai's ComfyUI-WanVideoWrapper implementation.
"""

from .controlnet import WanControlNet
from .load import (
    load_uni3c_checkpoint,
    load_uni3c_controlnet,
    download_uni3c_checkpoint_if_missing,
    get_uni3c_checkpoint_path,
    infer_config_from_checkpoint,
    UNI3C_REPO_ID,
    UNI3C_CHECKPOINT_FILENAME,
)

__all__ = [
    "WanControlNet",
    "load_uni3c_checkpoint",
    "load_uni3c_controlnet",
    "download_uni3c_checkpoint_if_missing",
    "get_uni3c_checkpoint_path",
    "infer_config_from_checkpoint",
    "UNI3C_REPO_ID",
    "UNI3C_CHECKPOINT_FILENAME",
]

