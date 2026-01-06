"""
Core generation parameters.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import ParamGroup


@dataclass
class GenerationConfig(ParamGroup):
    """
    Core generation parameters.
    
    These are the fundamental parameters for video generation
    that don't fit into specialized categories (LoRA, VACE, Phase).
    """
    prompt: str = ""
    negative_prompt: Optional[str] = None
    resolution: Optional[str] = None
    video_length: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    flow_shift: Optional[float] = None
    sample_solver: Optional[str] = None
    
    # Image inputs
    image_start: Optional[str] = None
    image_end: Optional[str] = None
    image_refs: Optional[List[str]] = None
    
    # Advanced parameters
    embedded_guidance_scale: Optional[float] = None
    denoising_strength: Optional[float] = None
    
    @classmethod
    def from_params(cls, params: Dict[str, Any], **context) -> 'GenerationConfig':
        """Parse generation config from parameters."""
        return cls(
            prompt=params.get('prompt', ''),
            negative_prompt=params.get('negative_prompt'),
            resolution=params.get('resolution'),
            video_length=params.get('video_length'),
            num_inference_steps=params.get('num_inference_steps') or params.get('steps'),
            guidance_scale=params.get('guidance_scale'),
            seed=params.get('seed'),
            flow_shift=params.get('flow_shift'),
            sample_solver=params.get('sample_solver'),
            image_start=params.get('image_start'),
            image_end=params.get('image_end'),
            image_refs=params.get('image_refs'),
            embedded_guidance_scale=params.get('embedded_guidance_scale'),
            denoising_strength=params.get('denoising_strength') or params.get('denoise_strength'),
        )
    
    def to_wgp_format(self) -> Dict[str, Any]:
        """Convert to WGP-compatible format."""
        result = {}
        
        if self.prompt:
            result['prompt'] = self.prompt
        if self.negative_prompt:
            result['negative_prompt'] = self.negative_prompt
        if self.resolution:
            result['resolution'] = self.resolution
        if self.video_length is not None:
            result['video_length'] = self.video_length
        if self.num_inference_steps is not None:
            result['num_inference_steps'] = self.num_inference_steps
        if self.guidance_scale is not None:
            result['guidance_scale'] = self.guidance_scale
        if self.seed is not None:
            result['seed'] = self.seed
        if self.flow_shift is not None:
            result['flow_shift'] = self.flow_shift
        if self.sample_solver:
            result['sample_solver'] = self.sample_solver
        if self.image_start:
            result['image_start'] = self.image_start
        if self.image_end:
            result['image_end'] = self.image_end
        if self.image_refs:
            result['image_refs'] = self.image_refs
        if self.embedded_guidance_scale is not None:
            result['embedded_guidance_scale'] = self.embedded_guidance_scale
        if self.denoising_strength is not None:
            result['denoising_strength'] = self.denoising_strength
        
        return result
    
    def validate(self) -> list:
        """Validate generation configuration."""
        errors = []
        
        if self.video_length is not None:
            if (self.video_length - 1) % 4 != 0:
                errors.append(f"video_length {self.video_length} is not 4N+1")
        
        return errors
