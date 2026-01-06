"""
VACE (Video-to-Video Control) configuration.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import ParamGroup


@dataclass
class VACEConfig(ParamGroup):
    """
    Video control parameters for VACE-style generation.
    
    Handles:
    - Guide video (structure/motion reference)
    - Mask video (inpainting regions)
    - Control weights
    """
    guide_path: Optional[str] = None
    mask_path: Optional[str] = None
    prompt_type: Optional[str] = None
    control_weight: Optional[float] = None
    control_weight2: Optional[float] = None
    
    @classmethod
    def from_params(cls, params: Dict[str, Any], **context) -> 'VACEConfig':
        """Parse VACE config from parameters."""
        return cls(
            guide_path=params.get('video_guide'),
            mask_path=params.get('video_mask'),
            prompt_type=params.get('video_prompt_type'),
            control_weight=params.get('control_net_weight'),
            control_weight2=params.get('control_net_weight2'),
        )
    
    def to_wgp_format(self) -> Dict[str, Any]:
        """Convert to WGP-compatible format."""
        result = {}
        
        if self.guide_path:
            result['video_guide'] = self.guide_path
        if self.mask_path:
            result['video_mask'] = self.mask_path
        if self.prompt_type:
            result['video_prompt_type'] = self.prompt_type
        if self.control_weight is not None:
            result['control_net_weight'] = self.control_weight
        if self.control_weight2 is not None:
            result['control_net_weight2'] = self.control_weight2
        
        return result
    
    def validate(self) -> list:
        """Validate VACE configuration."""
        errors = []
        # Could add path validation here if needed
        return errors
