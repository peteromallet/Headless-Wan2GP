"""
Phase configuration handling.

IMPORTANT: This wraps the existing parse_phase_config() function rather than
reimplementing its complex timestep/threshold calculation logic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

from .base import ParamGroup

logger = logging.getLogger(__name__)


@dataclass
class PhaseConfig(ParamGroup):
    """
    Phase-based generation configuration.
    
    Wraps the existing parse_phase_config() function to provide typed access
    while preserving battle-tested calculation logic.
    """
    raw_config: Dict[str, Any] = field(default_factory=dict)
    parsed_output: Dict[str, Any] = field(default_factory=dict)
    
    # Key parsed values (for typed access)
    num_phases: int = 0
    total_steps: int = 0
    model_name: Optional[str] = None
    
    # For model patching
    patch_config: Optional[Dict[str, Any]] = None
    patch_model_name: Optional[str] = None
    
    @classmethod
    def from_params(cls, params: Dict[str, Any], **context) -> 'PhaseConfig':
        """
        Parse phase config using the existing parse_phase_config() function.
        
        This wraps rather than reimplements the existing logic.
        """
        phase_config = params.get('phase_config')
        if not phase_config:
            return cls()
        
        task_id = context.get('task_id', '')
        model_name = context.get('model') or params.get('model_name')
        debug_mode = context.get('debug_mode', False)
        
        # Calculate total steps from steps_per_phase
        steps_per_phase = phase_config.get('steps_per_phase', [2, 2, 2])
        total_steps = sum(steps_per_phase)
        
        try:
            # Import and call existing parse_phase_config
            from source.task_conversion import parse_phase_config
            
            parsed = parse_phase_config(
                phase_config=phase_config,
                num_inference_steps=total_steps,
                task_id=task_id,
                model_name=model_name,
                debug_mode=debug_mode
            )
            
            # Extract patch config if present
            patch_config = parsed.pop('_patch_config', None)
            
            return cls(
                raw_config=phase_config,
                parsed_output=parsed,
                num_phases=len(steps_per_phase),
                total_steps=total_steps,
                model_name=model_name,
                patch_config=patch_config,
                patch_model_name=model_name,
            )
            
        except Exception as e:
            logger.error(f"Task {task_id}: Failed to parse phase_config: {e}")
            return cls(raw_config=phase_config)
    
    def is_empty(self) -> bool:
        """Check if this is an empty/unset phase config."""
        return not self.raw_config
    
    def has_patch_config(self) -> bool:
        """Check if this phase config requires model patching."""
        return self.patch_config is not None
    
    def to_wgp_format(self) -> Dict[str, Any]:
        """
        Convert to WGP-compatible format.
        
        Returns the parsed_output minus internal keys.
        """
        if not self.parsed_output:
            return {}
        
        result = {}
        
        # Copy all parsed values except internal ones
        internal_keys = {'_patch_config', '_parsed_phase_config', '_phase_config_model_name'}
        for key, value in self.parsed_output.items():
            if key not in internal_keys and value is not None:
                result[key] = value
        
        return result
    
    def get_lora_info(self) -> Dict[str, Any]:
        """
        Get LoRA-related info from parsed phase config.
        
        Returns dict with lora_names, lora_multipliers, additional_loras.
        """
        return {
            'lora_names': self.parsed_output.get('lora_names', []),
            'lora_multipliers': self.parsed_output.get('lora_multipliers', []),
            'additional_loras': self.parsed_output.get('additional_loras', {}),
        }
    
    def validate(self) -> list:
        """Validate phase configuration."""
        errors = []
        
        if self.raw_config and not self.parsed_output:
            errors.append("Phase config present but failed to parse")
        
        return errors
