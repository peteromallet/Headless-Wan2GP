"""
Phase-Based LoRA Multiplier Utilities

Shared utilities for parsing and converting phase-based LoRA multipliers.
Used by both Qwen hires fix and any future multi-phase implementations.

Format: "pass1_strength;pass2_strength" or "phase1;phase2;phase3"
Example: "1.0;0.5" means strength 1.0 in pass 1, 0.5 in pass 2
"""

from typing import List, Tuple, Optional


# Lightning/accelerator LoRA patterns for auto-detection
LIGHTNING_PATTERNS = [
    "lightning",
    "distill",
    "accelerator",
    "turbo",
    "fast",
    "speed"
]


def is_lightning_lora(lora_name: str) -> bool:
    """
    Check if a LoRA is a Lightning/accelerator type based on filename patterns.

    Lightning LoRAs are optimized for fast generation and typically should NOT
    be used in refinement/hires passes.

    Args:
        lora_name: LoRA filename or path

    Returns:
        True if LoRA matches Lightning patterns

    Examples:
        >>> is_lightning_lora("Qwen-Image-Edit-Lightning-8steps.safetensors")
        True
        >>> is_lightning_lora("style_transfer.safetensors")
        False
    """
    if not lora_name:
        return False

    lora_lower = str(lora_name).lower()
    return any(pattern in lora_lower for pattern in LIGHTNING_PATTERNS)


def parse_phase_multiplier(
    multiplier_str: str,
    num_phases: int = 2,
    lora_name: Optional[str] = None
) -> Tuple[List[float], bool]:
    """
    Parse a phase-based multiplier string into a list of float values.

    Supports:
    - Simple format: "1.0" → [1.0, 1.0] (same for all phases)
    - Phase format: "1.0;0.5" → [1.0, 0.5]
    - Partial: "1.0;" → [1.0, 0.0] (fills missing with 0)

    Args:
        multiplier_str: Multiplier string to parse
        num_phases: Expected number of phases (default: 2)
        lora_name: Optional LoRA name for better error messages

    Returns:
        Tuple of (multiplier_list, is_valid)
        - multiplier_list: List of float values, one per phase
        - is_valid: True if parsing succeeded, False if fallback was used

    Raises:
        ValueError: If multiplier_str is completely invalid

    Examples:
        >>> parse_phase_multiplier("1.0", 2)
        ([1.0, 1.0], True)

        >>> parse_phase_multiplier("1.0;0.5", 2)
        ([1.0, 0.5], True)

        >>> parse_phase_multiplier("1.0;0.5;0.3", 3)
        ([1.0, 0.5, 0.3], True)

        >>> parse_phase_multiplier("1.0;", 2)  # Missing value
        ([1.0, 0.0], True)
    """
    if not multiplier_str:
        # Default to 1.0 for all phases
        return ([1.0] * num_phases, True)

    mult_str = str(multiplier_str).strip()

    # Check if this is phase-based format (contains semicolon)
    if ";" not in mult_str:
        # Simple format - same value for all phases
        try:
            value = float(mult_str)
            return ([value] * num_phases, True)
        except (ValueError, TypeError) as e:
            lora_info = f" for LoRA '{lora_name}'" if lora_name else ""
            raise ValueError(f"Invalid multiplier '{mult_str}'{lora_info}: {e}")

    # Phase-based format - parse semicolon-separated values
    parts = mult_str.split(";")
    multipliers = []

    for i, part in enumerate(parts[:num_phases]):  # Only take first num_phases values
        part = part.strip()
        if not part:
            # Empty value, use 0.0
            multipliers.append(0.0)
        else:
            try:
                multipliers.append(float(part))
            except (ValueError, TypeError) as e:
                lora_info = f" for LoRA '{lora_name}'" if lora_name else ""
                raise ValueError(
                    f"Invalid phase {i+1} multiplier '{part}' in '{mult_str}'{lora_info}: {e}"
                )

    # Fill missing phases with 0.0
    while len(multipliers) < num_phases:
        multipliers.append(0.0)

    return (multipliers, True)


def convert_to_phase_format(
    multiplier: str,
    lora_name: str,
    num_phases: int = 2,
    auto_detect_lightning: bool = True
) -> str:
    """
    Convert a simple multiplier to phase-based format.

    Logic:
    - If already in phase format ("X;Y"), return as-is
    - If Lightning LoRA and auto_detect_lightning=True, return "X;0;...;0"
    - Otherwise, return "X;X;...;X" (same for all phases)

    Args:
        multiplier: Multiplier value (string or number)
        lora_name: LoRA filename for Lightning detection
        num_phases: Number of phases (default: 2)
        auto_detect_lightning: Auto-disable Lightning LoRAs in later phases

    Returns:
        Phase-based multiplier string (e.g., "1.0;0.5")

    Examples:
        >>> convert_to_phase_format("1.0", "style.safetensors", 2)
        "1.0;1.0"

        >>> convert_to_phase_format("1.0", "Lightning-8steps.safetensors", 2)
        "1.0;0"

        >>> convert_to_phase_format("1.0;0.5", "any.safetensors", 2)
        "1.0;0.5"  # Already in phase format, return as-is
    """
    mult_str = str(multiplier).strip()

    # Already in phase format - return as-is
    if ";" in mult_str:
        return mult_str

    # Check if Lightning LoRA
    is_lightning = auto_detect_lightning and is_lightning_lora(lora_name)

    if is_lightning:
        # Lightning: Only active in first phase
        phases = [mult_str] + ["0"] * (num_phases - 1)
    else:
        # Standard: Same strength all phases
        phases = [mult_str] * num_phases

    return ";".join(phases)


def format_phase_multipliers(
    lora_names: List[str],
    multipliers: List[str],
    num_phases: int = 2,
    auto_detect_lightning: bool = True
) -> List[str]:
    """
    Convert a list of multipliers to phase-based format.

    High-level helper that processes entire LoRA lists.

    Args:
        lora_names: List of LoRA filenames
        multipliers: List of multiplier values (strings or numbers)
        num_phases: Number of phases (default: 2)
        auto_detect_lightning: Auto-disable Lightning LoRAs in later phases

    Returns:
        List of phase-based multiplier strings

    Example:
        >>> format_phase_multipliers(
        ...     ["Lightning.safetensors", "style.safetensors"],
        ...     ["1.0", "1.1"],
        ...     num_phases=2
        ... )
        ["1.0;0", "1.1;1.1"]
    """
    result = []

    for i, mult in enumerate(multipliers):
        lora_name = lora_names[i] if i < len(lora_names) else ""
        converted = convert_to_phase_format(
            mult,
            lora_name,
            num_phases=num_phases,
            auto_detect_lightning=auto_detect_lightning
        )
        result.append(converted)

    return result


def extract_phase_values(
    multipliers: List[str],
    phase_index: int,
    num_phases: int = 2
) -> List[str]:
    """
    Extract single-phase values from phase-based multipliers.

    Useful for sending to systems that don't understand phase format.

    Args:
        multipliers: List of phase-based multiplier strings (e.g., ["1.0;0", "1.1;1.1"])
        phase_index: Which phase to extract (0-indexed)
        num_phases: Total number of phases

    Returns:
        List of simple multiplier strings for the specified phase

    Example:
        >>> extract_phase_values(["1.0;0", "1.1;1.2"], phase_index=0)
        ["1.0", "1.1"]

        >>> extract_phase_values(["1.0;0", "1.1;1.2"], phase_index=1)
        ["0", "1.2"]
    """
    result = []

    for mult_str in multipliers:
        try:
            phase_values, _ = parse_phase_multiplier(mult_str, num_phases)
            if phase_index < len(phase_values):
                result.append(str(phase_values[phase_index]))
            else:
                result.append("0")
        except Exception:
            # Fallback for malformed input
            result.append("1.0" if phase_index == 0 else "0")

    return result


def get_phase_loras(
    lora_names: List[str],
    multipliers: List[str],
    phase_index: int,
    num_phases: int = 2
) -> Tuple[List[str], List[str]]:
    """
    Filter LoRAs for a specific phase based on multipliers.

    Returns only LoRAs that have non-zero multipliers for the given phase.

    Args:
        lora_names: List of LoRA filenames/paths
        multipliers: List of phase-based multiplier strings
        phase_index: Which phase to filter for (0-indexed)
        num_phases: Total number of phases

    Returns:
        Tuple of (filtered_lora_names, filtered_multipliers)
        Only includes LoRAs with non-zero multipliers for this phase

    Example:
        >>> get_phase_loras(
        ...     ["lightning.safetensors", "style.safetensors", "detail.safetensors"],
        ...     ["1.0;0", "1.1;1.2", "0;0.8"],
        ...     phase_index=1,  # Pass 2
        ...     num_phases=2
        ... )
        (["style.safetensors", "detail.safetensors"], ["1.2", "0.8"])
    """
    if not lora_names:
        return ([], [])

    # Ensure multipliers list matches lora_names length
    if isinstance(multipliers, str):
        multipliers = [m.strip() for m in multipliers.replace(",", " ").split() if m.strip()]

    # Pad with defaults if needed
    while len(multipliers) < len(lora_names):
        multipliers.append("1.0")

    filtered_loras = []
    filtered_mults = []

    for i, lora_name in enumerate(lora_names):
        mult_str = multipliers[i] if i < len(multipliers) else "1.0"

        try:
            phase_values, _ = parse_phase_multiplier(mult_str, num_phases, lora_name)

            if phase_index < len(phase_values):
                phase_mult = phase_values[phase_index]

                # Only include if non-zero
                if phase_mult != 0:
                    filtered_loras.append(lora_name)
                    filtered_mults.append(str(phase_mult))
        except ValueError as e:
            # Skip malformed multipliers with warning
            print(f"⚠️ Skipping LoRA '{lora_name}' in phase {phase_index+1}: {e}")
            continue

    return (filtered_loras, filtered_mults)
