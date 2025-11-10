"""
Unified LoRA Format Resolution System

This module provides a class-based interface for resolving all LoRA formats
from various sources (task params, model presets, phase configs, feature flags)
into a normalized format ready for WGP generation.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
import subprocess

# Import logging utilities
from source.logging_utils import model_logger


class LoraResolver:
    """
    Unified LoRA resolution class that handles all 7 input formats.
    
    Handles:
    1. WGP Legacy: activated_loras (list) + loras_multipliers (string)
    2. CSV String: "lora1,lora2" from database/API
    3. Internal List: lora_names + lora_multipliers (both lists)
    4. Phase-Config: ["1.0;0;0", "0;0.8;1.0"] for multi-phase
    5. Additional Dict: {"https://url": 1.0} for downloads
    6. Qwen Task: [{"path": "...", "scale": 1.0}]
    7. Model Preset: LoRAs from JSON model definitions
    """
    
    def __init__(self, wan_root: str, task_id: str = "unknown", dprint=None):
        """
        Initialize the LoRA resolver.
        
        Args:
            wan_root: Path to Wan2GP root directory
            task_id: Task ID for logging
            dprint: Optional debug print function
        """
        self.wan_root = Path(wan_root).resolve()
        self.task_id = task_id
        self.dprint = dprint if dprint else model_logger.debug
        self.lora_dirs = self._get_lora_search_paths()
    
    def _get_lora_search_paths(self) -> List[Path]:
        """Define standard LoRA directories for searching."""
        paths = [
            self.wan_root / "loras",
            self.wan_root / "loras_i2v",
            self.wan_root / "loras_hunyuan_i2v",
            self.wan_root / "loras_qwen",
            self.wan_root.parent / "loras",  # Parent repo for dev
            self.wan_root.parent / "loras_qwen",
        ]
        return [p for p in paths if p.is_dir()]
    
    def _log_debug(self, message: str):
        """Log debug message."""
        if self.dprint:
            self.dprint(f"[LORA_RESOLVER] Task {self.task_id}: {message}")
        else:
            model_logger.debug(f"[LORA_RESOLVER] Task {self.task_id}: {message}")
    
    def _log_info(self, message: str):
        """Log info message."""
        model_logger.info(f"[LORA_RESOLVER] Task {self.task_id}: {message}")
    
    def _log_warning(self, message: str):
        """Log warning message."""
        model_logger.warning(f"[LORA_RESOLVER] Task {self.task_id}: {message}")
    
    def _check_lora_exists_locally(self, lora_filename: str) -> Optional[Path]:
        """Check if a LoRA file exists in any of the standard LoRA directories."""
        for lora_dir in self.lora_dirs:
            candidate_path = lora_dir / lora_filename
            if candidate_path.is_file():
                self._log_debug(f"Found local LoRA: {candidate_path}")
                return candidate_path.resolve()
        return None
    
    def _download_lora_from_url(self, lora_url: str, model_type: str) -> Optional[Path]:
        """
        Download a LoRA from a URL using wget.
        
        Args:
            lora_url: URL to download from
            model_type: Model type (determines target directory)
            
        Returns:
            Path to downloaded LoRA file, or None if download failed
        """
        filename = Path(lora_url).name
        
        # Determine target directory based on model type
        if "qwen" in model_type.lower():
            target_dir = self.wan_root / "loras_qwen"
        else:
            target_dir = self.wan_root / "loras"
        
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = target_dir / filename
        
        # Check if already exists
        if local_path.is_file():
            self._log_info(f"LoRA already exists locally: {local_path}")
            return local_path.resolve()
        
        self._log_info(f"Downloading LoRA from {lora_url} to {local_path}")
        try:
            result = subprocess.run(
                ["wget", "-O", str(local_path), lora_url],
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            self._log_info(f"Successfully downloaded {filename}")
            return local_path.resolve()
        except subprocess.CalledProcessError as e:
            self._log_warning(f"wget failed for {lora_url}: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            self._log_warning(f"wget timed out for {lora_url}")
            return None
        except Exception as e:
            self._log_warning(f"Failed to download LoRA {lora_url}: {e}")
            return None
    
    def _resolve_lora_path(self, lora_identifier: str, model_type: str) -> Optional[Path]:
        """
        Resolve a LoRA identifier to an absolute local path.
        
        Handles:
        - Absolute paths
        - Relative paths
        - Filenames (searches in standard directories)
        - URLs (downloads if needed)
        
        Args:
            lora_identifier: Path, filename, or URL to LoRA
            model_type: Model type (for directory selection)
            
        Returns:
            Absolute path to validated LoRA file, or None if not found/failed
        """
        # 1. Check if it's already an absolute path that exists
        p = Path(lora_identifier)
        if p.is_absolute() and p.is_file():
            self._log_debug(f"LoRA is valid absolute path: {p}")
            return p.resolve()
        
        # 2. Check if filename exists in standard directories
        local_path = self._check_lora_exists_locally(p.name)
        if local_path:
            return local_path
        
        # 3. If it's a URL, download it
        if lora_identifier.startswith(("http://", "https://")):
            downloaded_path = self._download_lora_from_url(lora_identifier, model_type)
            if downloaded_path:
                return downloaded_path
        
        # 4. Check if it's relative to wan_root
        relative_to_wan = self.wan_root / p
        if relative_to_wan.is_file():
            self._log_debug(f"Found LoRA relative to wan_root: {relative_to_wan}")
            return relative_to_wan.resolve()
        
        self._log_warning(f"Could not resolve LoRA '{lora_identifier}' to a valid file")
        return None
    
    def _parse_multipliers_string(self, multipliers_str: str) -> List[Union[float, str]]:
        """Parses a WGP-style multipliers string into a list of floats or phase strings."""
        if not multipliers_str:
            return []
        
        # Check for phase-config format (contains semicolons)
        if ";" in multipliers_str:
            # Phase-config format: space-separated strings like "1.0;0 0;1.0"
            # Prefer space as separator, but handle comma if needed
            if " " in multipliers_str and "," not in multipliers_str:
                return [x.strip() for x in multipliers_str.split(" ") if x.strip()]
            else:
                # Mixed or comma-only - try to intelligently split
                sep = " " if " " in multipliers_str else ","
                return [x.strip() for x in multipliers_str.split(sep) if x.strip()]
        else:
            # Regular format: try comma first, then space
            if "," in multipliers_str:
                try:
                    return [float(x.strip()) for x in multipliers_str.split(",") if x.strip()]
                except ValueError:
                    self._log_warning(f"Could not parse multipliers '{multipliers_str}' to floats")
                    return [x.strip() for x in multipliers_str.split(",") if x.strip()]
            else:
                try:
                    return [float(x.strip()) for x in multipliers_str.split() if x.strip()]
                except ValueError:
                    self._log_warning(f"Could not parse multipliers '{multipliers_str}' to floats")
                    return [x.strip() for x in multipliers_str.split() if x.strip()]
    
    def resolve_all_lora_formats(
        self, 
        params: Dict[str, Any], 
        model_type: str, 
        task_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Union[float, str]]]:
        """
        Resolves all incoming LoRA formats into validated absolute paths.
        
        This is the SINGLE SOURCE OF TRUTH for LoRA resolution. It:
        1. Normalizes all 8 input formats
        2. Downloads missing LoRAs from URLs
        3. Validates all LoRAs exist on disk
        4. Returns ABSOLUTE paths that are guaranteed to exist

        Args:
            params: The generation parameters dictionary (will be modified in place).
            model_type: The model type being used (e.g., "qwen_image_edit_20B").
            task_params: The original task parameters from the database (for Qwen-specific 'loras' key).

        Returns:
            Tuple of (absolute_paths, multipliers) where:
            - absolute_paths: List of absolute file paths that exist on disk
            - multipliers: List of float or phase-config strings (e.g., "1.0;0.5;0.3")
        """
        self._log_info(f"Starting comprehensive LoRA resolution for model '{model_type}'")
        
        # Phase 1: Collect all LoRA identifiers from all sources
        collected_loras: List[Tuple[str, Union[float, str]]] = []  # (identifier, multiplier)
        
        # --- 1. Process 'lora_names' and 'lora_multipliers' (lists) ---
        initial_lora_names = params.pop("lora_names", [])
        initial_lora_multipliers = params.pop("lora_multipliers", [])

        if initial_lora_names:
            self._log_debug(f"Collecting {len(initial_lora_names)} from lora_names")
            for i, name in enumerate(initial_lora_names):
                multiplier = initial_lora_multipliers[i] if i < len(initial_lora_multipliers) else 1.0
                collected_loras.append((name, multiplier))

        # --- 2. Process 'activated_loras' and 'loras_multipliers' (WGP format) ---
        activated_loras_raw = params.pop("activated_loras", None)
        loras_multipliers_str_raw = params.pop("loras_multipliers", None)

        if activated_loras_raw:
            if isinstance(activated_loras_raw, str):
                activated_loras_list = [l.strip() for l in activated_loras_raw.split(",") if l.strip()]
            elif isinstance(activated_loras_raw, list):
                activated_loras_list = activated_loras_raw
            else:
                activated_loras_list = []
            
            multipliers_list = self._parse_multipliers_string(loras_multipliers_str_raw) if loras_multipliers_str_raw else []

            self._log_debug(f"Collecting {len(activated_loras_list)} from activated_loras")
            for i, name in enumerate(activated_loras_list):
                multiplier = multipliers_list[i] if i < len(multipliers_list) else 1.0
                collected_loras.append((name, multiplier))

        # --- 3. Process 'additional_loras' (dict {url: multiplier}) ---
        additional_loras_dict = params.pop("additional_loras", {})
        if additional_loras_dict and isinstance(additional_loras_dict, dict):
            self._log_debug(f"Collecting {len(additional_loras_dict)} from additional_loras dict")
            for lora_identifier, multiplier in additional_loras_dict.items():
                collected_loras.append((lora_identifier, multiplier))

        # --- 4. Process 'loras' (list of dicts) from task_params (Qwen-specific) ---
        if task_params and "loras" in task_params:
            qwen_loras_list = task_params["loras"]
            if isinstance(qwen_loras_list, list):
                self._log_debug(f"Collecting {len(qwen_loras_list)} Qwen-specific LoRAs")
                for lora_entry in qwen_loras_list:
                    if isinstance(lora_entry, dict):
                        lora_path = lora_entry.get("path") or lora_entry.get("url")
                        lora_scale = lora_entry.get("scale") or lora_entry.get("strength", 1.0)
                        if lora_path:
                            collected_loras.append((lora_path, float(lora_scale)))

        # Phase 2: Resolve all identifiers to validated absolute paths
        validated_paths: List[str] = []
        validated_multipliers: List[Union[float, str]] = []
        seen_paths: set = set()  # Deduplicate by absolute path
        
        self._log_info(f"Resolving {len(collected_loras)} collected LoRAs to absolute paths...")
        
        for identifier, multiplier in collected_loras:
            # Skip empty identifiers (phase-config placeholders)
            if not identifier or not str(identifier).strip():
                self._log_debug("Skipping empty LoRA identifier")
                continue
            
            # Resolve to absolute path (downloads if URL, searches directories, etc.)
            resolved_path = self._resolve_lora_path(identifier, model_type)
            
            if resolved_path and resolved_path.exists():
                abs_path_str = str(resolved_path.absolute())
                
                # Deduplicate by absolute path
                if abs_path_str not in seen_paths:
                    validated_paths.append(abs_path_str)
                    validated_multipliers.append(multiplier)
                    seen_paths.add(abs_path_str)
                    self._log_debug(f"✅ Validated: {Path(abs_path_str).name} → {abs_path_str}")
                else:
                    self._log_debug(f"⏭️  Skipping duplicate: {abs_path_str}")
            else:
                self._log_warning(f"❌ Failed to resolve LoRA '{identifier}' - skipping")

        self._log_info(f"✅ LoRA resolution complete: {len(validated_paths)} validated LoRAs ready")
        if validated_paths:
            self._log_debug(f"Final paths: {[Path(p).name for p in validated_paths]}")
            self._log_debug(f"Final multipliers: {validated_multipliers}")

        return validated_paths, validated_multipliers
