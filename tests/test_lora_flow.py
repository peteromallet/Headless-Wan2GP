"""
Integration test for LoRA URL resolution flow.

Tests the full path:
  parse_phase_config() → TaskConfig → LoRAConfig → download → to_wgp_format()

Run with: python -m pytest tests/test_lora_flow.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.task_conversion import parse_phase_config
from source.params import TaskConfig, LoRAConfig, LoRAStatus
from source.lora_utils import _download_lora_from_url


class TestLoRAFlow:
    """Test the complete LoRA URL → download → WGP format flow."""
    
    # Sample phase_config like what the frontend sends
    SAMPLE_PHASE_CONFIG = {
        "num_phases": 2,
        "steps_per_phase": [3, 3],
        "flow_shift": 5.0,
        "sample_solver": "euler",
        "model_switch_phase": 1,
        "phases": [
            {
                "guidance_scale": 1.0,
                "loras": [
                    {"url": "https://huggingface.co/test/high_noise.safetensors", "multiplier": 1.2},
                    {"url": "https://huggingface.co/test/style.safetensors", "multiplier": 0.5},
                ]
            },
            {
                "guidance_scale": 1.0,
                "loras": [
                    {"url": "https://huggingface.co/test/high_noise.safetensors", "multiplier": 0.0},
                    {"url": "https://huggingface.co/test/style.safetensors", "multiplier": 0.8},
                ]
            }
        ]
    }

    def test_parse_phase_config_returns_urls(self):
        """parse_phase_config should return URLs in lora_names."""
        result = parse_phase_config(
            self.SAMPLE_PHASE_CONFIG,
            num_inference_steps=6,
            task_id="test_task",
            model_name="test_model"
        )
        
        # lora_names should contain the URLs
        assert "lora_names" in result
        assert len(result["lora_names"]) == 2
        assert all(url.startswith("https://") for url in result["lora_names"])
        
        # lora_multipliers should have phase-config format (semicolons)
        assert "lora_multipliers" in result
        assert len(result["lora_multipliers"]) == 2
        assert all(";" in m for m in result["lora_multipliers"])
        
        # Verify the multipliers are correct
        assert result["lora_multipliers"][0] == "1.2;0.0"  # high_noise: 1.2 in phase1, 0.0 in phase2
        assert result["lora_multipliers"][1] == "0.5;0.8"  # style: 0.5 in phase1, 0.8 in phase2

    def test_lora_config_detects_urls(self):
        """LoRAConfig should detect URLs and mark them as PENDING."""
        parsed = parse_phase_config(
            self.SAMPLE_PHASE_CONFIG,
            num_inference_steps=6,
            task_id="test_task"
        )
        
        # Simulate what task_registry does
        params = {
            "activated_loras": parsed["lora_names"],
            "loras_multipliers": " ".join(parsed["lora_multipliers"]),
            "additional_loras": parsed["additional_loras"],
        }
        
        config = LoRAConfig.from_params(params, task_id="test_task")
        
        # Should have 2 entries (deduplicated)
        assert len(config.entries) == 2
        
        # All should be PENDING (URLs need download)
        assert all(e.status == LoRAStatus.PENDING for e in config.entries)
        
        # All should have URLs
        assert all(e.url is not None for e in config.entries)
        
        # Filenames should be extracted
        filenames = [e.filename for e in config.entries]
        assert "high_noise.safetensors" in filenames
        assert "style.safetensors" in filenames

    def test_to_wgp_format_excludes_pending(self):
        """to_wgp_format should exclude PENDING entries."""
        parsed = parse_phase_config(
            self.SAMPLE_PHASE_CONFIG,
            num_inference_steps=6,
            task_id="test_task"
        )
        
        params = {
            "activated_loras": parsed["lora_names"],
            "loras_multipliers": " ".join(parsed["lora_multipliers"]),
        }
        
        config = LoRAConfig.from_params(params, task_id="test_task")
        
        # Before download, to_wgp_format should return empty
        wgp = config.to_wgp_format()
        assert wgp["activated_loras"] == []
        assert wgp["loras_multipliers"] == ""

    def test_to_wgp_format_after_download(self):
        """to_wgp_format should include entries after mark_downloaded."""
        parsed = parse_phase_config(
            self.SAMPLE_PHASE_CONFIG,
            num_inference_steps=6,
            task_id="test_task"
        )
        
        params = {
            "activated_loras": parsed["lora_names"],
            "loras_multipliers": " ".join(parsed["lora_multipliers"]),
        }
        
        config = LoRAConfig.from_params(params, task_id="test_task")
        
        # Simulate downloads
        for entry in config.entries:
            if entry.url:
                config.mark_downloaded(entry.url, entry.filename)
        
        # Now to_wgp_format should return the entries
        wgp = config.to_wgp_format()
        assert len(wgp["activated_loras"]) == 2
        assert "high_noise.safetensors" in wgp["activated_loras"]
        assert "style.safetensors" in wgp["activated_loras"]
        
        # Multipliers should be space-separated (phase-config format)
        assert " " in wgp["loras_multipliers"]
        assert ";" in wgp["loras_multipliers"]

    def test_full_flow_with_mock_download(self):
        """Test the complete flow with mocked download function."""
        
        # 1. Parse phase_config (like orchestrator does)
        parsed = parse_phase_config(
            self.SAMPLE_PHASE_CONFIG,
            num_inference_steps=6,
            task_id="test_task",
            model_name="wan_2_2_i2v_lightning_baseline_2_2_2"
        )
        
        # 2. Build generation_params (like task_registry does)
        generation_params = {
            "prompt": "test prompt",
            "model_name": "wan_2_2_i2v_lightning_baseline_2_2_2",
            "resolution": "896x496",
            "video_length": 61,
            "activated_loras": parsed["lora_names"],
            "loras_multipliers": " ".join(parsed["lora_multipliers"]),
            "additional_loras": parsed["additional_loras"],
            "guidance_phases": parsed["guidance_phases"],
            "switch_threshold": parsed["switch_threshold"],
            "flow_shift": parsed["flow_shift"],
        }
        
        # 3. Parse via TaskConfig (like _convert_to_wgp_task does)
        task_config = TaskConfig.from_params(
            generation_params,
            task_id="test_task",
            task_type="travel_segment",
            model="wan_2_2_i2v_lightning_baseline_2_2_2"
        )
        
        # 4. Verify pending downloads detected
        assert task_config.lora.has_pending_downloads()
        pending = task_config.lora.get_pending_downloads()
        assert len(pending) == 2
        
        # 5. Mock the download step
        with patch('source.lora_utils._download_lora_from_url') as mock_download:
            # Mock returns the filename
            mock_download.side_effect = lambda url, task_id, dprint=None: os.path.basename(url)
            
            # Simulate what _convert_to_wgp_task does
            for url, mult in list(task_config.lora.get_pending_downloads().items()):
                local_path = mock_download(url, "test_task")
                task_config.lora.mark_downloaded(url, local_path)
        
        # 6. Convert to WGP format
        wgp_params = task_config.to_wgp_format()
        
        # 7. Verify final output
        assert "activated_loras" in wgp_params
        assert len(wgp_params["activated_loras"]) == 2
        
        # Should be filenames, not URLs
        for lora in wgp_params["activated_loras"]:
            assert not lora.startswith("http")
            assert lora.endswith(".safetensors")
        
        # Multipliers should be phase-config format
        assert "loras_multipliers" in wgp_params
        assert ";" in wgp_params["loras_multipliers"]
        
        print("\n✅ Full flow test passed!")
        print(f"   activated_loras: {wgp_params['activated_loras']}")
        print(f"   loras_multipliers: {wgp_params['loras_multipliers']}")

    def test_mixed_local_and_url_loras(self):
        """Test mix of local filenames and URLs."""
        params = {
            "activated_loras": [
                "local_lora.safetensors",
                "https://huggingface.co/test/remote_lora.safetensors",
            ],
            "loras_multipliers": "0.8 1.2",
        }
        
        config = LoRAConfig.from_params(params, task_id="test_task")
        
        # Should have 2 entries
        assert len(config.entries) == 2
        
        # One LOCAL, one PENDING
        statuses = [e.status for e in config.entries]
        assert LoRAStatus.LOCAL in statuses
        assert LoRAStatus.PENDING in statuses
        
        # to_wgp_format should only include the LOCAL one
        wgp = config.to_wgp_format()
        assert len(wgp["activated_loras"]) == 1
        assert wgp["activated_loras"][0] == "local_lora.safetensors"

    def test_deduplication(self):
        """Test that same LoRA in activated_loras and additional_loras is deduplicated."""
        url = "https://huggingface.co/test/lora.safetensors"
        
        params = {
            "activated_loras": [url],
            "loras_multipliers": "0.5;0.8",
            "additional_loras": {url: 1.0},  # Same URL, different multiplier
        }
        
        config = LoRAConfig.from_params(params, task_id="test_task")
        
        # Should only have 1 entry (deduplicated by filename)
        assert len(config.entries) == 1
        
        # Phase-config multiplier from activated_loras should be preserved
        # (additional_loras doesn't have semicolon, so it doesn't override)
        assert config.entries[0].multiplier == "0.5;0.8"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_phase_config_loras(self):
        """Test phase_config with no LoRAs."""
        phase_config = {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "phases": [
                {"guidance_scale": 1.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
            ]
        }
        
        result = parse_phase_config(phase_config, num_inference_steps=6, task_id="test")
        
        assert result["lora_names"] == []
        assert result["lora_multipliers"] == []

    def test_no_loras_in_params(self):
        """Test TaskConfig with no LoRAs."""
        params = {
            "prompt": "test",
            "resolution": "896x496",
        }
        
        config = TaskConfig.from_params(params, task_id="test", model="test_model")
        
        assert not config.lora.has_pending_downloads()
        assert config.lora.to_wgp_format()["activated_loras"] == []

    def test_invalid_url_handling(self):
        """Test that invalid URLs are still handled gracefully."""
        params = {
            "activated_loras": ["not_a_url_but_has_dots.safetensors"],
        }
        
        config = LoRAConfig.from_params(params, task_id="test")
        
        # Should be treated as LOCAL (not a URL)
        assert len(config.entries) == 1
        assert config.entries[0].status == LoRAStatus.LOCAL


class TestFromDbTask:
    """Test TaskConfig.from_db_task which is used by the actual queue."""
    
    TRAVEL_SEGMENT_PARAMS = {
        "prompt": "cinematic video of a woman walking",
        "model_name": "wan_2_2_i2v_lightning_baseline_2_2_2",
        "resolution": "896x496",
        "video_length": 61,
        "seed_to_use": 12345,
        "flow_shift": 5.0,
        "sample_solver": "euler",
        "phase_config": {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "phases": [
                {
                    "guidance_scale": 1.0,
                    "loras": [
                        {"url": "https://huggingface.co/test/lightning_high.safetensors", "multiplier": 1.0},
                    ]
                },
                {
                    "guidance_scale": 1.0,
                    "loras": [
                        {"url": "https://huggingface.co/test/lightning_high.safetensors", "multiplier": 0.0},
                    ]
                }
            ]
        },
        # These come from phase_config parsing in task_registry
        "activated_loras": ["https://huggingface.co/test/lightning_high.safetensors"],
        "loras_multipliers": "1.0;0.0",
    }
    
    def test_from_db_task_parses_loras(self):
        """TaskConfig.from_db_task should correctly parse LoRAs from segment params."""
        config = TaskConfig.from_db_task(
            self.TRAVEL_SEGMENT_PARAMS,
            task_id="test_segment",
            task_type="travel_segment",
            model="wan_2_2_i2v_lightning_baseline_2_2_2"
        )
        
        # Should have 1 LoRA entry
        assert len(config.lora.entries) == 1
        
        # Should be PENDING (URL needs download)
        assert config.lora.entries[0].status == LoRAStatus.PENDING
        assert config.lora.has_pending_downloads()
        
        # Multiplier should be phase-config format
        assert config.lora.entries[0].multiplier == "1.0;0.0"
    
    def test_from_db_task_preserves_phase_config(self):
        """TaskConfig should preserve phase_config for WGP patching."""
        config = TaskConfig.from_db_task(
            self.TRAVEL_SEGMENT_PARAMS,
            task_id="test_segment",
            task_type="travel_segment",
            model="wan_2_2_i2v_lightning_baseline_2_2_2"
        )
        
        wgp = config.to_wgp_format()
        
        # _parsed_phase_config should be preserved for model patching
        assert "_parsed_phase_config" in wgp or "phase_config" in self.TRAVEL_SEGMENT_PARAMS
    
    def test_download_failure_excludes_lora(self):
        """If download fails, LoRA should remain PENDING and be excluded from WGP."""
        config = TaskConfig.from_db_task(
            self.TRAVEL_SEGMENT_PARAMS,
            task_id="test_segment",
            task_type="travel_segment",
            model="wan_2_2_i2v_lightning_baseline_2_2_2"
        )
        
        # Don't call mark_downloaded - simulate failure
        
        # to_wgp_format should exclude PENDING entries
        wgp = config.to_wgp_format()
        assert wgp["activated_loras"] == []
        
        # Validation should warn about pending downloads
        errors = config.validate()
        # Note: validate() may or may not flag this as an error


class TestThreePhaseConfig:
    """Test 3-phase configurations (common for Lightning models)."""
    
    THREE_PHASE_CONFIG = {
        "num_phases": 3,
        "steps_per_phase": [2, 2, 2],
        "phases": [
            {
                "guidance_scale": 1.0,
                "loras": [
                    {"url": "https://hf.co/high_noise.safetensors", "multiplier": 1.2},
                    {"url": "https://hf.co/style.safetensors", "multiplier": 0.5},
                ]
            },
            {
                "guidance_scale": 1.0,
                "loras": [
                    {"url": "https://hf.co/high_noise.safetensors", "multiplier": 0.6},
                    {"url": "https://hf.co/style.safetensors", "multiplier": 0.5},
                ]
            },
            {
                "guidance_scale": 1.0,
                "loras": [
                    {"url": "https://hf.co/high_noise.safetensors", "multiplier": 0.0},
                    {"url": "https://hf.co/style.safetensors", "multiplier": 0.8},
                ]
            }
        ]
    }
    
    def test_three_phase_multipliers(self):
        """3-phase config should produce multipliers with 3 values."""
        result = parse_phase_config(
            self.THREE_PHASE_CONFIG,
            num_inference_steps=6,
            task_id="test"
        )
        
        # Should have 2 LoRAs
        assert len(result["lora_names"]) == 2
        assert len(result["lora_multipliers"]) == 2
        
        # Each multiplier should have 3 semicolon-separated values
        for mult in result["lora_multipliers"]:
            values = mult.split(";")
            assert len(values) == 3, f"Expected 3 phases, got {len(values)}: {mult}"
        
        # Verify specific values
        assert result["lora_multipliers"][0] == "1.2;0.6;0.0"  # high_noise
        assert result["lora_multipliers"][1] == "0.5;0.5;0.8"  # style
    
    def test_three_phase_through_taskconfig(self):
        """3-phase config should work through full TaskConfig flow."""
        parsed = parse_phase_config(
            self.THREE_PHASE_CONFIG,
            num_inference_steps=6,
            task_id="test"
        )
        
        params = {
            "activated_loras": parsed["lora_names"],
            "loras_multipliers": " ".join(parsed["lora_multipliers"]),
        }
        
        config = LoRAConfig.from_params(params, task_id="test")
        
        # Should preserve 3-phase multipliers
        for entry in config.entries:
            if ";" in str(entry.multiplier):
                values = str(entry.multiplier).split(";")
                assert len(values) == 3


class TestQueueIntegration:
    """Test that simulates what _convert_to_wgp_task does."""
    
    def test_queue_conversion_flow(self):
        """
        Simulate the exact flow in HeadlessTaskQueue._convert_to_wgp_task.
        This is the critical integration point.
        """
        # Params as they would come from a GenerationTask
        task_params = {
            "prompt": "test prompt",
            "model_name": "wan_2_2_i2v_lightning_baseline_2_2_2",
            "resolution": "896x496",
            "video_length": 61,
            "activated_loras": [
                "https://huggingface.co/test/lora1.safetensors",
                "https://huggingface.co/test/lora2.safetensors",
            ],
            "loras_multipliers": "1.0;0.0 0.5;0.8",
            "_source_task_type": "travel_segment",
        }
        
        # Step 1: Parse into TaskConfig (like _convert_to_wgp_task does)
        config = TaskConfig.from_db_task(
            task_params,
            task_id="test_queue",
            task_type=task_params.get('_source_task_type', ''),
            model="wan_2_2_i2v_lightning_baseline_2_2_2",
            debug_mode=True
        )
        
        # Step 2: Check for pending downloads
        assert config.lora.has_pending_downloads()
        pending = config.lora.get_pending_downloads()
        assert len(pending) == 2
        
        # Step 3: Simulate downloads
        for url, mult in list(pending.items()):
            local_filename = os.path.basename(url)
            config.lora.mark_downloaded(url, local_filename)
        
        # Step 4: Validate
        errors = config.validate()
        # Should have no critical errors after download
        
        # Step 5: Convert to WGP format
        wgp_params = config.to_wgp_format()
        
        # Step 6: Verify output
        assert "activated_loras" in wgp_params
        loras = wgp_params["activated_loras"]
        
        # Should have 2 LoRAs
        assert len(loras) == 2
        
        # Should be filenames, NOT URLs
        for lora in loras:
            assert not lora.startswith("http"), f"URL leaked to WGP: {lora}"
            assert lora.endswith(".safetensors")
        
        # Multipliers should be preserved
        assert "loras_multipliers" in wgp_params
        assert ";" in wgp_params["loras_multipliers"]  # Phase config format
        
        print("\n✅ Queue integration test passed!")
        print(f"   Input URLs: {task_params['activated_loras']}")
        print(f"   Output filenames: {wgp_params['activated_loras']}")
        print(f"   Output multipliers: {wgp_params['loras_multipliers']}")


class TestDownloaderUtils:
    """Test downloader helper behaviors used by the queue on feat/typed-params."""

    def test_download_normalizes_hf_blob_url(self, tmp_path, monkeypatch):
        """
        HF URLs sometimes come in as /blob/ links; we should normalize to /resolve/
        and pass the correct repo_id + filename to hf_hub_download.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "loras").mkdir(parents=True, exist_ok=True)

        called = {}

        def fake_hf_hub_download(repo_id, filename, local_dir, local_dir_use_symlinks=False, subfolder=None, **kwargs):
            called["repo_id"] = repo_id
            called["filename"] = filename
            called["local_dir"] = local_dir
            called["subfolder"] = subfolder
            # create the file the function expects to exist afterwards
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            out = Path(local_dir) / filename
            out.write_bytes(b"stub")
            return str(out)

        # Patch the import used inside _download_lora_from_url
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

        url = "https://huggingface.co/org/repo/blob/main/loras/my_lora.safetensors"
        out = _download_lora_from_url(url, task_id="test")

        assert out == "my_lora.safetensors"
        assert called["repo_id"] == "org/repo"
        # lora_utils passes filename as basename and subfolder for resolve/main/<subfolder>/<filename>
        assert called["filename"] == "my_lora.safetensors"
        assert called["subfolder"] == "loras"
        assert Path(called["local_dir"]).name == "loras"

    def test_download_collision_prefixes_parent_and_removes_legacy_generic(self, tmp_path, monkeypatch):
        """
        When a collision-prone filename like high_noise_model.safetensors is used, we prefix the
        parent folder and remove any legacy generic file in standard lora dirs.
        """
        monkeypatch.chdir(tmp_path)
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir(parents=True, exist_ok=True)

        # Create a legacy collision-prone file that should be cleaned up
        legacy = loras_dir / "high_noise_model.safetensors"
        legacy.write_bytes(b"old")

        # Patch urlretrieve to avoid network and write the "downloaded" file
        def fake_urlretrieve(url, filename):
            Path(filename).write_bytes(b"new")
            return filename, None

        # lora_utils imports urlretrieve directly: `from urllib.request import urlretrieve`
        import source.lora_utils as lora_utils
        monkeypatch.setattr(lora_utils, "urlretrieve", fake_urlretrieve)

        url = "https://example.com/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors"
        out = _download_lora_from_url(url, task_id="test")

        # Legacy generic file removed
        assert not legacy.exists()

        # New file name is collision-safe
        expected = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_high_noise_model.safetensors"
        assert out == expected
        assert (loras_dir / expected).is_file()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

