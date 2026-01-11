"""
Phase 1 Gate Test: Uni3C ControlNet Forward Pass Verification

This test verifies that:
1. The checkpoint can be loaded (or model can be created with random weights)
2. The forward pass produces correct output shapes
3. All 20 block outputs have the expected dimension (5120)

Run with: python test_uni3c_controlnet.py
"""

import sys
import os

# We need to import the uni3c module without triggering Wan2GP's CUDA-dependent imports.
# The solution is to temporarily create a separate package structure for testing.

def setup_isolated_import():
    """Set up imports to avoid CUDA-dependent parent package imports."""
    # Get absolute path to uni3c module
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uni3c_dir = os.path.join(base_dir, "Wan2GP", "models", "wan", "uni3c")
    
    # Add parent of uni3c to path so we can import as standalone
    parent_dir = os.path.dirname(uni3c_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    return uni3c_dir


def test_controlnet_forward_pass():
    """Test forward pass with dummy data (no checkpoint required)."""
    import torch
    
    print("\n" + "=" * 60)
    print("Phase 1 Gate: Uni3C ControlNet Forward Pass Test")
    print("=" * 60 + "\n")
    
    # Import the module directly by loading its source
    setup_isolated_import()
    
    # Use importlib to load the module directly
    import importlib.util
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load controlnet module
    controlnet_path = os.path.join(base_dir, "Wan2GP", "models", "wan", "uni3c", "controlnet.py")
    spec = importlib.util.spec_from_file_location("controlnet", controlnet_path)
    controlnet_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controlnet_module)
    
    WanControlNet = controlnet_module.WanControlNet
    
    # Create model with expected config
    config = {
        "in_channels": 20,  # Uni3C expects 20 channels (may need padding from 16)
        "conv_out_dim": 5120,
        "time_embed_dim": 5120,
        "dim": 1024,
        "ffn_dim": 8192,
        "num_heads": 16,
        "num_layers": 20,
    }
    
    print(f"Creating WanControlNet with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    model = WanControlNet(**config)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created: {total_params:,} params ({trainable_params:,} trainable)")
    
    # Test dimensions
    batch_size = 1
    channels = config["in_channels"]
    frames = 13  # Typical for short video
    height, width = 60, 80  # Latent dimensions for 480x640
    
    print(f"\nTest input shape: [{batch_size}, {channels}, {frames}, {height}, {width}]")
    
    # Create dummy inputs
    render_latent = torch.randn(batch_size, channels, frames, height, width)
    temb = torch.randn(batch_size, 5120)  # Pre-projected time embedding
    
    print("Running forward pass...")
    
    with torch.no_grad():
        controlnet_states = model(
            render_latent=render_latent,
            render_mask=None,
            camera_embedding=None,
            temb=temb,
        )
    
    print(f"\nOutput: {len(controlnet_states)} states returned")
    
    # Verify outputs
    assert len(controlnet_states) == 20, f"Expected 20 states, got {len(controlnet_states)}"
    
    # Calculate expected sequence length
    # After patch embedding with stride (1, 2, 2): F * (H/2) * (W/2)
    expected_seq_len = frames * (height // 2) * (width // 2)
    print(f"Expected sequence length: {expected_seq_len}")
    
    all_shapes_correct = True
    for i, state in enumerate(controlnet_states):
        expected_shape = (batch_size, expected_seq_len, 5120)
        actual_shape = tuple(state.shape)
        is_correct = actual_shape == expected_shape
        
        if not is_correct:
            all_shapes_correct = False
            print(f"  ❌ State {i}: {actual_shape} (expected {expected_shape})")
        elif i == 0 or i == 19:  # Print first and last
            print(f"  ✓ State {i}: {actual_shape}")
    
    if len(controlnet_states) > 2:
        print(f"  ... (states 1-18 also verified)")
    
    if all_shapes_correct:
        print("\n" + "=" * 60)
        print("✅ Phase 1 Gate PASSED!")
        print("=" * 60)
        print("\nAll 20 controlnet states have correct shape.")
        print("Ready to proceed to Phase 2: Guide Video → Latents")
        return True
    else:
        print("\n" + "=" * 60)
        print("❌ Phase 1 Gate FAILED!")
        print("=" * 60)
        return False


def test_checkpoint_loading():
    """Test loading actual checkpoint (requires download)."""
    import torch
    import importlib.util
    
    print("\n" + "=" * 60)
    print("Phase 1 Optional: Checkpoint Loading Test")
    print("=" * 60 + "\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load modules directly
    load_path = os.path.join(base_dir, "Wan2GP", "models", "wan", "uni3c", "load.py")
    spec = importlib.util.spec_from_file_location("load", load_path)
    load_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(load_module)
    
    controlnet_path = os.path.join(base_dir, "Wan2GP", "models", "wan", "uni3c", "controlnet.py")
    spec = importlib.util.spec_from_file_location("controlnet", controlnet_path)
    controlnet_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controlnet_module)
    
    load_uni3c_checkpoint = load_module.load_uni3c_checkpoint
    WanControlNet = controlnet_module.WanControlNet
    
    # Use ckpts dir relative to Wan2GP
    ckpts_dir = os.path.join(base_dir, "Wan2GP", "ckpts")
    
    try:
        # Try to load checkpoint
        state_dict, config = load_uni3c_checkpoint(ckpts_dir=ckpts_dir)
        
        print(f"\nCreating model from checkpoint config:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        model = WanControlNet(**config)
        
        # Load weights
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"\n⚠️ Missing keys ({len(missing)}):")
            for k in missing[:5]:
                print(f"  - {k}")
            if len(missing) > 5:
                print(f"  ... and {len(missing) - 5} more")
        
        if unexpected:
            print(f"\n⚠️ Unexpected keys ({len(unexpected)}):")
            for k in unexpected[:5]:
                print(f"  - {k}")
            if len(unexpected) > 5:
                print(f"  ... and {len(unexpected) - 5} more")
        
        if not missing and not unexpected:
            print("\n✅ All weights loaded successfully!")
        
        model.eval()
        
        # Run forward pass
        batch_size = 1
        channels = config["in_channels"]
        frames = 13
        height, width = 60, 80
        
        render_latent = torch.randn(batch_size, channels, frames, height, width)
        temb = torch.randn(batch_size, 5120)
        
        print(f"\nRunning forward pass with loaded weights...")
        
        with torch.no_grad():
            controlnet_states = model(
                render_latent=render_latent,
                render_mask=None,
                camera_embedding=None,
                temb=temb,
            )
        
        print(f"✅ Forward pass successful! {len(controlnet_states)} states returned")
        return True
        
    except FileNotFoundError as e:
        print(f"⚠️ Checkpoint not found: {e}")
        print("Run with --download flag to download checkpoint first")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_checkpoint():
    """Download checkpoint from HuggingFace."""
    import importlib.util
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load module directly
    load_path = os.path.join(base_dir, "Wan2GP", "models", "wan", "uni3c", "load.py")
    spec = importlib.util.spec_from_file_location("load", load_path)
    load_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(load_module)
    
    ckpts_dir = os.path.join(base_dir, "Wan2GP", "ckpts")
    load_module.download_uni3c_checkpoint_if_missing(ckpts_dir=ckpts_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1 Gate: Uni3C ControlNet Test")
    parser.add_argument("--checkpoint", action="store_true", help="Also test checkpoint loading")
    parser.add_argument("--download", action="store_true", help="Download checkpoint if missing")
    args = parser.parse_args()
    
    # Always run the basic forward pass test
    success = test_controlnet_forward_pass()
    
    if args.download:
        print("\n" + "-" * 60)
        print("Downloading checkpoint...")
        print("-" * 60)
        download_checkpoint()
    
    if args.checkpoint or args.download:
        checkpoint_success = test_checkpoint_loading()
        if checkpoint_success is False:
            success = False
    
    sys.exit(0 if success else 1)
