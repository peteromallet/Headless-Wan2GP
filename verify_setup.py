#!/usr/bin/env python3
"""Quick verification that the environment is set up correctly."""

import os
import sys

print("=" * 60)
print("Headless-Wan2GP Setup Verification")
print("=" * 60)

# Check 1: Current directory
cwd = os.getcwd()
print(f"\n1. Current working directory: {cwd}")

# Check 2: Wan2GP directory
wan2gp_dir = os.path.join(cwd, "Wan2GP")
wan2gp_exists = os.path.isdir(wan2gp_dir)
print(f"2. Wan2GP directory exists: {wan2gp_exists}")
if wan2gp_exists:
    print(f"   Path: {wan2gp_dir}")

# Check 3: defaults directory
defaults_dir = os.path.join(wan2gp_dir, "defaults")
defaults_exists = os.path.isdir(defaults_dir)
print(f"3. defaults directory exists: {defaults_exists}")
if defaults_exists:
    json_files = [f for f in os.listdir(defaults_dir) if f.endswith('.json')]
    print(f"   Found {len(json_files)} JSON model definition files")

# Check 4: models directory
models_dir = os.path.join(wan2gp_dir, "models")
models_exists = os.path.isdir(models_dir)
print(f"4. models directory exists: {models_exists}")

# Check 5: Try importing torch and check CUDA
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"5. PyTorch CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   Device count: {torch.cuda.device_count()}")
        print(f"   Device 0: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"5. PyTorch check failed: {e}")

# Check 6: Verify we can change to Wan2GP directory
try:
    os.chdir(wan2gp_dir)
    new_cwd = os.getcwd()
    print(f"6. Successfully changed to Wan2GP directory: {new_cwd}")

    # Check from inside Wan2GP
    defaults_relative = os.path.isdir("defaults")
    print(f"   defaults/ visible from here: {defaults_relative}")

    if defaults_relative:
        json_files = [f for f in os.listdir("defaults") if f.endswith('.json')]
        print(f"   Can see {len(json_files)} model definition files")

    os.chdir(cwd)  # Change back
except Exception as e:
    print(f"6. Failed to change to Wan2GP directory: {e}")

print("\n" + "=" * 60)
if wan2gp_exists and defaults_exists and defaults_relative:
    print("✅ Setup looks good! The environment should work.")
else:
    print("❌ Setup issues detected. Check the errors above.")
print("=" * 60)
