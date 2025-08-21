#!/usr/bin/env python3
"""
CUDA Detection and Validation Script for Headless-Wan2GP
Run this before starting the worker to ensure proper CUDA setup.
"""

import sys
import subprocess

def check_nvidia_driver():
    """Check if NVIDIA drivers are installed."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA drivers detected")
            # Extract driver version
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    driver_info = line.strip()
                    print(f"   {driver_info}")
                    break
            return True
        else:
            print("❌ NVIDIA drivers not found or not working")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi command not found - NVIDIA drivers may not be installed")
        return False

def check_pytorch_cuda():
    """Check if PyTorch is compiled with CUDA support."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        
        if torch.cuda.is_available():
            print("✅ CUDA is available in PyTorch")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            return True
        else:
            print("❌ CUDA is NOT available in PyTorch")
            print("   This means you have the CPU-only version installed")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_vram_requirements():
    """Check if GPU has sufficient VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 12:
                print(f"✅ GPU VRAM: {gpu_memory:.1f} GB (sufficient for video generation)")
                return True
            elif gpu_memory >= 8:
                print(f"⚠️  GPU VRAM: {gpu_memory:.1f} GB (may work with smaller models)")
                return True
            else:
                print(f"❌ GPU VRAM: {gpu_memory:.1f} GB (insufficient for most video models)")
                return False
        return False
    except:
        return False

def main():
    print("🔍 Checking CUDA setup for Headless-Wan2GP...\n")
    
    # Check system components
    nvidia_ok = check_nvidia_driver()
    pytorch_ok = check_pytorch_cuda()
    vram_ok = check_vram_requirements()
    
    print("\n" + "="*50)
    
    if nvidia_ok and pytorch_ok and vram_ok:
        print("🎉 All checks passed! You're ready to run Headless-Wan2GP")
        print("\nRun the worker with:")
        print("python worker.py --db-type supabase [your-args] --debug")
    else:
        print("❌ Setup issues detected. Please fix the following:")
        
        if not nvidia_ok:
            print("\n📥 Install NVIDIA drivers:")
            print("   1. Visit https://www.nvidia.com/drivers")
            print("   2. Download latest drivers for your GPU")
            print("   3. Restart after installation")
        
        if not pytorch_ok:
            print("\n🔧 Install CUDA-enabled PyTorch:")
            print("   pip uninstall torch torchvision torchaudio")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        
        if not vram_ok:
            print("\n💾 GPU VRAM may be insufficient for large models")
            print("   Consider using smaller models or cloud GPU instances")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
