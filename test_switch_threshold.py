#!/usr/bin/env python3
"""
Switch Threshold Testing Script with Direct Execution

This script creates and RUNS 7 tasks using vace_14B_fake_cocktail_2_2 model with different 
switch_threshold values (100, 200, 300, 400, 500, 600, 700) to test the dual-model switching behavior.

Usage:
    # Use all defaults (video.mp4, mask.mp4, "camera rotates around him")
    python test_switch_threshold.py
    
    # Use custom files with default prompt
    python test_switch_threshold.py my_video.mp4 my_mask.mp4
    
    # Use custom files and prompt
    python test_switch_threshold.py my_video.mp4 my_mask.mp4 "custom prompt here"
"""

import json
import sys
import argparse
import time
import os
from pathlib import Path
from datetime import datetime

# Add our project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from source.common_utils import generate_unique_task_id
from headless_wgp import WanOrchestrator


def run_switch_threshold_tests(video_path: str, mask_path: str, prompt: str, 
                              base_output_dir: str = "outputs/switch_threshold_test",
                              project_id: str = None):
    """
    Create and RUN tasks testing different switch_threshold values.
    
    Args:
        video_path: Path to the guide video
        mask_path: Path to the mask video  
        prompt: Generation prompt
        base_output_dir: Base directory for outputs
        project_id: Optional project ID for organization
    """
    
    # Generate a unique test run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_id = f"switch_test_{timestamp}"
    
    if project_id is None:
        project_id = f"switch-threshold-test-{timestamp}"
    
    # Define specific threshold values to test
    threshold_values = [100, 200, 300, 400, 500, 600, 700]
    
    print(f"🧪 Starting Switch Threshold Test Suite")
    print(f"📁 Test Run ID: {test_run_id}")
    print(f"🎬 Video: {video_path}")
    print(f"🎭 Mask: {mask_path}")
    print(f"💭 Prompt: {prompt}")
    print(f"📊 Testing switch_threshold values: {threshold_values}")
    print(f"🚀 Running tasks directly via headless_wgp.py")
    print("=" * 60)
    
    # Initialize WanOrchestrator
    print("\n⚙️  Initializing WanOrchestrator...")
    wan_root = Path(__file__).parent / "Wan2GP"
    orchestrator = WanOrchestrator(str(wan_root))
    
    # Load the model once for all tests
    model_name = "vace_14B_fake_cocktail_2_2"
    print(f"📦 Loading model: {model_name}")
    print("This may take a few minutes on first load...")
    orchestrator.load_model(model_name)
    print(f"✅ Model loaded successfully!\n")
    
    # Create output directory with absolute path
    output_dir = (Path(base_output_dir).absolute() / test_run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    total_tasks = len(threshold_values)
    
    for i, switch_threshold in enumerate(threshold_values, 1):
        print(f"\n{'='*60}")
        print(f"📋 Task {i}/{total_tasks}: switch_threshold = {switch_threshold}")
        print(f"{'='*60}")
        
        task_id = generate_unique_task_id(f"switch_test_{switch_threshold}_")
        
        # Prepare output path with custom naming
        output_filename = f"switch_threshold_{switch_threshold}.mp4"
        
        print(f"🎯 Task ID: {task_id}")
        print(f"📊 Switch Threshold: {switch_threshold} → {switch_threshold/1000.0:.3f} (converted to 0-1)")
        print(f"💾 Output: {output_dir}/{output_filename}")
        print(f"⏳ Starting generation...")
        
        start_time = time.time()
        
        try:
            # Run the generation with the specific switch_threshold
            output_path = orchestrator.generate_vace(
                prompt=prompt,
                video_guide=video_path,
                video_mask=mask_path,
                video_prompt_type="VM",  # Video + Mask
                resolution="768x576",
                video_length=65,
                num_inference_steps=10,  # From model defaults
                guidance_scale=3,  # From model defaults  
                flow_shift=2,  # From model defaults
                seed=12345,  # Consistent seed for comparison
                negative_prompt="blurry, low quality, distorted",
                control_net_weight=1.0,
                control_net_weight2=1.0,
                switch_threshold=switch_threshold / 1000.0,  # 🎯 KEY TEST PARAMETER - Convert 0-1000 to 0-1 range
                guidance2_scale=1  # From model defaults
            )
            
            generation_time = time.time() - start_time
            
            # Move/rename the output to our custom location
            if output_path and Path(output_path).exists():
                final_path = output_dir / output_filename
                Path(output_path).rename(final_path)
                
                print(f"✅ Generation completed in {generation_time:.1f} seconds")
                print(f"📁 Saved to: {final_path}")
                
                # Get file size for analysis
                file_size_mb = final_path.stat().st_size / (1024 * 1024)
                
                results.append({
                    "task_id": task_id,
                    "switch_threshold": switch_threshold,
                    "output_path": str(final_path),
                    "generation_time": generation_time,
                    "file_size_mb": file_size_mb,
                    "status": "success"
                })
            else:
                print(f"⚠️  Generation completed but output path not found")
                results.append({
                    "task_id": task_id,
                    "switch_threshold": switch_threshold,
                    "status": "no_output"
                })
                
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            results.append({
                "task_id": task_id,
                "switch_threshold": switch_threshold,
                "status": "failed",
                "error": str(e)
            })
        
        # Brief pause between generations
        if i < total_tasks:
            print(f"\n⏸️  Pausing 5 seconds before next generation...")
            time.sleep(5)
    
    # Unload model to free memory
    print(f"\n🧹 Unloading model...")
    orchestrator.unload_model()
    
    return results, test_run_id, output_dir


def save_results(results: list, test_run_id: str, output_dir: Path):
    """Save test results to JSON and generate analysis report."""
    
    # Save raw results
    results_file = output_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "test_run_id": test_run_id,
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Generate analysis report
    report_file = output_dir / "analysis_report.md"
    
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    
    report = f"""# Switch Threshold Test Results
    
## Test Run: {test_run_id}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- Total Tests: {len(results)}
- Successful: {len(successful)}
- Failed: {len(failed)}

## Results by Threshold

| Threshold | Status | Time (s) | Size (MB) | Output File |
|-----------|--------|----------|-----------|-------------|
"""
    
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        time_str = f"{r.get('generation_time', 0):.1f}" if "generation_time" in r else "N/A"
        size_str = f"{r.get('file_size_mb', 0):.2f}" if "file_size_mb" in r else "N/A"
        output = Path(r["output_path"]).name if "output_path" in r else "N/A"
        
        report += f"| {r['switch_threshold']} | {status_icon} | {time_str} | {size_str} | {output} |\n"
    
    if successful:
        # Calculate averages
        avg_time = sum(r["generation_time"] for r in successful) / len(successful)
        avg_size = sum(r["file_size_mb"] for r in successful) / len(successful)
        
        report += f"""
## Performance Analysis

### Average Metrics
- Average Generation Time: {avg_time:.1f} seconds
- Average File Size: {avg_size:.2f} MB

### Timing Analysis by Threshold
"""
        for r in successful:
            pct_diff = ((r["generation_time"] - avg_time) / avg_time) * 100
            report += f"- **{r['switch_threshold']}**: {r['generation_time']:.1f}s ({pct_diff:+.1f}% from avg)\n"
    
    report += """
## Visual Quality Assessment

Review the generated videos and rate each on:

| Threshold | Quality (1-10) | Prompt Adherence | Detail | Structure | Notes |
|-----------|----------------|------------------|--------|-----------|-------|
| 100 | _ | _ | _ | _ | |
| 200 | _ | _ | _ | _ | |
| 300 | _ | _ | _ | _ | |
| 400 | _ | _ | _ | _ | |
| 500 | _ | _ | _ | _ | |
| 600 | _ | _ | _ | _ | |
| 700 | _ | _ | _ | _ | |

## Recommendations

Based on the results:
- **Optimal Quality**: ___ (fill in after review)
- **Best Speed/Quality Balance**: ___ (fill in after review)
- **Recommended Default**: ___ (fill in after review)
"""
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📊 Analysis report saved to: {report_file}")
    
    return results_file, report_file


def main():
    parser = argparse.ArgumentParser(description="Run switch_threshold tests for vace_14B_fake_cocktail_2_2")
    parser.add_argument("video", nargs='?', default="video.mp4", help="Path to guide video (default: video.mp4)")
    parser.add_argument("mask", nargs='?', default="mask.mp4", help="Path to mask video (default: mask.mp4)")
    parser.add_argument("prompt", nargs='?', default="camera rotates around him", help="Generation prompt (default: 'camera rotates around him')")
    parser.add_argument("--output-dir", default="outputs/switch_threshold_test", 
                       help="Base output directory")
    
    args = parser.parse_args()
    
    # Convert to absolute paths to handle working directory changes
    video_path = Path(args.video).absolute()
    mask_path = Path(args.mask).absolute()
    
    # Validate input files
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
        
    if not mask_path.exists():
        print(f"❌ Error: Mask file not found: {mask_path}")
        sys.exit(1)
    
    print(f"📁 Using absolute paths:")
    print(f"   Video: {video_path}")
    print(f"   Mask: {mask_path}")
    
    # Run the tests
    try:
        results, test_run_id, output_dir = run_switch_threshold_tests(
            video_path=str(video_path),  # Pass as string
            mask_path=str(mask_path),    # Pass as string
            prompt=args.prompt,
            base_output_dir=args.output_dir
        )
        
        # Save results and generate report
        results_file, report_file = save_results(results, test_run_id, output_dir)
        
        # Final summary
        print("\n" + "="*60)
        print("🎯 TEST SUITE COMPLETE!")
        print("="*60)
        
        successful = [r for r in results if r.get("status") == "success"]
        print(f"✅ Successful: {len(successful)}/{len(results)}")
        
        if successful:
            print(f"\n📁 All outputs saved to: {output_dir}")
            print(f"📊 Results: {results_file}")
            print(f"📝 Report: {report_file}")
            
            print("\n🎬 Generated videos:")
            for r in successful:
                print(f"  - switch_threshold_{r['switch_threshold']}.mp4")
            
            print("\n🔍 Next steps:")
            print("1. Review all generated videos side-by-side")
            print("2. Fill out the quality assessment in the report")
            print("3. Identify the optimal switch_threshold value")
            print("4. Update model defaults if needed")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()