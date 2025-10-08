#!/usr/bin/env python3
"""
Quick test script for visualization with switchover adjustment.
"""

import json
import sys
import tempfile
import requests
from pathlib import Path
from urllib.parse import urlparse

# Import the visualization module
sys.path.insert(0, str(Path(__file__).parent / "source"))
from visualization_utils import create_travel_visualization


def download_file(url: str, output_path: str) -> str:
    """Download a file from URL to local path."""
    print(f"  ⬇️  Downloading: {Path(output_path).name}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"    ✅ Downloaded ({file_size_mb:.2f} MB)")
    return output_path


def main():
    """Main entry point."""

    # Task data from user
    task_data = {
        "id": "test-viz-123",
        "task_type": "travel_orchestrator",
        "status": "Complete",
        "output_location": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/travel_final_224151_01213b.mp4",
        "params": {
            "orchestrator_details": {
                "steps": 6,
                "run_id": "20251007185551753",
                "shot_id": "d66e2581-7516-453c-8650-9b56a1baa480",
                "seed_base": 396723,
                "model_name": "lightning_baseline_2_2_2",
                "base_prompt": "zooming in on a bus that's driving through the countryside",
                "segment_frames_expanded": [50, 50, 50, 50],
                "frame_overlap_expanded": [10, 10, 10, 10],
                "enhanced_prompts_expanded": [
                    "zooming in on a bus that's driving through the countryside",
                    "zooming in on a bus that's driving through the countryside",
                    "zooming in on a bus that's driving through the countryside",
                    "zooming in on a bus that's driving through the countryside"
                ],
                "input_image_paths_resolved": [
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/27ba1029-a246-4fcb-a715-c62bc0969071-u1_8f22c97f-23d8-407b-94c4-98c3c62c9a59.jpeg",
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/749aa693-57b6-440f-a48d-93bf238388fc-u2_20dba911-7262-47f3-b52c-6384bfda0e3d.jpeg",
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/650357b4-6cf6-4453-9e34-833ac8869144-u2_f7921019-0ccf-49af-904d-276f4bdb1f21.jpeg",
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/f61cdd66-d282-4eee-9892-c25d8a7a2f55-u1_ee9039dd-442b-495d-8fae-14e1fc7158d5.jpeg",
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/23d4b5b0-0141-4d2f-bc9a-e44e82b9c867-u1_bf75e138-fb6e-49f4-bc4d-52e92a726a00.jpeg"
                ],
                "structure_video_path": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/guidance-videos/f3c36ed6-eeb4-4259-8f67-b8260efd1c0e/1759862831715-0o6xi.MOV",
                "structure_video_treatment": "adjust",
                "fps_helpers": 16
            }
        }
    }

    print("=" * 70)
    print("🎬 Testing Visualization with Updated Switchover Logic")
    print("=" * 70)

    # Parse params
    params = task_data["params"]
    orchestrator_details = params["orchestrator_details"]

    # Extract relevant info
    input_image_urls = orchestrator_details["input_image_paths_resolved"]
    structure_video_url = orchestrator_details["structure_video_path"]
    output_video_url = task_data["output_location"]
    segment_frames = orchestrator_details["segment_frames_expanded"]
    frame_overlaps = orchestrator_details["frame_overlap_expanded"]
    segment_prompts = orchestrator_details["enhanced_prompts_expanded"]
    structure_video_treatment = orchestrator_details["structure_video_treatment"]
    fps = orchestrator_details.get("fps_helpers", 16)

    # Calculate timeline
    total_raw_frames = sum(segment_frames)
    total_overlap_frames = sum(frame_overlaps)
    actual_final_frames = total_raw_frames - total_overlap_frames

    print(f"\n📊 Task Info:")
    print(f"  - Task ID: {task_data['id']}")
    print(f"  - FPS: {fps}")
    print(f"  - Segments: {len(segment_frames)}")
    print(f"  - Segment Frames: {segment_frames}")
    print(f"  - Frame Overlaps: {frame_overlaps}")
    print(f"  - Total Raw Frames: {total_raw_frames}")
    print(f"  - Total Overlap: {total_overlap_frames}")
    print(f"  - Actual Final Frames: {actual_final_frames}")
    print(f"  - Expected Duration: {actual_final_frames / fps:.2f}s")
    print(f"  - Input Images: {len(input_image_urls)}")

    # Create temp directory for downloads
    temp_dir = Path(tempfile.mkdtemp(prefix="test_viz_"))
    print(f"\n📁 Temp directory: {temp_dir}")

    try:
        # Download input images
        print(f"\n🖼️  Downloading {len(input_image_urls)} input images...")
        local_image_paths = []
        for i, img_url in enumerate(input_image_urls):
            ext = Path(urlparse(img_url).path).suffix or ".jpg"
            local_path = temp_dir / f"input_{i}{ext}"
            download_file(img_url, str(local_path))
            local_image_paths.append(str(local_path))

        # Download structure video
        print(f"\n🎥 Downloading structure video...")
        structure_ext = Path(urlparse(structure_video_url).path).suffix or ".mp4"
        local_structure_path = temp_dir / f"structure{structure_ext}"
        download_file(structure_video_url, str(local_structure_path))

        # Download output video
        print(f"\n🎥 Downloading output video...")
        output_ext = Path(urlparse(output_video_url).path).suffix or ".mp4"
        local_output_path = temp_dir / f"output{output_ext}"
        download_file(output_video_url, str(local_output_path))

        # Create visualization
        print(f"\n{'='*70}")
        print(f"🎨 Creating visualization with halfway switchover...")
        print("=" * 70)

        viz_output_path = Path(__file__).parent / "test_viz.mp4"

        result_path = create_travel_visualization(
            output_video_path=str(local_output_path),
            structure_video_path=str(local_structure_path),
            guidance_video_path=None,
            input_image_paths=local_image_paths,
            segment_frames=segment_frames,
            segment_prompts=segment_prompts,
            viz_output_path=str(viz_output_path),
            layout="triple",
            fps=fps,
            show_guidance=False,
            structure_video_treatment=structure_video_treatment,
            frame_overlaps=frame_overlaps
        )

        print(f"\n{'='*70}")
        print(f"✅ Visualization Complete!")
        print("=" * 70)
        print(f"\n📹 Output: {result_path}")

        result_file = Path(result_path)
        if result_file.exists():
            size_mb = result_file.stat().st_size / (1024 * 1024)
            print(f"📊 File size: {size_mb:.2f} MB")

        return result_path

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up temp files
        print(f"\n🧹 Cleaning up temp directory...")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    result = main()
    print(f"\n🎉 Done! Visualization saved to: {result}")
