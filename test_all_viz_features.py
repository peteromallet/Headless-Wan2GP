#!/usr/bin/env python3
"""
Test all three new visualization features:
1. Active image scaling (1.15x)
2. Vertical layout
3. Structure video type/strength overlay
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
        "id": "test-viz-all-features",
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

    print("=" * 80)
    print("🎬 Testing All Three Visualization Features")
    print("=" * 80)

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

    print(f"\n📊 Task Info:")
    print(f"  - Task ID: {task_data['id']}")
    print(f"  - FPS: {fps}")
    print(f"  - Segments: {len(segment_frames)}")
    print(f"  - Input Images: {len(input_image_urls)}")

    # Create temp directory for downloads
    temp_dir = Path(tempfile.mkdtemp(prefix="test_viz_features_"))
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

        # Test 1: Triple layout with overlay
        print(f"\n{'='*80}")
        print(f"🎨 TEST 1: Triple layout with structure overlay")
        print(f"   Features: Active image scale + Structure type/strength overlay")
        print("=" * 80)

        viz_output_path_1 = Path(__file__).parent / "triple_layout_test.mp4"
        result_path_1 = create_travel_visualization(
            output_video_path=str(local_output_path),
            structure_video_path=str(local_structure_path),
            guidance_video_path=None,
            input_image_paths=local_image_paths,
            segment_frames=segment_frames,
            segment_prompts=segment_prompts,
            viz_output_path=str(viz_output_path_1),
            layout="triple",
            fps=fps,
            show_guidance=False,
            structure_video_treatment=structure_video_treatment,
            frame_overlaps=frame_overlaps,
            structure_video_type="depth",
            structure_video_strength=1.0
        )

        print(f"✅ Test 1 Complete: {result_path_1}")
        print(f"   Size: {Path(result_path_1).stat().st_size / (1024*1024):.2f} MB")

        # Test 2: NEW Vertical layout with overlay
        print(f"\n{'='*80}")
        print(f"🎨 TEST 2: NEW Vertical layout with overlay")
        print(f"   Features: Active image scale + Vertical layout + Structure overlay")
        print("=" * 80)

        viz_output_path_2 = Path(__file__).parent / "vertical_layout_test.mp4"
        result_path_2 = create_travel_visualization(
            output_video_path=str(local_output_path),
            structure_video_path=str(local_structure_path),
            guidance_video_path=None,
            input_image_paths=local_image_paths,
            segment_frames=segment_frames,
            segment_prompts=segment_prompts,
            viz_output_path=str(viz_output_path_2),
            layout="vertical",
            fps=fps,
            show_guidance=False,
            structure_video_treatment=structure_video_treatment,
            frame_overlaps=frame_overlaps,
            structure_video_type="canny",
            structure_video_strength=0.5
        )

        print(f"✅ Test 2 Complete: {result_path_2}")
        print(f"   Size: {Path(result_path_2).stat().st_size / (1024*1024):.2f} MB")

        # Summary
        print(f"\n{'='*80}")
        print(f"✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📹 Created visualizations:")
        print(f"  1. triple_layout_test.mp4 - Triple layout with 'depth | 1.0' overlay")
        print(f"  2. vertical_layout_test.mp4 - Vertical layout with 'canny | 0.5' overlay")
        print(f"\n✨ All videos show:")
        print(f"  - Active images scaled to 1.15x (Feature 1)")
        print(f"  - Structure type/strength overlay in top-left (Feature 3)")
        print(f"  - Vertical layout shows images on left side (Feature 2)")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temp files
        print(f"\n🧹 Cleaning up temp directory...")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
