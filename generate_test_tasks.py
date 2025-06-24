#!/usr/bin/env python3
"""
Generate a set of end-to-end headless-server tasks that exercise the
most important travel-segment scenarios:

1. single image, no continue-video
2. 3 images, no continue-video
3. continue-video + single image
4. continue-video + 2 images

For each test a JSON payload (compatible with add_task.py) is written
under   tests/<test_name>/<test_name>_task.json  and the referenced
input files (images / video) are copied into that same directory with a
clean, descriptive name so the folder is self-contained and easy to
inspect.

After writing the JSON the script *optionally* enqueues the task by
invoking   python add_task.py --type travel_orchestrator --params
@<json_file>

Set  --enqueue   to actually perform the enqueue.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------
PROJECT_ID = "test_suite"
BASE_PROMPT = "Car driving through a city, sky morphing"
NEG_PROMPT = "chaotic"
MODEL_NAME = "vace_14B"
RESOLUTION = "500x500"
FPS = 16
SEGMENT_FRAMES_DEFAULT = 81  # will be quantised downstream (4n+1)
FRAME_OVERLAP_DEFAULT = 12
SEED_BASE = 11111
OUTPUT_DIR_DEFAULT = "./outputs"

SAMPLES_DIR = Path("samples")
ASSET_IMAGES = [
    SAMPLES_DIR / "1.png",
    SAMPLES_DIR / "2.png",
    SAMPLES_DIR / "3.png",
]
ASSET_VIDEO = SAMPLES_DIR / "test.test.mp4"

TESTS_ROOT = Path("tests")

# ---------------------------------------------------------------------

def make_orchestrator_payload(*, run_id: str,
                              images: list[Path],
                              continue_video: Path | None,
                              num_segments: int) -> dict:
    """Create the orchestrator_details dict used by headless server."""
    payload: dict = {
        "run_id": run_id,
        "input_image_paths_resolved": [str(p) for p in images],
        "parsed_resolution_wh": RESOLUTION,
        "model_name": MODEL_NAME,
        "num_new_segments_to_generate": num_segments,
        "base_prompts_expanded": [BASE_PROMPT] * num_segments,
        "negative_prompts_expanded": [NEG_PROMPT] * num_segments,
        "segment_frames_expanded": [SEGMENT_FRAMES_DEFAULT] * num_segments,
        "frame_overlap_expanded": [FRAME_OVERLAP_DEFAULT] * num_segments,
        "fps_helpers": FPS,
        "fade_in_params_json_str": json.dumps({
            "low_point": 0.0, "high_point": 1.0,
            "curve_type": "ease_in_out", "duration_factor": 0.0
        }),
        "fade_out_params_json_str": json.dumps({
            "low_point": 0.0, "high_point": 1.0,
            "curve_type": "ease_in_out", "duration_factor": 0.0
        }),
        "seed_base": SEED_BASE,
        "main_output_dir_for_run": OUTPUT_DIR_DEFAULT,
    }
    if continue_video is not None:
        payload["continue_from_video_resolved_path"] = str(continue_video)
    return payload


def write_test_case(name: str,
                    images: list[Path],
                    continue_video: Path | None,
                    num_segments: int,
                    enqueue: bool) -> None:
    """Materialise a test folder with JSON and asset copies."""
    test_dir = TESTS_ROOT / name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy / link assets with descriptive names and record new paths
    copied_images: list[Path] = []
    for idx, img in enumerate(images):
        dst = test_dir / f"{name}_image{idx+1}{img.suffix}"
        if dst.exists():
            dst.unlink()
        shutil.copy2(img, dst)
        copied_images.append(dst)
    if continue_video is not None:
        dst_vid = test_dir / f"{name}_continue_video{continue_video.suffix}"
        if dst_vid.exists():
            dst_vid.unlink()
        shutil.copy2(continue_video, dst_vid)
        continue_video_path = dst_vid
    else:
        continue_video_path = None

    # Build JSON payload (wrapper around orchestrator_details)
    run_id = f"{name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    orch_payload = make_orchestrator_payload(
        run_id=run_id,
        images=copied_images,
        continue_video=continue_video_path,
        num_segments=num_segments,
    )
    task_json: dict = {
        "project_id": PROJECT_ID,
        "orchestrator_details": orch_payload,
    }

    json_path = test_dir / f"{name}_task.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(task_json, fp, indent=2)

    try:
        print(f"[WRITE] {json_path.resolve().relative_to(Path.cwd().resolve())}")
    except ValueError:
        # Fallback if paths are on different drives or unrelated
        print(f"[WRITE] {json_path.resolve()}")

    if enqueue:
        cmd = [sys.executable, "add_task.py", "--type", "travel_orchestrator",
               "--params", f"@{json_path}"]
        print("[ENQUEUE]", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] add_task failed: {e}")


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser("generate_test_tasks")
    parser.add_argument("--enqueue", action="store_true",
                        help="Actually call add_task.py for each generated test case")
    args = parser.parse_args()

    TESTS_ROOT.mkdir(exist_ok=True)

    # --- scenario definitions ------------------------------------------------
    scenarios = [
        ("test_1_single_image",          [ASSET_IMAGES[0]],                 None,                 1),
        ("test_2_three_images",          ASSET_IMAGES,                      None,                 3),
        ("test_3_continue_plus_one",     [ASSET_IMAGES[0]],                 ASSET_VIDEO,          1),
        ("test_4_continue_plus_two",     ASSET_IMAGES[:2],                  ASSET_VIDEO,          2),
    ]

    for name, imgs, cont_vid, segments in scenarios:
        write_test_case(name, imgs, cont_vid, segments, enqueue=args.enqueue)

    print("\nAll test cases generated under", TESTS_ROOT.resolve())


if __name__ == "__main__":
    main() 