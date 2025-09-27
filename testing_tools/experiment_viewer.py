#!/usr/bin/env python3
"""
Streamlit web viewer for experiment results.
Run with: streamlit run experiment_viewer.py --server.port 5500
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import base64
import shutil

def get_base64_video(video_path):
    """Convert video file to base64 for HTML display."""
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        return base64.b64encode(video_bytes).decode()
    except:
        return None

def display_video(video_path, width=400):
    """Display video in Streamlit."""
    if video_path.exists():
        video_base64 = get_base64_video(video_path)
        if video_base64:
            video_html = f"""
            <video width="{width}" controls>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
            st.markdown(video_html, unsafe_allow_html=True)
        else:
            st.error(f"Could not load video: {video_path}")
    else:
        st.warning(f"Video not found: {video_path}")

def load_json_file(json_path):
    """Load JSON file safely."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except:
        return None

def load_text_file(file_path):
    """Load text file safely."""
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except:
        return None

def delete_experiment(exp_path):
    """Delete an entire experiment folder."""
    try:
        if exp_path.exists():
            shutil.rmtree(exp_path)
            return True
    except Exception as e:
        st.error(f"Error deleting experiment: {e}")
    return False

def scan_experiments(testing_path):
    """Scan testing folder for experiments."""
    experiments = {}

    for experiment_folder in testing_path.iterdir():
        if experiment_folder.is_dir() and not experiment_folder.name.startswith('.'):
            settings_file = experiment_folder / "settings.json"

            if settings_file.exists():
                exp_data = {
                    "name": experiment_folder.name,
                    "path": experiment_folder,
                    "settings": load_json_file(settings_file),
                    "generations": []
                }

                # Look for numbered generations in experiment folder
                numbered_files = {}
                for file in experiment_folder.iterdir():
                    if file.is_file() and '_' in file.name:
                        parts = file.name.split('_', 1)
                        if len(parts) == 2 and parts[0].isdigit():
                            num = int(parts[0])
                            file_type = parts[1]

                            if num not in numbered_files:
                                numbered_files[num] = {}
                            numbered_files[num][file_type] = file

                # Look for input files in parent testing directory
                input_files = {}
                for file in testing_path.iterdir():
                    if file.is_file() and '_' in file.name:
                        parts = file.name.split('_', 1)
                        if len(parts) == 2 and parts[0].isdigit():
                            num = int(parts[0])
                            file_type = parts[1]

                            if num not in input_files:
                                input_files[num] = {}
                            input_files[num][file_type] = file

                # Process numbered generations
                for num in sorted(numbered_files.keys()):
                    gen_data = {
                        "num": num,
                        "files": numbered_files[num].copy()
                    }

                    # Add input files from parent directory
                    if num in input_files:
                        gen_data["files"].update(input_files[num])

                    # Load metadata and run details
                    metadata_file = experiment_folder / f"{num}_metadata.json"
                    run_details_file = experiment_folder / f"{num}_run_details.json"

                    # Check for prompt files (.txt or .json)
                    prompt_txt_file = testing_path / f"{num}_prompt.txt"
                    prompt_json_file = testing_path / f"{num}_prompt.json"

                    if metadata_file.exists():
                        gen_data["metadata"] = load_json_file(metadata_file)
                    if run_details_file.exists():
                        gen_data["run_details"] = load_json_file(run_details_file)

                    # Load prompt from .txt or .json
                    if prompt_txt_file.exists():
                        gen_data["prompt"] = load_text_file(prompt_txt_file)
                    elif prompt_json_file.exists():
                        prompt_data = load_json_file(prompt_json_file)
                        if isinstance(prompt_data, dict) and "prompt" in prompt_data:
                            gen_data["prompt"] = prompt_data["prompt"]
                        elif isinstance(prompt_data, str):
                            gen_data["prompt"] = prompt_data

                    exp_data["generations"].append(gen_data)

                # Check for single-file experiments (text-to-video)
                output_file = experiment_folder / "output.mp4"
                if output_file.exists():
                    gen_data = {
                        "num": None,
                        "files": {"output.mp4": output_file}
                    }

                    metadata_file = experiment_folder / "metadata.json"
                    run_details_file = experiment_folder / "run_details.json"

                    if metadata_file.exists():
                        gen_data["metadata"] = load_json_file(metadata_file)
                    if run_details_file.exists():
                        gen_data["run_details"] = load_json_file(run_details_file)

                    exp_data["generations"].append(gen_data)

                experiments[experiment_folder.name] = exp_data

    return experiments

def main():
    st.set_page_config(
        page_title="Experiment Viewer",
        page_icon="🧪",
        layout="wide"
    )

    st.title("🧪 Experiment Results Viewer")
    st.markdown("---")

    # Folder selection
    st.sidebar.header("📁 Folder Selection")

    base_path = Path("testing")
    if not base_path.exists():
        st.error("Testing folder not found!")
        return

    # Get available experiment folders
    available_folders = [f.name for f in base_path.iterdir() if f.is_dir() and not f.name.startswith('.')]

    if not available_folders:
        st.error("No experiment folders found in testing/")
        return

    selected_folder = st.sidebar.selectbox(
        "Select Experiment Batch",
        available_folders,
        help="Choose which experiment batch to view"
    )

    if selected_folder:
        testing_path = base_path / selected_folder
        st.sidebar.success(f"📂 Viewing: {testing_path}")

        # Scan experiments
        with st.spinner("Loading experiments..."):
            experiments = scan_experiments(testing_path)

        if not experiments:
            st.warning("No experiments found in selected folder.")
            return

        st.header(f"Experiments in `{selected_folder}`")
        st.write(f"Found {len(experiments)} experiment(s)")

        # Display experiments
        for exp_name, exp_data in experiments.items():
            # Clear any old confirmation states for experiments that no longer exist
            confirm_key = f"confirm_delete_{exp_name}"

            with st.expander(f"🧪 {exp_name}", expanded=True):

                # Header with delete button
                col_header, col_delete = st.columns([4, 1])
                with col_header:
                    st.write("")  # Spacer
                with col_delete:
                    # Check if in confirmation state
                    if st.session_state.get(confirm_key, False):
                        # Show confirmation buttons
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("✅ YES", key=f"confirm_yes_{exp_name}",
                                        help="Confirm deletion"):
                                if delete_experiment(exp_data["path"]):
                                    st.success(f"✅ Experiment '{exp_name}' deleted!")
                                    # Clear confirmation state
                                    if confirm_key in st.session_state:
                                        del st.session_state[confirm_key]
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to delete '{exp_name}'")
                        with col_no:
                            if st.button("❌ NO", key=f"confirm_no_{exp_name}",
                                        help="Cancel deletion"):
                                # Clear confirmation state
                                if confirm_key in st.session_state:
                                    del st.session_state[confirm_key]
                                st.rerun()
                    else:
                        # Show delete button
                        if st.button("🗑️ Delete", key=f"delete_{exp_name}",
                                    help=f"Delete entire experiment folder: {exp_name}"):
                            st.session_state[confirm_key] = True
                            st.rerun()

                # Show confirmation warning if in confirmation state
                if st.session_state.get(confirm_key, False):
                    st.warning(f"⚠️ **CONFIRM DELETION** of experiment '{exp_name}' and all its data?")

                # Basic info
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("Settings")
                    if exp_data["settings"]:
                        st.json(exp_data["settings"])
                    else:
                        st.warning("No settings.json found")

                with col2:
                    st.subheader("Generations")

                    if not exp_data["generations"]:
                        st.info("No completed generations found")
                        continue

                    for gen in exp_data["generations"]:
                        if gen["num"] is not None:
                            st.markdown(f"#### Generation #{gen['num']}")
                        else:
                            st.markdown("#### Single Generation")

                        # Display prompt if available - prioritize metadata over static prompt files
                        prompt_text = None
                        if "metadata" in gen and gen["metadata"] and "settings" in gen["metadata"] and "prompt" in gen["metadata"]["settings"]:
                            prompt_text = gen["metadata"]["settings"]["prompt"]
                        elif "prompt" in gen and gen["prompt"]:
                            prompt_text = gen["prompt"]

                        if prompt_text:
                            st.write(f"**Prompt:** {prompt_text}")

                        # Display video outputs
                        video_cols = st.columns(3)

                        with video_cols[0]:
                            # Input video - look for N_video.mp4 or video.mp4
                            input_video_key = None
                            for key in gen["files"]:
                                if key.endswith("video.mp4"):
                                    input_video_key = key
                                    break

                            if input_video_key:
                                st.write("**Input Video:**")
                                display_video(gen["files"][input_video_key], width=250)

                        with video_cols[1]:
                            # Mask video - look for N_mask.mp4 or mask.mp4
                            mask_video_key = None
                            for key in gen["files"]:
                                if key.endswith("mask.mp4"):
                                    mask_video_key = key
                                    break

                            if mask_video_key:
                                st.write("**Input Mask:**")
                                display_video(gen["files"][mask_video_key], width=250)

                        with video_cols[2]:
                            # Output video - look for N_output.mp4 or output.mp4
                            output_key = None
                            for key in gen["files"]:
                                if key.endswith("output.mp4"):
                                    output_key = key
                                    break

                            if output_key:
                                st.write("**Generated Output:**")
                                display_video(gen["files"][output_key], width=250)
                            else:
                                st.warning("No output video found")

                        # Display metadata
                        if "run_details" in gen and gen["run_details"]:
                            details = gen["run_details"]

                            metric_cols = st.columns(4)
                            with metric_cols[0]:
                                # Try both old and new field names
                                gen_time = details.get('generation_time_seconds') or details.get('generation_time', 0)
                                st.metric("Generation Time", f"{gen_time:.1f}s")
                            with metric_cols[1]:
                                st.metric("Status", details.get('status', 'Unknown'))
                            with metric_cols[2]:
                                # Check metadata for file size
                                if gen.get("metadata") and 'file_size_mb' in gen["metadata"]:
                                    st.metric("File Size", f"{gen['metadata']['file_size_mb']:.1f}MB")
                            with metric_cols[3]:
                                # Try both timestamp field names
                                start_ts = details.get('start_timestamp') or gen.get("metadata", {}).get('timestamp')
                                if start_ts:
                                    try:
                                        start_time = datetime.fromisoformat(start_ts)
                                        st.metric("Started", start_time.strftime("%H:%M:%S"))
                                    except:
                                        pass

                        # Full metadata (collapsible)
                        if "metadata" in gen and gen["metadata"]:
                            with st.expander("View Full Metadata", expanded=False):
                                st.json(gen["metadata"])

                        st.markdown("---")

if __name__ == "__main__":
    main()