#!/bin/bash
cd /workspace/Headless-Wan2GP
/workspace/Headless-Wan2GP/venv/bin/python test_clip_offsets_exact.py \
  "Wan2GP/outputs/2025-11-05-21h18m47s_seed44929631_smooth motion.mp4" \
  "outputs/join_clips/unknown_task_1762377341.8336985/ending_video_211542_bd2742.mp4" 2>&1 | grep -v "ALSA\|XDG_RUNTIME\|FutureWarning\|pynvml"
