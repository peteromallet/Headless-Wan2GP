# Phase 1.5 Validation - Quick Start

## 🎯 Goal
Verify Phase 1 works correctly before implementing Phase 2.

## ⏱️ Time Required
- Quick: 30 minutes
- Thorough: 24 hours

## 📝 Quick Validation Checklist (30 min)

### Step 1: Restart Worker
```bash
# Stop current worker, then restart
python worker.py --main-output-dir ./outputs [your other args...]
```

### Step 2: Check Logs
```bash
tail -f logs/*.log | grep "OUTPUT_DIR"
```
**Expected:** See logs showing configured output directory

### Step 3: Run Test Tasks

**Test Image Generation:**
```bash
# Submit a t2i or flux task
# Check: outputs/ for new .png files
ls -lht outputs/*.png | head -5
```

**Test Video Generation:**
```bash
# Submit a vace or t2v task
# Check: outputs/ for new .mp4 files
ls -lht outputs/*.mp4 | head -5
```

### Step 4: Verify NO NEW Files in Old Location
```bash
# Should return 0 (no files in last hour)
find Wan2GP/outputs/ -name "*.mp4" -mmin -60 | wc -l
find Wan2GP/outputs/ -name "*.png" -mmin -60 | wc -l
```

### Step 5: Check for Errors
```bash
grep -i "error\|fail\|exception" logs/*.log | tail -20
```

## ✅ Success Criteria

All must be true:
- [ ] Worker started successfully
- [ ] Logs show `[OUTPUT_DIR]` messages
- [ ] New images in `outputs/` (NOT `Wan2GP/outputs/`)
- [ ] New videos in `outputs/` (NOT `Wan2GP/outputs/`)
- [ ] No path-related errors in logs
- [ ] Old `Wan2GP/outputs/` has NO new files

## 🚦 Decision Point

**All ✅?** → Ready for Phase 2 tomorrow!

**Any ❌?** → See troubleshooting in `FILE_OUTPUT_STANDARDIZATION_BRIEFING.md` Phase 1.5

## 📊 What to Report

After validation, note:
- ✅ or ❌ for each checklist item
- Any errors encountered
- Approximate time to complete
- Number of test tasks run

## 📖 Full Details

For complete validation steps and troubleshooting:
→ See `FILE_OUTPUT_STANDARDIZATION_BRIEFING.md` - Phase 1.5
