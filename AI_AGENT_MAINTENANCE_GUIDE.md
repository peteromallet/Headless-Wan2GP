# AI Agent Maintenance Guide for Headless-Wan2GP

## Overview

This guide provides step-by-step instructions for an AI agent to automatically maintain the Headless-Wan2GP repository. The agent will pull the latest changes from the Headless-Wan2GP fork, **update the upstream Wan2GP submodule from https://github.com/deepbeepmeep/Wan2GP.git**, test the system, and fix issues to ensure continuous functionality.

## Core Mission

**Primary Goal**: Update the Wan2GP submodule to the latest upstream version and maintain a stable, working Headless-Wan2GP system that can successfully run video generation tests without human intervention.

**Key Objectives**:
1. **Update Wan2GP submodule** from upstream https://github.com/deepbeepmeep/Wan2GP.git to latest version
2. **Ensure compatibility** between updated Wan2GP and the Headless-Wan2GP wrapper system
3. **Validate functionality** through comprehensive baseline testing

**Success Criteria**: 
- Wan2GP submodule successfully updated to latest upstream commit
- All model configurations in `test_model_comparison.py` generate videos successfully without errors (baseline test)
- Integration between Headless-Wan2GP and updated upstream Wan2GP remains functional

## Phase 1: Environment Setup & Wan2GP Analysis

### Step 1.1: Analyze Current vs Upstream Differences
**Goal**: Understand all changes before updating

**Agent Actions**:
- Clone fresh upstream Wan2GP to temporary location for comparison
- Run diff analysis between current Wan2GP/ and upstream
- Document new files, removed files, and modified files
- Focus on key integration files: wgp.py, defaults/*.json, requirements.txt
- Create `maintenance_analysis/wan2gp_diff_analysis.md` with findings

### Step 1.2: Document System Integration Points  
**Goal**: Map how our system depends on Wan2GP

**Agent Actions**:
- Analyze `headless_model_management.py` - identify all WGP imports and function calls
- Analyze `headless_wgp.py` - identify critical WGP functions we use (generate_video, load_models, etc.)
- Document model config dependencies (defaults/*.json files)
- Document path/import dependencies (Python path changes, working directory)
- Create `maintenance_analysis/system_integration_analysis.md`

### Step 1.3: Generate Update Checklist
**Goal**: Create actionable checklist based on analysis

**Agent Actions**:
- Based on diff analysis: identify files that need our attention
- Based on integration analysis: identify functions/APIs that might break
- Create verification items for each integration point
- Include post-update validation steps
- Create `maintenance_analysis/update_checklist.md`

### Step 1.4: Clean Slate - Update Wan2GP
**Goal**: Replace current Wan2GP with latest upstream

**Agent Actions**:
- Create backup of current Wan2GP directory (for rollback)
- Remove current Wan2GP directory completely  
- Clone fresh from https://github.com/deepbeepmeep/Wan2GP.git
- Record new commit hash for tracking

### Step 1.5: Update Documentation
**Goal**: Record the update for tracking

**Agent Actions**:
- Update STRUCTURE.md with new Wan2GP commit hash and timestamp
- Clean up temporary analysis files
- Document update completion

## Phase 2: Environment Validation

### Step 2.1: Python Environment Check
```bash
# Verify Python version
python --version
python3 --version

# Check if we're in the right directory (should be in Headless-Wan2GP root)
pwd
ls -la | grep -E "(test_model_comparison\.py|headless_wgp\.py|worker\.py|Wan2GP/)"
```

**Agent Behavior**:
- Ensure Python 3.8+ is available
- If multiple Python versions exist, prefer the one that works with existing requirements.txt
- If Python is missing, install it before proceeding

### Step 2.2: Install/Update Dependencies
```bash
# Install main project requirements
pip install -r requirements.txt

# Install Wan2GP requirements
cd Wan2GP/
pip install -r requirements.txt
cd ..
```

**Agent Behavior**:
- If pip fails, try `pip3` or create virtual environment
- If dependency conflicts occur, document them and attempt resolution
- Consider using `pip install --upgrade` for problematic packages

## Phase 3: Iterative Testing & Fixing Loop

### Step 3.1: Initial Baseline Test Run
```bash
# Run the baseline model comparison test
python test_model_comparison.py --output-dir outputs/baseline_test_$(date +%Y%m%d_%H%M%S)
```

**Expected Outcomes**:
- ✅ **Success**: All 3 models (vace_14B, vace_14B_fake_cocktail_2_2, optimised-t2i) generate videos → Proceed to Phase 4
- ⚠️ **Partial Success**: Some models work → Analyze and fix failing ones
- ❌ **Complete Failure**: System won't start → Debug environment issues

### Step 3.2: Log Analysis Protocol

**When tests fail, analyze logs systematically:**

1. **Import/Module Errors**:
   ```bash
   # Check for missing modules
   python -c "import sys; print('\n'.join(sys.path))"
   ```
   
2. **CUDA/GPU Errors**:
   ```bash
   # Check GPU availability
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
   ```

3. **File Path Errors**:
   ```bash
   # Verify required files exist
   ls -la samples/video.mp4 samples/mask.mp4 2>/dev/null || echo "Missing input files"
   ```

**Agent Response Pattern**:
- **Read the complete error message** - don't just look at the last line
- **Identify the root cause** - is it missing files, wrong paths, import errors, or parameter issues?
- **Apply the most upstream fix possible** - prefer fixes in the headless wrapper over Wan2GP modifications

### Step 3.3: Common Error Resolution Strategies

#### Strategy A: Missing Input Files
```bash
# Create dummy input files if missing
if [ ! -f "samples/video.mp4" ]; then
    echo "Creating dummy samples/video.mp4..."
    # Use ffmpeg to create a test video or copy from samples/
    cp samples/test.mp4 samples/video.mp4 2>/dev/null || echo "No sample video available"
fi

if [ ! -f "samples/mask.mp4" ]; then
    echo "Creating dummy samples/mask.mp4..."
    # Create a simple mask video
    cp samples/test.mp4 samples/mask.mp4 2>/dev/null || echo "No sample mask available"
fi
```

#### Strategy B: Parameter/Configuration Issues
```bash
# Check if model configs exist
ls -la Wan2GP/defaults/vace_14B*.json
ls -la Wan2GP/defaults/optimised-t2i.json

# Validate JSON syntax
python -c "import json; print('vace_14B:', json.load(open('Wan2GP/defaults/vace_14B.json')))"
```

**Agent Behavior**:
- If configs are missing, check if the model names in `test_model_comparison.py` need updating
- If JSON is malformed, attempt to fix or use a backup version
- If models don't exist, update the test to use available models

#### Strategy C: Memory/VRAM Issues
```bash
# Check available memory
free -h
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected"
```

**Agent Actions**:
- Reduce batch sizes in test configurations
- Lower resolution/video length parameters
- Enable CPU fallback if GPU memory insufficient

### Step 3.4: Fix Application Priority

**Priority 1 (Upstream - Preferred)**:
1. Fix paths in `headless_wgp.py` or `headless_model_management.py`
2. Update model configuration mappings
3. Adjust parameters in `test_model_comparison.py`
4. Install missing dependencies

**Priority 2 (Downstream - When Necessary)**:
1. Fix import paths in Wan2GP files
2. Patch configuration files in Wan2GP/defaults/
3. Update model loading logic in wgp.py (last resort)

### Step 3.5: Iteration Loop Protocol

```bash
# After each fix attempt:
echo "=== Fix Attempt $(date) ===" >> debug_log.txt
echo "Error: [describe the error]" >> debug_log.txt
echo "Fix Applied: [describe the fix]" >> debug_log.txt

# Re-run test
python test_model_comparison.py --output-dir outputs/test_attempt_$(date +%H%M%S)

# Record outcome
echo "Result: [SUCCESS/PARTIAL/FAILED]" >> debug_log.txt
echo "Next Action: [describe next step]" >> debug_log.txt
echo "" >> debug_log.txt
```

**Agent Persistence Rules**:
- Maximum 10 fix attempts per session
- If no progress after 5 attempts, take a different approach
- If system completely breaks, start fresh from Phase 1
- Always document what was tried

## Phase 4: Continuous Operation

### Step 4.1: Success Validation
```bash
# Verify all outputs were generated
ls -la outputs/model_comparison_*/
find outputs/ -name "*.mp4" -exec ls -lh {} \;

# Check results.json for success status
cat outputs/model_comparison_*/results.json | jq '.results[].status'
```

### Step 4.2: Monitoring Loop (Run Continuously)
```bash
#!/bin/bash
# monitoring_loop.sh

while true; do
    echo "=== Monitoring Run $(date) ==="
    
    # Run baseline test suite
    python test_model_comparison.py --output-dir outputs/monitor_$(date +%Y%m%d_%H%M%S)
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Baseline tests passed"
        sleep 3600  # Wait 1 hour before next check
    else
        echo "❌ Baseline tests failed, investigating..."
        # Apply fix strategies from Phase 3
        sleep 300   # Wait 5 minutes before retry
    fi
done
```

**Agent Behavior**:
- Run tests regularly (every 1-6 hours depending on stability)
- On failure, immediately apply Phase 3 debugging
- Keep logs of all runs for pattern analysis
- Auto-restart if system becomes completely unresponsive

### Step 4.3: Maintenance Tasks

#### Weekly Tasks:
```bash
# Check for upstream updates
cd Wan2GP/
git fetch origin
git log HEAD..origin/main --oneline

# If updates available, trigger fresh pull (Phase 1)
if [ "$(git rev-list HEAD..origin/main --count)" -gt 0 ]; then
    echo "Upstream updates detected, triggering refresh..."
    cd ..
    # Go back to Phase 1
fi
```

#### Daily Tasks:
```bash
# Clean old outputs
find outputs/ -name "test_*" -mtime +7 -exec rm -rf {} \;

# Rotate logs
if [ -f debug_log.txt ] && [ $(wc -l < debug_log.txt) -gt 1000 ]; then
    mv debug_log.txt debug_log_$(date +%Y%m%d).txt
    touch debug_log.txt
fi
```

## Phase 5: Advanced Troubleshooting

### Step 5.1: System Health Checks
```bash
# Comprehensive system diagnostic
python -c "
import torch
import sys
import subprocess
import os

print('=== System Diagnostic ===')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')

print(f'Working Dir: {os.getcwd()}')
print(f'Wan2GP Exists: {os.path.exists(\"Wan2GP/wgp.py\")}')
"
```

### Step 5.2: Emergency Recovery Procedures

**If system becomes completely unresponsive:**
1. Kill all Python processes: `pkill -f python`
2. Clear GPU memory: `nvidia-smi --gpu-reset` (if available)
3. Start fresh from Phase 1

**If Wan2GP repo becomes corrupted:**
1. `rm -rf Wan2GP/`
2. Clear pip cache: `pip cache purge`
3. Restart from Phase 1.2

**If dependencies break:**
1. Create new virtual environment
2. Install fresh dependencies
3. Test with minimal configuration first

## Behavioral Guidelines for AI Agents

### Core Principles:
1. **Be Persistent**: Don't give up after first failure
2. **Be Systematic**: Follow the phases in order
3. **Be Conservative**: Prefer upstream fixes over downstream changes
4. **Be Thorough**: Read complete error messages, not just summaries
5. **Be Documented**: Log every action and outcome

### Decision Making Framework:
1. **Analyze**: What exactly failed and why?
2. **Research**: Check logs, file existence, configurations
3. **Plan**: Choose the least invasive fix that addresses root cause
4. **Execute**: Apply fix and test immediately
5. **Verify**: Confirm fix worked and didn't break anything else
6. **Document**: Record what was done for future reference

### Escalation Triggers:
- More than 10 consecutive failures
- System errors that persist across fresh installs
- Hardware-related issues (GPU failure, insufficient memory)
- Upstream API changes that break core functionality

### Success Metrics:
- All 3 baseline models in `test_model_comparison.py` generate videos without errors
- Test completion time under 30 minutes total
- No memory leaks or resource exhaustion
- Stable operation for 24+ hours without intervention

## Example Conversation Flow

**Agent**: "Starting Headless-Wan2GP maintenance cycle. Analyzing current vs upstream Wan2GP differences..."

**Agent**: "Diff analysis complete. Found 15 modified files, 3 new files, 1 removed file. Documenting integration points..."

**Agent**: "Integration analysis complete. Our system uses 12 critical WGP functions. Generating update checklist..."

**Agent**: "Creating backup of current Wan2GP. Updating to latest upstream commit (abc123)..."

**Agent**: "Running baseline test suite (test_model_comparison.py)... vace_14B failed with 'CUDA out of memory'. Reducing video_length from 65 to 33 frames..."

**Agent**: "Retesting baseline... vace_14B now successful. All 3 baseline models working."

**Agent**: "All baseline tests passing. System is healthy and ready for video generation."

**Agent**: "Entering monitoring mode. Next baseline check in 60 minutes."

---

This guide ensures continuous, automated maintenance of the Headless-Wan2GP system with minimal human intervention while prioritizing stability and upstream Wan2GP compatibility.
