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

### Step 1.1: Analyze Current vs Upstream Wan2GP Differences
```bash
# Verify we're in the Headless-Wan2GP directory
pwd
ls -la | grep -E "(AI_AGENT_MAINTENANCE_GUIDE\.md|STRUCTURE\.md|worker\.py)"

# Create analysis directory for documentation
mkdir -p maintenance_analysis
cd maintenance_analysis

# Clone fresh upstream for comparison (temporary)
echo "Cloning fresh upstream Wan2GP for comparison..."
git clone https://github.com/deepbeepmeep/Wan2GP.git upstream_wan2gp_temp
cd ..

# Generate comprehensive diff analysis
echo "=== WAN2GP UPDATE ANALYSIS $(date) ===" > maintenance_analysis/wan2gp_diff_analysis.md
echo "" >> maintenance_analysis/wan2gp_diff_analysis.md

# Compare directory structures
echo "## Directory Structure Changes" >> maintenance_analysis/wan2gp_diff_analysis.md
echo "\`\`\`bash" >> maintenance_analysis/wan2gp_diff_analysis.md
echo "# Current Wan2GP structure:" >> maintenance_analysis/wan2gp_diff_analysis.md
find Wan2GP -type f -name "*.py" | head -20 >> maintenance_analysis/wan2gp_diff_analysis.md
echo "" >> maintenance_analysis/wan2gp_diff_analysis.md
echo "# Upstream Wan2GP structure:" >> maintenance_analysis/wan2gp_diff_analysis.md
find maintenance_analysis/upstream_wan2gp_temp -type f -name "*.py" | head -20 >> maintenance_analysis/wan2gp_diff_analysis.md
echo "\`\`\`" >> maintenance_analysis/wan2gp_diff_analysis.md
echo "" >> maintenance_analysis/wan2gp_diff_analysis.md

# Compare key integration files
echo "## Key File Changes Analysis" >> maintenance_analysis/wan2gp_diff_analysis.md
for file in "wgp.py" "defaults" "requirements.txt"; do
    if [ -f "Wan2GP/$file" ] && [ -f "maintenance_analysis/upstream_wan2gp_temp/$file" ]; then
        echo "### Changes in $file:" >> maintenance_analysis/wan2gp_diff_analysis.md
        echo "\`\`\`diff" >> maintenance_analysis/wan2gp_diff_analysis.md
        diff -u "Wan2GP/$file" "maintenance_analysis/upstream_wan2gp_temp/$file" | head -50 >> maintenance_analysis/wan2gp_diff_analysis.md
        echo "\`\`\`" >> maintenance_analysis/wan2gp_diff_analysis.md
        echo "" >> maintenance_analysis/wan2gp_diff_analysis.md
    fi
done

# Analyze new files
echo "## New Files in Upstream" >> maintenance_analysis/wan2gp_diff_analysis.md
echo "\`\`\`bash" >> maintenance_analysis/wan2gp_diff_analysis.md
comm -13 <(find Wan2GP -name "*.py" | sort) <(find maintenance_analysis/upstream_wan2gp_temp -name "*.py" | sort) >> maintenance_analysis/wan2gp_diff_analysis.md
echo "\`\`\`" >> maintenance_analysis/wan2gp_diff_analysis.md

# Analyze removed files  
echo "## Files Removed from Upstream" >> maintenance_analysis/wan2gp_diff_analysis.md
echo "\`\`\`bash" >> maintenance_analysis/wan2gp_diff_analysis.md
comm -23 <(find Wan2GP -name "*.py" | sort) <(find maintenance_analysis/upstream_wan2gp_temp -name "*.py" | sort) >> maintenance_analysis/wan2gp_diff_analysis.md
echo "\`\`\`" >> maintenance_analysis/wan2gp_diff_analysis.md

echo "Diff analysis complete. Review maintenance_analysis/wan2gp_diff_analysis.md"
```

**Agent Behavior**:
- Thoroughly analyze ALL differences before making changes
- Document new files, removed files, and modified files
- Pay special attention to changes in wgp.py, model definitions, and requirements
- Create detailed analysis for review before proceeding

### Step 1.2: Document Current System Integration
```bash
# Create integration analysis document
echo "=== HEADLESS-WAN2GP INTEGRATION ANALYSIS $(date) ===" > maintenance_analysis/system_integration_analysis.md
echo "" >> maintenance_analysis/system_integration_analysis.md

echo "## Our Integration Points with Wan2GP" >> maintenance_analysis/system_integration_analysis.md
echo "" >> maintenance_analysis/system_integration_analysis.md

echo "### 1. headless_model_management.py Integration" >> maintenance_analysis/system_integration_analysis.md
echo "- **Purpose**: Task queue manager that wraps WGP functionality" >> maintenance_analysis/system_integration_analysis.md
echo "- **Key WGP Dependencies**:" >> maintenance_analysis/system_integration_analysis.md
grep -n "import.*wgp\|from.*wgp\|wgp\." headless_model_management.py | head -10 >> maintenance_analysis/system_integration_analysis.md
echo "" >> maintenance_analysis/system_integration_analysis.md

echo "### 2. headless_wgp.py Integration" >> maintenance_analysis/system_integration_analysis.md  
echo "- **Purpose**: Direct orchestrator wrapper around wgp.generate_video()" >> maintenance_analysis/system_integration_analysis.md
echo "- **Key WGP Dependencies**:" >> maintenance_analysis/system_integration_analysis.md
grep -n "import.*wgp\|from.*wgp\|wgp\." headless_wgp.py | head -10 >> maintenance_analysis/system_integration_analysis.md
echo "" >> maintenance_analysis/system_integration_analysis.md

echo "### 3. Critical WGP Functions We Use" >> maintenance_analysis/system_integration_analysis.md
echo "\`\`\`python" >> maintenance_analysis/system_integration_analysis.md
echo "# From headless_wgp.py:" >> maintenance_analysis/system_integration_analysis.md
grep -A 2 -B 2 "generate_video\|load_models\|get_model_def\|test_vace_module" headless_wgp.py | head -20 >> maintenance_analysis/system_integration_analysis.md
echo "\`\`\`" >> maintenance_analysis/system_integration_analysis.md
echo "" >> maintenance_analysis/system_integration_analysis.md

echo "### 4. Model Configuration Dependencies" >> maintenance_analysis/system_integration_analysis.md
echo "- We rely on Wan2GP/defaults/*.json files for model configurations" >> maintenance_analysis/system_integration_analysis.md
echo "- Current model configs we use:" >> maintenance_analysis/system_integration_analysis.md
ls -la Wan2GP/defaults/*.json | head -10 >> maintenance_analysis/system_integration_analysis.md
echo "" >> maintenance_analysis/system_integration_analysis.md

echo "### 5. Path and Import Dependencies" >> maintenance_analysis/system_integration_analysis.md
echo "- Our system adds Wan2GP to Python path and changes working directory" >> maintenance_analysis/system_integration_analysis.md
echo "- This is critical for WGP's relative path assumptions" >> maintenance_analysis/system_integration_analysis.md

echo "Integration analysis complete. Review maintenance_analysis/system_integration_analysis.md"
```

### Step 1.3: Generate Update Checklist
```bash
# Create comprehensive update checklist based on analysis
echo "=== WAN2GP UPDATE CHECKLIST $(date) ===" > maintenance_analysis/update_checklist.md
echo "" >> maintenance_analysis/update_checklist.md

echo "## Pre-Update Analysis" >> maintenance_analysis/update_checklist.md
echo "- [ ] Diff analysis completed (wan2gp_diff_analysis.md)" >> maintenance_analysis/update_checklist.md
echo "- [ ] Integration analysis completed (system_integration_analysis.md)" >> maintenance_analysis/update_checklist.md
echo "- [ ] Backup current working system created" >> maintenance_analysis/update_checklist.md
echo "" >> maintenance_analysis/update_checklist.md

echo "## Critical Integration Points to Verify" >> maintenance_analysis/update_checklist.md
echo "- [ ] wgp.py generate_video() function signature unchanged" >> maintenance_analysis/update_checklist.md
echo "- [ ] wgp.py load_models() function still works with our orchestrator" >> maintenance_analysis/update_checklist.md
echo "- [ ] Model definition loading (get_model_def) still compatible" >> maintenance_analysis/update_checklist.md
echo "- [ ] VACE module detection (test_vace_module) still works" >> maintenance_analysis/update_checklist.md
echo "- [ ] LoRA discovery and loading mechanisms unchanged" >> maintenance_analysis/update_checklist.md
echo "- [ ] Model family detection functions still available" >> maintenance_analysis/update_checklist.md
echo "" >> maintenance_analysis/update_checklist.md

echo "## File-Specific Updates Needed" >> maintenance_analysis/update_checklist.md
echo "### Based on diff analysis:" >> maintenance_analysis/update_checklist.md
echo "- [ ] Review new Python files for integration impacts" >> maintenance_analysis/update_checklist.md
echo "- [ ] Check if removed files affect our imports" >> maintenance_analysis/update_checklist.md
echo "- [ ] Update any hardcoded paths or assumptions" >> maintenance_analysis/update_checklist.md
echo "- [ ] Verify model config JSON files still compatible" >> maintenance_analysis/update_checklist.md
echo "" >> maintenance_analysis/update_checklist.md

echo "## Our System Updates Required" >> maintenance_analysis/update_checklist.md
echo "### headless_model_management.py:" >> maintenance_analysis/update_checklist.md
echo "- [ ] Update WGP import statements if needed" >> maintenance_analysis/update_checklist.md
echo "- [ ] Verify task queue integration still works" >> maintenance_analysis/update_checklist.md
echo "- [ ] Check LoRA processing pipeline compatibility" >> maintenance_analysis/update_checklist.md
echo "" >> maintenance_analysis/update_checklist.md
echo "### headless_wgp.py:" >> maintenance_analysis/update_checklist.md
echo "- [ ] Update generate_video() call parameters if changed" >> maintenance_analysis/update_checklist.md
echo "- [ ] Verify model loading pattern still matches WGP" >> maintenance_analysis/update_checklist.md
echo "- [ ] Check parameter resolution logic compatibility" >> maintenance_analysis/update_checklist.md
echo "" >> maintenance_analysis/update_checklist.md
echo "### test_model_comparison.py:" >> maintenance_analysis/update_checklist.md
echo "- [ ] Update model names if changed in upstream" >> maintenance_analysis/update_checklist.md
echo "- [ ] Verify test parameters still valid" >> maintenance_analysis/update_checklist.md
echo "" >> maintenance_analysis/update_checklist.md

echo "## Post-Update Validation" >> maintenance_analysis/update_checklist.md
echo "- [ ] All 3 baseline models load successfully" >> maintenance_analysis/update_checklist.md
echo "- [ ] test_model_comparison.py runs without errors" >> maintenance_analysis/update_checklist.md
echo "- [ ] VACE generation works with video guides" >> maintenance_analysis/update_checklist.md
echo "- [ ] T2V generation works without video guides" >> maintenance_analysis/update_checklist.md
echo "- [ ] LoRA loading and application works" >> maintenance_analysis/update_checklist.md
echo "- [ ] No import errors in any of our modules" >> maintenance_analysis/update_checklist.md

echo "Update checklist created. Review maintenance_analysis/update_checklist.md"
```

**Agent Behavior**:
- Generate checklist based on actual diff analysis findings
- Include specific items for each integration point discovered
- Create actionable items that can be verified after update
- Focus on our system's specific dependencies on WGP

### Step 1.4: Clean Slate - Remove Current Wan2GP
```bash
# Backup current working Wan2GP for emergency rollback
echo "Creating backup of current Wan2GP..."
cp -r Wan2GP Wan2GP_backup_$(date +%Y%m%d_%H%M%S)

# Remove current Wan2GP directory to get fresh upstream
echo "Removing current Wan2GP directory..."
rm -rf Wan2GP/

# Clean up temporary analysis directory
rm -rf maintenance_analysis/upstream_wan2gp_temp

# Verify removal
ls -la | grep -i wan2gp
```

**Agent Behavior**: 
- ALWAYS create backup before deletion for emergency rollback
- This ensures we get the latest upstream version without any local modifications
- If removal fails, try `sudo rm -rf Wan2GP/` 
- If still fails, note the specific error and attempt file-by-file deletion
- Continue only when Wan2GP directory is completely gone

### Step 1.5: Fresh Clone from Upstream Wan2GP
```bash
# Clone latest from upstream deepbeepmeep/Wan2GP repository
echo "Cloning fresh upstream Wan2GP..."
git clone https://github.com/deepbeepmeep/Wan2GP.git

# Verify clone success and record version info
cd Wan2GP/
git status
git log --oneline -5
current_commit=$(git rev-parse HEAD)
echo "Updated to Wan2GP commit: $current_commit"
cd ..

# Document the update in our analysis
echo "## Update Completed" >> maintenance_analysis/wan2gp_diff_analysis.md
echo "- **Previous version**: (backed up as Wan2GP_backup_*)" >> maintenance_analysis/wan2gp_diff_analysis.md
echo "- **New version commit**: $current_commit" >> maintenance_analysis/wan2gp_diff_analysis.md
echo "- **Update timestamp**: $(date)" >> maintenance_analysis/wan2gp_diff_analysis.md
```

**Agent Behavior**:
- This is the **critical step** - getting the latest upstream Wan2GP code
- If clone fails, wait 30 seconds and retry up to 3 times
- Record the commit hash for tracking purposes and documentation
- Ensure we're using the absolute latest version from deepbeepmeep/Wan2GP

### Step 1.6: Update Structure Documentation
```bash
# Return to project root
cd ..

# Update structure.md with current Wan2GP commit
echo "# Last Wan2GP Update: $(date)" >> STRUCTURE.md
echo "Commit: $(cd Wan2GP && git rev-parse HEAD)" >> STRUCTURE.md
```

**Agent Behavior**:
- Always document what was done for future reference
- Include timestamp and commit hash for tracking

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
   ls -la video.mp4 mask.mp4 2>/dev/null || echo "Missing input files"
   ```

**Agent Response Pattern**:
- **Read the complete error message** - don't just look at the last line
- **Identify the root cause** - is it missing files, wrong paths, import errors, or parameter issues?
- **Apply the most upstream fix possible** - prefer fixes in the headless wrapper over Wan2GP modifications

### Step 3.3: Common Error Resolution Strategies

#### Strategy A: Missing Input Files
```bash
# Create dummy input files if missing
if [ ! -f "video.mp4" ]; then
    echo "Creating dummy video.mp4..."
    # Use ffmpeg to create a test video or copy from samples/
    cp samples/test.mp4 video.mp4 2>/dev/null || echo "No sample video available"
fi

if [ ! -f "mask.mp4" ]; then
    echo "Creating dummy mask.mp4..."
    # Create a simple mask video
    cp samples/test.mp4 mask.mp4 2>/dev/null || echo "No sample mask available"
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

**Agent**: "Integration analysis complete. Our system uses 12 critical WGP functions. Generating update checklist with 28 verification items..."

**Agent**: "Creating backup of current Wan2GP. Updating to latest upstream commit (abc123)..."

**Agent**: "Wan2GP updated successfully. Installing dependencies and running baseline test suite..."

**Agent**: "Running baseline test suite (test_model_comparison.py)... vace_14B failed with 'CUDA out of memory'. Reducing video_length from 65 to 33 frames..."

**Agent**: "Retesting baseline... vace_14B now successful. All 3 baseline models working. Verifying integration checklist..."

**Agent**: "✅ All 28 checklist items verified. System is healthy and ready for video generation."

**Agent**: "Entering monitoring mode. Next baseline check in 60 minutes."

---

This guide ensures continuous, automated maintenance of the Headless-Wan2GP system with minimal human intervention while prioritizing stability and upstream Wan2GP compatibility.
