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

## Phase 1: Wan2GP Analysis & Planning

### Step 1.1: Analyze Current vs Upstream Differences
**Goal**: Understand all changes before updating

**Agent Actions**:
- Clone fresh upstream Wan2GP to temporary location for comparison at `maintenance_analysis/tmp_upstream_Wan2GP/` (delete this temporary clone after diff is complete)
- Run diff analysis between current Wan2GP/ and upstream
- Document new files, removed files, and modified files
- Focus on key integration files: wgp.py, defaults/*.json, requirements.txt
- Create `maintenance_analysis/wan2gp_diff_analysis.md` with findings

### Step 1.2: Document System Integration Points  
**Goal**: Map how our system depends on Wan2GP

**Agent Actions**:
- Analyze `headless_model_management.py` - identify all WGP imports and function calls
- Analyze `headless_wgp.py` - identify critical WGP functions we use (generate_video, load_models, etc.)
- Analyze `worker.py` - identify how it integrates with WGP and any critical dependencies
- Document model config dependencies (defaults/*.json files)
- Document path/import dependencies (Python path changes, working directory)
- Create `maintenance_analysis/system_integration_analysis.md`

### Step 1.3: Generate Update Checklist & Progress Tracking
**Goal**: Create actionable checklist and progress tracking documents

**Agent Actions**:
- Based on diff analysis: identify files that need our attention
- Based on integration analysis: identify functions/APIs that might break
- Create verification items for each integration point
- Include post-update validation steps
- Create `maintenance_analysis/update_checklist.md`
- **Create `maintenance_analysis/milestone_progress.txt`** - for major milestone tracking
- **Use `attempt_log.txt`** - for detailed attempt summaries (structured logging format)

**CRITICAL**: These become your living documents throughout the maintenance process. Update them after every appropriate action and regularly sense-check your progress against them.

### Step 1.4: Clean Slate - Update Wan2GP
**Goal**: Replace current Wan2GP with latest upstream

**Agent Actions**:
- Create backup of current Wan2GP directory at `backups/Wan2GP_$(date +%Y%m%d_%H%M%S)/` (for rollback)
- Remove current Wan2GP directory completely  
- Clone fresh from https://github.com/deepbeepmeep/Wan2GP.git
- Record new commit hash for tracking

### Step 1.5: Update Documentation
**Goal**: Record the update for tracking

**Agent Actions**:
- Update STRUCTURE.md with new Wan2GP commit hash and timestamp
- Remove only the temporary upstream clone directory (`maintenance_analysis/tmp_upstream_Wan2GP/`); keep analysis documents
- Document update completion

## Phase 2: Environment Validation

### Step 2.1: Python Environment Setup
```bash
# Verify Python version
python --version
python3 --version

# Check if we're in the right directory (should be in Headless-Wan2GP root)
pwd
ls -la | grep -E "(test_model_comparison\.py|headless_wgp\.py|worker\.py|Wan2GP/)"

# Create and activate virtual environment for isolation
python3 -m venv venv_headless_wan2gp
source venv_headless_wan2gp/bin/activate  # On Windows: venv_headless_wan2gp\Scripts\activate
```

**Agent Behavior**:
- **Always create a fresh virtual environment** to avoid dependency conflicts
- Ensure Python 3.8+ is available
- If virtual environment creation fails, try `python -m venv` instead of `python3 -m venv`
- Activate the virtual environment before proceeding to dependency installation

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
- Use `python -m pip` inside the active virtual environment if `pip` alias is ambiguous
- If dependency conflicts occur, document them and attempt resolution (pin versions or recreate venv if needed)
- Consider using `pip install --upgrade` for problematic packages

## Phase 3: Iterative Testing & Fixing Loop

### Step 3.1: Initial Baseline Test Run
```bash
# Run the baseline model comparison test
mkdir -p outputs
python test_model_comparison.py --output-dir outputs/baseline_test_$(date +%Y%m%d_%H%M%S)
```

**Expected Outcomes**:
- ✅ **Success**: All 3 models (vace_14B, vace_14B_fake_cocktail_2_2, optimised-t2i) generate videos → Log success in `maintenance_analysis/milestone_progress.txt` and wrap up Phase 3
- ⚠️ **Partial Success**: Some models work → Analyze and fix failing ones
- ❌ **Complete Failure**: System won't start → Debug environment issues

### Step 3.2: Log Analysis Protocol

**When tests fail, analyze logs systematically**

**Agent Response Pattern**:
- **Read the complete error message** - don't just look at the last line
- **Identify the root cause** - is it missing files, wrong paths, import errors, or parameter issues?
- **🚨 CRITICAL: Apply the most upstream fix possible** - prefer fixes in the headless wrapper over Wan2GP modifications. This is EXTREMELY important to maintain system stability and upgradeability.

### Step 3.3: Fix Application Priority

**🚨 EXTREMELY IMPORTANT: Focus UPSTREAM of Wan2GP as much as possible! 🚨**

**Priority 1 (Upstream - STRONGLY Preferred)**:
1. Fix paths in `headless_wgp.py` or `headless_model_management.py`
2. Update model configuration mappings in our wrapper code
3. Adjust parameters in `test_model_comparison.py`
4. Install missing dependencies
5. Modify `worker.py` integration logic

**Priority 2 (Downstream - AVOID if possible, use only when absolutely necessary)**:
1. Fix import paths in Wan2GP files
2. Patch configuration files in Wan2GP/defaults/
3. Update model loading logic in wgp.py (absolute last resort)

**Why Upstream Fixes Matter**: 
- Preserves ability to update Wan2GP submodule cleanly
- Prevents merge conflicts on future updates
- Maintains clear separation between our wrapper and upstream code
- Makes debugging and maintenance significantly easier

### Step 3.4: Iteration Loop Protocol

**Each Attempt Must Be Clearly Documented In:**

```bash
# BEFORE each fix attempt - log the plan
ATTEMPT_NUM=$(( $(grep -c "=== Fix Attempt" attempt_log.txt 2>/dev/null || echo 0) + 1 ))
echo "=== Fix Attempt #${ATTEMPT_NUM} - $(date) ===" >> attempt_log.txt
echo "PROBLEM: [1-line description of what failed]" >> attempt_log.txt
echo "ROOT CAUSE: [your analysis of why it failed]" >> attempt_log.txt
echo "FIX STRATEGY: [specific action you're taking]" >> attempt_log.txt
echo "EXPECTED OUTCOME: [what should happen if this works]" >> attempt_log.txt

# Apply the fix
[your actual fix commands here]

# Re-run test
echo "TESTING: Running baseline test..." >> attempt_log.txt
mkdir -p outputs
python test_model_comparison.py --output-dir outputs/attempt_${ATTEMPT_NUM}_$(date +%H%M%S)
EXIT_CODE=$?

# AFTER test - record detailed outcome
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ RESULT: SUCCESS - All tests passed" >> attempt_log.txt
    echo "NEXT ACTION: Proceeding to validation phase" >> attempt_log.txt
else
    echo "❌ RESULT: FAILED (exit code: $EXIT_CODE)" >> attempt_log.txt
    echo "NEW ERROR: [describe what failed this time]" >> attempt_log.txt
    echo "ANALYSIS: [is this the same error, different error, or partial progress?]" >> attempt_log.txt
    echo "NEXT ACTION: [specific next step based on this outcome]" >> attempt_log.txt
fi
echo "CHECKLIST STATUS: [brief update on overall progress]" >> attempt_log.txt
echo "SANITY CHECK: Did I prioritize upstream fixes? [YES/NO + brief justification]" >> attempt_log.txt
echo "PHASE ALIGNMENT: Am I following the prescribed phases? [YES/NO + current phase/step]" >> attempt_log.txt
echo "----------------------------------------" >> attempt_log.txt
echo "" >> attempt_log.txt
```

**Agent Must Provide Clear Summary After Each Attempt:**
- **Before fixing**: State the problem, root cause, and strategy
- **After testing**: Report exact outcome and analysis
- **Progress tracking**: Update on overall mission progress
- **Next steps**: Clear plan for the next attempt

**Conversational Reporting (to User):**
After each attempt, provide a succinct but complete summary:

```
🔧 Attempt #3: [Brief description of what you tried]
❌ Result: [What happened - success/failure/partial]
📊 Analysis: [Key insight about why it failed/succeeded]
⏭️ Next: [What you'll try next, or if moving to next phase]
📋 Progress: [X/Y items completed on checklist]
```

**Attempt Summary Captured in `attempt_log.txt`:**
The bash logging protocol above captures all necessary attempt details in a structured format that appends to `attempt_log.txt`. This creates a complete chronological record of all attempts.

**Document Update Required After Each Each Achievement:**

**Update `maintenance_analysis/milestone_progress.txt`** (when reaching major milestones):
```
=== [Phase/Milestone Name] - [Timestamp] ===
STATUS: [COMPLETED/IN_PROGRESS/BLOCKED]
KEY ACHIEVEMENTS:
- [Achievement 1]
- [Achievement 2]
BLOCKERS RESOLVED:
- [Blocker that was overcome]
OVERALL PROGRESS WITH REASONING:
- [Brief Update on Progress]
NEXT MAJOR MILESTONE: [What's next]
========================================

```

**Major Milestones Requiring Milestone Document Updates:**
- ✅ **Phase 1 Complete**: All analysis documents created, Wan2GP updated
- ✅ **Environment Ready**: Virtual environment set up, all dependencies installed
- ✅ **First Successful Test**: At least one model generates video successfully
- ✅ **All Models Working**: Complete baseline test suite passes
- ✅ **Integration Verified**: All headless wrapper functions working with updated Wan2GP
- ✅ **System Stable**: Multiple consecutive successful test runs
- ❌ **Critical Blocker Resolved**: Any major obstacle that was blocking progress
- ❌ **Recovery from Failure**: Successfully recovered from system-breaking error

**Agent Persistence Rules**:
_Definition_: A "loop" refers to a full Phase 3 cycle from the initial baseline run through successive fix attempts until success, reset, or escalation.
- Maximum 10 fix attempts per loop
- If no progress after 5 attempts, take a different approach
- If system completely breaks, start fresh from Phase 1
- Always document what was tried
- **Update your checklist after each attempt** - mark progress, add new discovered issues, reassess priorities
- **All attempt details are automatically logged to `attempt_log.txt`** via the structured bash logging protocol
- **Update `milestone_progress.txt` after major milestones** - track phase completions, significant breakthroughs
- **Sanity check after each attempt**: Answer YES/NO with justification - Did I prioritize upstream fixes? Am I following the prescribed phases?
- **After each cycle of changes**: Sense-check your current approach against this base document - are you following the prescribed phases and protocols?

## Phase 5: Advanced Troubleshooting

### Step 5.1: Emergency Recovery Procedures

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
3. **🚨 Be Conservative: ALWAYS prefer upstream fixes over downstream changes** - This is CRITICAL for maintainability
4. **Be Thorough**: Read complete error messages, not just summaries
5. **Be Documented**: Log every action and outcome
6. **Maintain Living Checklist**: Update your checklist after every completed task and regularly sense-check your progress
7. **Focus UPSTREAM**: When in doubt, fix in headless wrapper code, NOT in Wan2GP submodule

### Checklist Management Protocol:
**After Every Task Completion**:
1. **Update Status**: Mark completed items as ✅ DONE in your checklist
2. **Add New Items**: If you discovered additional tasks, add them to the checklist
3. **Sense-Check Progress**: Review the overall checklist - are you on track? Missing anything critical?
4. **Prioritize Remaining**: Re-order remaining tasks by importance/dependencies
5. **Estimate Completion**: Assess how much work remains

**Regular Sense-Check Questions**:
- Am I making meaningful progress toward the core mission (functional Wan2GP update)?
- Have I been stuck on the same issue for too long without trying a different approach?
- Are there critical integration points I haven't verified yet?
- Is my current approach the most efficient path to success?
- **After each cycle of changes**: Does my current state align with the guidance in this base document? Am I following the prescribed phases and protocols?

### Decision Making Framework:
1. **Analyze**: What exactly failed and why?
2. **Research**: Check logs, file existence, configurations
3. **Plan**: Choose the least invasive fix that addresses root cause
4. **Execute**: Apply fix and test immediately
5. **Verify**: Confirm fix worked and didn't break anything else
6. **Document**: Record what was done for future reference
7. **Update Checklist**: Mark completion and add any new discovered tasks
8. **Update Milestones** (if applicable): Add entry to `maintenance_analysis/milestone_progress.txt` for major achievements
9. **Sanity Check**: Answer YES/NO with justification - Did I prioritize upstream fixes? Am I following the prescribed phases?
10. **Sense-Check vs Base Document**: After each cycle of changes, verify you're still following the prescribed phases and protocols in this guide

*Note: Attempt details are automatically captured in `attempt_log.txt` via the structured bash logging protocol.*

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

**Agent**: "Logging milestone and concluding the maintenance cycle."

---

This guide ensures continuous, automated maintenance of the Headless-Wan2GP system with minimal human intervention while prioritizing stability and upstream Wan2GP compatibility.
