# Uni3C Integration — Starting Point & Status

> **This is the entry point.** All Uni3C docs are in this folder.

---

## 🎬 START HERE

**Ready to implement?** → Open **[Phase 1: Port ControlNet](./PHASE_1_PORT_CONTROLNET.md)**

---

## 📊 Project Status Dashboard

> **Last Updated**: _[DATE]_  
> **Overall Status**: 🟡 Planning Complete / Implementation Not Started  
> **Current Phase**: Phase 0 (Pre-Implementation)  
> **Blocking Issues**: None  
> **Next Action**: Begin Phase 1 - Port ControlNet architecture

### Progress Summary

| Phase | Description | Status | Est. Days | Owner | Doc |
|-------|-------------|--------|-----------|-------|-----|
| 0 | Planning & Validation | ✅ Done | - | - | - |
| 1 | Port Uni3C ControlNet | 🔴 Not Started | 2-3 | TBD | [→ Phase 1](./PHASE_1_PORT_CONTROLNET.md) |
| 2 | Guide Video → Latents | 🔴 Not Started | 1 | TBD | [→ Phase 2](./PHASE_2_GUIDE_VIDEO_LATENTS.md) |
| 3 | Model Integration | 🔴 Not Started | 1-2 | TBD | [→ Phase 3](./PHASE_3_MODEL_INTEGRATION.md) |
| 4 | Headless Param Wiring | 🔴 Not Started | 0.5 | TBD | [→ Phase 4](./PHASE_4_HEADLESS_WIRING.md) |
| 5 | Testing & Validation | 🔴 Not Started | 1-2 | TBD | [→ Phase 5](./PHASE_5_TESTING.md) |

**Legend**: ✅ Done | 🟢 On Track | 🟡 In Progress | 🟠 Blocked | 🔴 Not Started

---

## 🚀 Quick Start: What Do I Do?

| If you're... | Do this |
|--------------|---------|
| **Starting implementation** | 1. Open [Phase 1 doc](./PHASE_1_PORT_CONTROLNET.md)<br>2. Assign yourself as Owner in the task table<br>3. Change first task status from 🔴 → 🟡<br>4. Update "Last Updated" date above<br>5. Start coding |
| **Resuming work** | 1. Check Progress Summary above for current phase<br>2. Open that phase's doc<br>3. Find 🟡 task, complete it, mark ✅<br>4. Update "Last Updated" date |
| **Checking progress** | Look at Progress Summary table above |
| **Blocked** | In phase doc: change task to 🟠, add note. Here: update "Blocking Issues" |
| **Finishing a phase** | Update this doc's Progress Summary, then open next phase doc |

---

## 🎯 Definition of Done (Acceptance Criteria)

For Uni3C integration to be considered **complete**, ALL of the following must pass:

### Must Have
- [ ] Task with `use_uni3c=true` + `uni3c_guide_video` produces visually different output than same task without
- [ ] Guide video motion is reflected in generated output (human-verified)
- [ ] `[UNI3C]` logs appear at all 6 layers showing params flowed through
- [ ] No silent param drops (deliberately broken param triggers warning log)
- [ ] Works with existing `wan_2_2_i2v_lightning_baseline_3_3` preset

### Should Have
- [ ] `uni3c_strength=0` produces output identical to `use_uni3c=false`
- [ ] `uni3c_start_percent` / `uni3c_end_percent` correctly gate application window
- [ ] Guide video with different frame count than output still works (frame policy)

### Won't Have (Deferred)
- render_mask support
- camera_embedding support
- Custom Uni3C checkpoint path override

---

## 🚨 Risk Register

| Risk | Severity | Status | Mitigation | Owner |
|------|----------|--------|------------|-------|
| Checkpoint weight mismatch | High | ✅ Mitigated | Using Kijai's verified fp16 checkpoint | - |
| Silent param filtering | High | 🟡 Open | 6-layer logging strategy defined; must implement | TBD |
| VRAM overflow | Medium | 🟡 Open | Offload flag; needs testing on target GPU | TBD |
| temb shape mismatch (diffusion-forcing) | Medium | 🟡 Open | Guard for `_flag_df` case identified | TBD |
| 16→20 channel padding needed | Low | 🟡 Open | Padding code pattern identified from Kijai | TBD |

---

## 📚 Reference Documents

| Doc | Purpose | When to Use |
|-----|---------|-------------|
| [Sense Check](./_reference/SENSE_CHECK.md) | Validation of plan against source implementations | Verifying assumptions |
| [Kijai Appendix](./_reference/KIJAI_APPENDIX.md) | Code snippets from Kijai's ComfyUI impl | Porting code |
| [Parameter Definitions](./_reference/PARAM_DEFINITIONS.md) | Uni3C parameter table and defaults | API design |

---

## 📝 How to Update This Doc

| Event | What to Update |
|-------|----------------|
| Starting a phase | Change phase status in Progress Summary |
| Phase complete | Change phase status to ✅, update "Current Phase" |
| Blocked | Update "Blocking Issues" field |
| Risk status change | Update Risk Register |
| Definition of Done item achieved | Check the box |
| End of day/session | Update "Last Updated" date |

**Status Legend**:
```
🔴 Not Started    - Work hasn't begun
🟡 In Progress    - Actively being worked on  
🟢 On Track       - Phase/overall status healthy
🟠 Blocked        - Can't proceed; needs resolution
✅ Done           - Complete and verified
```

