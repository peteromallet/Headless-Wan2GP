# Uni3C Integration — Starting Point & Status

> **This is the entry point.** All Uni3C docs are in this folder.

---

## 🎬 START HERE

**Ready to implement?** → Open **[Phase 1: Port ControlNet](./PHASE_1_PORT_CONTROLNET.md)**

---

## 📊 Project Status Dashboard

> **Last Updated**: 2026-01-11  
> **Overall Status**: 🟡 Implementation In Progress  
> **Current Phase**: Phase 5 (Testing & Validation)  
> **Blocking Issues**: None  
> **Next Action**: Create test task with `use_uni3c=true` and verify logs at all 6 layers

### Progress Summary

| Phase | Description | Status | Est. Days | Owner | Doc |
|-------|-------------|--------|-----------|-------|-----|
| 0 | Planning & Validation | ✅ Done | - | - | - |
| 1 | Port Uni3C ControlNet | ✅ Done | 2-3 | - | [→ Phase 1](./PHASE_1_PORT_CONTROLNET.md) |
| 2 | Guide Video → Latents | ✅ Done | 1 | - | [→ Phase 2](./PHASE_2_GUIDE_VIDEO_LATENTS.md) |
| 3 | Model Integration | ✅ Done | 1-2 | - | [→ Phase 3](./PHASE_3_MODEL_INTEGRATION.md) |
| 4 | Headless Param Wiring | ✅ Done | 0.5 | - | [→ Phase 4](./PHASE_4_HEADLESS_WIRING.md) |
| 5 | Testing & Validation | 🟡 In Progress | 1-2 | - | [→ Phase 5](./PHASE_5_TESTING.md) |

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
| Silent param filtering | High | ✅ Mitigated | 6-layer logging implemented (Layer 1-3 in Phase 4, Layer 4-6 in Phase 2-3) | - |
| VRAM overflow | Medium | ✅ Mitigated | Offload flag implemented in `_compute_uni3c_states()` | - |
| temb shape mismatch (diffusion-forcing) | Medium | ✅ Mitigated | Guard added in `_compute_uni3c_states()`: `if temb.dim() == 1: temb = temb.unsqueeze(0)` | - |
| 16→20 channel padding needed | Low | ✅ Mitigated | Padding implemented in `_compute_uni3c_states()` as fallback | - |
| Temporal/spatial grid mismatch | Medium | ✅ Mitigated | Trilinear interpolation of render_latent implemented | - |
| Hidden-dim mismatch | Low | ✅ Mitigated | Guard added at injection site; logs warning and skips | - |

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

