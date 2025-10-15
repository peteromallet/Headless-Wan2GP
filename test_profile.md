# Profile Switching Test

Current situation: Profile 3 loads to CPU instead of VRAM, causing OOM

## Recommendation: Try Profile 1

Profile 1 (HighRAM_HighVRAM):
- Forces everything into VRAM
- Most aggressive preloading
- Your 56GB RAM meets minimum (64GB recommended but should work)

## Test Command:
python worker.py --wgp-profile 1 [other args]

## Expected behavior with Profile 1:
- Models load directly to VRAM (not CPU)
- No "pinned to reserved RAM" messages
- VRAM usage: 16-20GB immediately
- Text encoding: 1-2 seconds

## If Profile 1 also fails:
- Try Profile 4 (known working, just slower)
- Or investigate why Profile 3 loads models to CPU
