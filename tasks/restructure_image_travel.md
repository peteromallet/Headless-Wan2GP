# Restructuring plan: Travel-Between-Images → Fully-Queued Workflow

This document describes the **structural changes** required to let `steerable_motion.travel_between_images` enqueue _all_ segment tasks at once and let `headless.py` take care of creating guide videos & chaining, instead of the current _serial_ approach.

---

## 1. Goals

1. **No per-segment polling in `travel_between_images.py`**  – it should finish quickly after pushing **N** segment tasks (and one orchestrator task) into the DB.
2. **`headless.py` becomes responsible for the dependency chain**:
   • wait for a segment's video to finish
   • create its guide artefacts for the _next_ segment
   • enqueue the next segment automatically
3. Maintain existing CLI/behaviour for callers (only runtime improves).


## 2. High-Level Design

```
steerable_motion.py      headless.py (server loop)              wgp.py
┌──────────────┐        ┌────────────────────────────┐        ┌─────────┐
│travel_between│ 1. N+1 │  (A) orchestrator task      │        │ video   │
│   images     │ push → │  (B) segment-0 task         │  ----> │gen.     │
└──────────────┘        └────────────┬───────────────┘        └─────────┘
                                    2. segment-0 done
                                     ├─▶ create guide-1
                                     ├─▶ enqueue segment-1
                                     └─▶ repeat …
```

* **Orchestrator task** (new `task_type = "travel_orchestrator"`) carries the full sequence definition (image list, prompts, overlaps, etc.).
* **Segment task** (`task_type = "travel_segment"`) generates the actual video for a single leg.  It stores its output location in DB for successor lookup.


## 3. Database / Schema Updates

1. **tasks table** already has `task_type` – good.
2. Add **`payload` JSONB fields** (optional) if we need to store large sequencing info separate from `params` (optional).
3. No blocking change if we piggy-back on existing `params`.


## 4. Changes in `travel_between_images.py`

### 4.1  Remove per-segment loop / polling
* Build a single `orchestrator_payload` containing:
  * ordered `input_images`
  * `{base_prompts, negative_prompts, segment_frames, frame_overlap}` (expanded to per-segment arrays so headless has no math to do)
  * common settings from `common_args`
* Enqueue:
  ```python
  add_task_to_db(orchestrator_payload, db_path, task_type="travel_orchestrator")
  ```
* **Do _not_** call `poll_task_status`.
* Exit immediately (status msgs only).

### 4.2  Delete big helper code blocks now destined for `headless.py`:
* guide-video construction
* frame extraction helpers
* cross-fade stitching (these belong to server side now)

> Keep generic utils (e.g. `_get_unique_target_path`) in `sm_functions.common_utils` so both modules can import.


## 5. Add logic to `headless.py`

### 5.1  New task handlers
1. **`_handle_travel_orchestrator_task()`**
   * Reads sequence definition.
   * Immediately enqueues **segment-0** (`travel_segment`, `segment_index=0`, no guide video yet if `continue_from_video` absent).
2. **`_handle_travel_segment_task()`**  (refactor of current generation part)
   * Runs WGP like today (no cross-fade/stitch yet).
   * After completion:
     1. Save output path to DB (already happens).
     2. Look inside parent orchestrator payload to determine **if more segments remain**.
        * If yes: create guide video for **next segment** by:
          * extracting overlap frames from the just-rendered mp4 (reuse existing helper).
          * building guide video frames (reuse logic moved from `travel_between_images`).
        * Enqueue next segment task referencing new guide path and `previous_segment_task_id` for dependency clarity.
     3. If last segment: enqueue **stitch task** (`task_type="travel_stitch"`) that waits for all segment mp4s then cross-fades & (optionally) upscales.

### 5.2  Guide / Stitch utilities
* Move these from `travel_between_images`:
  * `extract_frames_from_video`, `cross_fade_overlap_frames`, `create_video_from_frames_list`, `_apply_saturation_to_video_ffmpeg`, etc.
* Place in `sm_functions.common_utils` _or_ a new `sm_functions.video_utils` to keep headless lean.

### 5.3  DB Dependency helpers (optional but nice)
* Add `depends_on_task_id` column **OR** store list in `params`.
* When claiming tasks headless should:
  * skip tasks whose `depends_on_task_id` isn't `STATUS_COMPLETE` yet.
  * simple where-clause + ORDER BY is enough.

### 5.4  Polling / Concurrency
* Because tasks are now independent rows, multiple workers can process parallel segments (except each one waits on previous due to depends_on).


## 6. Transitional Considerations

* Provide **feature flag** (e.g. `--legacy_travel_flow`) to fall back to old behaviour until new path is proven.
* Update unit tests & CI.
* Ensure migration script adds any new DB columns.


## 7. File/Module Moves Summary

| From `travel_between_images.py`                        | To                                |
|--------------------------------------------------------|------------------------------------|
| `extract_frames_from_video`, `cross_fade_overlap_frames` | `sm_functions.video_utils`         |
| Guide-video build logic (both first & subsequent segs) | `_handle_travel_segment_task`      |
| Final stitching (`create_video_from_frames_list`, etc.)| new `_handle_travel_stitch_task`   |

Other helper funcs (`_get_unique_target_path`, `_adjust_frame_brightness`, …) should reside in `common_utils` if not already.


## 8. CLI / API Changes

* **No change** for end-user calling `steerable_motion.py travel_between_images …` – all params are forwarded inside orchestrator payload.
* New internal `task_type` strings: `travel_orchestrator`, `travel_segment`, `travel_stitch`.


## 9. Estimated Work Breakdown

1. ✂️ Refactor helpers into shared util module – **½ day**
2. 📝 Create new handlers in `headless.py` – **1 day**
3. 🔄 Rewrite `travel_between_images.py` to enqueue orchestrator only – **½ day**
4. 🔗 Add dependency tracking & DB schema migration – **½ day**
5. 🧪 Testing (unit + e2e) – **1 day**

_Total: ~3 days of focused work._

---

### End of Plan 