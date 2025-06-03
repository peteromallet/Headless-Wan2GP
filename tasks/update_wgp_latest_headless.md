# Wan2GP compatibility update – headless.py vs. upstream commit 7670af9

## Context
The upstream `Wan2GP` repository has been updated (local diff `6706709` → `7670af9`).  The main breaking changes impacting `headless.py` are inside `wgp.py`:

* `generate_video(...)` renamed a keyword argument:
  * **`remove_background_image_ref` → `remove_background_images_ref`** (plural)
* `generate_video(...)` now expects `state["gen"]["file_settings_list"]` to be present (list of per-file settings), similar to existing `file_list`.

Without adapting our wrapper these changes cause `TypeError` (unexpected keyword) or `KeyError`.

---

## Required code changes (high-priority)

- [ ] **Update the call into `wgp_mod.generate_video` (`process_single_task`):**
  ```python
  # BEFORE
  remove_background_image_ref = ui_params.get("remove_background_image_ref", 1)

  # AFTER (support both keys)
  remove_background_images_ref = ui_params.get("remove_background_images_ref", ui_params.get("remove_background_image_ref", 1))
  ```

- [ ] **Rename the keyword in the actual function call** so we pass `remove_background_images_ref=...`.

- [ ] **Augment the generated state dicts**
  * In `build_task_state(...)` and the minimal state inside `_handle_rife_interpolate_task`, add an empty list entry:
    ```python
    "file_settings_list": []
    ```
    This prevents `KeyError` at the top of the new `generate_video`.

- [ ] **Optionally map legacy task JSON**
  * When building `ui_defaults`, copy any incoming `remove_background_image_ref` value into `remove_background_images_ref` for forward-compat.

---

## Nice-to-have / follow-up

- [ ] Confirm no other renamed parameters (scan changed signature lines).
- [ ] Run an integration test generating a short clip to validate that `headless.py` operates end-to-end after the patch.
- [ ] Bump internal version / changelog entry: "Compatibility with Wan2GP `7670af9`".

---

## Environment / CI notes
No new environment variables are required; the update is purely within API surface. 