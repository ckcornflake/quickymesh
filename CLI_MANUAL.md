# quickymesh CLI Manual

`main.py` is the command-line client for the quickymesh pipeline. It communicates with the quickymesh API server over HTTP — you can run it on the same machine as the server or point it at a remote one.

---

## Starting the CLI

```bash
python main.py
```

By default the server runs with **authentication disabled**, so no API key is needed. If the server was started with `--auth-file users.yaml` (see [users.yaml.example](users.yaml.example)), the CLI resolves the bearer token in this order:

1. `--api-key` flag
2. `QUICKYMESH_API_KEY` environment variable
3. Saved token file at `~/.config/quickymesh/token` (Linux/macOS) or `%APPDATA%/quickymesh/token` (Windows) — you currently need to create this file manually

Similarly for the server URL: `--server`, then `QUICKYMESH_SERVER`, then default `http://localhost:8000`.

---

## Main menu

```
--- quickymesh ---
[n] Start a new 2D pipeline
[3] Start a 3D pipeline from a local image
[m] Manage a pipeline  ( [re]submit / edit / hide / kill )
[u] Unhide a pipeline
[w] Watch for status updates
[t] Retry failed tasks
[q] Quit
```

If any pipeline is waiting for your attention (concept art or mesh review), the menu shows a notice and pressing Enter jumps straight to the first one. Numeric shortcuts `1`–`7` are accepted as aliases for the letters above.

---

## [n] Start a new pipeline

Walks you through creating a new pipeline:

1. **Pipeline name** — a short identifier with no spaces (e.g., `red_dragon`). Must be unique.

2. **Concept art backend:**
   ```
   1. Gemini Flash  — requires API key, small cost per image, very accurate, can use an existing image
   2. FLUX.1 [dev]  — runs locally via ComfyUI, ~25 GB models, ~16 GB VRAM
   ```

3. **Base image (Gemini only)** — optionally provide a path to an existing image on disk. The AI will modify or adapt it based on your description instead of generating from scratch.

4. **Description** — plain English description of the 3-D object. A background suffix is automatically appended to ensure clean Trellis background removal. You'll be told what was added.

5. **Polygon count** — press Enter to use the default (8000). Higher poly counts produce more detailed meshes but take longer and use more VRAM.

6. **Symmetry** — optionally symmetrize the final mesh. Enter an axis (`x-`, `x+`, `y-`, `y+`, `z-`, `z+`) to enable, or leave blank to skip. Blank means no symmetrization; pick an explicit axis (e.g. `x-` to mirror the right side to the left) if you want it on.

After confirming, concept art generation queues immediately and runs in the background.

---

## Concept art review

When concept art is ready the CLI interrupts the idle menu with a review prompt:

```
Concept art for 'red_dragon' is ready for review.
Review sheet saved to: pipeline_root/uncompleted_pipelines/red_dragon/concept_arts/review_sheet.png
(Opening image viewer...)

Actions:
  approve <indices>   — e.g. 'approve 1 3' to send to mesh generation
  regenerate          — pick an image to regenerate (or 'regenerate all')
  modify              — modify one image via Gemini (Gemini backend only)
  restyle             — restyle image shape/silhouette via ControlNet
  menu                — return to the main menu
  quit                — exit the program

Enter action
>
```

Indices are **1-based** (matching the labels on the review sheet).

### Actions

**`approve <indices>`**
```
> approve 1 3
```
Sends images 1 and 3 to mesh generation. You can approve multiple at once — each becomes a separate mesh. The pipeline continues in the background.

**`regenerate`**
```
> regenerate
Which image to regenerate? (1–4): 2
```
or, to regenerate every concept art in one shot:
```
> regenerate all
```
Re-queues generation for the selected image(s). With `regenerate all` the CLI first asks whether you want to change the description; with single-image regenerate it uses the current description as-is. If you've made earlier edits you want to keep, prefer single-image regenerate.

**`modify`** (Gemini backend only)
```
> modify
Which image to modify? (1–4): 2
Describe the change to make to image 2: Remove the wings and add more detail to the claws
```
Sends the image to Gemini with an edit instruction. A new version is saved alongside the original (concept art history is preserved). Use this for targeted changes without regenerating from scratch. If the image has prior versions, the CLI will also ask which version you want to base the modification on.

**`restyle`** (requires ControlNet restyle worker)
```
> restyle
Which image to restyle? (1–4): 1
Positive prompt: fantasy oil painting, dramatic lighting, highly detailed
Negative prompt (Enter for default):
Denoise strength 0.1–1.0 (Enter for default 0.75):
```
Applies a new art style to the image while preserving its shape via ControlNet Canny edge guidance. Like `modify`, this creates a new version rather than overwriting the original.

**`menu`** — returns to the main menu. The pipeline stays in `concept_art_review` on the server, so you can come back later (via `[m] Manage` or `[w] Watch`) to finish reviewing.

**`quit`** — exits the program. Running pipelines continue on the server.

---

## Mesh review

Once mesh generation, texturing, and screenshots are complete for a mesh, the CLI surfaces a review:

```
3D mesh review for 'red_dragon_1_0'
  Status: awaiting_approval
  Mesh: pipeline_root/pipelines/red_dragon_1_0/meshes/.../textured.glb

Actions:
  approve       — export mesh to final assets
  regenerate    — re-queue mesh generation (optionally with different poly count)
  menu          — return to main menu
  quit          — exit the program

Enter action
>
```

The review sheet (6-angle screenshots) and an HTML 3-D viewer are opened automatically. Open `preview.html` in a browser for the interactive viewer if it doesn't open automatically.

**`approve`** — exports the mesh to `pipeline_root/final_game_ready_assets/` and hides the pipeline. Use `[u] Unhide` from the main menu if you want to access it again later.

**`regenerate`** — rejects the current mesh and re-queues mesh generation. The CLI optionally lets you update the polygon count and symmetry settings before re-queuing.

**`menu`** — returns to the main menu without taking action. The pipeline stays in `awaiting_approval` on the server.

**`quit`** — exits the program.

---

## [w] Watch mode

Polls the server every 3 seconds for status changes. When a pipeline needs approval, it surfaces the review prompt automatically.

```
Watch mode active. Polling for updates every 3s.
Press 'q' + Enter to return to the menu.

[10:05:12] !!! APPROVAL NEEDED: red_dragon !!!
```

Use this after starting a pipeline when you want to be notified as soon as concept art or meshes are ready, without having to check manually.

---

## Watch mode during generation

When you start a pipeline or approve images, the CLI optionally enters a live generation watch mode:

```
Watching generation progress... (press 'q' + Enter to return to menu)

[10:02:15] red_dragon: concept_art_generating  (pending×3  generating×1)
[10:02:45] red_dragon: concept_art_generating  (generating×2  completed×2)
[10:03:10] red_dragon: concept_art_review

[10:03:10] Ready for review!
```

The status line updates whenever any per-image status changes. Press `q` + Enter to return to the menu at any time — the pipeline continues running on the server.

---

## Tips

- **Multiple pipelines:** Start several pipelines and they all run concurrently. The CLI shows whichever needs attention next.
- **Disconnect and reconnect:** The server keeps running after you exit the CLI. Reconnect later with `python main.py` and your pipelines will still be there.
- **Check the review sheet first:** The 6-angle screenshot review sheet opens automatically. But open `preview.html` in a browser for an interactive 3-D view before approving.
- **Reject to iterate:** Rejected meshes automatically re-queue with the same (or updated) settings. You can reject many times — there's no limit.
- **VRAM lock:** Only one GPU-heavy task runs at a time. If Trellis is generating, FLUX concept art generation waits. This prevents out-of-memory errors.
