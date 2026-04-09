# quickymesh CLI Manual

`main.py` is the command-line client for the quickymesh pipeline. It communicates with the quickymesh API server over HTTP — you can run it on the same machine as the server or point it at a remote one.

---

## Starting the CLI

```bash
python main.py
```

By default the server runs with **authentication disabled**, so no API key is needed. If the server was started with `--auth-file tokens.json`, the CLI resolves the bearer token in this order:

1. `--api-key` flag
2. `QUICKYMESH_API_KEY` environment variable
3. Saved token file at `~/.config/quickymesh/token` (Linux/macOS) or `%APPDATA%/quickymesh/token` (Windows)

Similarly for the server URL: `--server`, then `QUICKYMESH_SERVER`, then default `http://localhost:8000`.

---

## Main menu

```
--- quickymesh ---
[n] Start a new pipeline
[e] Edit a pipeline
[p] Pause / Resume / Cancel a pipeline
[s] Status (workers + pipelines)
[w] Watch for approvals
[r] Retry failed tasks
[q] Quit
```

If any pipeline is waiting for your attention (concept art or mesh review), the menu shows a notice and pressing Enter jumps straight to the first one.

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

6. **Symmetry** — optionally symmetrize the final mesh. Enter an axis (`x-`, `x+`, `y-`, `y+`, `z-`, `z+`) to enable, or leave blank to skip. Default: `x-` (mirror the right side to the left, useful for characters and vehicles).

After confirming, concept art generation queues immediately and runs in the background.

---

## Concept art review

When concept art is ready the CLI interrupts the idle menu with a review prompt:

```
Concept art for 'red_dragon' is ready for review.
Review sheet saved to: pipeline_root/uncompleted_pipelines/red_dragon/concept_arts/review_sheet.png
(Opening image viewer...)

Actions:
  approve <indices>      — e.g. 'approve 1 3' to send to mesh generation
  regenerate [indices]   — e.g. 'regenerate 2 4' or just 'regenerate' for all
  modify <idx>           — edit one image via Gemini (replaces original)
  restyle <idx>          — restyle via ControlNet Canny (changes art style)
  cancel                 — cancel this pipeline
  quit                   — exit the program

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

**`regenerate [indices]`**
```
> regenerate 2 4    # regenerate specific images
> regenerate        # regenerate all images
```
Re-queues generation for the specified images. You can optionally change the description first — the CLI asks before queueing.

**`modify <idx>`** (Gemini backend only)
```
> modify 2
Modification instruction: Remove the wings and add more detail to the claws
```
Sends the image to Gemini with an edit instruction. The modified image replaces the original at that index. Use this for targeted changes without regenerating.

**`restyle <idx>`** (requires ControlNet restyle worker)
```
> restyle 1
Positive prompt: fantasy oil painting, dramatic lighting, highly detailed
Negative prompt (Enter for default):
Denoise strength 0.1-1.0 (Enter for 0.75):
```
Applies a new art style to the image while preserving its shape via ControlNet Canny edge guidance.

**`cancel`** — cancels the pipeline. The pipeline directory is kept but no further work is done.

**`quit`** — exits the program. Running pipelines continue on the server.

---

## Mesh review

Once mesh generation, texturing, and screenshots are complete for a mesh, the CLI surfaces a review:

```
Mesh 'red_dragon_1' is ready for review.
(Opening screenshots...)

Actions:
  approve <asset_name> [format]  — e.g. 'approve red_dragon_final' or 'approve red_dragon_final obj'
  reject                          — send back for regeneration (pipeline continues)
  cancel                          — cancel the pipeline entirely
  quit                            — exit the program

Enter action
>
```

The review sheet (6-angle screenshots) and an HTML 3-D viewer are opened automatically. Open `review_<name>.html` in a browser for the interactive viewer if it doesn't open automatically.

**`approve <asset_name> [format]`**
```
> approve red_dragon_final
> approve red_dragon_final obj    # export as .obj instead of .glb
```
Approves the mesh and assigns it the final asset name. If multiple meshes are pending, the next one appears automatically. When all meshes are reviewed, approved ones are exported to `pipeline_root/final_game_ready_assets/`.

**`reject`** — rejects the mesh. The CLI optionally lets you update the polygon count and symmetry settings before re-queuing generation.

---

## [e] Edit a pipeline

Edit a pipeline's settings before mesh generation starts. Editing is only allowed while the pipeline is in `initializing`, `concept_art_generating`, or `concept_art_review` status.

```
Editable pipelines:
  1. red_dragon  [concept_art_review]

Enter pipeline number:
> 1

Current description: "a red dragon breathing fire"
New description (Enter to keep):
> a blue dragon exhaling ice

Current polygon target: 8000
New polygon count (Enter to keep):
> 12000

Current symmetry: on, axis=x-
Enable symmetrize? (y/n, Enter to keep):
> n

Pipeline 'red_dragon' updated.
```

---

## [p] Pause / Resume / Cancel

Manage a pipeline's lifecycle:

```
Pipelines:
  1. red_dragon  [concept_art_generating]
  2. spaceship   [mesh_generating]

Enter pipeline number:
> 1

'red_dragon' is concept_art_generating. [p]ause or [c]ancel?
> p
Pipeline 'red_dragon' paused.
```

- **Pause** — halts the pipeline after the current in-flight task completes. Resume it later.
- **Resume** — restarts a paused pipeline.
- **Cancel** — cancels all pending tasks. The pipeline directory is kept for reference.

---

## [s] Status

Shows a snapshot of all workers and pipelines:

```
=== Workers ===
  ConceptArtWorkerThread: running
  TrellisWorkerThread: running
  ScreenshotWorkerThread: running

=== Pipelines ===
  red_dragon: concept_art_review
  spaceship: mesh_generating  [2 queued]
```

Brackets indicate queued or failed tasks. If a pipeline shows `[1 FAILED]`, use `[r] Retry` to reset those tasks.

---

## [w] Watch mode

Polls the server every 3 seconds for status changes. When a pipeline needs approval, it surfaces the review prompt automatically.

```
Watch mode active. Polling for updates every 3s.
Press Ctrl-C to return to the menu.

[10:05:12] !!! APPROVAL NEEDED: red_dragon !!!
```

Use this after starting a pipeline when you want to be notified as soon as concept art or meshes are ready, without having to check manually.

---

## [r] Retry failed tasks

If a pipeline shows failed tasks (e.g., from a Gemini timeout or ComfyUI error), retry resets them so the workers try again:

```
Pipelines with failed tasks:
  1. red_dragon  [1 FAILED]

Enter pipeline number:
> 1
Reset 1 failed task(s) for 'red_dragon'.
```

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
