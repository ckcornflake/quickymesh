# quickymesh

A parallel 3-D asset pipeline: Gemini Flash → Trellis (ComfyUI) → Blender → game-ready `.glb`.

Multiple pipelines run concurrently. Background worker threads handle GPU work while the CLI surfaces approvals only when the user is needed.

---

## Prerequisites

| Tool | Version / Notes |
|------|----------------|
| Python | **3.14** (packages installed here; 3.13 is on PATH but has no packages) |
| ComfyUI (portable, NVIDIA) | Running on port 8188 with Trellis workflows loaded |
| Blender | 4.5, at `C:/Program Files/Blender Foundation/Blender 4.5/blender.exe` |
| Gemini API key | Set in `.env` as `GEMINI_API_KEY=...` |

Install Python deps (run once from repo root):

```
/c/Users/jmkel/AppData/Local/Programs/Python/Python314/python.exe -m pip install -r requirements.txt
```

---

## Configuration

All defaults live in `defaults.yaml`. Key settings:

| Key | Default | Env override |
|-----|---------|-------------|
| `gemini.model` | `gemini-2.5-flash-image` | `GEMINI_MODEL` |
| `generation.num_concept_arts` | `4` | `NUM_CONCEPT_ARTS` |
| `generation.num_polys` | `8000` | `NUM_POLYS` |
| `output.root` | `pipeline_root/` | `OUTPUT_ROOT` |
| `infrastructure.comfyui_url` | `http://localhost:8188` | `COMFYUI_URL` |
| `infrastructure.blender_path` | (see above) | `BLENDER_PATH` |

Secrets go in `.env` (gitignored):

```
GEMINI_API_KEY=your_key_here
```

---

## Running the pipeline

```bash
/c/Users/jmkel/AppData/Local/Programs/Python/Python314/python.exe main.py
```

### CLI idle menu

When no pipeline needs your attention:
s
```
--- quickymesh ---
[n] Start a new pipeline
[s] Status (workers + pipelines)
[q] Quit
```

`[s]` shows which worker threads are alive and the current status of every pipeline — use it to check progress without waiting for a review prompt.

### Starting a new pipeline

Choose `[n]`. You will be asked:
1. **Pipeline name** — a short identifier with no spaces (e.g. `red_dragon`)
2. **Description** — plain English (e.g. `a red dragon breathing fire`). The suffix `plain white background` is automatically appended so Trellis background removal works.
3. **Polygon count** — press Enter to use the default (8000).

Concept art generation queues immediately and runs in the background.

### Concept art review

When images are ready the CLI interrupts the idle menu automatically:

```
Actions:
  approve <indices>  — e.g. 'approve 1 3' to send to mesh gen
  regenerate <idx>   — e.g. 'regenerate 2 4'
  modify <idx>       — modify one image via Gemini
  cancel             — cancel this pipeline
  quit               — exit the program
```

Indices are **1-based** and match the labels on the review sheet PNG. Approving immediately queues mesh generation — you will be told.

### Mesh review

Once mesh + texturing + screenshots are done the CLI surfaces the mesh review. You can:

```
approve <sub_name> [format]  — e.g. 'approve red_dragon_1' or 'approve red_dragon_1 obj'
reject                       — send mesh back (pipeline stays alive)
cancel                       — cancel pipeline entirely
quit
```

Approved meshes are exported to `pipeline_root/final_game_ready_assets/`.

---

## Output layout

```
pipeline_root/
  tasks.db                         ← SQLite task queue (crash-safe)
  uncompleted_pipelines/
    <name>/
      state.json                   ← full pipeline state
      concept_arts/                ← generated + padded PNGs
      meshes/<name>_1/             ← per-approved-image
        mesh.glb
        textured.glb
        screenshots/
        review_<name>_1.png
        preview.html               ← Three.js viewer
  completed_pipelines/             ← moved here after final export
  final_game_ready_assets/
    <final_name>/
      <final_name>.glb
```

---

## Testing

Run the full test suite (318 tests, all mocked — no real API or GPU needed):

```bash
/c/Users/jmkel/AppData/Local/Programs/Python/Python314/python.exe -m pytest tests/ -v
```

### Smoke testing individual phases

Each smoke script exercises real external services. Run them in order to validate your environment before using the full pipeline.

**Phase 2 — Gemini concept art**

```bash
/c/Users/jmkel/AppData/Local/Programs/Python/Python314/python.exe scripts/smoke_test_gemini.py
```

Saves a padded 1024×1024 PNG to `smoke_gemini/`. Open it to confirm the image looks right.

**Phase 3 — Trellis mesh generation**

> Requires ComfyUI running on port 8188 with Trellis nodes loaded.

```bash
/c/Users/jmkel/AppData/Local/Programs/Python/Python314/python.exe scripts/smoke_test_trellis.py \
    --image path/to/concept_art.png \
    --name my_test_mesh
```

Prints the path to the generated `.glb`. Inspect it in Blender or the HTML preview.

**Phase 4 — Blender screenshots**

> Requires Blender 4.5 installed.

```bash
/c/Users/jmkel/AppData/Local/Programs/Python/Python314/python.exe scripts/smoke_test_screenshots.py \
    --mesh path/to/mesh.glb \
    --hdri
```

Outputs 6 view PNGs, a review sheet, and `preview.html`. Open the HTML in a browser to confirm the 3-D viewer works.

**Phase 5 — Full pipeline end-to-end**

Start `main.py`, create a pipeline, let concept art generate (takes ~30 s), approve images, wait for mesh (~5–10 min in Trellis), approve the mesh. Check `pipeline_root/final_game_ready_assets/` for the exported `.glb`.

---

## Architecture overview

```
main.py
└── PipelineAgent
    ├── ConceptArtWorkerThread   — polls broker for concept_art_generate / concept_art_modify
    ├── TrellisWorkerThread      — polls broker for mesh_generate / mesh_texture  ┐ VRAM lock
    └── ScreenshotWorkerThread   — polls broker for screenshot                    ┘ (one at a time)

Broker (SQLite tasks.db)         — crash-safe task queue shared by all threads
VRAMArbiter (threading.Lock)     — prevents concurrent GPU-heavy tasks (OOM guard)
PipelineState (state.json)       — Pydantic model; saved after every mutation
PromptInterface                  — abstracted: CLIPromptInterface today, email/agent later
```

**Priority order** when multiple pipelines are active:
1. Concept art review (highest)
2. Mesh review
3. Idle menu (lowest)

---

## Known issues / deferred

- **White-on-white background removal**: Trellis background removal only strips the padded border, not white areas that are part of the subject itself (e.g. a white teapot on a white background). Workaround: use a coloured subject or add `on a dark background` to your description override. Fix deferred post-phase-5.
- **ComfyUI auto-start**: The doc specifies auto-starting ComfyUI if unreachable — not yet implemented. Start ComfyUI manually before running the pipeline.
- **Email / agent prompt interface**: `PromptInterface` is abstract and ready; only `CLIPromptInterface` is implemented.
- **Symmetrize on export**: DOC specifies optional mesh symmetrization at export time — not yet implemented.
