# quickymesh

Generate game-ready 3-D assets from a text description — or an existing image — using a local AI pipeline.

**Current state:** Phase 1 complete. The pipeline runs as a REST API server backed by worker threads. A CLI client (`qm_cli.py`) provides the full user experience over HTTP. A Dockerized ComfyUI+Trellis container handles all GPU-intensive generation. Blender handles screenshots and mesh cleanup on the host.

---

## What it does

1. You describe a 3-D object (or point to an existing image).
2. Gemini Flash 2.5 (or FLUX.1-dev locally) generates several concept art images.
3. You approve the ones you like.
4. Trellis (running in Docker via ComfyUI) turns each approved image into a textured 3-D mesh.
5. Blender renders screenshots from 6 angles and generates an interactive HTML preview.
6. You approve or reject each mesh, then export the winners as `.glb` (or `.obj`).

Multiple pipelines run concurrently. Background worker threads handle GPU work so you're never waiting idle.

---

## System requirements

| Component | Requirement |
|---|---|
| OS | Windows 10/11 or Linux. macOS not supported (CUDA required). |
| Python | **3.12+** (for the server and CLI) |
| NVIDIA GPU | **8 GB+ VRAM** for FLUX / restyle; **16 GB+ VRAM** for Trellis. RTX 3000 series or newer recommended. |
| NVIDIA driver | **550+** (CUDA 12.x) |
| Docker | Docker Desktop (Windows) or Docker Engine (Linux) with GPU passthrough |
| WSL 2 | Required on Windows for Docker with NVIDIA GPU support |
| Blender | Included in the Docker image — nothing to install on the host. |
| Disk | **~60 GB** free for Docker image (~20 GB) + model weights (~40 GB) |

**Windows users:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and enable the WSL 2 backend. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU passthrough inside Docker.

---

## Concept art backends

Two backends are available for concept art generation:

| | Gemini Flash 2.5 | FLUX.1-dev (local) |
|---|---|---|
| **Runs on** | Google Cloud API | Local GPU (Docker) |
| **Models to download** | None | ~8 GB (`flux1-dev-fp8.safetensors`) |
| **VRAM required** | 0 (cloud) | ~16 GB |
| **Speed** | ~10–30 s per image | ~30–90 s per image |
| **Cost** | ~$0.01–0.05 per image | Free (electricity) |
| **Image quality** | Excellent — follows prompts precisely | Very good |
| **Image modification** | Yes (`modify` command) | No |
| **Use existing image as base** | Yes | No |
| **API key required** | Yes | No |

**To get a Gemini API key:** Visit [aistudio.google.com](https://aistudio.google.com/), sign in, go to API keys, and go to "Create API key". Image generation costs a small amount per request, but may provide a small number of initial images free — see [Google AI pricing](https://ai.google.dev/pricing).

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/ckcornflake/quickymesh
cd quickymesh
```

### 2. Install Python dependencies (for the CLI only)

```bash
pip install -r requirements.txt
```

The server, Blender, and ComfyUI all run inside Docker — nothing else is needed on the host.

---

## Docker setup

The Docker container is the complete runtime — ComfyUI, the quickymesh API server, and Blender all run inside one image. Nothing else needs to be installed on the host beyond Python for `qm_cli.py`.

> **Time warning:** The first build downloads PyTorch, ComfyUI, model dependencies, and Blender. Expect **30–60 minutes** on a typical connection. Subsequent builds use the Docker layer cache and finish in seconds unless the Dockerfile changes.

### Step 1 — Configure

```bash
cp docker/.env.example docker/.env
```

Edit `docker/.env` and fill in at minimum:

```
API_KEY=your-secret-api-key         # any string — used by the CLI to authenticate
GEMINI_API_KEY=your_gemini_key      # only needed for the Gemini concept art backend
```

### Step 2 — Build the image

From the repo root:

```powershell
# PowerShell (Windows)
.\docker\build_run.ps1 build
```

```bash
# Bash / WSL / Linux
bash docker/build_run.sh build
```

### Step 3 — Download model weights

Models are not bundled in the image (~40 GB total). Run once:

```bash
# Download everything (FLUX + restyle + Trellis)
bash docker/download_models.sh

# Or selectively:
bash docker/download_models.sh flux      # FLUX.1-dev FP8 (~8 GB)
bash docker/download_models.sh restyle   # ControlNet Canny + Juggernaut-XL (~8 GB)
bash docker/download_models.sh trellis   # Trellis2 + DINOv2 (~25 GB)
```

> **FLUX.1-dev** requires accepting the license at [huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and logging in with `huggingface-cli login` first.

> **Trellis weights** also download automatically on first container start. Skip `download_models.sh trellis` if you prefer that or have an existing local copy.

### Step 4 — Start the container

```powershell
.\docker\build_run.ps1 start
```

```bash
bash docker/build_run.sh start
```

Once running:

| Service | URL |
|---|---|
| quickymesh API | `http://localhost:8000` |
| API docs (Swagger) | `http://localhost:8000/docs` |
| ComfyUI (for debugging) | `http://localhost:8190` |

Watch startup logs with:

```bash
.\docker\build_run.ps1 logs
bash docker/build_run.sh logs
```

> **First start:** Trellis model weights download on first startup if not already present — can take **10–30 minutes**.

---

## Using the CLI

```bash
python qm_cli.py
```

Connects to `http://localhost:8000` by default. To connect elsewhere:

```bash
python qm_cli.py --server http://10.0.0.5:8000 --key your-api-key
```

**Save defaults in `~/.qm_config`** (JSON) so you don't need flags every time:

```json
{
  "server": "http://localhost:8000",
  "api_key": "your-secret-api-key-here"
}
```

Or set environment variables:

```bash
export QM_SERVER=http://localhost:8000
export QM_API_KEY=your-secret-api-key-here
```

See [CLI_MANUAL.md](CLI_MANUAL.md) for a full walkthrough of all commands and flows.

---

## Configuration reference

All defaults live in `defaults.yaml`. Key settings and their environment overrides:

| Setting | Default | Env variable |
|---|---|---|
| `infrastructure.comfyui_url` | `http://localhost:8190` | `COMFYUI_URL` |
| `infrastructure.blender_path` | `C:/Program Files/.../blender.exe` | `BLENDER_PATH` |
| `infrastructure.comfyui_output_dir` | _(required for Docker)_ | `COMFYUI_OUTPUT_DIR` |
| `generation.num_concept_arts` | `4` | `NUM_CONCEPT_ARTS` |
| `generation.num_polys` | `8000` | `NUM_POLYS` |
| `gemini.model` | `gemini-2.5-flash-preview-04-17` | `GEMINI_MODEL` |
| `output.root` | `pipeline_root/` | `OUTPUT_ROOT` |

When using the Docker container, set `COMFYUI_OUTPUT_DIR` in `.env` to the host path that is volume-mounted into the container as `/app/output`. This allows the Python server (running on the host) to read generated files.

---

## Workflow tuning

The ComfyUI workflow JSONs live in `comfyui_workflows/`. Most settings are fixed at sensible defaults, but the following are worth knowing about if you want to tune quality, speed, or VRAM usage. Edit the JSON files directly and rebuild the container.

### Mesh generation (`trellis_generate.json`)

| Setting | Node | Default | Notes |
|---|---|---|---|
| `pipeline_type` | Generator | `"512"` | Voxel resolution: `"512"` or `"768"`. Higher = more geometric detail, more VRAM. |
| `sparse_structure_steps` | Generator | `12` | Step count for the sparse structure diffusion pass. Lower (e.g. `6`) = faster, less accurate overall shape. |
| `shape_steps` | Generator | `12` | Step count for the detailed shape SLat diffusion pass. Lower = faster, less surface detail. |
| `max_views` | Generator | `4` | Number of viewpoints sampled during generation. Higher = more geometrically consistent from all angles, slower. |
| `low_vram` | LoadModel | `false` | Set `true` on 16 GB cards if you get OOM errors during mesh generation. |
| `dual_contouring_resolution` | Remesh | `"Auto"` | Force `"256"` for faster remesh, `"512"` for more detail. |
| `keep_models_loaded` | LoadModel | `true` | Keeps Trellis models in VRAM between meshes. Set `false` if you need VRAM freed immediately after generation. |

### Texturing (`trellis_texture.json`)

| Setting | Node | Default | Notes |
|---|---|---|---|
| `texture_size` | Texturing | `4096` | Output texture atlas resolution: `512`/`1024`/`2048`/`4096`. Lower = faster, smaller file. |
| `texture_steps` | Texturing | `12` | Diffusion step count for texture generation. Lower = faster but lower quality. |
| `resolution` | Texturing | `1024` | Multi-view render resolution used during texturing. Lower = faster. |
| `front_axis` | Texturing | `"z"` | Orientation of the input image relative to the mesh. Change to `"x"` or `"y"` if the texture appears rotated. |
| `keep_models_loaded` | LoadModel | `true` | Same as above — keeps models in VRAM between texture passes. |

### VRAM notes

With `keep_models_loaded: true` (default), Trellis models (~12 GB) stay in VRAM across consecutive mesh generations — significantly reducing per-mesh time since model loading is skipped. When you switch to FLUX concept art generation, quickymesh automatically evicts the Trellis models first.

On 16 GB cards running both FLUX and Trellis, you may need to set `keep_models_loaded: false` in both workflow files if you encounter OOM errors.

---

## Output layout

```
pipeline_root/
  tasks.db                               — SQLite task queue (crash-safe)
  uncompleted_pipelines/
    <name>/
      state.json                         — pipeline state (Pydantic)
      concept_arts/                      — generated PNGs (1024×1024)
      meshes/<name>_1/
        mesh.glb                         — initial Trellis mesh
        textured.glb                     — textured mesh
        screenshots/                     — 6-angle Blender renders
        review_<name>_1.png              — screenshot review sheet
        preview.html                     — interactive Three.js viewer
  completed_pipelines/                   — moved here after export
  final_game_ready_assets/
    <asset_name>/
      <asset_name>.glb                   — exported game-ready mesh
```

---

## Testing

All 503 tests are fully mocked — no real API or GPU needed:

```bash
python -m pytest tests/ -v
```

---

## Architecture overview

```
┌──────────────────────────────────────── Docker: quickymesh-runtime ──┐
│                                                                        │
│  startup.sh                                                            │
│  ├── ComfyUI (:8190)          FLUX / ControlNet / Trellis workflows   │
│  └── api_server.py (:8000)                                            │
│      ├── FastAPI app          REST API, OpenAPI docs at /docs          │
│      └── PipelineAgent                                                 │
│          ├── ConceptArtWorkerThread   — Gemini or FLUX concept art     │
│          ├── TrellisWorkerThread      — mesh generation + texturing  ┐ │
│          └── ScreenshotWorkerThread   — Blender screenshots          ┘ │
│                                       VRAM lock (one GPU task at once) │
│  Blender 4.3.2 (headless)     screenshots, mesh cleanup, export        │
│                                                                        │
│  Volumes:                                                              │
│    /app/models      — model weights (FLUX, Juggernaut, Trellis)        │
│    /app/output      — ComfyUI output (images, meshes)                  │
│    /quickymesh/pipeline_root  — pipeline state, assets                 │
│    /quickymesh/logs           — API server logs                        │
└────────────────────────────────────────────────────────────────────────┘

         ▲ HTTP :8000
         │
┌────────┴──────────┐
│   qm_cli.py       │  runs anywhere with network access (host, LAN, etc.)
└───────────────────┘

Broker (SQLite tasks.db)     — crash-safe task queue, survives process crashes
VRAMArbiter (threading.Lock) — prevents concurrent GPU-heavy tasks (OOM guard)
PipelineState (state.json)   — Pydantic model, saved to disk after every mutation
```

---

## Further reading

- [CLI_MANUAL.md](CLI_MANUAL.md) — full CLI user guide
- [API.md](API.md) — HTTP API reference for building frontends or integrations
- [ARCHITECTURE_PLAN.md](ARCHITECTURE_PLAN.md) — full design document and roadmap

---

## Roadmap

| Phase | Status |
|---|---|
| 1 — FastAPI + CLI HTTP client | **Complete** |
| 1.5 — Unified runtime Docker container (API + Blender + ComfyUI) | **Complete** |
| 2 — Svelte web UI | Planned |
| 3 — Local proxy server (Nginx + WireGuard) | Planned |
| 4 — Public cloud deployment (quickymesh.ai) | Planned |
