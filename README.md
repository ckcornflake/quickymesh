# quickymesh

Generate game-ready 3-D assets from a text description — or an existing image — using a local AI pipeline.

**Status:** v0.1.0 — usable, single-user. quickymesh runs as a REST API server (`api_server.py`) backed by worker threads, with an interactive CLI client (`main.py`) that talks to it over HTTP. A single Docker container runs ComfyUI, Trellis, and Blender; the CLI runs on the host. See the [roadmap](#roadmap) for what's next.

---

## What it does

1. You describe a 3-D object (or point to an existing image).
2. Gemini Flash 2.5 (or FLUX.1-dev locally) generates several concept art images.
3. You approve the ones you like.
4. Trellis (running in Docker via ComfyUI) turns each approved image into a textured 3-D mesh.
5. Blender renders screenshots from 6 angles and generates an interactive HTML preview.
6. You approve or reject each mesh, then export the winners as `.glb` (or `.obj`).

Queue up as many pipelines as you want — background worker threads drain them while you keep using the CLI. GPU-heavy tasks are serialized by a VRAM arbiter (no OOM), so pipelines mostly run one after another on the GPU, with some CPU-bound overlap.

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
| **License** | Google API ToS (commercial OK) | FLUX.1 [dev] **Non-Commercial** License |

> **Heads-up:** FLUX.1-dev weights are released under a **non-commercial** license.
> Research, personal projects, and evaluation are fine; commercial use requires a
> paid license from [Black Forest Labs](https://blackforestlabs.ai/). If you plan to
> use quickymesh commercially, use the Gemini backend instead (or bring your own
> commercially-licensed image model). See the full terms at
> [huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).

**To get a Gemini API key:** Visit [aistudio.google.com](https://aistudio.google.com/), sign in, go to API keys, and click "Create API key". Image generation costs a small amount per request, but may provide a small number of initial images free — see [Google AI pricing](https://ai.google.dev/pricing).

> **Gemini request failures?** If your concept art requests come back with auth
> or quota errors even though the key is valid, the most common fix is enabling
> billing on the key's Google Cloud project. Image generation models are often
> gated behind a billing-enabled project even when you have free-tier credit
> available. Open the key in AI Studio, click through to its Cloud project, and
> attach a billing account.

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

The Docker container is the complete runtime — ComfyUI, the quickymesh API server, and Blender all run inside one image. Nothing else needs to be installed on the host beyond Python for the CLI client (`main.py`).

> **Time warning:** The first build downloads PyTorch, ComfyUI, model dependencies, and Blender. Expect **30–60 minutes** on a typical connection. Subsequent builds use the Docker layer cache and finish in seconds unless the Dockerfile changes.

### Step 1 — Configure

```bash
cp docker/.env.example docker/.env          # bash / WSL / Linux
```

```powershell
Copy-Item docker\.env.example docker\.env   # PowerShell
```

Edit `docker/.env` and fill in at minimum:

```
GEMINI_API_KEY=your_gemini_key      # only needed for the Gemini concept art backend
```

See the comments in `docker/.env.example` for the full list of variables you can
override (model paths, pipeline root, etc.) — defaults are sensible for a
repo-relative setup.

> **Auth is off by default.** Set `API_KEY` in `docker/.env` to enable
> single-user bearer-token auth, or mount a `users.yaml` file (see
> [users.yaml.example](users.yaml.example) and the commented volume line in
> [docker-compose.yml](docker/docker-compose.yml)) for multi-user mode. With
> neither set, every request is treated as a local admin — fine for localhost,
> not fine if you expose port 8000.

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
>
> **Trellis weights** also download automatically on first container start if you skip the `trellis` target above.

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

```powershell
.\docker\build_run.ps1 logs
```

```bash
bash docker/build_run.sh logs
```

> **First start:** if you skipped `download_models.sh trellis`, Trellis model
> weights download on first container start — expect **10–30 minutes** before
> the first mesh generation task will run.

### Just want a working Trellis2 + ComfyUI?

Installing Trellis2 into a native ComfyUI setup is notoriously fiddly — custom
wheels, CUDA toolchain matching, PyTorch versions, extension dependencies, the
lot. If you don't care about the quickymesh pipeline and just want a ComfyUI
instance with Trellis2 working out of the box, this container is a reasonable
way to get there:

1. Follow the Docker setup above through Step 4 (you can leave `GEMINI_API_KEY`
   blank in `docker/.env`).
2. Skip `main.py` entirely.
3. Open ComfyUI directly at [http://localhost:8190](http://localhost:8190) and
   use it like any other ComfyUI install — your own workflows, your own nodes,
   your own models dropped into `docker/models/trellis/`.
4. The quickymesh API server will also be running on `:8000` but you can ignore
   it, or disable it by editing [docker/comfyui-trellis/startup.sh](docker/comfyui-trellis/startup.sh).

The `comfyui_workflows/trellis_generate.json` and `trellis_texture.json`
workflows in this repo are also usable as standalone ComfyUI workflows if you
want a working starting point.

### Advanced: native Windows install (not officially supported)

Docker is the **supported install path** for quickymesh — everything below is at
your own risk and is not covered by issue triage. That said, if you'd rather run
ComfyUI + Trellis natively on Windows (no Docker), the community video below walks
through the ComfyUI + Trellis2 side of the install:

- **[Installing Trellis2 for ComfyUI on Windows](https://www.youtube.com/watch?v=OkK-BfLiS2Q)** — *Atelier Darren, Jan 2026*. Not affiliated with this project. Covers ComfyUI + Trellis2 only; you'll still need to install Blender separately, set `BLENDER_PATH` and `COMFYUI_OUTPUT_DIR` in your environment, and run `python api_server.py` yourself. FLUX.1-dev and ControlNet restyle weights are also your responsibility (see `docker/download_models.sh` for the URLs and target paths).

---

## Using the CLI

quickymesh is a two-process app. Start the API server (inside Docker, or via `python api_server.py`), then launch the CLI client in a separate terminal:

```bash
python main.py
```

Connects to `http://localhost:8000` by default. To connect elsewhere:

```bash
python main.py --server http://10.0.0.5:8000 --api-key your-api-key
```

**Authentication is off by default.** To require a bearer token, either start
the server with `--auth-file path/to/users.yaml` (see
[users.yaml.example](users.yaml.example) for the expected format), or set the
`API_KEY` env var for a single-user fallback. In Docker, set `API_KEY` in
`docker/.env`.

Once enabled, the CLI picks the token up from (in order):

1. `--api-key` command-line flag
2. `QUICKYMESH_API_KEY` environment variable
3. Saved token file at `~/.config/quickymesh/token` (Linux/macOS) or `%APPDATA%/quickymesh/token` (Windows) — you currently need to create this file manually; the CLI does not prompt to save a token on first connect.

The server URL can also be set via `QUICKYMESH_SERVER`.

See [CLI_MANUAL.md](CLI_MANUAL.md) for a full walkthrough of all commands and flows.

---

## Configuration reference

All defaults live in `defaults.yaml`. Key settings and their environment overrides:

| Setting | Default | Env variable |
|---|---|---|
| `infrastructure.comfyui_url` | `http://localhost:8190` | `COMFYUI_URL` |
| `infrastructure.blender_path` | `C:/Program Files/Blender Foundation/Blender 4.5/blender.exe` | `BLENDER_PATH` |
| `infrastructure.comfyui_output_dir` | _(required for Docker)_ | `COMFYUI_OUTPUT_DIR` |
| `generation.num_concept_arts` | `4` | `NUM_CONCEPT_ARTS` |
| `generation.num_polys` | `8000` | `NUM_POLYS` |
| `gemini.model` | `gemini-2.5-flash-image` | `GEMINI_MODEL` |
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
  pipelines/
    <name>/                              — 2D pipeline: concept art → N child 3D pipelines
      state.json                         — pipeline state (Pydantic)
      concept_art/                       — generated PNGs + review sheet
    <name>_<i>/                          — 3D pipeline spawned from concept art index i
      state.json
      meshes/
        textured_mesh.glb                — final textured mesh from Trellis
        screenshots/                     — 6-angle Blender renders
      preview.html                       — interactive Three.js viewer
  final_game_ready_assets/
    <asset_name>_v<N>.glb                — exported game-ready mesh (flat files)
```

2D pipelines (`[n]` in the CLI) go through concept art review and then spawn
one or more 3D child pipelines (one per approved image). 3D-only pipelines
(`[3]` in the CLI, starting from a local image) skip concept art entirely and
use the same `<name>_<i>` directory layout.

---

## Testing

All 600 tests are fully mocked — no real API or GPU needed:

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
│   main.py (CLI)   │  runs anywhere with network access (host, LAN, etc.)
└───────────────────┘

Broker (SQLite tasks.db)     — crash-safe task queue, survives process crashes
VRAMArbiter (threading.Lock) — prevents concurrent GPU-heavy tasks (OOM guard)
PipelineState (state.json)   — Pydantic model, saved to disk after every mutation
```

---

## Further reading

- [CLI_MANUAL.md](CLI_MANUAL.md) — full CLI user guide
- [API.md](API.md) — HTTP API reference for building frontends or integrations
- [examples/prompts.md](examples/prompts.md) — sample descriptions that produce good results
- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup and PR workflow
- [CHANGELOG.md](CHANGELOG.md) — release notes

---

## License and third-party components

quickymesh itself (the Python code in this repository) is released under the
[MIT License](LICENSE). You can use, modify, and redistribute the code freely,
including for commercial purposes, subject to the terms of that license.

**However, quickymesh orchestrates a number of third-party tools and models, each
with its own license.** The MIT license on this project's code does **not** cover
those dependencies. If you plan to use quickymesh in a commercial context, review
each of the following and make sure your use complies with its terms:

| Component | Used for | License | Commercial use |
|---|---|---|---|
| **FLUX.1-dev** (weights) | Local concept art backend | FLUX.1 [dev] Non-Commercial License | ❌ Not permitted without a paid license from [Black Forest Labs](https://blackforestlabs.ai/) |
| **Trellis / Trellis2** (weights) | 3D mesh generation | See the model card on [HuggingFace](https://huggingface.co/microsoft/TRELLIS-image-large) | Verify before commercial use |
| **Juggernaut-XL** (weights) | ControlNet restyle | CreativeML Open RAIL++-M | Permitted with use-based restrictions |
| **ControlNet Canny SDXL** (weights) | ControlNet restyle | OpenRAIL / Apache-2.0 (varies by variant) | Verify variant before use |
| **DINOv2** (weights, used internally by Trellis) | 3D mesh generation | Apache-2.0 | ✅ |
| **ComfyUI** | Workflow runtime in the Docker image | GPL-3.0 | ✅ (invoked as a separate process) |
| **Blender** | Headless mesh cleanup, screenshots | GPL-3.0 | ✅ (invoked as a subprocess, not linked) |
| **Gemini API** | Cloud concept art backend | Google API Terms of Service + Gemini additional terms | ✅ subject to Google's terms and per-request billing |
| **PyTorch, FastAPI, Pydantic, and other Python deps** | Runtime | Various permissive (BSD / MIT / Apache-2.0) | ✅ |

The most important item above is **FLUX.1-dev**, which is the only dependency with
an outright non-commercial restriction. If that's a problem for your use case, use
the Gemini backend (or supply your own commercially-licensed image model) for
concept art and leave the local FLUX weights un-downloaded.

This list is provided in good faith but is **not legal advice**. License terms for
model weights in particular change over time — verify the current terms on each
project's page before relying on them.

---

## Roadmap

| Phase | Status |
|---|---|
| 1 — FastAPI + CLI HTTP client | **Complete** |
| 1.5 — Unified runtime Docker container (API + Blender + ComfyUI) | **Complete** |
| 2 — Svelte web UI | Planned |
| 3 — Local proxy server (Nginx + WireGuard) | Planned |
| 4 — Public cloud deployment (quickymesh.ai) | Planned |
