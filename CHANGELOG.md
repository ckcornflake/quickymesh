# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — Initial public release

First public release of quickymesh — a local AI pipeline that turns text prompts or
reference images into game-ready 3D assets.

### Features

- **Two-process architecture.** FastAPI server (`api_server.py`) runs the pipeline
  agent and worker threads in-process; `main.py` is an interactive HTTP CLI client
  that can connect to a local or remote server.
- **Concept art generation.** Two interchangeable backends:
  - Gemini Flash 2.5 (cloud, supports image-based prompts and in-place modification)
  - FLUX.1-dev (local, runs inside the ComfyUI container)
- **ControlNet restyling** of concept art via a local Juggernaut-XL + Canny workflow.
- **3D mesh generation** via Trellis (running inside ComfyUI) with separate
  `mesh_generate` and `mesh_texture` passes.
- **Blender post-processing.** Headless mesh cleanup, symmetrization, 6-angle
  screenshot rendering, and interactive Three.js HTML preview generation.
- **Per-pipeline FIFO task queue** backed by SQLite (`tasks.db`) — crash-safe and
  survives server restarts.
- **VRAM arbiter** (mutex) prevents concurrent GPU-heavy tasks from OOMing the card.
- **2D / 3D pipeline separation.** Users can either start a full 2D→3D pipeline from a
  text prompt, or submit their own image directly for 3D mesh generation.
- **Concept art versioning.** Modify/restyle preserves history; older versions can be
  used as the source for further edits or resubmitted for 3D generation.
- **Unified Docker runtime.** Single container bundles ComfyUI, Blender, and the
  quickymesh API server. CLI runs on the host.
- **Optional bearer-token auth** on the API server.
- **600 unit tests**, fully mocked — no GPU or external API access required to run
  the suite.

### Known limitations

- Server requires an NVIDIA GPU with CUDA 12.x. macOS is not supported.
- Docker is the supported install path; native (non-Docker) installs are best-effort.
- Gemini concept art backend requires a `GEMINI_API_KEY` and incurs per-image cost.
- Phase 2 (web UI), Phase 3 (proxy), and Phase 4 (hosted deployment) are planned but
  not yet implemented.

[Unreleased]: https://github.com/ckcornflake/quickymesh/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ckcornflake/quickymesh/releases/tag/v0.1.0
