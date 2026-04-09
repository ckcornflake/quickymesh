# Contributing to quickymesh

Thanks for your interest in contributing! This document covers local development
setup, running the test suite, and the PR workflow.

## Code of conduct

Be kind. This is a small project; assume good faith.

## Getting set up

quickymesh is a two-process app:

- **Server side** (FastAPI + ComfyUI + Blender + Trellis) runs inside Docker.
- **Client side** (`main.py`) is a plain Python CLI that talks to the server over
  HTTP. It can run on any host with network access to the server.

For most contributions you only need the CLI running locally, and you can point it
at a running server. If you're working on the server, pipeline agent, or worker
threads, you'll need the Docker container running too.

### 1. Clone and install Python deps

```bash
git clone https://github.com/ckcornflake/quickymesh
cd quickymesh
pip install -r requirements.txt
```

Python **3.12+** is required. See the "Running tests" section below for a note on
Python 3.14 and pytest.

### 2. Build and start the Docker container (server-side work only)

Follow the Docker setup in [README.md](README.md#docker-setup). The short version:

```bash
cp docker/.env.example docker/.env        # fill in GEMINI_API_KEY if using Gemini
bash docker/build_run.sh build            # ~30–60 min first time
bash docker/download_models.sh            # ~40 GB of weights
bash docker/build_run.sh start
```

The API server listens on `http://localhost:8000`; ComfyUI on `http://localhost:8190`.

### 3. Run the CLI

```bash
python main.py
```

## Running tests

The full suite is mocked — no GPU, ComfyUI, or external APIs required.

```bash
python -m pytest tests/ -v
```

> **Note on Python 3.14.** On Windows, `python` / `pytest` on `PATH` may resolve to
> a 3.13 install that doesn't have the project's dependencies. If you hit
> `ModuleNotFoundError` while running the tests, invoke pytest via the full path to
> the 3.14 interpreter where dependencies are installed, e.g.:
>
> ```bash
> "C:/Users/<you>/AppData/Local/Programs/Python/Python314/python.exe" -m pytest tests/ -v
> ```

To run a single test file or test:

```bash
python -m pytest tests/test_broker.py -v
python -m pytest tests/test_broker.py::test_claim_next_respects_fifo -v
```

## Project layout

```
main.py               HTTP CLI client entry point
api_server.py         FastAPI server entry point
src/
  cli/                CLI client code (talks to API over HTTP)
  api/                FastAPI routers, models, auth, event bus
  agent/              PipelineAgent + worker threads
  workers/            Actual worker implementations (ComfyUI, Trellis, Blender)
  broker.py           SQLite-backed task queue
  state.py            Pipeline state (Pydantic)
  config.py           Config loader (defaults.yaml + env overrides)
  ...
tests/                Unit tests (all mocked)
docker/               Dockerfile, compose file, startup scripts, model downloader
comfyui_workflows/    ComfyUI workflow JSONs (FLUX, Trellis, ControlNet)
```

See [README.md](README.md#architecture-overview) for a runtime architecture diagram.

## Submitting a change

1. **Branch** from `main`: `git checkout -b short-description`.
2. **Make the change.** Keep the diff focused — one bug fix or one feature per PR.
3. **Run the tests** (see above). If you're touching a worker, add a mocked test.
4. **Commit.** Use a concise, imperative subject line (`Fix X`, `Add Y`). Describe
   the *why* in the body if it isn't obvious from the diff.
5. **Open a PR** against `main`. Describe what changed and how you tested it.

## Style

- No formal linter config yet. Follow the style of the surrounding code.
- Prefer `pathlib.Path` over `os.path`.
- Type hints are encouraged but not required on internal helpers.
- Keep public module interfaces narrow; underscore-prefix internal helpers.
- Avoid adding new top-level dependencies unless they're clearly necessary.

## Reporting bugs

Open an issue on GitHub with:

- What you were doing (CLI menu choice / API endpoint)
- What you expected
- What actually happened (with logs if available — `docker/logs/quickymesh.log`
  for the server side, terminal output for the CLI)
- Your OS, Python version, GPU/driver version

## Areas that need help

- Phase 2 (Svelte web UI) is unstarted — see the roadmap in README.
- Cross-platform Blender path auto-discovery.
- Additional concept art backends.
- GitHub Actions CI workflow (run pytest on push).
