# quickymesh — Web Architecture Plan

## Overview

This document describes the four-phase plan to evolve quickymesh from a local
CLI tool into a web-accessible service backed by the 5090 generation box, with
a clean API, web UI, and a hardened public-facing proxy server.

The guiding principles:
- **Preserve what works.** The SQLite broker, worker threads, and VRAMArbiter
  are sound. We build around them, not over them.
- **No premature complexity.** Each phase delivers a working system. Later
  phases layer on without breaking earlier ones.
- **The CLI stays alive.** It becomes a first-class API client, useful for
  local testing and as a reference implementation.
- **The 5090 box is never internet-reachable.** All external traffic goes
  through the proxy server.

### Phases

| Phase | Goal |
|---|---|
| 1 | FastAPI API layer + CLI HTTP client refactor |
| 1.5 | Docker consolidation — one runtime container (ComfyUI + Blender) |
| 2 | Svelte + Vite web UI |
| 3 | Local proxy server (Nginx + WireGuard) |
| 4 | Public cloud deployment (quickymesh.ai) |

---

## Answers to Open Questions

1. **User management** — config file for now. A `users.yaml` in the server config directory maps usernames to API keys and roles (`admin` / `user`). No registration endpoint needed until Phase 4.

2. **Pipeline visibility** — users see only their own pipelines. An `admin` role sees all users' pipelines across a unified view.

3. **Frontend framework** — **Svelte + Vite** (not SvelteKit — plain Svelte with Vite is sufficient; SvelteKit adds server-side routing complexity we don't need).

4. **Domain** — **quickymesh.ai** is available for registration (~$60–100/year). Use a self-signed cert in Phase 3 local testing, swap to Let's Encrypt when deploying in Phase 4.

---

## Technology Decisions

### FastAPI — Yes
The right choice for the API layer. Async-native, minimal boilerplate, automatic
OpenAPI docs at `/docs`, excellent file-streaming support for serving images and
meshes, and plays well with the existing synchronous worker thread model.

### Celery — No (not now)
Celery is a distributed task queue for horizontal scale. We already have a
working task queue (SQLite broker + daemon threads). Replacing it with Celery
would mean adding Redis/RabbitMQ as required infrastructure and rewriting the
worker integration — solving problems we don't have. **If we ever need
distributed workers across multiple machines, the migration path is: swap the
SQLite broker for Redis, keep everything else.** That's a much smaller change
than introducing Celery today.

### Server-Sent Events (SSE) — for real-time push
SSE is one-directional (server → client), works over plain HTTP, requires no
WebSocket upgrade, passes transparently through Nginx, and browsers reconnect
automatically on drop. For our use case — notifying the client that a review
is ready, or that generation has completed — it is exactly the right tool and
simpler to implement than WebSockets. We get responsive UX without meaningful
added complexity.

### httpx — CLI HTTP client
Replaces direct `PipelineAgent` calls in the CLI. Supports async and sync
modes, ships with a clean API, and handles streaming responses for SSE.

### Nginx — reverse proxy / static file server (Phase 3+)
Handles SSL termination, serves the frontend HTML/JS/CSS as static files, and
proxies API calls to the FastAPI backend. Standard, battle-tested.

### WireGuard — secure tunnel (Phase 3+)
The only channel between the public proxy server and the 5090 box. The 5090
machine's firewall blocks everything except the WireGuard UDP port. The proxy
server reaches the generation API over the private WireGuard network. No port
forwarding, no public IP on the 5090.

---

## Phase 1 — API Layer + CLI Refactor

**Goal:** The 5090 box runs a FastAPI server. The CLI becomes an HTTP client
that talks to it. Everything still works locally.   If this was an open source project
it would be ready to interface with any arbitrary front end within reason.

### 1.1 Process Architecture

FastAPI and the worker threads run **in the same process**. This is the minimal
change — the PipelineAgent, broker, and workers are unchanged. FastAPI
handlers simply call the same PipelineAgent methods the CLI used to call
directly.

```
┌─────────────────────────────────────────────── 5090 box ──┐
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              api_server.py (single process)          │  │
│  │                                                      │  │
│  │  ┌──────────────┐    ┌──────────────────────────┐   │  │
│  │  │  FastAPI app │    │     PipelineAgent         │   │  │
│  │  │  (uvicorn /  │───▶│  (same as before)        │   │  │
│  │  │   asyncio)   │    └──────────┬───────────────┘   │  │
│  │  └──────────────┘               │                    │  │
│  │                          ┌──────▼──────┐             │  │
│  │                          │   Broker    │             │  │
│  │                          │  (SQLite)   │             │  │
│  │                          └──────┬──────┘             │  │
│  │                    ┌───────────┼───────────┐         │  │
│  │               ┌────▼───┐  ┌───▼───┐  ┌────▼────┐   │  │
│  │               │Concept │  │Trell  │  │ Screen  │   │  │
│  │               │ArtWkr  │  │isWkr  │  │ shotWkr │   │  │
│  │               └────────┘  └───────┘  └─────────┘   │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           pipeline_root/ (filesystem)                │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘

         ▲ HTTP (localhost or LAN only)
         │
┌────────┴──────────┐
│   CLI client      │  (Python / httpx — runs anywhere with network access)
│   (qm_cli.py)     │
└───────────────────┘
```

### 1.2 Authentication

A simple **API key** scheme. Each user gets a key stored in a config file
(or a small `users` table in SQLite). The key is sent in the `Authorization`
header as `Bearer <key>`. FastAPI middleware validates it on every request.
Users are identified by the key, so pipelines are owned by user — each user
sees only their own pipelines.

We deliberately avoid JWT or OAuth for now. Those are Phase 3 concerns.

### 1.3 API Endpoints

All endpoints are under `/api/v1/`.

**Pipelines**
```
POST   /pipelines                         Create new pipeline
GET    /pipelines                         List pipelines (user's own)
GET    /pipelines/{name}                  Get pipeline state
DELETE /pipelines/{name}                  Cancel pipeline
PATCH  /pipelines/{name}                  Edit (description, polys, symmetry)
POST   /pipelines/{name}/pause            Pause
POST   /pipelines/{name}/resume           Resume
POST   /pipelines/{name}/retry            Retry failed tasks
```

**Concept art review**
```
GET    /pipelines/{name}/concept_art/{idx}          Download image (PNG)
POST   /pipelines/{name}/concept_art/approve        Approve selected indices
POST   /pipelines/{name}/concept_art/regenerate     Regenerate indices
POST   /pipelines/{name}/concept_art/modify         Gemini modify (if supported)
POST   /pipelines/{name}/concept_art/restyle        ControlNet restyle
```

**Mesh review**
```
GET    /pipelines/{name}/meshes/{mesh_name}/screenshot/{view}   Download PNG
GET    /pipelines/{name}/meshes/{mesh_name}/preview             Download HTML
GET    /pipelines/{name}/meshes/{mesh_name}/mesh                Download GLB
POST   /pipelines/{name}/meshes/{mesh_name}/approve             Approve mesh
POST   /pipelines/{name}/meshes/{mesh_name}/reject              Reject mesh
```

**Assets (completed pipelines)**
```
GET    /assets                            List completed/exported pipelines
GET    /assets/{name}/mesh               Download final GLB
```

**Real-time updates**
```
GET    /pipelines/{name}/events           SSE stream — pipeline state changes
GET    /events                            SSE stream — all user's pipelines
```

**System**
```
GET    /status                            Worker health, queue depth
```

### 1.4 SSE Event Format

```json
{ "event": "status_change",
  "pipeline": "my_spaceship",
  "status": "concept_art_review",
  "message": "Concept art ready for review" }

{ "event": "task_complete",
  "pipeline": "my_spaceship",
  "task_type": "mesh_generate" }
```

The worker threads already `print()` to stdout as a side-channel. In Phase 1
we replace those prints with calls to an **event bus** — a simple
`asyncio.Queue` per connected SSE client. Worker threads post events to it;
the SSE endpoint drains it and streams to the client. This requires a small
thread-safe bridge since workers are sync threads but FastAPI is async.

### 1.5 CLI Refactor

`qm_cli.py` replaces `main.py` as the user-facing entry point. It uses
`httpx` to call the FastAPI server. The user experience is identical — same
prompts, same review flows — but all state reads/writes go through the API
instead of touching files or the broker directly.

The CLI takes a `--server` flag (default `http://localhost:8000`) and a
`--key` flag (or reads from `~/.qm_config`). This lets it work against a
local server for testing or a remote proxy in Phase 3.

**The existing `main.py` and `run_cli()` are kept as the server-side launch
entry point.** The file-touching CLI code is not deleted — it becomes the
server's internal implementation, not the user-facing interface.

### 1.6 New File Structure

```
quickymesh/
  src/
    api/
      app.py            # FastAPI app, router registration
      auth.py           # API key middleware
      routers/
        pipelines.py    # Pipeline CRUD endpoints
        review.py       # Concept art + mesh review endpoints
        assets.py       # Completed pipeline downloads
        events.py       # SSE endpoint
        status.py       # System health
      event_bus.py      # Thread-safe bridge: worker threads → SSE clients
    agent/              # (unchanged)
    workers/            # (unchanged)
    ...
  api_server.py         # Entry point: starts FastAPI + worker threads
  qm_cli.py             # New CLI client using httpx
  main.py               # Kept for reference / local standalone use
```

---

## Phase 1.5 — Docker Consolidation

**Goal:** Package the generation environment (ComfyUI + Blender + model weights)
into a single Docker image so that anyone cloning the project has a working
setup with one command. This phase happens after Phase 1's API is stable,
before Phase 2 begins.

### Decision: One Runtime Container

All generation dependencies live in a **single container** (`quickymesh-runtime`).
There is no benefit to splitting Blender into a separate container:

- Blender's geometry operations (cleanup, symmetrize) are CPU-only — no CUDA
  required, no resource conflict with ComfyUI.
- Screenshot rendering uses the GPU, but the container already needs
  `--gpus all` for ComfyUI/Trellis. No additional config.
- `subprocess.run()` works identically inside Docker containers — `BlenderScreenshotWorker`
  requires zero changes.
- Blender adds ~500 MB to an image that is already multi-GB. Not a concern.
- For open-source users: `docker compose up` and everything works. Splitting
  into multiple containers adds orchestration complexity for zero benefit.

### Container Split

```
┌──────────────────────────────────────────────────────────────┐
│  quickymesh-runtime  (GPU container, --gpus all)             │
│                                                               │
│  Base: Trellis base image (Python 3.12, PyTorch 2.8.0+cu128) │
│  + ComfyUI + Trellis custom nodes + model weights            │
│  + Blender binary (~500 MB)                                   │
│  + Python packages (requirements.txt)                         │
│                                                               │
│  Exposes: :8188 (ComfyUI API)                                 │
│           :8000 (quickymesh FastAPI — api_server.py)          │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  quickymesh-frontend  (Phase 2+, optional at Phase 1.5)       │
│                                                               │
│  Nginx serving the Svelte dist/ bundle                        │
│  Proxies /api/* → quickymesh-runtime:8000                     │
└──────────────────────────────────────────────────────────────┘
```

The FastAPI server (`api_server.py`) and all worker threads run **inside**
`quickymesh-runtime`. They call ComfyUI at `localhost:8188` (same container)
and call Blender via subprocess (same container). No inter-container networking
required for generation.

### Why Trellis as the Base Image

CUDA extension wheels (for Flash Attention, xformers, and Trellis' own ops) are
compiled against a specific Python + PyTorch + CUDA driver combination. The
Trellis Docker image ships pre-built wheels for Python 3.12 / PyTorch 2.8.0 /
cu128. Rebuilding them from source takes 20–40 minutes and breaks on CUDA
driver version mismatches. Using the Trellis image as the base avoids this
entirely.

**SM 12.0 (Blackwell / RTX 5000 series) note:** The RTX 5090 uses SM 12.0.
Pre-built wheels in the Trellis image may target sm_86/sm_89. If generation
fails with CUDA kernel errors, the solution is to rebuild the affected wheels
with `-DTORCH_CUDA_ARCH_LIST="12.0"` added to the build-wheels Dockerfile.
Track this in the `docker/` directory.

### Docker Compose (generation box)

```yaml
services:
  runtime:
    build:
      context: .
      dockerfile: docker/Dockerfile.runtime
    image: quickymesh-runtime:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    ports:
      - "8000:8000"   # quickymesh API
      - "8188:8188"   # ComfyUI (LAN access for debugging)
    volumes:
      - ./pipeline_root:/app/pipeline_root
      - ./logs:/app/logs
      - comfyui-models:/app/ComfyUI/models  # model weights volume
    restart: unless-stopped

volumes:
  comfyui-models:
    driver: local
```

### Model Weights

Model weights (tens of GB) are not baked into the image. They live in a
named Docker volume (`comfyui-models`) and are downloaded on first run via
a `docker/download_models.sh` script. This script is idempotent — re-running
it skips files that already exist. The `README` documents the exact models
required and provides the script.

### Open-Source User Flow

```bash
git clone https://github.com/ckcornflake/quickymesh
cd quickymesh
cp .env.example .env          # add GEMINI_API_KEY if using Gemini backend
docker compose run runtime bash docker/download_models.sh
docker compose up
# → API available at http://localhost:8000
# → ComfyUI available at http://localhost:8188
```

### Files Added

```
quickymesh/
  docker/
    Dockerfile.runtime          # extends Trellis base, adds Blender + our code
    download_models.sh          # idempotent model weight downloader
    build_wheels.sh             # optional: rebuild CUDA wheels for SM 12.0
  docker-compose.yml            # generation box services
  docker-compose.proxy.yml      # proxy server services (Phase 3)
  .env.example                  # template with all required env vars documented
```

---

## Phase 2 — Web UI

**Goal:** A web frontend that lets users do everything the CLI can do, with a
nicer interface. Runs against the Phase 1 API. Tested on localhost.

### Stack
- **Svelte + Vite** (not SvelteKit — plain Svelte with Vite gives component
  structure and reactivity without the server-side routing complexity of
  SvelteKit). Reactivity is built into the language — no hooks, no virtual DOM,
  much smaller output bundle than React.
- SSE integration is one line: `const es = new EventSource('/api/v1/events')`.
- Static files (`dist/`) served by Nginx in Phase 3. In Phase 2, served by
  FastAPI's `StaticFiles` mount during development.

### Key UI Flows

**Pipeline creation** — form matching the CLI prompts: name, backend choice
(Gemini / FLUX), description, poly count, symmetry. Submit → server creates
pipeline → SSE connection opens for that pipeline → progress appears live.

**Concept art review** — when SSE signals `concept_art_review`, the review
panel activates showing the generated images. Approve/regenerate/restyle
buttons call the API. Restyle opens a modal with positive/negative/denoise
fields.

**Mesh review** — screenshot carousel per mesh, approve/reject per mesh.
Rejected meshes show the poly count + symmetry update form inline.

**Asset library** — list of completed pipelines with download buttons for
the final GLB.

---

## Phase 3 — Local Proxy Server

**Goal:** Everything runs on the local machine (or WSL/Docker) using the same
technology stack it will use when deployed. The 5090 box stays on the LAN,
never internet-reachable.

### Architecture

```
Internet (browser)
      │ HTTPS :443
      ▼
┌─────────────────────────────────── Proxy server (WSL/Docker/VPS) ──┐
│                                                                      │
│  ┌──────────┐     ┌────────────────────────────────────────────┐   │
│  │  Nginx   │────▶│  FastAPI (proxy_server.py)                 │   │
│  │  :443    │     │  - Serve frontend static files             │   │
│  │  :80     │     │  - Auth gateway (validate API keys /       │   │
│  └──────────┘     │    sessions, map to user identities)       │   │
│                   │  - Forward API requests to 5090 box        │   │
│                   │  - Rate limiting, request logging          │   │
│                   └────────────────┬───────────────────────────┘   │
│                                    │ WireGuard VPN tunnel           │
└────────────────────────────────────┼───────────────────────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │   5090 box (LAN only)   │
                        │   api_server.py          │
                        │   (Phase 1 server)       │
                        └─────────────────────────┘
```

### WireGuard Tunnel

The proxy server and 5090 box form a private WireGuard network
(e.g. `10.0.0.0/24`). The 5090 firewall allows:
- WireGuard UDP port (inbound from proxy server IP only)
- Nothing else inbound from the internet

The proxy server forwards API requests to `http://10.0.0.2:8000` (the 5090's
WireGuard address). From the outside world's perspective the 5090 does not
exist.

### Auth Strategy

The proxy server owns authentication. Users log in via the web UI and get a
**session cookie** (HttpOnly, Secure). The proxy maps sessions to user
identities and adds the appropriate API key header before forwarding to the
5090 box. This keeps the 5090's API key scheme simple and unchanged.

Alternatively for the CLI: users authenticate directly with their API key
(passed in the `Authorization` header), which the proxy validates and forwards.

### Docker Compose Layout (proxy server)

```yaml
services:
  nginx:
    image: nginx:alpine
    ports: ["80:80", "443:443"]
    volumes:
      - ./frontend/dist:/srv/static
      - ./nginx.conf:/etc/nginx/nginx.conf
      - certbot-certs:/etc/letsencrypt

  api_proxy:
    build: ./proxy_server
    environment:
      - GENERATION_BOX_URL=http://10.0.0.2:8000  # WireGuard address
      - SECRET_KEY=...

  certbot:
    image: certbot/certbot
    # Handles Let's Encrypt cert renewal
```

---

## Phase 4 — Public Cloud Deployment

**Goal:** Move the proxy server from local to a cloud VPS. No changes to the
5090 box or the frontend. The WireGuard tunnel stretches from the cloud VPS
to your home network.

### What Changes

- Provision a small VPS (e.g. DigitalOcean Droplet, Hetzner CX22 — the proxy
  does very little compute work). ~$5–10/month is sufficient.
- Set up WireGuard on the VPS to tunnel to the 5090 box. On the home router,
  forward the WireGuard UDP port to the 5090 (or to a dedicated WireGuard
  router/device).
- Point the domain's DNS to the VPS IP.
- Let's Encrypt certificate via Certbot (already in the Docker Compose).
- The Docker Compose from Phase 3 deploys with no changes.

### What Does Not Change

- The 5090 box (`api_server.py`) — zero changes.
- The frontend — zero changes.
- The CLI (`qm_cli.py`) — users point `--server` at the public domain.

---

## Migration Risk Assessment

| Phase | Risk | Mitigation |
|---|---|---|
| 1 | FastAPI + worker threads in same process — asyncio/sync mixing | Use `run_in_executor` for blocking PipelineAgent calls; well-understood FastAPI pattern |
| 1 | CLI refactor breaks existing workflows | Keep `main.py` standalone path working; new CLI is additive |
| 2 | SSE in browser — reconnection on network blip | Browser EventSource reconnects automatically; no special handling needed |
| 3 | WireGuard setup complexity | WireGuard is simpler than it looks; good tooling on both Linux and Windows |
| 4 | Home IP changes (dynamic DNS) | Use a DDNS service (Cloudflare, DuckDNS) on the home router |

---

## What We Are Not Doing (and Why)

**Celery** — Adds Redis/RabbitMQ as required infrastructure, designed for
horizontal worker scale we don't need. Rewrite cost >> benefit.

**JWT / OAuth** — Over-engineered for the initial user base. API keys for the
CLI, session cookies for the browser — both are standard and sufficient. Can
be added in Phase 4 if needed.

**Separate process per worker** — The current same-process model with daemon
threads is correct at this scale. The VRAMArbiter (a threading.Lock) only
works in-process; moving workers to separate processes would require replacing
it with a file lock or named semaphore, which adds complexity for no gain.

**gRPC / WebSockets for CLI** — HTTP + SSE handles everything we need.
gRPC would complicate the proxy server and add a code generation step.

---

---

## Logging

Good logging is essential for debugging distributed systems where things go
wrong silently across threads, processes, and network hops. Every component
writes structured logs so that when something fails you can say "check the
logs" and get a clear picture of what happened and where.

### Log Levels

| Level | Used for |
|---|---|
| `DEBUG` | Per-request detail, task claim/release, SSE subscriber add/remove |
| `INFO` | Task lifecycle (claimed, completed, failed), pipeline status changes, API requests |
| `WARNING` | Retries, VRAM lock timeouts, ComfyUI free_memory failures, missing optional files |
| `ERROR` | Unhandled exceptions, task failures, auth failures |

Production runs at `INFO`. Set `LOG_LEVEL=DEBUG` in the environment to get
full detail during debugging.

### Log Format

**Structured JSON** for all server-side components. Human-readable `key=value`
is harder to grep, harder to feed into a log viewer, and harder to parse
programmatically. Every log line includes:

```json
{
  "ts": "2026-04-03T12:34:56.789Z",
  "level": "INFO",
  "logger": "quickymesh.workers.trellis",
  "msg": "mesh_generate task completed",
  "pipeline": "my_spaceship",
  "task_id": 42,
  "duration_ms": 18432,
  "user": "jmkel"
}
```

A small `src/logging_config.py` module sets this up once at process start using
Python's standard `logging` + a JSON formatter (e.g. `python-json-logger`).
Both `api_server.py` and `qm_cli.py` call it on startup.

### What Gets Logged, by Component

**API server (`src/api/`)**
- Every request: method, path, user, status code, duration_ms (via FastAPI middleware)
- Auth failures: IP address, attempted key prefix (never the full key)
- SSE: client connected/disconnected, number of active subscribers
- File downloads: pipeline, file type, size_bytes

**Worker threads (`src/agent/worker_threads.py`)**
- Task claimed: task_id, task_type, pipeline, user
- Task completed: task_id, duration_ms
- Task failed: task_id, exception type, message (full traceback at DEBUG)
- VRAM lock acquired/released: which worker, wait_ms
- ComfyUI `free_memory()` called, success/failure

**Broker (`src/broker.py`)**
- Currently minimal — add: enqueue (pipeline, task_type), claim, done, failed, retry

**Concept art workers**
- Generation attempt: pipeline, model/backend, prompt (truncated to 100 chars)
- Retry: attempt number, delay_s, error code
- Success: duration_ms, image_size_bytes

**Trellis / ControlNet workers**
- ComfyUI workflow queued: prompt_id, job_id
- Workflow completed: prompt_id, duration_ms
- Output file found/not found

**CLI client (`qm_cli.py`)**
- Logs to `~/.quickymesh/cli.log` (file) at INFO, and to stderr at WARNING+
- Every API call: method, endpoint, status code, duration_ms
- SSE events received (DEBUG)

**Proxy server (`proxy_server/`, Phase 3)**
- Forwarded requests: upstream URL, response status, duration_ms
- Auth: session validated, session expired
- WireGuard tunnel: upstream unreachable (ERROR with retry count)

### Log Output

| Context | Output |
|---|---|
| `api_server.py` local dev | stderr (pretty-printed) |
| `api_server.py` production | `logs/api.log` (JSON, rotated daily, 7-day retention) |
| Worker threads | Same file as API server (they share the process) |
| `qm_cli.py` | `~/.quickymesh/cli.log` + stderr for warnings/errors |
| Proxy server | `logs/proxy.log` (separate file, same rotation policy) |
| Nginx | Standard access log + error log (managed by Docker) |

**Log rotation** via Python's `logging.handlers.TimedRotatingFileHandler` —
no external logrotate config needed for Phase 1/2. In Phase 3/4 Docker captures
stdout/stderr and log files are volume-mounted for persistence.

### Viewing Logs During Development

```bash
# Tail the API server log with pretty JSON output (requires jq)
tail -f logs/api.log | jq .

# Filter to one pipeline
tail -f logs/api.log | jq 'select(.pipeline == "my_spaceship")'

# Filter to errors only
tail -f logs/api.log | jq 'select(.level == "ERROR")'

# See all VRAM lock events
tail -f logs/api.log | jq 'select(.msg | contains("VRAM"))'
```

### Error Correlation

Every API request gets a `request_id` (a short UUID) injected into the
response header (`X-Request-Id`) and into every log line produced during that
request's lifetime. When debugging a reported failure:

1. Find the `request_id` from the client's log or response headers
2. `grep request_id logs/api.log | jq .`
3. See exactly what happened across middleware, route handler, and any worker
   notifications that fired during that request