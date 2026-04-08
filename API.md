# quickymesh HTTP API Reference

The quickymesh API is a REST API served by `api_server.py`. It is the interface between any frontend (CLI, web app, script, or agent) and the generation pipeline.

Interactive docs (OpenAPI/Swagger) are available at `http://localhost:8000/docs` when the server is running.

---

## Base URL

All endpoints are under `/api/v1/`.

```
http://localhost:8000/api/v1/
```

---

## Authentication

Every request must include an API key in the `Authorization` header:

```
Authorization: Bearer <your-api-key>
```

API keys are configured in `users.yaml` (see `users.yaml.example`). For single-user setups, set the `API_KEY` environment variable instead.

**Roles:**
- `admin` — sees all pipelines across all users, full access.
- `user` — sees only their own pipelines.

**Error responses:**
- `401 Unauthorized` — missing or invalid API key.
- `403 Forbidden` — valid key but insufficient role.

---

## Pipelines

### `POST /api/v1/pipelines` — Create pipeline

Start a new pipeline. Concept art generation is queued immediately and runs in the background.

**Request body:**

```json
{
  "name": "red_dragon",
  "description": "a red dragon breathing fire",
  "num_polys": 8000,
  "input_image_path": null,
  "symmetrize": false,
  "symmetry_axis": "x-",
  "concept_art_backend": "gemini"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Unique identifier, no spaces. |
| `description` | string | Yes | Plain English description of the object. |
| `num_polys` | int | No | Target polygon count. Defaults to `defaults.yaml` setting (8000). |
| `input_image_path` | string | No | Absolute path to an existing image on the server. Used as a base for Gemini modification. Gemini backend only. |
| `symmetrize` | bool | No | Apply mesh symmetrize after approval. Default `false`. |
| `symmetry_axis` | string | No | Axis for symmetrize: `x-`, `x+`, `y-`, `y+`, `z-`, `z+`. Default `x-`. |
| `concept_art_backend` | string | No | `"gemini"` or `"flux"`. Default `"gemini"`. |

**Response:** `201 Created` — full pipeline state object (same as `GET /pipelines/{name}`).

**Errors:** `409 Conflict` if a pipeline with that name already exists.

---

### `GET /api/v1/pipelines` — List pipelines

Returns a summary list of all pipelines visible to the authenticated user.

**Response:**

```json
[
  {
    "name": "red_dragon",
    "status": "concept_art_review",
    "description": "a red dragon breathing fire",
    "concept_art_backend": "gemini",
    "created_at": "2026-04-04T10:00:00",
    "updated_at": "2026-04-04T10:02:30"
  }
]
```

---

### `GET /api/v1/pipelines/{name}` — Get pipeline state

Returns the full pipeline state.

**Response:** Pipeline state object. Key fields:

```json
{
  "name": "red_dragon",
  "description": "a red dragon breathing fire",
  "status": "concept_art_review",
  "num_polys": 8000,
  "symmetrize": false,
  "symmetry_axis": "x-",
  "concept_art_backend": "gemini",
  "concept_arts": [
    {
      "index": 0,
      "status": "completed",
      "image_path": "/path/to/concept_arts/0.png"
    }
  ],
  "meshes": [],
  "created_at": "2026-04-04T10:00:00",
  "updated_at": "2026-04-04T10:02:30"
}
```

**Pipeline statuses:** `initializing`, `concept_art_generating`, `concept_art_review`, `mesh_generating`, `mesh_review`, `approved`, `cancelled`, `paused`, `failed`.

**Concept art statuses:** `pending`, `generating`, `completed`, `approved`, `regenerating`, `failed`.

**Mesh statuses:** `queued`, `generating`, `texturing`, `screenshotting`, `awaiting_approval`, `approved`, `failed`.

**Errors:** `404` if the pipeline does not exist.

---

### `DELETE /api/v1/pipelines/{name}` — Cancel pipeline

Cancels all pending tasks and marks the pipeline as cancelled. The pipeline directory is not deleted.

**Response:** `{"status": "ok"}`

---

### `PATCH /api/v1/pipelines/{name}` — Edit pipeline settings

Update description, polygon count, or symmetry settings. Only allowed before mesh generation starts (statuses: `initializing`, `concept_art_generating`, `concept_art_review`).

**Request body (all fields optional):**

```json
{
  "description": "a blue dragon breathing ice",
  "num_polys": 12000,
  "symmetrize": true,
  "symmetry_axis": "x-"
}
```

**Response:** Updated pipeline state object.

**Errors:** `409 Conflict` if the pipeline is past the concept art stage.

---

### `POST /api/v1/pipelines/{name}/pause` — Pause pipeline

Pauses a running pipeline. In-flight worker tasks complete before the pipeline halts.

**Response:** `{"status": "ok"}`

---

### `POST /api/v1/pipelines/{name}/resume` — Resume pipeline

Resumes a paused pipeline.

**Response:** `{"status": "ok"}`

**Errors:** `409 Conflict` if the pipeline is not paused.

---

### `POST /api/v1/pipelines/{name}/retry` — Retry failed tasks

Resets failed broker tasks so workers will pick up again.

**Response:** `{"status": "ok", "tasks_reset": 2}`

---

## 3D Pipelines
These endpoints handle 3D mesh generation, which can be triggered from 2D pipeline references or direct image uploads.

### `POST /api/v1/3d-pipelines/from-ref` — Create from 2D reference
Start a 3D pipeline using an existing 2D pipeline's concept art.

**Request body:**
```json
{
  "pipeline_name": "red_dragon",
  "concept_art_index": 0,
  "concept_art_version": "1",
  "num_polys": 8000,
  "symmetrize": true,
  "symmetry_axis": "x-"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `pipeline_name` | string | Yes | Name of the source 2D pipeline. |
| `concept_art_index` | int | Yes | 0-based index of the concept art. |
| `concept_art_version` | string | No | Version of the concept art. |
| `num_pol_count` | int | No | Target polygon count. |
| `symmetrize` | bool | No | Apply symmetry. |
| `symmetry_axis` | string | No | `x-`, `x+`, `y-`, `y+`, `z-`, `z+`. |

### `POST /api/v1/3d-pipelines/from-upload` — Create from upload
Start a 3D pipeline by uploading an image.

**Request (multipart/form-data):**
- `name`: Unique name for the 3D pipeline.
- `image`: The image file.
- `num_polys`: (Optional) Target polygon count.
- `symmetrize`: (Optional) Boolean.
- `symmetry_axis`: (Optional) Axis.

### `GET /api/v1/3d-pipelines` — List 3D pipelines
Returns a list of all 3D pipelines.

### `GET /api/v1/3d-pipelines/{name}` — Get 3D pipeline state
Returns the full state of a 3D pipeline.

### `DELETE /api/v1/3d-pipelines/{name}` — Cancel 3D pipeline
Cancels the 3D pipeline.

### `GET /api/v1/3d-pipelines/{name}/sheet` — Review sheet
Returns the mesh review sheet PNG.

### `GET /api/v1/3d-pipelines/{name}/screenshot/{filename}` — Single screenshot
Returns a specific screenshot PNG.

### `GET /api/v1/3d-pipelines/{name}/preview` — HTML preview
Returns the Three.js HTML preview.

### `GET /api/v1/3d-pipelines/{name}/mesh` — Download GLB
Returns the textured `.glb` mesh.

### `POST /api/v1/3d-pipelines/{name}/approve` — Approve mesh
Approves the mesh and exports it.

**Request body:**
```json
{
  "asset_name": "dragon_final",
  "export_format": "glb"
}
```

### `POST /api/v1/3d-pipelines/{name}/reject` — Reject mesh
Rejects the mesh and re-queues generation with updated settings.

---

## Concept art review

### `GET /api/v1/pipelines/{name}/concept_art/sheet` — Download review sheet

Returns a PNG grid of all concept art images with labels (256×256 per image).

**Response:** `image/png`

---

### `GET /api/v1/pipelines/{name}/concept_art/{idx}` — Download one image

Returns a single concept art PNG by 0-based index.

**Response:** `image/png`

**Errors:** `404` if the index is out of range or the image is not yet generated.

---

### `POST /api/v1/pipelines/{name}/concept_art/approve` — Approve images

Approves the specified concept art images and queues mesh generation. The pipeline transitions to `mesh_generating`.

**Request body:**

```json
{
  "indices": [0, 2]
}
```

Indices are **0-based**.

**Response:** `{"status": "ok"}`

---

### `POST /api/v1/pipelines/{name}/concept_art/regenerate` — Regenerate images

Re-queues generation for specific images (or all if `indices` is omitted).

**Request body:**

```json
{
  "indices": [1, 3],
  "description_override": "a darker red dragon"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `indices` | int[] | No | 0-based indices to regenerate. Omit to regenerate all. |
| `description_override` | string | No | Replace the pipeline description for this regeneration only. |

**Response:** `{"status": "accepted", "message": "Regenerating 2 image(s)"}`

---

### `POST /api/v1/pipelines/{name}/concept_art/modify` — Modify one image (Gemini only)

Modifies a concept art image using Gemini's image editing API. The modified image replaces the original at that index.

**Request body:**

```json
{
  "index": 1,
  "instruction": "Remove the wings and make the body larger"
}
```

**Response:** `{"status": "ok"}`

**Errors:** `409 Conflict` if the active concept art backend does not support modification (e.g., FLUX).

---

### `POST /api/v1/pipelines/{name}/concept_art/restyle` — Restyle image (ControlNet)

Restyls a concept art image using ControlNet Canny + Juggernaut-XL. Useful for changing the art style while preserving the shape.

**Request body:**

```json
{
  "index": 0,
  "positive": "fantasy oil painting, highly detailed, dramatic lighting",
  "negative": "blurry, low quality, text, watermark",
  "denoise": 0.75
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `index` | int | Yes | 0-based index of the image to restyle. |
| `positive` | string | Yes | Positive style prompt. |
| `negative` | string | No | Negative prompt. Defaults to common quality exclusions. |
| `denoise` | float | No | Denoising strength (0.1–1.0). Higher = more style change. Default 0.75. |

**Response:** `{"status": "ok"}`

**Errors:** `409 Conflict` if the ControlNet restyle worker is not configured.

---

## Mesh review

### `GET /api/v1/pipelines/{name}/meshes/{mesh_name}/sheet` — Review sheet

Returns the mesh review sheet PNG (6-view screenshot grid).

**Response:** `image/png`

---

### `GET /api/v1/pipelines/{name}/meshes/{mesh_name}/screenshot/{filename}` — Single screenshot

Returns a single screenshot PNG by filename (e.g., `render_front.png`).

**Response:** `image/png`

---

### `GET /api/v1/pipelines/{name}/meshes/{mesh_name}/preview` — HTML preview

Returns the Three.js HTML file for an interactive 3-D viewer in a browser.

**Response:** `text/html`

---

### `GET /api/v1/pipelines/{name}/meshes/{mesh_name}/mesh` — Download GLB

Returns the textured `.glb` mesh file.

**Response:** `model/gltf-binary`

---

### `POST /api/v1/pipelines/{name}/meshes/{mesh_name}/approve` — Approve mesh

Approves a mesh and gives it a final asset name. When all pending meshes are reviewed, the approved ones are exported to `final_game_ready_assets/`.

**Request body:**

```json
{
  "asset_name": "red_dragon_final",
  "export_format": "glb"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `asset_name` | string | Yes | Name for the exported asset file (without extension). |
| `export_format` | string | No | `"glb"` or `"obj"`. Defaults to server config. |

**Response:** `{"status": "ok"}`

**Errors:** `409 Conflict` if the mesh is not in `awaiting_approval` status.

---

### `POST /api/v1/pipelines/{name}/meshes/{mesh_name}/reject` — Reject mesh

Rejects a mesh. Optionally updates generation settings for the retry. When all pending meshes are reviewed, remaining approved ones are exported; if all were rejected, mesh generation is re-queued.

**Request body (all fields optional):**

```json
{
  "num_polys": 12000,
  "symmetrize": true,
  "symmetry_axis": "x-"
}
```

**Response:** `{"status": "ok"}`

---

## Assets (completed pipelines)

### `GET /api/v1/assets` — List exported assets

Returns a list of all exported game-ready assets.

**Response:**

```json
[
  {
    "name": "red_dragon_final",
    "filename": "red_dragon_final.glb",
    "size_bytes": 2048000
  }
]
```

---

### `GET /api/v1/assets/{name}/mesh` — Download exported asset

Downloads the final exported mesh by asset name (without extension).

**Response:** `model/gltf-binary` or `application/octet-stream`

**Errors:** `404` if no asset with that name exists.

---

## System

### `GET /api/v1/status` — Server health

Returns worker thread health and pipeline queue summary.

**Response:**

```json
{
  "workers": [
    {"name": "ConceptArtWorkerThread", "alive": true},
    {"name": "TrellisWorkerThread", "alive": true},
    {"name": "ScreenshotWorkerThread", "alive": true}
  ],
  "all_workers_alive": true,
  "pipeline_count": 2,
  "pipelines": [
    {
      "name": "red_dragon",
      "status": "concept_art_review",
      "queued_tasks": 0,
      "failed_tasks": 0
    }
  ]
}
```

---

## Real-time events (SSE)

The server pushes events over Server-Sent Events (SSE). Connect to an event stream to receive live updates without polling.

### `GET /api/v1/pipelines/{name}/events` — Single pipeline stream

Subscribe to events for one pipeline.

### `GET /api/v1/events` — Global stream

Subscribe to events for all of the authenticated user's pipelines.

### Using SSE

```javascript
const es = new EventSource('/api/v1/events', {
  headers: { 'Authorization': 'Bearer your-api-key' }
});

es.onmessage = (e) => {
  const event = JSON.parse(e.data);
  console.log(event);
};
```

### Event types

All events are JSON objects. The `event` field identifies the type.

**`status_change`** — Pipeline transitioned to a new status.
```json
{
  "event": "status_change",
  "pipeline": "red_dragon",
  "status": "concept_art_review",
  "message": "Concept art ready for review"
}
```

**`pipeline_created`** — A new pipeline was started.
```json
{
  "event": "pipeline_created",
  "pipeline": "red_dragon",
  "status": "initializing",
  "user": "alice"
}
```

**`concept_art_updated`** — A concept art image was modified or restyled.
```json
{
  "event": "concept_art_updated",
  "pipeline": "red_dragon",
  "index": 1
}
```

**`heartbeat`** — Sent every 15 seconds to keep the connection alive.
```json
{"event": "heartbeat"}
```

The browser `EventSource` API reconnects automatically if the connection drops. No special handling is needed.

---

## Error format

All errors return a JSON body:

```json
{
  "detail": "Pipeline 'red_dragon' not found"
}
```

| Status | Meaning |
|---|---|
| `400` | Bad request — malformed JSON or invalid field value |
| `401` | Unauthorized — missing or invalid API key |
| `404` | Not found — pipeline, mesh, or asset does not exist |
| `409` | Conflict — action not allowed in the current pipeline state |
| `422` | Unprocessable entity — index out of range or invalid parameter |
| `500` | Internal server error — check `logs/api.log` |

---

## Quick example: full pipeline flow

```python
import httpx
import time

BASE = "http://localhost:8000/api/v1"
HEADERS = {"Authorization": "Bearer your-api-key"}

# 1. Create pipeline
r = httpx.post(f"{BASE}/pipelines", headers=HEADERS, json={
    "name": "my_dragon",
    "description": "a red dragon breathing fire",
    "concept_art_backend": "gemini",
})
print(r.json()["status"])  # "initializing"

# 2. Poll until concept art is ready
while True:
    state = httpx.get(f"{BASE}/pipelines/my_dragon", headers=HEADERS).json()
    if state["status"] == "concept_art_review":
        break
    time.sleep(5)

# 3. Download review sheet
sheet = httpx.get(f"{BASE}/pipelines/my_dragon/concept_art/sheet", headers=HEADERS)
with open("review_sheet.png", "wb") as f:
    f.write(sheet.content)

# 4. Approve images 0 and 2
httpx.post(f"{BASE}/pipelines/my_dragon/concept_art/approve", headers=HEADERS,
           json={"indices": [0, 2]})

# 5. Poll until mesh review is ready
while True:
    state = httpx.get(f"{BASE}/pipelines/my_dragon", headers=HEADERS).json()
    if state["status"] == "mesh_review":
        break
    time.sleep(10)

# 6. Approve the first mesh
mesh_name = state["meshes"][0]["sub_name"]
httpx.post(f"{BASE}/pipelines/my_dragon/meshes/{mesh_name}/approve",
           headers=HEADERS, json={"asset_name": "dragon_final"})

# 7. Download the exported asset
asset = httpx.get(f"{BASE}/assets/dragon_final/mesh", headers=HEADERS)
with open("dragon_final.glb", "wb") as f:
    f.write(asset.content)
```
