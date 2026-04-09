"""
Integration tests for the quickymesh FastAPI routers.

Uses FastAPI's TestClient (synchronous httpx transport — no real uvicorn needed).
The PipelineAgent is replaced with a lightweight fake that operates on real
state.json files in a tmp_path, so pipeline logic is exercised but no workers
run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

import src.api.auth as auth_mod
from src.api.app import create_app
from src.api.event_bus import event_bus
from src.broker import Broker
from src.config import Config
from src.state import (
    ConceptArtItem,
    ConceptArtStatus,
    Pipeline3DState,
    Pipeline3DStatus,
    PipelineState,
    PipelineStatus,
)
from src.workers.concept_art import MockConceptArtWorker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_API_KEY = "test-key-abc"
_AUTH = {"Authorization": f"Bearer {_API_KEY}"}


@pytest.fixture
def users_yaml(tmp_path) -> Path:
    f = tmp_path / "users.yaml"
    f.write_text(
        f"users:\n"
        f"  testuser:\n"
        f"    api_key: \"{_API_KEY}\"\n"
        f"    role: admin\n"
    )
    return f


@pytest.fixture
def cfg(tmp_path) -> Config:
    defaults = {
        "gemini": {"model": "m", "alternative_model": "a"},
        "generation": {
            "num_concept_arts": 2,
            "num_polys": 8000,
            "review_sheet_thumb_size": 64,
            "html_preview_size": 256,
            "export_format": "glb",
            "background_suffix": "plain white background",
            "concept_art_image_size": 512,
        },
        "infrastructure": {
            "comfyui_url": "http://localhost:8188",
            "comfyui_install_dir": "/fake",
            "comfyui_output_dir": "/fake/output",
            "comfyui_poll_interval": 2.0,
            "comfyui_timeout": 600.0,
            "blender_path": "/fake/blender",
            "vram_lock_timeout": 30.0,
        },
        "output": {"root": str(tmp_path / "output")},
    }
    d = tmp_path / "defaults.yaml"
    d.write_text(yaml.dump(defaults))
    e = tmp_path / ".env"
    e.write_text("GEMINI_API_KEY=fake\n")
    return Config(defaults_path=d, env_path=e)


@pytest.fixture
def broker(cfg) -> Broker:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    b = Broker(cfg.output_root / "tasks.db")
    yield b
    b.close()


@pytest.fixture
def agent(cfg, broker):
    from src.agent.pipeline_agent import PipelineAgent
    from src.vram_arbiter import VRAMArbiter
    from src.workers.screenshot import MockScreenshotWorker
    from src.workers.trellis import MockTrellisWorker

    a = PipelineAgent(
        broker=broker,
        arbiter=VRAMArbiter(),
        cfg=cfg,
        concept_worker=MockConceptArtWorker(),
        trellis_worker=MockTrellisWorker(),
        screenshot_worker=MockScreenshotWorker(),
    )
    a._threads = [MagicMock(is_alive=MagicMock(return_value=True),
                            __class__=MagicMock(__name__="FakeWorker"))]
    return a


@pytest.fixture
def client(agent, cfg, users_yaml):
    app = create_app(
        agent=agent,
        cfg=cfg,
        concept_worker=MockConceptArtWorker(),
        restyle_worker=None,
        users_file=str(users_yaml),
    )
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper: seed a pipeline into the filesystem
# ---------------------------------------------------------------------------


def _seed_pipeline(
    cfg: Config,
    name: str = "testpipe",
    status: PipelineStatus = PipelineStatus.CONCEPT_ART_GENERATING,
    concept_arts: list[ConceptArtItem] | None = None,
) -> PipelineState:
    pipeline_dir = cfg.pipelines_dir / name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    state = PipelineState(
        name=name,
        description="a dragon",
        num_polys=8000,
        status=status,
        concept_arts=concept_arts or [],
        pipeline_dir=str(pipeline_dir.relative_to(cfg.output_root)),
    )
    state.save(pipeline_dir / "state.json")
    return state


def _seed_concept_art_image(cfg: Config, name: str, idx: int, version: int = 0) -> Path:
    """Create a minimal 1×1 PNG at the versioned path so FileResponse can serve it."""
    ca_dir = cfg.pipelines_dir / name / "concept_art"
    ca_dir.mkdir(parents=True, exist_ok=True)
    img_path = ca_dir / f"concept_art_{idx + 1}_{version}.png"
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    img.save(str(img_path))
    return img_path


def _seed_3d_pipeline(
    cfg: Config,
    name: str,
    status: Pipeline3DStatus = Pipeline3DStatus.AWAITING_APPROVAL,
    glb_content: bytes = b"glTF",
) -> Pipeline3DState:
    """Seed a 3D pipeline with a fake GLB file."""
    pipeline_dir = cfg.pipelines_dir / name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    # Create a fake input image
    img_path = pipeline_dir / "input.png"
    Image.new("RGB", (64, 64)).save(str(img_path))
    # Create a fake GLB
    glb_path = pipeline_dir / "meshes" / "textured.glb"
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(glb_content)

    state = Pipeline3DState(
        name=name,
        input_image_path=str(img_path),
        num_polys=8000,
        status=status,
        textured_mesh_path=str(glb_path),
        pipeline_dir=str(pipeline_dir.relative_to(cfg.output_root)),
    )
    state.save(pipeline_dir / "state.json")
    return state


# ===========================================================================
# Auth tests
# ===========================================================================


class TestAuth:
    def test_missing_auth_header_returns_401(self, client):
        resp = client.get("/api/v1/pipelines")
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self, client):
        resp = client.get("/api/v1/pipelines",
                          headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

    def test_valid_api_key_returns_200(self, client):
        resp = client.get("/api/v1/pipelines", headers=_AUTH)
        assert resp.status_code == 200


class TestAuthDisabled:
    """When create_app is called with auth_enabled=False (the OSS default),
    every request should be accepted and treated as a synthetic local admin."""

    @pytest.fixture
    def noauth_client(self, agent, cfg):
        app = create_app(
            agent=agent,
            cfg=cfg,
            concept_worker=MockConceptArtWorker(),
            restyle_worker=None,
            auth_enabled=False,
        )
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    def test_no_header_succeeds(self, noauth_client):
        resp = noauth_client.get("/api/v1/pipelines")
        assert resp.status_code == 200

    def test_garbage_header_succeeds(self, noauth_client):
        resp = noauth_client.get(
            "/api/v1/pipelines",
            headers={"Authorization": "Bearer literally-anything"},
        )
        assert resp.status_code == 200

    def test_default_is_disabled_when_no_users_file(self, agent, cfg):
        """Without a users_file and without explicit auth_enabled, auth is OFF."""
        app = create_app(
            agent=agent,
            cfg=cfg,
            concept_worker=MockConceptArtWorker(),
            restyle_worker=None,
        )
        with TestClient(app, raise_server_exceptions=True) as c:
            resp = c.get("/api/v1/pipelines")
            assert resp.status_code == 200


# ===========================================================================
# Pipelines CRUD
# ===========================================================================


class TestCreatePipeline:
    def test_create_returns_201(self, client):
        resp = client.post("/api/v1/pipelines", headers=_AUTH, json={
            "name": "mypipe",
            "description": "a spaceship",
        })
        assert resp.status_code == 201
        assert resp.json()["name"] == "mypipe"

    def test_create_sets_concept_art_backend(self, client, cfg):
        client.post("/api/v1/pipelines", headers=_AUTH, json={
            "name": "fluxpipe",
            "description": "a robot",
            "concept_art_backend": "flux",
        })
        state = client.get("/api/v1/pipelines/fluxpipe", headers=_AUTH).json()
        assert state["concept_art_backend"] == "flux"

    def test_create_duplicate_returns_409(self, client, cfg):
        _seed_pipeline(cfg, "dup")
        resp = client.post("/api/v1/pipelines", headers=_AUTH, json={
            "name": "dup",
            "description": "collision",
        })
        assert resp.status_code == 409

    def test_create_publishes_sse_event(self, client, cfg):
        import asyncio
        loop = asyncio.new_event_loop()
        event_bus.set_loop(loop)
        q = event_bus.subscribe(pipeline_name="newpipe")

        client.post("/api/v1/pipelines", headers=_AUTH, json={
            "name": "newpipe",
            "description": "test",
        })

        async def drain():
            return await asyncio.wait_for(q.get(), timeout=1.0)

        event = loop.run_until_complete(drain())
        event_bus.unsubscribe(q)
        loop.close()
        assert event["event"] == "pipeline_created"
        assert event["pipeline"] == "newpipe"


class TestListPipelines:
    def test_empty_returns_empty_list(self, client):
        resp = client.get("/api/v1/pipelines", headers=_AUTH)
        assert resp.json() == []

    def test_returns_seeded_pipelines(self, client, cfg):
        _seed_pipeline(cfg, "pipe1")
        _seed_pipeline(cfg, "pipe2")
        resp = client.get("/api/v1/pipelines", headers=_AUTH)
        names = [p["name"] for p in resp.json()]
        assert "pipe1" in names
        assert "pipe2" in names

    def test_does_not_return_3d_pipelines(self, client, cfg):
        _seed_pipeline(cfg, "twodpipe")
        _seed_3d_pipeline(cfg, "threedpipe")
        resp = client.get("/api/v1/pipelines", headers=_AUTH)
        names = [p["name"] for p in resp.json()]
        assert "twodpipe" in names
        assert "threedpipe" not in names


class TestGetPipeline:
    def test_returns_state(self, client, cfg):
        _seed_pipeline(cfg, "mypipe")
        resp = client.get("/api/v1/pipelines/mypipe", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json()["name"] == "mypipe"

    def test_missing_pipeline_returns_404(self, client):
        resp = client.get("/api/v1/pipelines/ghost", headers=_AUTH)
        assert resp.status_code == 404


class TestCancelPipeline:
    def test_cancel_sets_status_cancelled(self, client, cfg):
        _seed_pipeline(cfg, "topipe")
        client.delete("/api/v1/pipelines/topipe", headers=_AUTH)
        state = client.get("/api/v1/pipelines/topipe", headers=_AUTH).json()
        assert state["status"] == "cancelled"

    def test_cancel_missing_returns_404(self, client):
        assert client.delete("/api/v1/pipelines/ghost", headers=_AUTH).status_code == 404


class TestPatchPipeline:
    def test_patch_description(self, client, cfg):
        _seed_pipeline(cfg, "editpipe", status=PipelineStatus.CONCEPT_ART_REVIEW)
        resp = client.patch("/api/v1/pipelines/editpipe", headers=_AUTH,
                            json={"description": "updated desc"})
        assert resp.status_code == 200
        assert resp.json()["description"] == "updated desc"

    def test_patch_num_polys(self, client, cfg):
        _seed_pipeline(cfg, "polypipe", status=PipelineStatus.CONCEPT_ART_REVIEW)
        client.patch("/api/v1/pipelines/polypipe", headers=_AUTH, json={"num_polys": 12000})
        state = client.get("/api/v1/pipelines/polypipe", headers=_AUTH).json()
        assert state["num_polys"] == 12000

    def test_patch_cancelled_pipeline_returns_409(self, client, cfg):
        _seed_pipeline(cfg, "cancelpipe", status=PipelineStatus.CANCELLED)
        resp = client.patch("/api/v1/pipelines/cancelpipe", headers=_AUTH,
                            json={"num_polys": 5000})
        assert resp.status_code == 409


class TestPausePipeline:
    def test_pause_sets_paused_status(self, client, cfg):
        _seed_pipeline(cfg, "runpipe", status=PipelineStatus.CONCEPT_ART_GENERATING)
        client.post("/api/v1/pipelines/runpipe/pause", headers=_AUTH)
        state = client.get("/api/v1/pipelines/runpipe", headers=_AUTH).json()
        assert state["status"] == "paused"

    def test_resume_non_paused_returns_409(self, client, cfg):
        _seed_pipeline(cfg, "activepipe", status=PipelineStatus.CONCEPT_ART_GENERATING)
        resp = client.post("/api/v1/pipelines/activepipe/resume", headers=_AUTH)
        assert resp.status_code == 409


class TestRetryPipeline:
    def test_retry_resets_failed_tasks(self, client, cfg, broker):
        _seed_pipeline(cfg, "failpipe")
        state_path = cfg.pipelines_dir / "failpipe" / "state.json"
        task_id = broker.enqueue("failpipe", "concept_art_generate",
                                 {"pipeline_name": "failpipe",
                                  "state_path": str(state_path)})
        broker.mark_failed(task_id, "test error")
        resp = client.post("/api/v1/pipelines/failpipe/retry", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json()["tasks_reset"] == 1


# ===========================================================================
# Concept art review
# ===========================================================================


class TestConceptArtRegenerate:
    def _make_state(self, cfg, name="regenpipe"):
        arts = []
        for i in range(2):
            _seed_concept_art_image(cfg, name, i)
            arts.append(ConceptArtItem(index=i, status=ConceptArtStatus.READY))
        _seed_pipeline(cfg, name, status=PipelineStatus.CONCEPT_ART_REVIEW,
                       concept_arts=arts)

    def test_regenerate_specific_indices_queues_task(self, client, cfg, broker):
        self._make_state(cfg)
        resp = client.post("/api/v1/pipelines/regenpipe/concept_art/regenerate",
                           headers=_AUTH, json={"indices": [0]})
        assert resp.status_code == 200
        tasks = broker.get_tasks(task_type="concept_art_generate")
        assert len(tasks) == 1

    def test_regenerate_all_queues_task_with_no_indices(self, client, cfg, broker):
        self._make_state(cfg)
        resp = client.post("/api/v1/pipelines/regenpipe/concept_art/regenerate",
                           headers=_AUTH, json={})
        assert resp.status_code == 200
        tasks = broker.get_tasks(task_type="concept_art_generate")
        assert len(tasks) == 1

    def test_regenerate_updates_description(self, client, cfg):
        self._make_state(cfg)
        client.post("/api/v1/pipelines/regenpipe/concept_art/regenerate",
                    headers=_AUTH, json={"description_override": "a new dragon"})
        state = client.get("/api/v1/pipelines/regenpipe", headers=_AUTH).json()
        assert state["description"] == "a new dragon"


class TestConceptArtModify:
    def _make_state_with_image(self, cfg, name="modpipe"):
        _seed_concept_art_image(cfg, name, 0)
        arts = [ConceptArtItem(index=0, status=ConceptArtStatus.READY)]
        _seed_pipeline(cfg, name, status=PipelineStatus.CONCEPT_ART_REVIEW,
                       concept_arts=arts)

    def test_modify_with_supporting_worker_returns_200(self, client, cfg):
        self._make_state_with_image(cfg)
        resp = client.post("/api/v1/pipelines/modpipe/concept_art/modify",
                           headers=_AUTH,
                           json={"index": 0, "instruction": "make it red"})
        assert resp.status_code == 200

    def test_modify_with_non_supporting_worker_returns_409(self, client, cfg, agent, users_yaml):
        from src.workers.concept_art import FluxComfyUIConceptArtWorker
        mock_flux = MagicMock(spec=FluxComfyUIConceptArtWorker)
        mock_flux.supports_modify = False

        from src.api.app import create_app as _ca
        app2 = _ca(agent=agent, cfg=cfg, concept_worker=mock_flux,
                   users_file=str(users_yaml))
        with TestClient(app2) as c:
            self._make_state_with_image(cfg)
            resp = c.post("/api/v1/pipelines/modpipe/concept_art/modify",
                          headers=_AUTH,
                          json={"index": 0, "instruction": "make it blue"})
        assert resp.status_code == 409

    def test_modify_out_of_range_returns_422(self, client, cfg):
        self._make_state_with_image(cfg)
        resp = client.post("/api/v1/pipelines/modpipe/concept_art/modify",
                           headers=_AUTH,
                           json={"index": 99, "instruction": "make it gold"})
        assert resp.status_code == 422

    def test_modify_with_source_version_forwards_to_task(self, client, cfg, broker):
        # Slot 0 is on version 2; seed images for both v0 and v2 on disk.
        _seed_concept_art_image(cfg, "modpipev", 0, version=0)
        _seed_concept_art_image(cfg, "modpipev", 0, version=2)
        arts = [ConceptArtItem(index=0, version=2, status=ConceptArtStatus.READY)]
        _seed_pipeline(cfg, "modpipev", status=PipelineStatus.CONCEPT_ART_REVIEW,
                       concept_arts=arts)
        resp = client.post(
            "/api/v1/pipelines/modpipev/concept_art/modify",
            headers=_AUTH,
            json={"index": 0, "instruction": "redo", "source_version": 0},
        )
        assert resp.status_code == 200
        tasks = broker.get_tasks(task_type="concept_art_modify")
        assert len(tasks) == 1
        assert tasks[0].payload["source_version"] == 0

    def test_modify_with_bad_source_version_returns_422(self, client, cfg):
        self._make_state_with_image(cfg)  # slot 0 at version 0
        resp = client.post(
            "/api/v1/pipelines/modpipe/concept_art/modify",
            headers=_AUTH,
            json={"index": 0, "instruction": "x", "source_version": 5},
        )
        assert resp.status_code == 422

    def test_modify_with_missing_source_version_file_returns_404(self, client, cfg):
        # Slot at version 2 but the v1 file is missing on disk
        _seed_concept_art_image(cfg, "modpipemissing", 0, version=0)
        _seed_concept_art_image(cfg, "modpipemissing", 0, version=2)
        arts = [ConceptArtItem(index=0, version=2, status=ConceptArtStatus.READY)]
        _seed_pipeline(cfg, "modpipemissing",
                       status=PipelineStatus.CONCEPT_ART_REVIEW,
                       concept_arts=arts)
        resp = client.post(
            "/api/v1/pipelines/modpipemissing/concept_art/modify",
            headers=_AUTH,
            json={"index": 0, "instruction": "x", "source_version": 1},
        )
        assert resp.status_code == 404


# ===========================================================================
# Concept art image download
# ===========================================================================


class TestConceptArtDownload:
    def test_download_existing_image_returns_png(self, client, cfg):
        _seed_concept_art_image(cfg, "dlpipe", 0)
        arts = [ConceptArtItem(index=0, status=ConceptArtStatus.READY)]
        _seed_pipeline(cfg, "dlpipe", status=PipelineStatus.CONCEPT_ART_REVIEW,
                       concept_arts=arts)
        resp = client.get("/api/v1/pipelines/dlpipe/concept_art/0", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_download_missing_index_returns_404(self, client, cfg):
        _seed_pipeline(cfg, "dlpipe2")
        resp = client.get("/api/v1/pipelines/dlpipe2/concept_art/5", headers=_AUTH)
        assert resp.status_code == 404


# ===========================================================================
# Assets
# ===========================================================================


class TestAssets:
    def test_empty_assets_dir_returns_empty_list(self, client, cfg):
        resp = client.get("/api/v1/assets", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_glb_files(self, client, cfg):
        assets_dir = cfg.final_assets_dir
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / "my_ship.glb").write_bytes(b"GLB")
        resp = client.get("/api/v1/assets", headers=_AUTH)
        names = [a["name"] for a in resp.json()]
        assert "my_ship" in names

    def test_download_asset_returns_file(self, client, cfg):
        assets_dir = cfg.final_assets_dir
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / "my_tank.glb").write_bytes(b"GLB_DATA")
        resp = client.get("/api/v1/assets/my_tank/mesh", headers=_AUTH)
        assert resp.status_code == 200

    def test_download_missing_asset_returns_404(self, client, cfg):
        resp = client.get("/api/v1/assets/ghost/mesh", headers=_AUTH)
        assert resp.status_code == 404


# ===========================================================================
# Status
# ===========================================================================


class TestStatus:
    def test_status_returns_worker_info(self, client):
        resp = client.get("/api/v1/status", headers=_AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "workers" in data
        assert "pipeline_count" in data

    def test_status_includes_pipeline_list(self, client, cfg):
        _seed_pipeline(cfg, "statuspipe")
        resp = client.get("/api/v1/status", headers=_AUTH)
        names = [p["name"] for p in resp.json()["pipelines"]]
        assert "statuspipe" in names


class TestCreatePipelineFromUpload:
    def _png_bytes(self) -> bytes:
        import io
        buf = io.BytesIO()
        Image.new("RGB", (4, 4), color=(0, 255, 0)).save(buf, format="PNG")
        return buf.getvalue()

    def test_create_from_upload_returns_201(self, client, cfg):
        resp = client.post(
            "/api/v1/pipelines/from-upload",
            headers=_AUTH,
            data={
                "name": "uppipe",
                "description": "a robot",
                "num_polys": "8000",
                "symmetrize": "false",
                "symmetry_axis": "x-",
                "concept_art_backend": "gemini",
            },
            files={"image": ("base.png", self._png_bytes(), "image/png")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "uppipe"
        # File was saved into the pipeline dir
        saved = list((cfg.pipelines_dir / "uppipe").glob("input_*.png"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == self._png_bytes()

    def test_create_from_upload_409_on_duplicate(self, client, cfg):
        _seed_pipeline(cfg, "dup")
        resp = client.post(
            "/api/v1/pipelines/from-upload",
            headers=_AUTH,
            data={"name": "dup", "description": "x"},
            files={"image": ("b.png", self._png_bytes(), "image/png")},
        )
        assert resp.status_code == 409


class TestPatchHidden:
    def test_patch_2d_hidden_true(self, client, cfg):
        _seed_pipeline(cfg, "hpipe", status=PipelineStatus.CONCEPT_ART_REVIEW)
        resp = client.patch(
            "/api/v1/pipelines/hpipe", headers=_AUTH, json={"hidden": True}
        )
        assert resp.status_code == 200
        assert resp.json()["hidden"] is True

    def test_patch_2d_hidden_allowed_when_cancelled(self, client, cfg):
        # hidden must be settable even when the editable window is closed
        _seed_pipeline(cfg, "hpipe2", status=PipelineStatus.CANCELLED)
        resp = client.patch(
            "/api/v1/pipelines/hpipe2", headers=_AUTH, json={"hidden": True}
        )
        assert resp.status_code == 200
        assert resp.json()["hidden"] is True

    def test_patch_2d_description_blocked_when_cancelled(self, client, cfg):
        _seed_pipeline(cfg, "hpipe3", status=PipelineStatus.CANCELLED)
        resp = client.patch(
            "/api/v1/pipelines/hpipe3", headers=_AUTH,
            json={"description": "new"},
        )
        assert resp.status_code == 409

    def test_patch_3d_hidden_true(self, client, cfg):
        _seed_3d_pipeline(cfg, "h3d")
        resp = client.patch(
            "/api/v1/3d-pipelines/h3d", headers=_AUTH, json={"hidden": True}
        )
        assert resp.status_code == 200
        assert resp.json()["hidden"] is True

    def test_patch_3d_hidden_allowed_at_idle(self, client, cfg):
        _seed_3d_pipeline(cfg, "h3d2", status=Pipeline3DStatus.IDLE)
        resp = client.patch(
            "/api/v1/3d-pipelines/h3d2", headers=_AUTH, json={"hidden": True}
        )
        assert resp.status_code == 200
        assert resp.json()["hidden"] is True

    def test_patch_3d_num_polys_at_awaiting(self, client, cfg):
        _seed_3d_pipeline(cfg, "h3d3", status=Pipeline3DStatus.AWAITING_APPROVAL)
        resp = client.patch(
            "/api/v1/3d-pipelines/h3d3", headers=_AUTH, json={"num_polys": 1234}
        )
        assert resp.status_code == 200
        assert resp.json()["num_polys"] == 1234

    def test_patch_3d_num_polys_blocked_at_idle(self, client, cfg):
        _seed_3d_pipeline(cfg, "h3d4", status=Pipeline3DStatus.IDLE)
        resp = client.patch(
            "/api/v1/3d-pipelines/h3d4", headers=_AUTH, json={"num_polys": 1234}
        )
        assert resp.status_code == 409

    def test_patch_3d_404_for_missing(self, client):
        resp = client.patch(
            "/api/v1/3d-pipelines/ghost", headers=_AUTH, json={"hidden": True}
        )
        assert resp.status_code == 404


class TestRetry3DPipeline:
    def test_retry_resets_failed_3d_tasks(self, client, cfg, broker):
        _seed_3d_pipeline(cfg, "r3d")
        tid1 = broker.enqueue("r3d", "mesh_generate", {})
        tid2 = broker.enqueue("r3d", "mesh_texture", {})
        broker.mark_failed(tid1, "boom")
        broker.mark_failed(tid2, "boom")
        resp = client.post("/api/v1/3d-pipelines/r3d/retry", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json()["tasks_reset"] == 2

    def test_retry_404_for_missing_3d_pipeline(self, client):
        resp = client.post("/api/v1/3d-pipelines/ghost/retry", headers=_AUTH)
        assert resp.status_code == 404


class TestPipelinesWithFailures:
    def test_empty_when_no_failures(self, client):
        resp = client.get("/api/v1/pipelines-with-failures", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json() == {"pipelines": []}

    def test_lists_pipelines_with_failed_tasks(self, client, cfg, broker):
        _seed_pipeline(cfg, "pa")
        _seed_pipeline(cfg, "pb")
        tid_a = broker.enqueue("pa", "concept_art_generate", {})
        tid_b = broker.enqueue("pb", "concept_art_generate", {})
        broker.mark_failed(tid_a, "boom")
        # leave pb's task pending
        del tid_b
        resp = client.get("/api/v1/pipelines-with-failures", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json()["pipelines"] == ["pa"]

    def test_excludes_cancelled_tasks(self, client, cfg, broker):
        _seed_pipeline(cfg, "pc")
        tid = broker.enqueue("pc", "concept_art_generate", {})
        broker.mark_failed(tid, "cancelled")
        resp = client.get("/api/v1/pipelines-with-failures", headers=_AUTH)
        assert resp.json()["pipelines"] == []


class TestTasksEndpoints:
    def test_2d_tasks_returns_broker_tasks(self, client, cfg, broker):
        _seed_pipeline(cfg, "tpipe")
        broker.enqueue("tpipe", "concept_art_generate", {"foo": "bar"})
        resp = client.get("/api/v1/pipelines/tpipe/tasks", headers=_AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["task_type"] == "concept_art_generate"
        assert data[0]["status"] == "pending"
        # payload must NOT be exposed
        assert "payload" not in data[0]

    def test_2d_tasks_404_for_missing_pipeline(self, client):
        resp = client.get("/api/v1/pipelines/ghost/tasks", headers=_AUTH)
        assert resp.status_code == 404

    def test_3d_tasks_returns_broker_tasks(self, client, cfg, broker):
        _seed_3d_pipeline(cfg, "t3d")
        broker.enqueue("t3d", "mesh_generate", {})
        broker.enqueue("t3d", "mesh_texture", {})
        resp = client.get("/api/v1/3d-pipelines/t3d/tasks", headers=_AUTH)
        assert resp.status_code == 200
        types = sorted(t["task_type"] for t in resp.json())
        assert types == ["mesh_generate", "mesh_texture"]

    def test_3d_tasks_404_for_missing_pipeline(self, client):
        resp = client.get("/api/v1/3d-pipelines/ghost/tasks", headers=_AUTH)
        assert resp.status_code == 404


class TestConfig:
    def test_config_returns_expected_keys(self, client):
        resp = client.get("/api/v1/config", headers=_AUTH)
        assert resp.status_code == 200
        data = resp.json()
        for key in (
            "output_root",
            "pipelines_dir",
            "final_assets_dir",
            "background_suffix",
            "num_polys_default",
            "num_concept_arts_default",
            "concept_art_image_size",
            "export_format",
            "gemini_api_key_present",
        ):
            assert key in data, f"missing key: {key}"

    def test_config_never_returns_api_key(self, client):
        resp = client.get("/api/v1/config", headers=_AUTH)
        body = resp.text
        data = resp.json()
        # The fake config fixture sets GEMINI_API_KEY=fake; ensure that value
        # never appears in the response payload, and no key field other than
        # the boolean presence flag is exposed.
        assert "fake" not in body
        assert "gemini_api_key" not in data
        assert "api_key" not in data
        assert data["gemini_api_key_present"] in (True, False)

    def test_config_gemini_present_reflects_env(self, client, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        resp = client.get("/api/v1/config", headers=_AUTH)
        assert resp.json()["gemini_api_key_present"] is True
        monkeypatch.delenv("GEMINI_API_KEY")
        resp = client.get("/api/v1/config", headers=_AUTH)
        assert resp.json()["gemini_api_key_present"] is False


# ===========================================================================
# Resume / Pause
# ===========================================================================


class TestResumePipeline:
    def test_resume_paused_pipeline_succeeds(self, client, cfg):
        _seed_pipeline(cfg, "pausedpipe", status=PipelineStatus.PAUSED)
        resp = client.post("/api/v1/pipelines/pausedpipe/resume", headers=_AUTH)
        assert resp.status_code == 200

    def test_resume_missing_pipeline_returns_404(self, client):
        resp = client.post("/api/v1/pipelines/ghost/resume", headers=_AUTH)
        assert resp.status_code == 404


class TestPatchSymmetry:
    def test_patch_symmetrize_true(self, client, cfg):
        _seed_pipeline(cfg, "sympipe", status=PipelineStatus.CONCEPT_ART_REVIEW)
        client.patch("/api/v1/pipelines/sympipe", headers=_AUTH,
                     json={"symmetrize": True, "symmetry_axis": "x-"})
        state = client.get("/api/v1/pipelines/sympipe", headers=_AUTH).json()
        assert state["symmetrize"] is True
        assert state["symmetry_axis"] == "x-"

    def test_patch_symmetrize_false(self, client, cfg):
        _seed_pipeline(cfg, "sympipe2", status=PipelineStatus.CONCEPT_ART_REVIEW)
        client.patch("/api/v1/pipelines/sympipe2", headers=_AUTH,
                     json={"symmetrize": False})
        state = client.get("/api/v1/pipelines/sympipe2", headers=_AUTH).json()
        assert state["symmetrize"] is False


# ===========================================================================
# Concept art sheet endpoint
# ===========================================================================


class TestConceptArtSheet:
    def test_sheet_endpoint_with_ready_images_returns_png(self, client, cfg):
        _seed_concept_art_image(cfg, "sheetpipe", 0)
        arts = [ConceptArtItem(index=0, status=ConceptArtStatus.READY)]
        _seed_pipeline(cfg, "sheetpipe", status=PipelineStatus.CONCEPT_ART_REVIEW,
                       concept_arts=arts)
        resp = client.get("/api/v1/pipelines/sheetpipe/concept_art/sheet", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_sheet_endpoint_with_no_images_returns_409(self, client, cfg):
        _seed_pipeline(cfg, "emptypipe", status=PipelineStatus.CONCEPT_ART_REVIEW)
        resp = client.get("/api/v1/pipelines/emptypipe/concept_art/sheet", headers=_AUTH)
        assert resp.status_code == 409

    def test_sheet_missing_pipeline_returns_404(self, client):
        resp = client.get("/api/v1/pipelines/ghost/concept_art/sheet", headers=_AUTH)
        assert resp.status_code == 404


class TestConceptArtImageEdgeCases:
    def test_image_file_not_yet_generated_returns_404(self, client, cfg):
        arts = [ConceptArtItem(index=0, status=ConceptArtStatus.GENERATING)]
        _seed_pipeline(cfg, "genpipe", status=PipelineStatus.CONCEPT_ART_GENERATING,
                       concept_arts=arts)
        resp = client.get("/api/v1/pipelines/genpipe/concept_art/0", headers=_AUTH)
        assert resp.status_code == 404

    def test_negative_index_returns_404(self, client, cfg):
        _seed_pipeline(cfg, "negpipe")
        resp = client.get("/api/v1/pipelines/negpipe/concept_art/-1", headers=_AUTH)
        assert resp.status_code in (404, 422)


# ===========================================================================
# Restyle endpoint
# ===========================================================================


class TestRestyleConceptArt:
    def _make_state(self, cfg, name="restylepipe"):
        _seed_concept_art_image(cfg, name, 0)
        arts = [ConceptArtItem(index=0, status=ConceptArtStatus.READY)]
        _seed_pipeline(cfg, name, status=PipelineStatus.CONCEPT_ART_REVIEW,
                       concept_arts=arts)

    def test_restyle_without_worker_returns_409(self, client, cfg):
        self._make_state(cfg)
        resp = client.post("/api/v1/pipelines/restylepipe/concept_art/restyle",
                           headers=_AUTH, json={
                               "index": 0,
                               "positive": "zerg, biomechanical",
                           })
        assert resp.status_code == 409

    def test_restyle_with_worker_returns_200(self, client, cfg, agent, users_yaml):
        mock_restyle = MagicMock()
        mock_restyle.restyle_image = MagicMock(
            return_value=open(str(_seed_concept_art_image(cfg, "rw2pipe", 0)), "rb").read()
        )
        from src.api.app import create_app as _ca
        app2 = _ca(agent=agent, cfg=cfg, concept_worker=MockConceptArtWorker(),
                   restyle_worker=mock_restyle, users_file=str(users_yaml))
        with TestClient(app2) as c:
            self._make_state(cfg)
            resp = c.post("/api/v1/pipelines/restylepipe/concept_art/restyle",
                          headers=_AUTH, json={
                              "index": 0,
                              "positive": "zerg, biomechanical",
                              "negative": "blurry",
                              "denoise": 0.7,
                          })
        assert resp.status_code == 200

    def test_restyle_out_of_range_returns_422(self, client, cfg, agent, users_yaml):
        mock_restyle = MagicMock()
        from src.api.app import create_app as _ca
        app2 = _ca(agent=agent, cfg=cfg, concept_worker=MockConceptArtWorker(),
                   restyle_worker=mock_restyle, users_file=str(users_yaml))
        with TestClient(app2) as c:
            self._make_state(cfg)
            resp = c.post("/api/v1/pipelines/restylepipe/concept_art/restyle",
                          headers=_AUTH, json={"index": 99, "positive": "test"})
        assert resp.status_code == 422


# ===========================================================================
# 3D pipeline endpoints
# ===========================================================================


class TestList3DPipelines:
    def test_empty_returns_empty_list(self, client):
        resp = client.get("/api/v1/3d-pipelines", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_seeded_3d_pipeline(self, client, cfg):
        _seed_3d_pipeline(cfg, "ship3d")
        resp = client.get("/api/v1/3d-pipelines", headers=_AUTH)
        names = [p["name"] for p in resp.json()]
        assert "ship3d" in names

    def test_does_not_return_2d_pipelines(self, client, cfg):
        _seed_pipeline(cfg, "twodpipe")
        resp = client.get("/api/v1/3d-pipelines", headers=_AUTH)
        names = [p["name"] for p in resp.json()]
        assert "twodpipe" not in names


class TestGet3DPipeline:
    def test_returns_state(self, client, cfg):
        _seed_3d_pipeline(cfg, "myship3d")
        resp = client.get("/api/v1/3d-pipelines/myship3d", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json()["name"] == "myship3d"

    def test_missing_returns_404(self, client):
        resp = client.get("/api/v1/3d-pipelines/ghost", headers=_AUTH)
        assert resp.status_code == 404


class TestCancel3DPipeline:
    def test_cancel_sets_cancelled(self, client, cfg):
        _seed_3d_pipeline(cfg, "cancelship")
        client.delete("/api/v1/3d-pipelines/cancelship", headers=_AUTH)
        state = client.get("/api/v1/3d-pipelines/cancelship", headers=_AUTH).json()
        assert state["status"] == "cancelled"

    def test_cancel_missing_returns_404(self, client):
        assert client.delete("/api/v1/3d-pipelines/ghost", headers=_AUTH).status_code == 404


class TestCreate3DPipelineFromRef:
    def _setup_2d(self, cfg, name="mypipe"):
        """Seed a 2D pipeline with one approved concept art image."""
        ca_dir = cfg.pipelines_dir / name / "concept_art"
        ca_dir.mkdir(parents=True, exist_ok=True)
        img_path = ca_dir / "concept_art_1_0.png"
        Image.new("RGB", (64, 64)).save(str(img_path))
        arts = [ConceptArtItem(index=0, status=ConceptArtStatus.APPROVED, version=0)]
        _seed_pipeline(cfg, name, status=PipelineStatus.CONCEPT_ART_REVIEW, concept_arts=arts)
        return name

    def test_creates_3d_pipeline(self, client, cfg):
        self._setup_2d(cfg)
        resp = client.post("/api/v1/3d-pipelines/from-ref", headers=_AUTH, json={
            "pipeline_name": "mypipe",
            "concept_art_index": 0,
        })
        assert resp.status_code == 201
        assert resp.json()["status"] == "queued"

    def test_copies_source_image_into_3d_dir(self, client, cfg):
        """
        The 3D pipeline must own its own copy of the source image so it is
        decoupled from any later changes to the 2D pipeline's concept art
        folder (regenerate, delete, etc.).
        """
        self._setup_2d(cfg)
        resp = client.post("/api/v1/3d-pipelines/from-ref", headers=_AUTH, json={
            "pipeline_name": "mypipe",
            "concept_art_index": 0,
        })
        assert resp.status_code == 201
        name_3d = "mypipe_1_0"
        pipeline_dir_3d = cfg.pipelines_dir / name_3d
        copied = pipeline_dir_3d / "source_concept_art_1_0.png"
        assert copied.exists(), f"Expected source image copied to {copied}"
        # State must reference the copied path, not the original 2D path.
        state_data = resp.json()
        assert state_data["input_image_path"] == str(copied)
        assert "pipelines/mypipe/concept_art" not in state_data["input_image_path"].replace("\\", "/")

    def test_missing_2d_pipeline_returns_404(self, client, cfg):
        resp = client.post("/api/v1/3d-pipelines/from-ref", headers=_AUTH, json={
            "pipeline_name": "ghost",
            "concept_art_index": 0,
        })
        assert resp.status_code == 404

    def test_out_of_range_index_returns_422(self, client, cfg):
        self._setup_2d(cfg)
        resp = client.post("/api/v1/3d-pipelines/from-ref", headers=_AUTH, json={
            "pipeline_name": "mypipe",
            "concept_art_index": 99,
        })
        assert resp.status_code == 422

    def test_duplicate_3d_name_returns_409(self, client, cfg):
        self._setup_2d(cfg)
        client.post("/api/v1/3d-pipelines/from-ref", headers=_AUTH, json={
            "pipeline_name": "mypipe",
            "concept_art_index": 0,
        })
        resp = client.post("/api/v1/3d-pipelines/from-ref", headers=_AUTH, json={
            "pipeline_name": "mypipe",
            "concept_art_index": 0,
        })
        assert resp.status_code == 409


class TestApprove3DMesh:
    def test_approve_awaiting_approval_succeeds(self, client, cfg):
        _seed_3d_pipeline(cfg, "approveship", status=Pipeline3DStatus.AWAITING_APPROVAL)
        resp = client.post("/api/v1/3d-pipelines/approveship/approve", headers=_AUTH,
                           json={"asset_name": "my_ship"})
        assert resp.status_code == 200

    def test_approve_wrong_status_returns_409(self, client, cfg):
        _seed_3d_pipeline(cfg, "notreadyship", status=Pipeline3DStatus.QUEUED)
        resp = client.post("/api/v1/3d-pipelines/notreadyship/approve", headers=_AUTH,
                           json={"asset_name": "bad"})
        assert resp.status_code == 409

    def test_approve_missing_returns_404(self, client):
        resp = client.post("/api/v1/3d-pipelines/ghost/approve", headers=_AUTH,
                           json={"asset_name": "x"})
        assert resp.status_code == 404


class TestReject3DMesh:
    def test_reject_requeues_mesh_generation(self, client, cfg, broker):
        _seed_3d_pipeline(cfg, "rejectship", status=Pipeline3DStatus.AWAITING_APPROVAL)
        resp = client.post("/api/v1/3d-pipelines/rejectship/reject", headers=_AUTH,
                           json={})
        assert resp.status_code == 200
        tasks = broker.get_tasks(task_type="mesh_generate")
        assert len(tasks) == 1

    def test_reject_updates_num_polys(self, client, cfg):
        _seed_3d_pipeline(cfg, "rejectship2", status=Pipeline3DStatus.AWAITING_APPROVAL)
        client.post("/api/v1/3d-pipelines/rejectship2/reject", headers=_AUTH,
                    json={"num_polys": 4000})
        state = client.get("/api/v1/3d-pipelines/rejectship2", headers=_AUTH).json()
        assert state["num_polys"] == 4000

    def test_reject_wrong_status_returns_409(self, client, cfg):
        _seed_3d_pipeline(cfg, "queuedship", status=Pipeline3DStatus.QUEUED)
        resp = client.post("/api/v1/3d-pipelines/queuedship/reject", headers=_AUTH,
                           json={})
        assert resp.status_code == 409


class Test3DPipelineDownloads:
    def test_mesh_endpoint_returns_binary(self, client, cfg):
        _seed_3d_pipeline(cfg, "dlship", status=Pipeline3DStatus.AWAITING_APPROVAL)
        resp = client.get("/api/v1/3d-pipelines/dlship/mesh", headers=_AUTH)
        assert resp.status_code == 200

    def test_mesh_not_yet_generated_returns_404(self, client, cfg):
        state = _seed_3d_pipeline(cfg, "nomeshship", status=Pipeline3DStatus.QUEUED)
        # Remove textured_mesh_path that _seed_3d_pipeline sets
        state.textured_mesh_path = None
        state.mesh_path = None
        state.save(cfg.pipelines_dir / "nomeshship" / "state.json")
        resp = client.get("/api/v1/3d-pipelines/nomeshship/mesh", headers=_AUTH)
        assert resp.status_code == 404

    def test_sheet_not_generated_returns_404(self, client, cfg):
        _seed_3d_pipeline(cfg, "nosheetship")
        resp = client.get("/api/v1/3d-pipelines/nosheetship/sheet", headers=_AUTH)
        assert resp.status_code == 404

    def test_preview_not_generated_returns_404(self, client, cfg):
        _seed_3d_pipeline(cfg, "nopreviewship")
        resp = client.get("/api/v1/3d-pipelines/nopreviewship/preview", headers=_AUTH)
        assert resp.status_code == 404

    def test_screenshot_missing_file_returns_404(self, client, cfg):
        _seed_3d_pipeline(cfg, "screenshotship")
        resp = client.get("/api/v1/3d-pipelines/screenshotship/screenshot/front.png",
                          headers=_AUTH)
        assert resp.status_code == 404
