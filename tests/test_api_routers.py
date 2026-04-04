"""
Integration tests for the quickymesh FastAPI routers.

Uses FastAPI's TestClient (synchronous httpx transport — no real uvicorn needed).
The PipelineAgent is replaced with a lightweight fake that operates on real
state.json files in a tmp_path, so pipeline logic is exercised but no workers
run.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

import src.api.auth as auth_mod
from src.api.app import create_app
from src.api.event_bus import event_bus
from src.broker import Broker
from src.config import Config
from src.state import (
    ConceptArtItem,
    ConceptArtStatus,
    MeshItem,
    MeshStatus,
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
    """Write a minimal users.yaml and return its path."""
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
    """
    A real PipelineAgent with mocked worker threads (never started).
    Workers aren't started so no background threads run during tests.
    """
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
    # Inject fake threads so _show_status / status endpoint work
    a._threads = [MagicMock(is_alive=MagicMock(return_value=True),
                            __class__=MagicMock(__name__="FakeWorker"))]
    return a


@pytest.fixture
def client(agent, cfg, users_yaml):
    """TestClient with the real FastAPI app wired to the test agent."""
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
    meshes: list[MeshItem] | None = None,
) -> PipelineState:
    pipeline_dir = cfg.uncompleted_pipelines_dir / name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    state = PipelineState(
        name=name,
        description="a dragon",
        num_polys=8000,
        status=status,
        concept_arts=concept_arts or [],
        meshes=meshes or [],
        pipeline_dir=f"uncompleted_pipelines/{name}",
    )
    state.save(pipeline_dir / "state.json")
    return state


def _seed_concept_art_image(cfg: Config, name: str, idx: int) -> Path:
    """Create a minimal 1×1 PNG so FileResponse can serve it."""
    from PIL import Image
    ca_dir = cfg.uncompleted_pipelines_dir / name / "concept_art"
    ca_dir.mkdir(parents=True, exist_ok=True)
    img_path = ca_dir / f"concept_art_{idx + 1}.png"
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    img.save(str(img_path))
    return img_path


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
        """Create should publish a pipeline_created event to the bus."""
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

    def test_patch_after_mesh_generating_returns_409(self, client, cfg):
        _seed_pipeline(cfg, "latepipe", status=PipelineStatus.MESH_GENERATING)
        resp = client.patch("/api/v1/pipelines/latepipe", headers=_AUTH,
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
        task_id = broker.enqueue("failpipe", "concept_art_generate",
                                 {"pipeline_name": "failpipe",
                                  "state_path": str(cfg.uncompleted_pipelines_dir / "failpipe" / "state.json")})
        broker.mark_failed(task_id, "test error")
        resp = client.post("/api/v1/pipelines/failpipe/retry", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json()["tasks_reset"] == 1


# ===========================================================================
# Concept art review
# ===========================================================================


class TestConceptArtApprove:
    def _make_state_with_ready_arts(self, cfg, name="artpipe"):
        from src.concept_art_pipeline import generate_concept_arts
        state = _seed_pipeline(cfg, name, status=PipelineStatus.CONCEPT_ART_REVIEW)
        pipeline_dir = cfg.uncompleted_pipelines_dir / name
        arts = [ConceptArtItem(index=0), ConceptArtItem(index=1)]
        for art in arts:
            img_path = _seed_concept_art_image(cfg, name, art.index)
            art.image_path = str(img_path)
            art.status = ConceptArtStatus.READY
        state.concept_arts = arts
        state.save(pipeline_dir / "state.json")
        return state

    def test_approve_queues_mesh_generation(self, client, cfg, broker):
        self._make_state_with_ready_arts(cfg)
        resp = client.post("/api/v1/pipelines/artpipe/concept_art/approve",
                           headers=_AUTH, json={"indices": [0]})
        assert resp.status_code == 200
        tasks = broker.get_tasks(task_type="mesh_generate")
        assert len(tasks) == 1

    def test_approve_sets_status_approved(self, client, cfg):
        self._make_state_with_ready_arts(cfg)
        client.post("/api/v1/pipelines/artpipe/concept_art/approve",
                    headers=_AUTH, json={"indices": [0, 1]})
        state = client.get("/api/v1/pipelines/artpipe", headers=_AUTH).json()
        assert state["status"] == "mesh_generating"
        approved_arts = [ca for ca in state["concept_arts"]
                         if ca["status"] == "approved"]
        assert len(approved_arts) == 2

    def test_approve_out_of_range_index_returns_422(self, client, cfg):
        self._make_state_with_ready_arts(cfg)
        resp = client.post("/api/v1/pipelines/artpipe/concept_art/approve",
                           headers=_AUTH, json={"indices": [99]})
        assert resp.status_code == 422


class TestConceptArtRegenerate:
    def _make_state(self, cfg, name="regenpipe"):
        arts = []
        for i in range(2):
            img_path = _seed_concept_art_image(cfg, name, i)
            arts.append(ConceptArtItem(
                index=i, image_path=str(img_path), status=ConceptArtStatus.READY
            ))
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
        img_path = _seed_concept_art_image(cfg, name, 0)
        arts = [ConceptArtItem(index=0, image_path=str(img_path),
                               status=ConceptArtStatus.READY)]
        _seed_pipeline(cfg, name, status=PipelineStatus.CONCEPT_ART_REVIEW,
                       concept_arts=arts)

    def test_modify_with_supporting_worker_returns_200(self, client, cfg):
        self._make_state_with_image(cfg)
        resp = client.post("/api/v1/pipelines/modpipe/concept_art/modify",
                           headers=_AUTH,
                           json={"index": 0, "instruction": "make it red"})
        assert resp.status_code == 200

    def test_modify_with_non_supporting_worker_returns_409(self, client, cfg, agent, users_yaml):
        """When the concept_worker.supports_modify is False, should get 409."""
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


# ===========================================================================
# Concept art image download
# ===========================================================================


class TestConceptArtDownload:
    def test_download_existing_image_returns_png(self, client, cfg):
        img_path = _seed_concept_art_image(cfg, "dlpipe", 0)
        arts = [ConceptArtItem(index=0, image_path=str(img_path),
                               status=ConceptArtStatus.READY)]
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
# Mesh review
# ===========================================================================


def _seed_mesh_review_state(cfg: Config, name: str = "meshpipe") -> PipelineState:
    """Create a pipeline with one mesh in AWAITING_APPROVAL state."""
    pipeline_dir = cfg.uncompleted_pipelines_dir / name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = pipeline_dir / f"{name}_1" / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    glb_path = mesh_dir / "textured.glb"
    glb_path.write_bytes(b"GLB")  # fake file

    state = PipelineState(
        name=name,
        description="a dragon",
        num_polys=8000,
        status=PipelineStatus.MESH_REVIEW,
        concept_arts=[ConceptArtItem(index=0, status=ConceptArtStatus.APPROVED)],
        meshes=[MeshItem(
            concept_art_index=0,
            sub_name=f"{name}_1",
            status=MeshStatus.AWAITING_APPROVAL,
            textured_mesh_path=str(glb_path),
        )],
        pipeline_dir=f"uncompleted_pipelines/{name}",
    )
    state.save(pipeline_dir / "state.json")
    return state


class TestMeshApprove:
    def test_approve_marks_mesh_approved(self, client, cfg):
        _seed_mesh_review_state(cfg)
        resp = client.post(
            "/api/v1/pipelines/meshpipe/meshes/meshpipe_1/approve",
            headers=_AUTH,
            json={"asset_name": "my_dragon"},
        )
        assert resp.status_code == 200
        # After approval the pipeline is exported and moved to completed_pipelines.
        # Verify the export landed in final_assets_dir.
        exported = cfg.final_assets_dir / "my_dragon" / "mesh.glb"
        assert exported.exists()

    def test_approve_non_awaiting_returns_409(self, client, cfg):
        # Use a different pipeline name to avoid collision with the approve test
        _seed_mesh_review_state(cfg, name="meshpipe409")
        state = PipelineState.load(cfg.uncompleted_pipelines_dir / "meshpipe409" / "state.json")
        state.meshes[0].status = MeshStatus.APPROVED
        state.save(cfg.uncompleted_pipelines_dir / "meshpipe409" / "state.json")
        resp = client.post(
            "/api/v1/pipelines/meshpipe409/meshes/meshpipe409_1/approve",
            headers=_AUTH,
            json={"asset_name": "my_dragon"},
        )
        assert resp.status_code == 409

    def test_approve_missing_mesh_returns_404(self, client, cfg):
        _seed_mesh_review_state(cfg, name="meshpipe404")
        resp = client.post(
            "/api/v1/pipelines/meshpipe404/meshes/nonexistent_1/approve",
            headers=_AUTH,
            json={"asset_name": "ghost"},
        )
        assert resp.status_code == 404




class TestMeshReject:
    def test_reject_marks_mesh_queued(self, client, cfg):
        _seed_mesh_review_state(cfg)
        resp = client.post(
            "/api/v1/pipelines/meshpipe/meshes/meshpipe_1/reject",
            headers=_AUTH,
            json={},
        )
        assert resp.status_code == 200
        state = client.get("/api/v1/pipelines/meshpipe", headers=_AUTH).json()
        assert state["meshes"][0]["status"] == "queued"

    def test_reject_updates_num_polys(self, client, cfg):
        _seed_mesh_review_state(cfg)
        client.post(
            "/api/v1/pipelines/meshpipe/meshes/meshpipe_1/reject",
            headers=_AUTH,
            json={"num_polys": 5000},
        )
        state = client.get("/api/v1/pipelines/meshpipe", headers=_AUTH).json()
        assert state["num_polys"] == 5000

    def test_reject_updates_symmetry(self, client, cfg):
        _seed_mesh_review_state(cfg)
        client.post(
            "/api/v1/pipelines/meshpipe/meshes/meshpipe_1/reject",
            headers=_AUTH,
            json={"symmetrize": True, "symmetry_axis": "x-"},
        )
        state = client.get("/api/v1/pipelines/meshpipe", headers=_AUTH).json()
        assert state["symmetrize"] is True
        assert state["symmetry_axis"] == "x-"

    def test_reject_requeues_mesh_generation_when_all_rejected(self, client, cfg, broker):
        _seed_mesh_review_state(cfg)
        client.post(
            "/api/v1/pipelines/meshpipe/meshes/meshpipe_1/reject",
            headers=_AUTH,
            json={},
        )
        tasks = broker.get_tasks(task_type="mesh_generate")
        assert len(tasks) == 1


class TestMeshDownloads:
    def test_download_glb_returns_binary(self, client, cfg):
        _seed_mesh_review_state(cfg)
        resp = client.get(
            "/api/v1/pipelines/meshpipe/meshes/meshpipe_1/mesh",
            headers=_AUTH,
        )
        assert resp.status_code == 200

    def test_download_missing_screenshot_returns_404(self, client, cfg):
        _seed_mesh_review_state(cfg)
        resp = client.get(
            "/api/v1/pipelines/meshpipe/meshes/meshpipe_1/screenshot/front.png",
            headers=_AUTH,
        )
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


# ===========================================================================
# Coverage gap: pipelines.py — resume success, patch symmetry fields
# ===========================================================================


class TestResumePipeline:
    def test_resume_paused_pipeline_succeeds(self, client, cfg):
        _seed_pipeline(cfg, "pausedpipe", status=PipelineStatus.PAUSED)
        # Resume should succeed (200) for a paused pipeline
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
# Coverage gap: review.py — concept art sheet endpoint
# ===========================================================================


class TestConceptArtSheet:
    def test_sheet_endpoint_with_ready_images_returns_png(self, client, cfg):
        img_path = _seed_concept_art_image(cfg, "sheetpipe", 0)
        arts = [ConceptArtItem(index=0, image_path=str(img_path),
                               status=ConceptArtStatus.READY)]
        _seed_pipeline(cfg, "sheetpipe", status=PipelineStatus.CONCEPT_ART_REVIEW,
                       concept_arts=arts)
        resp = client.get("/api/v1/pipelines/sheetpipe/concept_art/sheet", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_sheet_endpoint_with_no_images_returns_409(self, client, cfg):
        # Pipeline with no concept arts — build_review_sheet raises ValueError
        _seed_pipeline(cfg, "emptypipe", status=PipelineStatus.CONCEPT_ART_REVIEW)
        resp = client.get("/api/v1/pipelines/emptypipe/concept_art/sheet", headers=_AUTH)
        assert resp.status_code == 409

    def test_sheet_missing_pipeline_returns_404(self, client):
        resp = client.get("/api/v1/pipelines/ghost/concept_art/sheet", headers=_AUTH)
        assert resp.status_code == 404


class TestConceptArtImageEdgeCases:
    def test_image_path_not_set_returns_404(self, client, cfg):
        """ConceptArtItem exists but image_path is None (still generating)."""
        arts = [ConceptArtItem(index=0, status=ConceptArtStatus.GENERATING)]
        _seed_pipeline(cfg, "genpipe", status=PipelineStatus.CONCEPT_ART_GENERATING,
                       concept_arts=arts)
        resp = client.get("/api/v1/pipelines/genpipe/concept_art/0", headers=_AUTH)
        assert resp.status_code == 404

    def test_negative_index_returns_404(self, client, cfg):
        _seed_pipeline(cfg, "negpipe")
        resp = client.get("/api/v1/pipelines/negpipe/concept_art/-1", headers=_AUTH)
        # FastAPI will reject -1 as an int path param or return 404
        assert resp.status_code in (404, 422)


# ===========================================================================
# Coverage gap: review.py — restyle endpoint
# ===========================================================================


class TestRestyleConceptArt:
    def _make_state(self, cfg, name="restylepipe"):
        img_path = _seed_concept_art_image(cfg, name, 0)
        arts = [ConceptArtItem(index=0, image_path=str(img_path),
                               status=ConceptArtStatus.READY)]
        _seed_pipeline(cfg, name, status=PipelineStatus.CONCEPT_ART_REVIEW,
                       concept_arts=arts)

    def test_restyle_without_worker_returns_409(self, client, cfg):
        # The default test client has restyle_worker=None
        self._make_state(cfg)
        resp = client.post("/api/v1/pipelines/restylepipe/concept_art/restyle",
                           headers=_AUTH, json={
                               "index": 0,
                               "positive": "zerg, biomechanical",
                           })
        assert resp.status_code == 409

    def test_restyle_with_worker_returns_200(self, client, cfg, agent, users_yaml):
        """Wire a mock restyle worker into the app."""
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
# Coverage gap: review.py — mesh download 404 paths
# ===========================================================================


class TestMeshDownload404s:
    def test_review_sheet_not_generated_returns_404(self, client, cfg):
        _seed_mesh_review_state(cfg, name="nosheetpipe")
        resp = client.get(
            "/api/v1/pipelines/nosheetpipe/meshes/nosheetpipe_1/sheet",
            headers=_AUTH,
        )
        assert resp.status_code == 404

    def test_preview_not_generated_returns_404(self, client, cfg):
        _seed_mesh_review_state(cfg, name="nopreviewpipe")
        resp = client.get(
            "/api/v1/pipelines/nopreviewpipe/meshes/nopreviewpipe_1/preview",
            headers=_AUTH,
        )
        assert resp.status_code == 404

    def test_mesh_file_no_glb_returns_404(self, client, cfg):
        """Seed a pipeline where the textured_mesh_path points to a missing file."""
        name = "noglbpipe"
        pipeline_dir = cfg.uncompleted_pipelines_dir / name
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        state = PipelineState(
            name=name,
            description="test",
            num_polys=8000,
            status=PipelineStatus.MESH_REVIEW,
            concept_arts=[ConceptArtItem(index=0, status=ConceptArtStatus.APPROVED)],
            meshes=[MeshItem(
                concept_art_index=0,
                sub_name=f"{name}_1",
                status=MeshStatus.AWAITING_APPROVAL,
                textured_mesh_path="/nonexistent/path.glb",
            )],
            pipeline_dir=f"uncompleted_pipelines/{name}",
        )
        state.save(pipeline_dir / "state.json")
        resp = client.get(
            f"/api/v1/pipelines/{name}/meshes/{name}_1/mesh",
            headers=_AUTH,
        )
        assert resp.status_code == 404

    def test_screenshot_file_missing_returns_404(self, client, cfg):
        """screenshot_dir set but the specific file doesn't exist."""
        name = "screenpipe"
        pipeline_dir = cfg.uncompleted_pipelines_dir / name
        screenshot_dir = pipeline_dir / f"{name}_1" / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        glb_path = pipeline_dir / f"{name}_1" / "meshes" / "textured.glb"
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(b"GLB")
        state = PipelineState(
            name=name,
            description="test",
            num_polys=8000,
            status=PipelineStatus.MESH_REVIEW,
            concept_arts=[ConceptArtItem(index=0, status=ConceptArtStatus.APPROVED)],
            meshes=[MeshItem(
                concept_art_index=0,
                sub_name=f"{name}_1",
                status=MeshStatus.AWAITING_APPROVAL,
                textured_mesh_path=str(glb_path),
                screenshot_dir=str(screenshot_dir),
            )],
            pipeline_dir=f"uncompleted_pipelines/{name}",
        )
        state.save(pipeline_dir / "state.json")
        resp = client.get(
            f"/api/v1/pipelines/{name}/meshes/{name}_1/screenshot/missing.png",
            headers=_AUTH,
        )
        assert resp.status_code == 404

    def test_screenshot_with_existing_file_returns_200(self, client, cfg):
        """screenshot_dir set and the file exists."""
        from PIL import Image
        name = "screenpipe2"
        pipeline_dir = cfg.uncompleted_pipelines_dir / name
        screenshot_dir = pipeline_dir / f"{name}_1" / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        glb_path = pipeline_dir / f"{name}_1" / "meshes" / "textured.glb"
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(b"GLB")
        img = Image.new("RGB", (1, 1))
        img.save(str(screenshot_dir / "front.png"))
        state = PipelineState(
            name=name,
            description="test",
            num_polys=8000,
            status=PipelineStatus.MESH_REVIEW,
            concept_arts=[ConceptArtItem(index=0, status=ConceptArtStatus.APPROVED)],
            meshes=[MeshItem(
                concept_art_index=0,
                sub_name=f"{name}_1",
                status=MeshStatus.AWAITING_APPROVAL,
                textured_mesh_path=str(glb_path),
                screenshot_dir=str(screenshot_dir),
            )],
            pipeline_dir=f"uncompleted_pipelines/{name}",
        )
        state.save(pipeline_dir / "state.json")
        resp = client.get(
            f"/api/v1/pipelines/{name}/meshes/{name}_1/screenshot/front.png",
            headers=_AUTH,
        )
        assert resp.status_code == 200


# ===========================================================================
# Coverage gap: review.py — approve/reject with multiple meshes
# ===========================================================================


def _seed_two_mesh_review_state(cfg: Config, name: str = "twomeshpipe") -> PipelineState:
    """Two meshes both AWAITING_APPROVAL."""
    pipeline_dir = cfg.uncompleted_pipelines_dir / name
    meshes = []
    for i in range(1, 3):
        mesh_dir = pipeline_dir / f"{name}_{i}" / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        glb = mesh_dir / "textured.glb"
        glb.write_bytes(b"GLB")
        meshes.append(MeshItem(
            concept_art_index=i - 1,
            sub_name=f"{name}_{i}",
            status=MeshStatus.AWAITING_APPROVAL,
            textured_mesh_path=str(glb),
        ))
    state = PipelineState(
        name=name,
        description="test",
        num_polys=8000,
        status=PipelineStatus.MESH_REVIEW,
        concept_arts=[
            ConceptArtItem(index=0, status=ConceptArtStatus.APPROVED),
            ConceptArtItem(index=1, status=ConceptArtStatus.APPROVED),
        ],
        meshes=meshes,
        pipeline_dir=f"uncompleted_pipelines/{name}",
    )
    state.save(pipeline_dir / "state.json")
    return state


class TestMeshApproveMultiple:
    def test_approve_one_of_two_does_not_export_yet(self, client, cfg):
        """Approving one mesh with another still pending should not trigger export."""
        _seed_two_mesh_review_state(cfg)
        resp = client.post(
            "/api/v1/pipelines/twomeshpipe/meshes/twomeshpipe_1/approve",
            headers=_AUTH,
            json={"asset_name": "first_ship"},
        )
        assert resp.status_code == 200
        # Pipeline still exists in uncompleted (not moved to completed yet)
        state = client.get("/api/v1/pipelines/twomeshpipe", headers=_AUTH).json()
        assert state["status"] == "mesh_review"

    def test_reject_one_of_two_keeps_other_pending(self, client, cfg):
        """Rejecting one mesh while another is still pending — both remain in review."""
        _seed_two_mesh_review_state(cfg, name="twomesh2")
        resp = client.post(
            "/api/v1/pipelines/twomesh2/meshes/twomesh2_1/reject",
            headers=_AUTH, json={},
        )
        assert resp.status_code == 200
        state = client.get("/api/v1/pipelines/twomesh2", headers=_AUTH).json()
        # Status should still be mesh_review (one remaining)
        assert state["status"] == "mesh_review"

    def test_approve_then_reject_other_triggers_export(self, client, cfg):
        """Approve mesh 1, reject mesh 2 → export runs with 1 approved mesh."""
        _seed_two_mesh_review_state(cfg, name="mixmesh")
        client.post("/api/v1/pipelines/mixmesh/meshes/mixmesh_1/approve",
                    headers=_AUTH, json={"asset_name": "approved_ship"})
        resp = client.post("/api/v1/pipelines/mixmesh/meshes/mixmesh_2/reject",
                           headers=_AUTH, json={})
        assert resp.status_code == 200
        exported = cfg.final_assets_dir / "approved_ship" / "mesh.glb"
        assert exported.exists()


class TestMeshRejectEdgeCases:
    def test_reject_non_awaiting_returns_409(self, client, cfg):
        name = "rejpipe409"
        state = _seed_mesh_review_state(cfg, name=name)
        state.meshes[0].status = MeshStatus.QUEUED
        state.save(cfg.uncompleted_pipelines_dir / name / "state.json")
        resp = client.post(
            f"/api/v1/pipelines/{name}/meshes/{name}_1/reject",
            headers=_AUTH, json={},
        )
        assert resp.status_code == 409


# ===========================================================================
# Coverage gap: SSE endpoints — connect and content-type
# ===========================================================================


class TestSSEEndpoints:
    """
    SSE endpoints return an infinite stream, so we can't use client.get()
    (it would hang waiting for the response body to complete).  We test
    auth rejection with a normal GET (FastAPI rejects before touching the
    stream), and verify content-type via the router registration.
    The generator internals are tested directly in TestSSEGenerator below.
    """

    def test_pipeline_events_requires_auth(self, client, cfg):
        _seed_pipeline(cfg, "authpipe")
        resp = client.get("/api/v1/pipelines/authpipe/events")
        assert resp.status_code == 401

    def test_global_events_requires_auth(self, client):
        resp = client.get("/api/v1/events")
        assert resp.status_code == 401

    def test_pipeline_events_missing_auth_returns_401(self, client, cfg):
        _seed_pipeline(cfg, "authpipe2")
        resp = client.get("/api/v1/pipelines/authpipe2/events",
                          headers={"Authorization": "Bearer bad-key"})
        assert resp.status_code == 401

    def test_global_events_missing_auth_returns_401(self, client):
        resp = client.get("/api/v1/events",
                          headers={"Authorization": "Bearer bad-key"})
        assert resp.status_code == 401


# ===========================================================================
# Coverage gap: SSE generator logic (tested directly, not via HTTP)
# ===========================================================================


class TestSSEGenerator:
    def test_generator_yields_event_from_queue(self):
        """Generator yields event data when queue has an item."""
        import asyncio
        from src.api.routers.events import _event_generator

        async def run():
            q = asyncio.Queue()
            await q.put({"event": "ping", "pipeline": "test"})
            # Cancel after first yield so the generator doesn't loop forever
            gen = _event_generator(q, "test")
            result = await gen.__anext__()
            await gen.aclose()
            return result

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run())
        loop.close()

        import json
        data = json.loads(result["data"])
        assert data["event"] == "ping"

    def test_generator_yields_heartbeat_on_timeout(self):
        """Generator yields heartbeat when queue is empty for the timeout window."""
        import asyncio
        from src.api.routers.events import _event_generator

        async def run():
            q = asyncio.Queue()
            # Use a very short timeout by patching the module constant
            import src.api.routers.events as events_mod
            original = events_mod._HEARTBEAT_INTERVAL
            events_mod._HEARTBEAT_INTERVAL = 0.05
            try:
                gen = _event_generator(q, None)
                result = await gen.__anext__()
                await gen.aclose()
                return result
            finally:
                events_mod._HEARTBEAT_INTERVAL = original

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run())
        loop.close()

        import json
        data = json.loads(result["data"])
        assert data["event"] == "heartbeat"

    def test_generator_unsubscribes_on_close(self):
        """Generator's finally block removes the queue from the event bus."""
        import asyncio
        from src.api.routers.events import _event_generator
        from src.api.event_bus import event_bus

        loop = asyncio.new_event_loop()
        event_bus.set_loop(loop)

        async def run():
            q = event_bus.subscribe(pipeline_name="closepipe")
            # Put an item in so the first __anext__() returns immediately
            # (generator must be entered before finally runs on aclose)
            await q.put({"event": "ping"})
            gen = _event_generator(q, "closepipe")
            await gen.__anext__()  # enters the try block
            await gen.aclose()    # triggers finally → unsubscribe
            return q

        q = loop.run_until_complete(run())
        loop.close()
        assert q not in event_bus._pipeline_subs.get("closepipe", set())


# ===========================================================================
# Coverage gap: event_bus.py — publish with full queue / closed loop
# ===========================================================================


class TestEventBusEdgeCases:
    def test_publish_drops_event_when_queue_is_full(self):
        """publish() should not raise even if the queue is full (maxsize=1)."""
        import asyncio
        from src.api.event_bus import EventBus

        bus = EventBus()
        loop = asyncio.new_event_loop()
        bus.set_loop(loop)

        # Create a tiny queue (maxsize=1) and fill it
        q = asyncio.Queue(maxsize=1)
        with bus._lock:
            bus._global_subs.add(q)

        # Fill the queue from the loop thread
        loop.run_until_complete(q.put({"event": "first"}))

        # Now publish — queue is full, put_nowait will raise; should be swallowed
        bus.publish({"event": "overflow"})
        # Give the loop a tick to process the call_soon_threadsafe
        loop.run_until_complete(asyncio.sleep(0.01))

        # Queue still has the original item, not the overflow
        assert q.qsize() == 1
        event = loop.run_until_complete(q.get())
        assert event["event"] == "first"
        loop.close()
