"""
Completed asset endpoints.

GET  /assets               List completed pipelines (exported meshes)
GET  /assets/{name}/mesh   Download the final exported GLB
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from src.api.auth import CurrentUser

log = logging.getLogger(__name__)
router = APIRouter(tags=["assets"])


def _cfg(request: Request):
    return request.app.state.cfg


@router.get("/assets")
async def list_assets(request: Request, user: CurrentUser):
    """List all completed pipelines that have exported meshes."""
    cfg = _cfg(request)
    assets_dir = cfg.final_assets_dir
    if not assets_dir.exists():
        return []

    results = []
    for item in sorted(assets_dir.iterdir()):
        if item.is_file() and item.suffix.lower() in (".glb", ".obj", ".fbx"):
            results.append({
                "name": item.stem,
                "filename": item.name,
                "size_bytes": item.stat().st_size,
            })
    return results


@router.get("/assets/{name}/mesh")
async def download_asset(name: str, request: Request, user: CurrentUser):
    """Download a final exported mesh by asset name (without extension)."""
    cfg = _cfg(request)
    assets_dir = cfg.final_assets_dir

    # Try common extensions
    for ext in (".glb", ".obj", ".fbx"):
        candidate = assets_dir / f"{name}{ext}"
        if candidate.exists():
            return FileResponse(
                str(candidate),
                media_type="model/gltf-binary" if ext == ".glb" else "application/octet-stream",
                filename=candidate.name,
            )

    raise HTTPException(status_code=404, detail=f"Asset '{name}' not found")
