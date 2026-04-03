# build_run.ps1 — Build and manage the ComfyUI Docker container.
# Run from PowerShell terminal in VSCode.
#
# Usage:
#   .\docker\build_run.ps1            — build + start
#   .\docker\build_run.ps1 build      — build only
#   .\docker\build_run.ps1 start      — start (image already built)
#   .\docker\build_run.ps1 stop       — stop and remove container
#   .\docker\build_run.ps1 logs       — tail container logs
#   .\docker\build_run.ps1 shell      — open bash inside container

param([string]$Cmd = "all")

$ComposeFile = "$PSScriptRoot\comfyui\docker-compose.yml"

switch ($Cmd) {
    "build" {
        Write-Host "==> Building ComfyUI image (this will take a while on first run) ..."
        docker compose -f $ComposeFile build
    }
    "start" {
        Write-Host "==> Starting ComfyUI container on port 8189 ..."
        docker compose -f $ComposeFile up -d
        Write-Host "==> ComfyUI starting at http://localhost:8189"
        Write-Host "    Run '.\docker\build_run.ps1 logs' to watch startup progress."
    }
    "all" {
        Write-Host "==> Building ComfyUI image ..."
        docker compose -f $ComposeFile build
        Write-Host "==> Starting ComfyUI container on port 8189 ..."
        docker compose -f $ComposeFile up -d
        Write-Host ""
        Write-Host "==> ComfyUI is starting at http://localhost:8189"
        Write-Host "    Run '.\docker\build_run.ps1 logs' to watch startup progress."
        Write-Host "    Run 'python docker\test_comfyui.py' to submit a test workflow."
    }
    "stop" {
        Write-Host "==> Stopping ComfyUI container ..."
        docker compose -f $ComposeFile down
    }
    "logs" {
        docker compose -f $ComposeFile logs -f
    }
    "shell" {
        docker exec -it comfyui bash
    }
    default {
        Write-Host "Unknown command: $Cmd"
        Write-Host "Usage: .\docker\build_run.ps1 [build|start|all|stop|logs|shell]"
        exit 1
    }
}
