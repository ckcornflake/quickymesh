# build_run.ps1 — Build and manage the ComfyUI Trellis Docker container.
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

$ComposeFile = "$PSScriptRoot\docker-compose.yml"
$Svc = "comfyui-trellis"

switch ($Cmd) {
    "build" {
        Write-Host "==> Building comfyui-trellis image (this will take a while on first run) ..."
        docker compose -f $ComposeFile build $Svc
    }
    "start" {
        Write-Host "==> Starting quickymesh-runtime container ..."
        docker compose -f $ComposeFile up -d $Svc
        Write-Host ""
        Write-Host "  ComfyUI        ->  http://localhost:8190"
        Write-Host "  quickymesh API ->  http://localhost:8000"
        Write-Host "  API docs       ->  http://localhost:8000/docs"
        Write-Host ""
        Write-Host "    Run '.\docker\build_run.ps1 logs' to watch startup progress."
    }
    "all" {
        Write-Host "==> Building quickymesh-runtime image ..."
        docker compose -f $ComposeFile build $Svc
        Write-Host "==> Starting quickymesh-runtime container ..."
        docker compose -f $ComposeFile up -d $Svc
        Write-Host ""
        Write-Host "  ComfyUI        ->  http://localhost:8190"
        Write-Host "  quickymesh API ->  http://localhost:8000"
        Write-Host "  API docs       ->  http://localhost:8000/docs"
        Write-Host ""
        Write-Host "    Run '.\docker\build_run.ps1 logs' to watch startup progress."
    }
    "stop" {
        Write-Host "==> Stopping comfyui-trellis container ..."
        docker compose -f $ComposeFile down $Svc
    }
    "logs" {
        docker compose -f $ComposeFile logs -f $Svc
    }
    "shell" {
        docker exec -it comfyui-trellis bash
    }
    default {
        Write-Host "Unknown command: $Cmd"
        Write-Host "Usage: .\docker\build_run.ps1 [build|start|all|stop|logs|shell]"
        exit 1
    }
}
