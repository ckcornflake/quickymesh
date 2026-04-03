#!/usr/bin/env bash
# build_run.sh — Build and manage ComfyUI Docker containers.
#
# Usage:
#   bash docker/build_run.sh [command] [service]
#
# Commands:
#   build   — build image(s)
#   start   — start container(s)
#   all     — build + start  (default)
#   stop    — stop and remove container(s)
#   logs    — tail container logs
#   shell   — open bash inside a container
#
# Services (optional, defaults to both):
#   sdxl    — ComfyUI SDXL on port 8189
#   trellis — ComfyUI Trellis2 on port 8190
#
# Examples:
#   bash docker/build_run.sh                    # build + start both
#   bash docker/build_run.sh build trellis      # build Trellis image only
#   bash docker/build_run.sh start sdxl         # start SDXL container only
#   bash docker/build_run.sh stop               # stop both
#   bash docker/build_run.sh logs trellis       # tail Trellis logs
#   bash docker/build_run.sh shell trellis      # bash inside Trellis container

set -e

COMPOSE_FILE="$(dirname "$0")/docker-compose.yml"
CMD="${1:-all}"

# Translate short names (sdxl, trellis) to docker-compose service names
case "${2:-}" in
  sdxl)    SVC="comfyui-sdxl" ;;
  trellis) SVC="comfyui-trellis" ;;
  *)       SVC="${2:-}" ;;
esac

print_urls() {
  [[ -z "$SVC" || "$SVC" == "comfyui-sdxl" ]]    && echo "  SDXL    → http://localhost:8189"
  [[ -z "$SVC" || "$SVC" == "comfyui-trellis" ]] && echo "  Trellis → http://localhost:8190"
}

case "$CMD" in
  build)
    echo "==> Building image(s) ..."
    docker compose -f "$COMPOSE_FILE" build $SVC
    ;;
  start)
    echo "==> Starting container(s) ..."
    docker compose -f "$COMPOSE_FILE" up -d $SVC
    print_urls
    ;;
  all)
    echo "==> Building image(s) ..."
    docker compose -f "$COMPOSE_FILE" build $SVC
    echo "==> Starting container(s) ..."
    docker compose -f "$COMPOSE_FILE" up -d $SVC
    echo ""
    print_urls
    ;;
  stop)
    echo "==> Stopping container(s) ..."
    docker compose -f "$COMPOSE_FILE" down $SVC
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" logs -f $SVC
    ;;
  shell)
    if [[ -z "$SVC" ]]; then
      echo "Please specify a service: bash docker/build_run.sh shell [sdxl|trellis]"
      exit 1
    fi
    docker exec -it "$SVC" bash
    ;;
  *)
    echo "Unknown command: $CMD"
    echo "Usage: bash docker/build_run.sh [build|start|all|stop|logs|shell] [sdxl|trellis]"
    exit 1
    ;;
esac
