#!/usr/bin/env bash
# build_run.sh — Build and manage the ComfyUI Trellis Docker container.
#
# Usage:
#   bash docker/build_run.sh [command]
#
# Commands:
#   build  — build the image
#   start  — start the container
#   all    — build + start  (default)
#   stop   — stop and remove the container
#   logs   — tail container logs
#   shell  — open bash inside the container
#
# Examples:
#   bash docker/build_run.sh           # build + start
#   bash docker/build_run.sh build     # build image only
#   bash docker/build_run.sh stop      # stop container
#   bash docker/build_run.sh logs      # tail logs
#   bash docker/build_run.sh shell     # bash inside container

set -e

COMPOSE_FILE="$(dirname "$0")/docker-compose.yml"
CMD="${1:-all}"
SVC="comfyui-trellis"

case "$CMD" in
  build)
    echo "==> Building image ..."
    docker compose -f "$COMPOSE_FILE" build $SVC
    ;;
  start)
    echo "==> Starting container ..."
    docker compose -f "$COMPOSE_FILE" up -d $SVC
    echo "  Trellis → http://localhost:8190"
    ;;
  all)
    echo "==> Building image ..."
    docker compose -f "$COMPOSE_FILE" build $SVC
    echo "==> Starting container ..."
    docker compose -f "$COMPOSE_FILE" up -d $SVC
    echo ""
    echo "  Trellis → http://localhost:8190"
    ;;
  stop)
    echo "==> Stopping container ..."
    docker compose -f "$COMPOSE_FILE" down $SVC
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" logs -f $SVC
    ;;
  shell)
    docker exec -it "$SVC" bash
    ;;
  *)
    echo "Unknown command: $CMD"
    echo "Usage: bash docker/build_run.sh [build|start|all|stop|logs|shell]"
    exit 1
    ;;
esac
