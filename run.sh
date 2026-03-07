#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# nanobot 生产环境一键部署/更新脚本 (Mac Mini)
# 用法:
#   ./run.sh          拉取最新镜像并启动
#   ./run.sh stop     停止服务
#   ./run.sh status   查看状态
#   ./run.sh logs     查看日志
#   ./run.sh update   拉取最新镜像并重启
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.prod.yml"
IMAGE="ghcr.io/jacob-qu/nanobot:latest"

# 确保 config 目录存在
mkdir -p ~/.nanobot

dc() {
    docker compose -f "$COMPOSE_FILE" "$@"
}

do_start() {
    echo "==> 拉取最新镜像..."
    docker pull "$IMAGE"
    echo "==> 启动服务..."
    dc up -d
    echo "==> 服务已启动"
    dc ps
}

do_stop() {
    echo "==> 停止服务..."
    dc down
    echo "==> 服务已停止"
}

do_update() {
    echo "==> 拉取最新镜像..."
    docker pull "$IMAGE"

    local current
    current=$(docker inspect --format='{{.Image}}' nanobot-gateway 2>/dev/null || echo "none")
    local latest
    latest=$(docker inspect --format='{{.Id}}' "$IMAGE" 2>/dev/null || echo "new")

    if [ "$current" = "$latest" ]; then
        echo "==> 已是最新版本，无需更新"
    else
        echo "==> 检测到新版本，重启服务..."
        dc up -d
        echo "==> 更新完成"
    fi
    dc ps
}

do_status() {
    dc ps
}

do_logs() {
    dc logs -f --tail=100
}

case "${1:-start}" in
    start)   do_start  ;;
    stop)    do_stop   ;;
    update)  do_update ;;
    status)  do_status ;;
    logs)    do_logs   ;;
    *)
        echo "Usage: $0 {start|stop|update|status|logs}"
        exit 1
        ;;
esac
