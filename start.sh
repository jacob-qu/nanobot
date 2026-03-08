#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# nanobot gateway launcher — background daemon with auto-restart
# Usage:  ./start.sh          Start the service
#         ./start.sh stop     Stop the service
#         ./start.sh status   Check service status
#         ./start.sh restart  Restart the service
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NANOBOT_HOME="${HOME}/.nanobot"
LOG_DIR="${NANOBOT_HOME}/logs"
PID_FILE="${NANOBOT_HOME}/gateway.pid"
LOG_FILE="${LOG_DIR}/gateway.log"

PYTHON="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8

# ── Auto-restart settings ─────────────────────────────────────
MAX_RESTARTS=50            # max restarts within the window
RESTART_WINDOW=3600        # window in seconds (1 hour)
RESTART_DELAY=5            # seconds between restarts
HEALTH_INTERVAL=30         # health-check interval (seconds)
HEALTH_TIMEOUT=10          # curl timeout for health probe

mkdir -p "$LOG_DIR"

# ── Helpers ────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }

is_running() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid=$(<"$PID_FILE")
    kill -0 "$pid" 2>/dev/null
}

# ── Preflight: model & token check ────────────────────────────
preflight() {
    log "=== Preflight check ==="

    local output
    output=$("$PYTHON" -c "
from nanobot.config.loader import load_config
c = load_config()
model = c.agents.defaults.model
provider = c.get_provider_name()
p = c.get_provider()
api_base = c.get_api_base() or 'default'
key = (p.api_key[:8] + '***') if p and p.api_key else ''
print(f'MODEL={model}')
print(f'PROVIDER={provider}')
print(f'API_BASE={api_base}')
print(f'API_KEY={key}')
" 2>&1) || { log "ERROR: failed to load config"; return 1; }

    local model provider api_base api_key
    model=$(echo "$output"   | grep '^MODEL='    | cut -d= -f2-)
    provider=$(echo "$output"| grep '^PROVIDER=' | cut -d= -f2-)
    api_base=$(echo "$output"| grep '^API_BASE=' | cut -d= -f2-)
    api_key=$(echo "$output" | grep '^API_KEY='  | cut -d= -f2-)

    if [[ -z "$api_key" ]]; then
        log "ERROR: No API key configured for provider=${provider}"
        return 1
    fi

    log "Model:    ${model}"
    log "Provider: ${provider}"
    log "API Base: ${api_base}"
    log "API Key:  ${api_key}"

    # Validate token with a minimal chat completion
    log "Verifying token with a test request..."
    local test_result
    test_result=$("$PYTHON" -c "
import asyncio, sys
from nanobot.config.loader import load_config
from nanobot.cli.commands import _make_provider
c = load_config()
provider = _make_provider(c)
async def probe():
    r = await provider.chat(
        messages=[{'role':'user','content':'hi'}],
        model=c.agents.defaults.model,
        max_tokens=1,
    )
    print('OK' if r.content or r.finish_reason else 'FAIL')
asyncio.run(probe())
" 2>&1) || true

    if echo "$test_result" | grep -q "^OK"; then
        log "Token verification: OK"
    else
        log "WARNING: Token verification failed — ${test_result}"
        log "Service will start anyway (network issues may resolve)"
    fi

    log "=== Preflight passed ==="
}

# ── Daemon: run gateway with auto-restart ─────────────────────
run_daemon() {
    local restart_count=0
    local window_start
    window_start=$(date +%s)

    trap 'log "Daemon received signal, shutting down..."; cleanup; exit 0' SIGTERM SIGINT SIGHUP

    while true; do
        local now
        now=$(date +%s)

        # Reset counter when window expires
        if (( now - window_start >= RESTART_WINDOW )); then
            restart_count=0
            window_start=$now
        fi

        if (( restart_count >= MAX_RESTARTS )); then
            log "ERROR: Reached ${MAX_RESTARTS} restarts within ${RESTART_WINDOW}s — giving up"
            cleanup
            exit 1
        fi

        log "Starting nanobot gateway (attempt $((restart_count + 1)))..."

        "$PYTHON" -m nanobot gateway >> "$LOG_FILE" 2>&1 &
        local gw_pid=$!
        echo "$gw_pid" > "$PID_FILE.gw"

        # Wait briefly then check it didn't crash immediately
        sleep 2
        if ! kill -0 "$gw_pid" 2>/dev/null; then
            log "ERROR: Gateway exited immediately"
            restart_count=$((restart_count + 1))
            sleep "$RESTART_DELAY"
            continue
        fi

        log "Gateway running (PID=${gw_pid})"

        # Health-check loop
        while kill -0 "$gw_pid" 2>/dev/null; do
            sleep "$HEALTH_INTERVAL"

            # Check process is alive
            if ! kill -0 "$gw_pid" 2>/dev/null; then
                log "WARNING: Gateway process (PID=${gw_pid}) disappeared"
                break
            fi

            # Check log for fatal signs (last 20 lines)
            if tail -20 "$LOG_FILE" 2>/dev/null | grep -qi "CRITICAL\|panic\|out of memory"; then
                log "WARNING: Detected fatal error in logs, killing gateway"
                kill "$gw_pid" 2>/dev/null
                wait "$gw_pid" 2>/dev/null
                break
            fi
        done

        wait "$gw_pid" 2>/dev/null
        local exit_code=$?
        log "Gateway exited with code ${exit_code}"

        restart_count=$((restart_count + 1))
        log "Restarting in ${RESTART_DELAY}s... (${restart_count}/${MAX_RESTARTS} in window)"
        sleep "$RESTART_DELAY"
    done
}

cleanup() {
    if [[ -f "$PID_FILE.gw" ]]; then
        local gw_pid
        gw_pid=$(<"$PID_FILE.gw")
        kill "$gw_pid" 2>/dev/null && log "Stopped gateway (PID=${gw_pid})"
        rm -f "$PID_FILE.gw"
    fi
    rm -f "$PID_FILE"
}

# ── Commands ───────────────────────────────────────────────────
do_start() {
    if is_running; then
        local pid=$(<"$PID_FILE")
        log "Service is already running (PID=${pid})"
        exit 0
    fi

    cd "$SCRIPT_DIR"

    # Clear Python bytecode cache to ensure fresh code is loaded
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
    log "Cleared __pycache__"

    preflight || exit 1

    log "Launching daemon..."
    nohup bash "$0" _daemon >> "$LOG_FILE" 2>&1 &
    local daemon_pid=$!
    echo "$daemon_pid" > "$PID_FILE"
    disown "$daemon_pid"

    log "Service started (daemon PID=${daemon_pid})"
    log "Logs: tail -f ${LOG_FILE}"
}

do_stop() {
    if ! is_running; then
        log "Service is not running"
        rm -f "$PID_FILE" "$PID_FILE.gw"
        return 0
    fi

    local pid=$(<"$PID_FILE")
    log "Stopping service (PID=${pid})..."
    kill "$pid" 2>/dev/null

    # Wait for graceful shutdown
    local waited=0
    while kill -0 "$pid" 2>/dev/null && (( waited < 15 )); do
        sleep 1
        waited=$((waited + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        log "Force killing..."
        kill -9 "$pid" 2>/dev/null
    fi

    # Also kill gateway child if still around
    if [[ -f "$PID_FILE.gw" ]]; then
        local gw_pid=$(<"$PID_FILE.gw")
        kill "$gw_pid" 2>/dev/null
        rm -f "$PID_FILE.gw"
    fi

    rm -f "$PID_FILE"
    log "Service stopped"
}

do_status() {
    if is_running; then
        local pid=$(<"$PID_FILE")
        local gw_pid="?"
        [[ -f "$PID_FILE.gw" ]] && gw_pid=$(<"$PID_FILE.gw")
        echo "nanobot is running (daemon=${pid}, gateway=${gw_pid})"
        echo "Logs: ${LOG_FILE}"
    else
        echo "nanobot is not running"
        rm -f "$PID_FILE" "$PID_FILE.gw"
        return 1
    fi
}

# ── Main ───────────────────────────────────────────────────────
case "${1:-start}" in
    start)    do_start   ;;
    stop)     do_stop    ;;
    restart)  do_stop; sleep 2; do_start ;;
    status)   do_status  ;;
    _daemon)  run_daemon ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
