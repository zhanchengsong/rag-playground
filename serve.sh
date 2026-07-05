#!/usr/bin/env bash
# Start/stop the two local llama.cpp servers this repo talks to:
#
#   chat  : Qwen3-8B            -> http://localhost:8080/v1  (config.CHAT_BASE_URL)
#   embed : Qwen3-Embedding-0.6B -> http://localhost:8081/v1  (config.EMBED_BASE_URL)
#
# Usage:
#   ./serve.sh start          # launch both (skips ones already running)
#   ./serve.sh stop           # stop both
#   ./serve.sh status         # health + PIDs
#   ./serve.sh logs [chat|embed]
#
# Ports/models are overridable: CHAT_PORT=9090 ./serve.sh start
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$REPO_DIR/.serve"           # pidfiles + logs (gitignored)
mkdir -p "$RUN_DIR"

CHAT_MODEL_PATH="${CHAT_MODEL_PATH:-$REPO_DIR/models/Qwen3-8B-Q8_0.gguf}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-$REPO_DIR/models/Qwen3-Embedding-0.6B-Q8_0.gguf}"
CHAT_PORT="${CHAT_PORT:-8080}"
EMBED_PORT="${EMBED_PORT:-8081}"
CHAT_CTX="${CHAT_CTX:-16384}"        # room for k=12 experiments in M3
NGL="${NGL:-99}"                     # offload everything to Metal

start_one() {  # name port pidfile logfile args...
    local name=$1 port=$2; shift 2
    local pidfile="$RUN_DIR/$name.pid" logfile="$RUN_DIR/$name.log"

    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "[$name] already running (pid $(cat "$pidfile"), port $port)"
        return
    fi

    echo "[$name] starting on port $port ..."
    nohup llama-server --port "$port" -ngl "$NGL" "$@" >"$logfile" 2>&1 &
    echo $! > "$pidfile"

    # llama-server answers /health with 503 while loading, 200 when ready
    for _ in $(seq 1 120); do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "[$name] ready: http://localhost:$port/v1"
            return
        fi
        if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            echo "[$name] DIED during startup -- last log lines:" >&2
            tail -n 20 "$logfile" >&2
            rm -f "$pidfile"
            return 1
        fi
        sleep 1
    done
    echo "[$name] still not healthy after 120s; check $logfile" >&2
    return 1
}

stop_one() {  # name
    local pidfile="$RUN_DIR/$1.pid"
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        kill "$(cat "$pidfile")"
        echo "[$1] stopped (pid $(cat "$pidfile"))"
    else
        echo "[$1] not running"
    fi
    rm -f "$pidfile"
}

status_one() {  # name port
    local pidfile="$RUN_DIR/$1.pid" state="stopped"
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        if curl -sf "http://localhost:$2/health" >/dev/null 2>&1; then
            state="ready (pid $(cat "$pidfile"))"
        else
            state="loading (pid $(cat "$pidfile"))"
        fi
    fi
    echo "[$1] port $2: $state"
}

case "${1:-start}" in
    start)
        # Embedding model first: it's small, loads in seconds, and rag.py needs
        # it before anything else. Qwen3-Embedding requires last-token pooling.
        start_one embed "$EMBED_PORT" \
            -m "$EMBED_MODEL_PATH" --embeddings --pooling last -c 2048 -b 2048 --ubatch-size 2048
        start_one chat "$CHAT_PORT" \
            -m "$CHAT_MODEL_PATH" -c "$CHAT_CTX" --jinja
        echo
        echo "rag.py defaults already point here; run: python rag.py"
        ;;
    stop)    stop_one chat; stop_one embed ;;
    status)  status_one chat "$CHAT_PORT"; status_one embed "$EMBED_PORT" ;;
    logs)    tail -f "$RUN_DIR/${2:-chat}.log" ;;
    *)       echo "usage: $0 {start|stop|status|logs [chat|embed]}" >&2; exit 1 ;;
esac
