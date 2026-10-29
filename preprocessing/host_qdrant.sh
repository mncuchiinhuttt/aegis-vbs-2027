#!/bin/bash
# Shell script to start Qdrant self-hosted server
# It will use Docker if available, otherwise it automatically downloads
# the precompiled standalone Qdrant binary for the local OS/Arch and runs it.

set -eu

QDRANT_VERSION="v1.10.1"
PORT=6333
GRPC_PORT=6334

echo "=== Checking host environment for Qdrant self-hosting ==="

if command -v docker-compose &> /dev/null || (command -v docker &> /dev/null && docker compose version &> /dev/null); then
    echo "[INFO] Docker Compose detected. Starting Qdrant via Docker..."
    if command -v docker-compose &> /dev/null; then
        COMPOSE=(docker-compose)
    else
        COMPOSE=(docker compose)
    fi
    if "${COMPOSE[@]}" up -d; then
        echo "=== Qdrant started successfully via Docker ==="
        echo "Access Dashboard at: http://localhost:$PORT/dashboard"
        exit 0
    fi
    echo "[WARNING] Docker Compose failed (is the Docker daemon running?). Falling back to the standalone binary."
elif command -v docker &> /dev/null; then
    echo "[INFO] Docker detected (without Compose). Starting Qdrant container..."
    # Ports are published on localhost only: Qdrant runs without authentication here.
    if docker run -d --name qdrant_server \
        -p 127.0.0.1:$PORT:6333 \
        -p 127.0.0.1:$GRPC_PORT:6334 \
        -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
        --restart always \
        -e QDRANT__SERVICE__ENABLE_STATIC_CONTENT=true \
        qdrant/qdrant:${QDRANT_VERSION}; then
        echo "=== Qdrant started successfully via Docker ==="
        echo "Access Dashboard at: http://localhost:$PORT/dashboard"
        exit 0
    fi
    echo "[WARNING] Docker failed (is the Docker daemon running?). Falling back to the standalone binary."
fi

# Docker is unavailable or failed. Proceed with the standalone binary.
echo "[INFO] Preparing the standalone Qdrant binary..."

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

DOWNLOAD_NAME=""
if [ "$OS" = "darwin" ]; then
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        DOWNLOAD_NAME="qdrant-aarch64-apple-darwin.tar.gz"
    else
        DOWNLOAD_NAME="qdrant-x86_64-apple-darwin.tar.gz"
    fi
elif [ "$OS" = "linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        DOWNLOAD_NAME="qdrant-x86_64-unknown-linux-gnu.tar.gz"
    else
        echo "[ERROR] Unsupported Linux architecture: $ARCH. Please install Docker to host Qdrant."
        exit 1
    fi
else
    echo "[ERROR] Unsupported OS: $OS. Please install Docker to host Qdrant."
    exit 1
fi

DOWNLOAD_URL="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/${DOWNLOAD_NAME}"
BIN_DIR="./qdrant_bin"
mkdir -p "$BIN_DIR"
mkdir -p "./qdrant_storage"

if [ -x "${BIN_DIR}/qdrant" ]; then
    echo "[INFO] Found existing Qdrant binary at ${BIN_DIR}/qdrant. Skipping download."
else
    echo "Detected OS: $OS, Arch: $ARCH"
    echo "Downloading Qdrant binary from: $DOWNLOAD_URL"

    # Download binary
    if command -v curl &> /dev/null; then
        curl --fail --location "$DOWNLOAD_URL" -o "${BIN_DIR}/qdrant.tar.gz"
    elif command -v wget &> /dev/null; then
        wget "$DOWNLOAD_URL" -O "${BIN_DIR}/qdrant.tar.gz"
    else
        echo "[ERROR] curl or wget is required to download the binary. Please install either tool or run with Docker."
        exit 1
    fi

    # Extract binary
    echo "Extracting Qdrant binary..."
    tar -xzf "${BIN_DIR}/qdrant.tar.gz" -C "$BIN_DIR"
    rm "${BIN_DIR}/qdrant.tar.gz"
    chmod +x "${BIN_DIR}/qdrant"
fi

# Run local binary
echo "Starting Qdrant standalone binary in the background..."
# Create a config directory/file if needed, or run with env overrides
export QDRANT__STORAGE__STORAGE_PATH="./qdrant_storage"
export QDRANT__SERVICE__HOST="127.0.0.1"
export QDRANT__SERVICE__HTTP_PORT=$PORT
export QDRANT__SERVICE__GRPC_PORT=$GRPC_PORT
export QDRANT__SERVICE__ENABLE_STATIC_CONTENT="true"

nohup "${BIN_DIR}/qdrant" > qdrant_local.log 2>&1 &
qdrant_pid=$!
for attempt in {1..30}; do
    if ! kill -0 "$qdrant_pid" 2>/dev/null; then
        echo "[ERROR] Qdrant exited during startup. See qdrant_local.log." >&2
        exit 1
    fi
    if { command -v curl &> /dev/null && curl --fail --silent --max-time 1 "http://127.0.0.1:$PORT/readyz" >/dev/null; } ||
       { command -v wget &> /dev/null && wget --quiet --timeout=1 --tries=1 --output-document=- "http://127.0.0.1:$PORT/readyz" >/dev/null; }; then
        echo "=== Qdrant is ready on localhost (PID $qdrant_pid) ==="
        echo "Logs: qdrant_local.log"
        echo "Dashboard: http://localhost:$PORT/dashboard"
        exit 0
    fi
    sleep 1
done
kill "$qdrant_pid" 2>/dev/null || true
echo "[ERROR] Qdrant did not become ready within 30 seconds. See qdrant_local.log." >&2
exit 1
